# SAT + UF integration (DPLL(T) via IPASIR-UP)

## Goal

Until now the C++ solver assumed top-level QF_UF: every assertion was
either `(= a b)` or `(not (= a b))`. There was no Boolean structure — `or`,
`and`, `ite`, etc. were not supported. This document describes the change
that turns the solver into a real DPLL(T) procedure: CaDiCaL drives the
search over the Boolean skeleton, and our existing `EGraph` plays the role
of the QF_UF theory propagator behind the IPASIR-UP API.

## Layering

```
                   ┌──────────────────────────────────────────────┐
                   │                solve_smt(...)                │
                   │   parse  →  Tseitin CNF  →  CaDiCaL.solve()  │
                   └────────────────────┬─────────────────────────┘
                                        │ IPASIR-UP callbacks
                                        ▼
              ┌──────────────────────────────────────────────────┐
              │                UfPropagator                      │
              │   atom var ↔ (Id, Id)   |   snapshot stack       │
              │   merge / check / explain                        │
              └────────────────────┬─────────────────────────────┘
                                   ▼
                              EGraph (existing)
```

The Boolean side (CaDiCaL + Tseitin) is a fresh layer. The QF_UF side
(`EGraph`, `ConcurrentUnionFind`, `parallel_rebuild`) is unchanged — we
explicitly **do not** modify the congruence-closure algorithms. The only
new thing the e-graph has to support is **state restoration on backtrack**
(snapshot/restore), described below.

## Pipeline, in code

`pe::solve_with_mode(input, parallel)` becomes the wrapper for both modes:

1. `parse_smtlib(input)` — overhauled parser that handles the QF_UF
   Boolean fragment (`and`, `or`, `not`, `=>`, `xor`, `ite`-over-Bool,
   `distinct`, plus Bool atoms). Term::Kind grows to cover these.
2. `cnf_encode(script)` — walk every assertion, give each unique
   `(= a b)` syntactic atom a fresh CaDiCaL variable, then Tseitin-encode
   the Boolean skeleton above the atoms. Result:
     - `vector<Clause>` for CaDiCaL
     - `unordered_map<atom_var, (Id, Id)>` ("var → theory atom")
     - `vector<atom_var>` of variables that *are* theory atoms (the rest
       are aux Tseitin vars, which the propagator ignores)
3. `UfPropagator prop(eg, atom_map, parallel)` — implements the IPASIR-UP
   `ExternalPropagator` interface.
4. `solver.connect_external_propagator(&prop)` and `solver.solve()` —
   CaDiCaL searches; whenever it assigns a literal, propagator gets a
   callback and may decide to merge in the e-graph or signal a conflict.

## Theory atom model

Every distinct equality atom `(= t1 t2)` that appears anywhere in the
formula is hashed canonically (using e-class ids of `t1` and `t2`, sorted
to ignore symmetry) and assigned a single CaDiCaL variable `v`:

- `v = true`  ⇒ assert `t1 = t2`     ⇒ in propagator: `eg.merge(a, b)`
- `v = false` ⇒ assert `t1 ≠ t2`    ⇒ in propagator: record disequality;
   conflict if `eg.find(a) == eg.find(b)`

Boolean structure (`or`, `and`, `ite`-over-Bool, `=>`, `xor`) only touches
`v` indirectly via Tseitin clauses. The propagator never sees aux vars.

`distinct(x1, ..., xk)` desugars at clausification time into the
conjunction of all pairwise `(not (= xi xj))` atoms. Bool-typed
uninterpreted constants get one fresh CaDiCaL var each (no theory
meaning).

## Tseitin clausification (vs naive distributive expansion)

We do Tseitin: for every internal Boolean connective, introduce a fresh
aux variable `x` and clauses defining `x ↔ subformula`. This is
**equisatisfiable**, not logically equivalent — it has more variables
than the original formula but only O(formula size) clauses, vs the
exponential blowup of naive CNF. Standard for every modern SMT solver.

| connective       | aux var | clauses                                                       |
|------------------|---------|---------------------------------------------------------------|
| `x = a ∧ b`      | x       | (¬x ∨ a), (¬x ∨ b), (x ∨ ¬a ∨ ¬b)                             |
| `x = a ∨ b`      | x       | (¬x ∨ a ∨ b), (x ∨ ¬a), (x ∨ ¬b)                              |
| `x = ¬a`         | (none)  | re-use `¬a` directly, no aux var                              |
| `x = a → b`      | x       | (¬x ∨ ¬a ∨ b), (x ∨ a), (x ∨ ¬b)                              |
| `x = a ↔ b`      | x       | 4-clause biconditional                                         |
| `x = ite(c,a,b)` | x       | (¬x ∨ ¬c ∨ a), (¬x ∨ c ∨ b), (x ∨ ¬c ∨ ¬a), (x ∨ c ∨ ¬b)      |

Top-level `(assert φ)` is encoded as `cnf(φ) ∧ unit_clause(top_var(φ))`.

## IPASIR-UP propagator contract

The propagator implements CaDiCaL's `ExternalPropagator` interface. Key
callbacks we use:

- `notify_assignment(lit, is_fixed)` — CaDiCaL assigns a literal. If `lit`
  is a theory atom var, push the implied `merge` (or disequality
  observation) into a per-level work buffer. **No e-graph mutation yet.**
- `notify_new_decision_level()` — CaDiCaL just decided. Snapshot the
  current `EGraph` to the level stack before any further propagation.
- `notify_backtrack(level)` — CaDiCaL is jumping back. Pop snapshots above
  `level` and replace `current` with the snapshot at `level`. Drop any
  buffered work associated with discarded levels.
- `cb_check_found_model(model)` — at the end of search, CaDiCaL has a
  full assignment. Apply all theory atoms, run congruence closure, and
  return false if any disequality has the two sides in the same e-class.
  (This is the "trailing check" that lets us be lazy about per-step
  propagation in v1.)
- `cb_has_external_clause` / `cb_add_external_clause_lit` — when we
  detect a theory conflict, return the conflict clause. v1 returns the
  full assignment of theory atoms negated — correct, but not minimal.

What we do **not** implement in v1:

- `cb_propagate` — pushing theory-implied literals back to CaDiCaL.
  Without this, the SAT solver won't learn theory facts directly; it'll
  re-discover them by assignment + check. Slower but correct.
- `cb_add_reason_clause_lit` — minimal explanation for an implied
  literal. Skipped because v1 doesn't propagate.

## Snapshot strategy (v1: clone-per-level)

The simplest correct strategy: **before every new decision level, deep-clone
the e-graph; on backtrack, drop the current and replace it with the snapshot
at the target level.**

Concretely, `UfPropagator` owns:

```cpp
std::vector<std::unique_ptr<EGraph>> snapshots;  // size == decision_level
std::unique_ptr<EGraph>               current;
```

`EGraph::clone()` is a new method that constructs a fresh `EGraph` of the
same capacity and replays the public mutating history (recorded `add` and
`merge` calls). This is needed because `EGraph` contains `std::atomic<bool>`
and `std::atomic<uint32_t>` arrays, which are non-copyable and non-movable —
no `= default` copy ctor will work.

**Cost.** O(history size) per clone. For a formula with N theory merges and
search depth D, total work is O(N·D) in the worst case — can blow up
catastrophically on deep search trees. For correctness this is fine; for
performance it isn't. See "Future work" below.

**Why this is OK as v1.** Most QF_UF benchmarks have small Boolean
structure (the SMT-COMP `eq/`, `NEQ/`, `PEQ/`, `SEQ/` corpora) and decision
depth is bounded by the number of theory atoms. The clone overhead lives
in the *search* phase, not the closure phase, so it doesn't interact with
parallelization. We can replace it without touching `parallel_rebuild`.

## How this interacts with parallel mode

CaDiCaL is single-threaded. The e-graph it talks to (via the propagator)
is whichever the user picked: `EGraph(capacity, parallel=true)` exposes
the parallel rebuild paths; `parallel=false` keeps Nelson. The propagator
calls `eg.merge(a, b)` per assignment and `eg.rebuild()` (sequential) or
`eg.parallel_rebuild()` at the end of each round.

For workloads where the rebuild dominates and the SAT skeleton is small,
the parallel rebuild still pays off. For SAT-heavy workloads with many
small theory checks, the per-rebuild parallelism overhead may dominate —
that's expected and consistent with what we already see on small inputs.

## Future work

These are out of scope for v1, but the design above leaves room for them:

1. **Trail-based backtracking.** Replace clone-per-level with an undo
   trail. Every CAS in `ConcurrentUnionFind` records (slot, old_value);
   `parent_index_` and `nodes_` get append-only trails. On backtrack,
   rewind the trail to the saved cursor. Cost per backtrack: O(work since
   that level), not O(state size). Standard DPLL(T) implementation.

2. **Theory propagation back to SAT.** Implement `cb_propagate`: after a
   merge, scan registered disequality atoms whose endpoints are now in
   the same class — those vars are forced false; push them back to
   CaDiCaL. Big SAT-side speedup; needs reason clauses (small congruence
   proof) for `cb_add_reason_clause_lit`.

3. **Parallel backtracking e-graph.** With trail-based undo, each branch
   in the SAT search could be explored in parallel with its own e-graph
   view. Open research direction; left for later.

4. **Watched literal compression for theory atoms.** Many duplicate
   `(= t1 t2)` atoms appear in QF_UF benchmarks — currently each gets
   its own var. Canonicalizing in the parser first would shrink the
   formula significantly.

5. **Distinct optimization.** `(distinct x1 ... xk)` currently expands
   to k(k−1)/2 atoms. For large k, a single specialized constraint
   (color-the-vertices) is dramatically cheaper.

## Files

New (added by this change):

- `cpp/include/parallel_egraph/sat_solve.hpp`
- `cpp/src/sat_solve.cpp` — Tseitin + IPASIR-UP propagator + driver +
  ITE lifting pass (see "Implementation notes" below)
- `cpp/SAT_INTEGRATION.md` — this doc
- `cpp/tests/bool/*.smt2` — 17 Boolean-fragment regression cases
  (one per connective, both `_sat` and `_unsat` variants)
- `cpp/tests/bool_regression_test.cpp` — driver for the above
- CaDiCaL FetchContent block in `cpp/CMakeLists.txt`

Modified:

- `cpp/include/parallel_egraph/smtlib.hpp` — `Term::Kind` gains the new
  Boolean kinds; `is_pure_conjunctive` predicate exposed.
- `cpp/src/smtlib.cpp` — parser handles the full Boolean fragment;
  also supports SMT-LIB `let` via parse-time substitution (see notes).
- `cpp/include/parallel_egraph/egraph.hpp` — `EGraph::clone()`.
- `cpp/src/egraph.cpp` — `EGraph::clone()` impl.
- `cpp/include/parallel_egraph/unionfind.hpp` and `cpp/src/unionfind.cpp`
  — `ConcurrentUnionFind::copy_state_from(other)` (atomics aren't copy-
  assignable, so cloning needs an explicit element-wise mirror).
- `cpp/src/solve.cpp` — `solve_with_mode` dispatches into the SAT-driven
  pipeline. Old top-level-only path stays as a fast-path when the
  formula has no Boolean structure (every assert is `=` or `not =`
  *with pure UF args on both sides* — see "implementation notes" below).
- `cpp/CMakeLists.txt` — adds C language, `kitten.c` (CaDiCaL's sub-
  solver), and several CaDiCaL build patches (see notes).

`cpp/src/main.cpp` is **not modified**: the SAT pipeline is invoked
transparently through the unchanged `solve_with_mode`/`solve_timed` API.

## Implementation notes (post-v1 design)

This section records decisions made during implementation that diverged
from or extended the original design above.

### ITE lifting (UF-typed ite)

The base Tseitin encoder only handles **Bool-typed** `ite`. UF-typed
`ite` like `(= u (ite c v w))` requires lifting the `ite` out to Boolean
position before clausification:

```text
C[ite c x y]   →   (ite c C[x] C[y])     (if C is Bool-typed context)
                →   (and (=> c C[x]) (=> (not c) C[y]))   (otherwise)
```

The implementation in `sat_solve.cpp` is a single recursive walk
(`split_uf` / `lift_in_bool`) that returns each UF subterm as a list of
`(guard, value)` pairs. An `(= a b)` parent then becomes the disjunction
of `(guard_a_i ∧ guard_b_j ∧ (val_a_i = val_b_j))` over all index pairs.
Cartesian-product expansion handles `ite`s buried inside UF `App`s.
Cost: exponential in the number of nested `ite`s within a single
equality, but real benchmarks tend to nest shallowly.

### `is_pure_assertion` requires pure UF args

The dispatcher in `solve.cpp` routes pure-conjunctive QF_UF formulas to
the legacy fast path. The legacy `add_term()` accepts only `Const` and
`App` — so an assertion like `(= u (ite c v w))` (which is syntactically
an `Eq` at top) had to be classified *non-pure* to avoid a segfault when
it reached `add_term`. The current rule:

> An assertion is pure iff it is `(= a b)` (binary) or `(not (= a b))`
> AND both sides are recursively pure UF terms (no `ite`, no Bool ops,
> no `let` references that resolved to anything but UF).

### `let` support via parse-time substitution

QF_UF benchmarks routinely use `let` to share subexpressions. Rather
than introduce a new `Term::Kind`, the parser inlines bound names as it
encounters them: a stack of `(name, Term)` frames is pushed for each
`(let ((x e1) (y e2) ...) body)` and popped after the body is parsed.
SMT-LIB `let` is parallel — `e2` is parsed in the *outer* scope, before
the frame is pushed.

Trade-off: each occurrence of a bound name produces a deep copy of its
RHS in the AST. For benchmarks with many uses of a deeply-nested
binding (e.g. the SMT-COMP `PEQ018_size7.smt2` where one assertion
`let`s ~250 names sharing common subterms), this fans out into a tree
of millions of nodes and the SAT path stalls. A future optimization
would be to keep `let` as an explicit AST node and let the encoder
hashcons duplicates downstream — but for the benchmarks we routinely
care about, parse-time substitution is fine.

### CaDiCaL build glue

CaDiCaL ships with `./configure` + `makefile.in`, not CMake. We pull it
via `FetchContent_MakeAvailable` and compile its sources directly. Three
adjustments were necessary:

1. **`build.hpp` synthesis.** CaDiCaL's `version.cpp` includes a
   generated `build.hpp` defining `VERSION`, `IDENTIFIER`, etc. Our
   CMakeLists writes one with the values from `VERSION` and CMake
   variables.
2. **macOS `closefrom(3)` patch.** `file.cpp` calls `::closefrom(3)`,
   which is Linux/BSD-only. CMake patches the call site at configure
   time (idempotent — re-running CMake just overwrites with the same
   patch).
3. **`kitten.c`.** CaDiCaL's `sweep` preprocessing uses kitten, a sister
   solver written in C. The CMakeLists globs both `*.cpp` and `*.c`
   from CaDiCaL's `src/` and the project enables both `C` and `CXX`.
4. **Disable `factor` preprocessing.** CaDiCaL 3.x's `factor` pass
   asserts that every variable is pre-declared. Setting
   `solver.set("factor", 0)` skips that check; we don't gain anything
   from `factor` on the small CNFs the encoder produces.

### EGraph cloning

`EGraph` contains `parlay::sequence<std::atomic<bool>>` (per-class
"changed" flags) and `ConcurrentUnionFind` (vector of `atomic<u32>`).
Atomics are non-copy-assignable, so `EGraph::clone()` cannot use the
default copy constructor. Implementation:

- `ConcurrentUnionFind::copy_state_from(other)` — element-wise atomic
  load/store of `data_`, asserting `capacity` matches.
- `EGraph::clone()` — constructs a fresh `EGraph` of the same capacity,
  calls `copy_state_from`, copy-assigns the non-atomic containers
  (`nodes_`, `parent_index_`, `classes_`, `parents_`, `worklist_`,
  `hashcons_`), then mirrors the `changed_` atomics element-wise.

Returns a `unique_ptr<EGraph>` because `EGraph` is non-movable.

### Tests

The `bool_regression` ctest target runs every `.smt2` in
`cpp/tests/bool/` in both sequential and parallel modes, exercising
each Boolean connective with matched `_sat`/`_unsat` cases:

- `or`, `and`, `=>`, `xor` — standard Tseitin templates.
- `ite_*` — Bool-typed and UF-typed (the latter exercises the lifting
  pass; UF-typed ites buried inside `App` exercise Cartesian expansion).
- `distinct` — desugared to ∧ of pairwise disequalities.

17 cases × 2 modes = 34 sub-tests. Run with `ctest --test-dir cpp/build`.
