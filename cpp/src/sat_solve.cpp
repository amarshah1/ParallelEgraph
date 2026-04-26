// SAT-driven QF_UF solver. See cpp/SAT_INTEGRATION.md for design.
//
// Three concerns in this file:
//   1. cnf_encode: walk the parsed Term tree, give every (= a b) atom a
//      CaDiCaL variable, and Tseitin-encode the Boolean structure above.
//      Theory atoms are looked up in an EGraph built up-front (one pass
//      over the AST adds every UF subterm).
//   2. UfPropagator: implements CaDiCaL::ExternalPropagator. Tracks
//      per-decision-level snapshots of the e-graph and replays merges
//      as theory atoms are assigned. v1 strategy is "lazy plus final
//      check" — we only fire conflicts when CaDiCaL hands us a complete
//      model via cb_check_found_model.
//   3. sat_solve_timed: glue. Build EGraph + atom map, encode CNF,
//      hand it all to CaDiCaL, return SolveResult.

#include "parallel_egraph/sat_solve.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <cadical.hpp>

#include "parallel_egraph/egraph.hpp"
#include "parallel_egraph/smtlib.hpp"

namespace pe {

namespace {

using Lit = int;  // CaDiCaL convention: positive = var, negative = ¬var, 0 = end

// Canonical key for an equality atom: (lo_id, hi_id) with lo <= hi to
// dedup symmetric eqs.
struct EqKey {
  Id a, b;
  bool operator==(const EqKey& o) const { return a == o.a && b == o.b; }
};
struct EqKeyHash {
  std::size_t operator()(const EqKey& k) const noexcept {
    // Simple 64-bit splitmix on packed (a,b).
    std::uint64_t x = (static_cast<std::uint64_t>(k.a) << 32) |
                       static_cast<std::uint64_t>(k.b);
    x ^= x >> 33;
    x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33;
    x *= 0xc4ceb9fe1a85ec53ULL;
    x ^= x >> 33;
    return static_cast<std::size_t>(x);
  }
};

// ---------------------------------------------------------------------------
// add_uf_term — port of solve.cpp's add_term, lifted here for sat path use.
// Iterative; no stack overflow on deeply nested terms.
// ---------------------------------------------------------------------------
Id add_uf_term(EGraph& eg, const Term& term) {
  enum class WorkKind { Process, Build };
  struct Work {
    WorkKind kind;
    const Term* term;
    std::string op;
    std::size_t nargs;
  };
  std::vector<Work> stack;
  std::vector<Id> results;
  stack.push_back({WorkKind::Process, &term, {}, 0});
  while (!stack.empty()) {
    Work w = std::move(stack.back());
    stack.pop_back();
    if (w.kind == WorkKind::Process) {
      const Term& t = *w.term;
      switch (t.kind) {
        case Term::Kind::Const: {
          ENode leaf;
          leaf.op = t.op;
          results.push_back(eg.add(std::move(leaf)));
          break;
        }
        case Term::Kind::App: {
          stack.push_back({WorkKind::Build, nullptr, t.op, t.args.size()});
          for (auto it = t.args.rbegin(); it != t.args.rend(); ++it) {
            stack.push_back({WorkKind::Process, &*it, {}, 0});
          }
          break;
        }
        default:
          throw std::runtime_error(
              "add_uf_term: only Const and App may appear in a UF term");
      }
    } else {
      std::size_t start = results.size() - w.nargs;
      std::vector<Id> children(results.begin() + start, results.end());
      results.erase(results.begin() + start, results.end());
      ENode node;
      node.op = std::move(w.op);
      node.children = std::move(children);
      results.push_back(eg.add(std::move(node)));
    }
  }
  return results.back();
}

// Recursively walks a Term tree and adds every UF subterm to `eg`. Boolean
// connectives are walked through but only their UF arguments end up in the
// e-graph. Used as a preflight pass before CNF encoding so we know the
// e-class id of every atom endpoint up front.
void preflight_add_uf(EGraph& eg, const Term& t) {
  switch (t.kind) {
    case Term::Kind::Const:
    case Term::Kind::App:
      add_uf_term(eg, t);
      break;
    case Term::Kind::Eq:
      add_uf_term(eg, t.args[0]);
      add_uf_term(eg, t.args[1]);
      break;
    case Term::Kind::Distinct:
      for (const auto& a : t.args) add_uf_term(eg, a);
      break;
    case Term::Kind::Not:
    case Term::Kind::And:
    case Term::Kind::Or:
    case Term::Kind::Implies:
    case Term::Kind::Xor:
    case Term::Kind::Ite:
      for (const auto& a : t.args) preflight_add_uf(eg, a);
      break;
    case Term::Kind::True:
    case Term::Kind::False:
      break;
  }
}

// Counts the number of distinct e-class IDs we'd allocate for `t`.
// Conservative upper bound (every subterm distinct) — used to size the
// fixed-cap union-find before construction.
std::size_t count_subterms_total(const Term& t) {
  std::size_t n = 1;
  for (const auto& a : t.args) n += count_subterms_total(a);
  return n;
}

// ---------------------------------------------------------------------------
// Encoder: builds CNF from the assertion list. Allocates a CaDiCaL var for
// each distinct theory atom and for each Tseitin aux gate.
// ---------------------------------------------------------------------------
class Encoder {
 public:
  Encoder(EGraph& eg) : eg_(eg) {}

  // Encode a single (assert φ): emits clauses for φ and asserts the top
  // var as a unit. Recurses through Boolean connectives, threading aux
  // vars through Tseitin templates.
  void encode_assertion(const Term& t) {
    Lit top = encode(t);
    add_clause({top});
  }

  // Returns the CaDiCaL variable count actually allocated.
  int num_vars() const { return next_var_ - 1; }

  // List of clauses, each terminated by 0.
  const std::vector<int>& cnf() const { return cnf_; }

  // Theory atom map: var → (a_id, b_id). Negative-polarity assignments
  // mean the disequality; positive means the equality.
  const std::unordered_map<int, std::pair<Id, Id>>& atom_endpoints() const {
    return atom_endpoints_;
  }
  // Reverse: tells us if a CaDiCaL var is one of our theory atoms.
  bool is_theory_var(int var) const {
    return atom_endpoints_.find(var) != atom_endpoints_.end();
  }

 private:
  // Mint a fresh CaDiCaL variable id.
  int new_var() { return next_var_++; }

  // Mint a Tseitin aux variable.
  int new_aux() { return new_var(); }

  // Get-or-create theory atom var for canonical (a,b). a,b are e-class
  // ids of the two sides; we sort them to canonicalize symmetric eqs.
  int atom_var_for(Id a, Id b) {
    Id lo = std::min(a, b), hi = std::max(a, b);
    EqKey key{lo, hi};
    auto it = atom_var_.find(key);
    if (it != atom_var_.end()) return it->second;
    int v = new_var();
    atom_var_.emplace(key, v);
    atom_endpoints_.emplace(v, std::make_pair(lo, hi));
    return v;
  }

  void add_clause(std::initializer_list<Lit> lits) {
    for (Lit l : lits) cnf_.push_back(l);
    cnf_.push_back(0);
  }
  void add_clause(const std::vector<Lit>& lits) {
    for (Lit l : lits) cnf_.push_back(l);
    cnf_.push_back(0);
  }

  // Returns a literal whose truth = truth of `t`. May allocate aux vars
  // and emit clauses. Memoizes shared subtrees by (kind, op, child-lits).
  Lit encode(const Term& t) {
    switch (t.kind) {
      case Term::Kind::True: {
        if (true_lit_ == 0) {
          true_lit_ = new_var();
          add_clause({true_lit_});
        }
        return true_lit_;
      }
      case Term::Kind::False: {
        if (true_lit_ == 0) {
          true_lit_ = new_var();
          add_clause({true_lit_});
        }
        return -true_lit_;
      }
      case Term::Kind::Const: {
        // A Bool-typed UF constant. Each distinct identifier gets a fresh
        // SAT var (no UF interaction since we never see it inside a UF
        // App in the QF_UF Boolean fragment we accept).
        auto it = bool_const_.find(t.op);
        if (it != bool_const_.end()) return it->second;
        int v = new_var();
        bool_const_.emplace(t.op, v);
        return v;
      }
      case Term::Kind::App: {
        // Bool-typed UF predicate (e.g. (p x)). We treat it as an opaque
        // atom: each distinct (op, child-eclass-ids) tuple gets a SAT var.
        // Theory consistency for these is not enforced in v1 — the only
        // uses we encounter in our test corpus are top-level booleans.
        // If two predicate calls have congruent arguments they should
        // share a var; we use the e-graph's add() to canonicalize.
        Id eid = add_uf_term(eg_, t);
        auto it = bool_app_.find(eid);
        if (it != bool_app_.end()) return it->second;
        int v = new_var();
        bool_app_.emplace(eid, v);
        return v;
      }
      case Term::Kind::Eq: {
        Id a = add_uf_term(eg_, t.args[0]);
        Id b = add_uf_term(eg_, t.args[1]);
        if (a == b) {
          // Reflexive: trivially true. Use the shared true literal.
          if (true_lit_ == 0) {
            true_lit_ = new_var();
            add_clause({true_lit_});
          }
          return true_lit_;
        }
        return atom_var_for(a, b);
      }
      case Term::Kind::Not: {
        return -encode(t.args[0]);
      }
      case Term::Kind::And: {
        // (and l1 l2 ... lk) → fresh x; clauses ¬x∨li (k of them) and
        // x ∨ ¬l1 ∨ ¬l2 ∨ ... ∨ ¬lk.
        if (t.args.empty()) {  // (and) = true
          if (true_lit_ == 0) {
            true_lit_ = new_var();
            add_clause({true_lit_});
          }
          return true_lit_;
        }
        if (t.args.size() == 1) return encode(t.args[0]);
        std::vector<Lit> lits;
        lits.reserve(t.args.size());
        for (const auto& a : t.args) lits.push_back(encode(a));
        int x = new_aux();
        std::vector<Lit> big = {x};
        for (Lit l : lits) {
          add_clause({-x, l});
          big.push_back(-l);
        }
        add_clause(big);
        return x;
      }
      case Term::Kind::Or: {
        if (t.args.empty()) {  // (or) = false
          if (true_lit_ == 0) {
            true_lit_ = new_var();
            add_clause({true_lit_});
          }
          return -true_lit_;
        }
        if (t.args.size() == 1) return encode(t.args[0]);
        std::vector<Lit> lits;
        lits.reserve(t.args.size());
        for (const auto& a : t.args) lits.push_back(encode(a));
        int x = new_aux();
        std::vector<Lit> big = {-x};
        for (Lit l : lits) {
          add_clause({x, -l});
          big.push_back(l);
        }
        add_clause(big);
        return x;
      }
      case Term::Kind::Implies: {
        // Right-associative: (=> a b c d) = (=> a (=> b (=> c d))).
        Lit acc = encode(t.args.back());
        for (auto it = t.args.rbegin() + 1; it != t.args.rend(); ++it) {
          Lit l = encode(*it);
          // x ↔ (l → acc)  =  x ↔ (¬l ∨ acc)
          int x = new_aux();
          add_clause({-x, -l, acc});
          add_clause({x, l});
          add_clause({x, -acc});
          acc = x;
        }
        return acc;
      }
      case Term::Kind::Xor: {
        // Pairwise xor: (xor a b c) = (xor (xor a b) c). For each pair
        // x ↔ (a XOR b): four clauses.
        Lit acc = encode(t.args[0]);
        for (std::size_t i = 1; i < t.args.size(); ++i) {
          Lit l = encode(t.args[i]);
          int x = new_aux();
          // x ↔ (acc XOR l)
          add_clause({-x, acc, l});
          add_clause({-x, -acc, -l});
          add_clause({x, -acc, l});
          add_clause({x, acc, -l});
          acc = x;
        }
        return acc;
      }
      case Term::Kind::Ite: {
        Lit c = encode(t.args[0]);
        Lit a = encode(t.args[1]);
        Lit b = encode(t.args[2]);
        // x ↔ ite(c, a, b)
        //  c → (x ↔ a):   ¬c ∨ ¬x ∨ a   ;   ¬c ∨ x ∨ ¬a
        // ¬c → (x ↔ b):    c ∨ ¬x ∨ b   ;    c ∨ x ∨ ¬b
        int x = new_aux();
        add_clause({-c, -x, a});
        add_clause({-c, x, -a});
        add_clause({c, -x, b});
        add_clause({c, x, -b});
        return x;
      }
      case Term::Kind::Distinct: {
        // (distinct x1 ... xk)  ≡  ∧_{i<j} ¬(xi = xj). In CNF: an aux
        // var = ∧ of all pairwise diseqs.
        std::vector<Lit> diseqs;
        for (std::size_t i = 0; i < t.args.size(); ++i) {
          Id xi = add_uf_term(eg_, t.args[i]);
          for (std::size_t j = i + 1; j < t.args.size(); ++j) {
            Id xj = add_uf_term(eg_, t.args[j]);
            if (xi == xj) {
              // Two syntactically congruent terms in distinct(): formula
              // is unsat (constant false). Emit a forced-false literal.
              if (true_lit_ == 0) {
                true_lit_ = new_var();
                add_clause({true_lit_});
              }
              return -true_lit_;
            }
            diseqs.push_back(-atom_var_for(xi, xj));
          }
        }
        // distinct = and(diseqs). Use the And template.
        if (diseqs.size() == 1) return diseqs[0];
        int x = new_aux();
        std::vector<Lit> big = {x};
        for (Lit l : diseqs) {
          add_clause({-x, l});
          big.push_back(-l);
        }
        add_clause(big);
        return x;
      }
    }
    throw std::runtime_error("Encoder::encode: unhandled Term::Kind");
  }

  EGraph& eg_;
  int next_var_ = 1;                    // CaDiCaL vars are 1-indexed
  int true_lit_ = 0;                    // shared "true" sentinel; lazy alloc
  std::vector<int> cnf_;                // 0-terminated clauses
  std::unordered_map<EqKey, int, EqKeyHash> atom_var_;
  std::unordered_map<int, std::pair<Id, Id>> atom_endpoints_;
  std::unordered_map<std::string, int> bool_const_;
  std::unordered_map<Id, int> bool_app_;
};

// ---------------------------------------------------------------------------
// UfPropagator — IPASIR-UP wrapper around EGraph with snapshot stack.
// ---------------------------------------------------------------------------
class UfPropagator : public CaDiCaL::ExternalPropagator {
 public:
  UfPropagator(std::unique_ptr<EGraph> root_eg,
               std::unordered_map<int, std::pair<Id, Id>> atoms,
               bool parallel)
      : current_(std::move(root_eg)),
        atom_endpoints_(std::move(atoms)),
        parallel_(parallel) {
    // Snapshot at level 0 is the propagator's "ground state" — what we
    // restore to whenever CaDiCaL backtracks all the way out.
    snapshots_.push_back(current_->clone());
  }

  // CaDiCaL pushes a vector of newly-true literals (positive = atom asserted,
  // negative = atom negated).  We translate theory atoms into either
  // merges or pending disequality observations at the current level.
  void notify_assignment(const std::vector<int>& lits) override {
    for (int lit : lits) apply_lit(lit);
  }

  // (We don't override notify_fixed_assignment in v1: that listener is on a
  // separate FixedAssignmentListener interface and is an optimization. Fixed
  // assignments still arrive via notify_assignment in batch form.)

  void notify_new_decision_level() override {
    // CaDiCaL is about to make a new decision. Snapshot the e-graph as
    // it stands NOW so a future backtrack to this level can restore it.
    snapshots_.push_back(current_->clone());
  }

  void notify_backtrack(std::size_t level) override {
    // CaDiCaL wants to be back at decision level `level`. Drop any
    // higher-level snapshots and replace `current` with the saved one.
    while (snapshots_.size() > level + 1) snapshots_.pop_back();
    current_ = snapshots_.back()->clone();
    // Discard any pending disequalities collected above `level`.
    while (diseq_levels_.size() > snapshots_.size() - 1) {
      diseq_levels_.pop_back();
    }
  }

  bool cb_check_found_model(const std::vector<int>& model) override {
    // CaDiCaL has a complete assignment. Apply every theory atom in the
    // model into a fresh e-graph (so we don't mutate the level-stack
    // snapshots) and run congruence closure. If any disequality has
    // both sides in the same e-class, the model is theory-inconsistent
    // — we return false and produce a conflict clause.
    auto check_eg = current_->clone();
    std::vector<std::pair<Id, Id>> pos_eqs;
    std::vector<std::pair<Id, Id>> neg_eqs;
    for (int lit : model) {
      int v = std::abs(lit);
      auto it = atom_endpoints_.find(v);
      if (it == atom_endpoints_.end()) continue;
      Id a = it->second.first, b = it->second.second;
      if (lit > 0) pos_eqs.emplace_back(a, b);
      else         neg_eqs.emplace_back(a, b);
    }
    for (auto& [a, b] : pos_eqs) check_eg->merge(a, b);
    if (parallel_) check_eg->parallel_rebuild();
    else           check_eg->rebuild();
    for (auto& [a, b] : neg_eqs) {
      if (check_eg->equiv(a, b)) {
        // Build conflict clause: negate every theory literal in the
        // current model. A trivially-correct (but redundant) reason —
        // the SAT solver will minimize.
        conflict_clause_.clear();
        for (int lit : model) {
          int v = std::abs(lit);
          if (atom_endpoints_.find(v) != atom_endpoints_.end()) {
            conflict_clause_.push_back(-lit);
          }
        }
        conflict_clause_.push_back(0);
        conflict_pending_ = true;
        return false;
      }
    }
    return true;
  }

  // We do not implement theory-directed propagation in v1.
  int cb_decide() override { return 0; }
  int cb_propagate() override { return 0; }
  int cb_add_reason_clause_lit(int /*propagated_lit*/) override { return 0; }

  bool cb_has_external_clause(bool& is_forgettable) override {
    is_forgettable = false;
    return conflict_pending_;
  }

  int cb_add_external_clause_lit() override {
    if (!conflict_pending_) return 0;
    if (conflict_emit_idx_ >= conflict_clause_.size()) {
      // Whole clause delivered — clear state.
      conflict_pending_ = false;
      conflict_clause_.clear();
      conflict_emit_idx_ = 0;
      return 0;
    }
    int lit = conflict_clause_[conflict_emit_idx_++];
    if (lit == 0) {
      // Clause terminator — also clears state for next time.
      conflict_pending_ = false;
      conflict_clause_.clear();
      conflict_emit_idx_ = 0;
    }
    return lit;
  }

 private:
  void apply_lit(int lit) {
    int v = std::abs(lit);
    auto it = atom_endpoints_.find(v);
    if (it == atom_endpoints_.end()) return;  // aux Tseitin var
    Id a = it->second.first, b = it->second.second;
    if (lit > 0) {
      // Asserted equality: fold it into the current e-graph.
      current_->merge(a, b);
    } else {
      // Asserted disequality: stash for the final consistency check.
      while (diseq_levels_.size() < snapshots_.size()) {
        diseq_levels_.emplace_back();
      }
      diseq_levels_.back().emplace_back(a, b);
    }
  }

  std::unique_ptr<EGraph> current_;
  std::vector<std::unique_ptr<EGraph>> snapshots_;
  std::vector<std::vector<std::pair<Id, Id>>> diseq_levels_;
  std::unordered_map<int, std::pair<Id, Id>> atom_endpoints_;
  bool parallel_;

  // Pending conflict clause to deliver to CaDiCaL via cb_add_external_clause_lit.
  std::vector<int> conflict_clause_;
  std::size_t conflict_emit_idx_ = 0;
  bool conflict_pending_ = false;
};

double secs_since(std::chrono::steady_clock::time_point t) {
  using namespace std::chrono;
  return duration<double>(steady_clock::now() - t).count();
}

}  // namespace

std::pair<SolveResult, SolveTimings> sat_solve_timed(const std::string& input,
                                                     bool parallel) {
  using clk = std::chrono::steady_clock;
  SolveTimings timings;
  auto total_start = clk::now();

  // ---- Parse ----
  auto parse_start = clk::now();
  Script script = parse_smtlib(input);
  std::vector<const Term*> assertions;
  for (const auto& c : script.commands) {
    if (c.kind == Command::Kind::Assert) assertions.push_back(&c.term);
  }
  timings.parse_s = secs_since(parse_start);

  // ---- Build EGraph ----
  // Capacity: every subterm could become an e-class. Sum across all
  // assertions for a safe upper bound.
  std::size_t capacity = 0;
  for (const Term* t : assertions) capacity += count_subterms_total(*t);
  if (capacity == 0) capacity = 1;
  auto eg = std::make_unique<EGraph>(capacity, parallel);

  auto build_start = clk::now();
  // Preflight: walk each assertion and add every UF subterm to the e-graph.
  // This guarantees that by the time the encoder runs, every endpoint of
  // every (= ti tj) atom already has a stable e-class id.
  for (const Term* t : assertions) preflight_add_uf(*eg, *t);

  // ---- Encode CNF ----
  Encoder enc(*eg);
  for (const Term* t : assertions) enc.encode_assertion(*t);
  timings.build_s = secs_since(build_start);

  // ---- Hand to CaDiCaL ----
  auto merge_start = clk::now();  // we'll attribute SAT search to "merge"

  CaDiCaL::Solver solver;
  // CaDiCaL 3.x's "factor" preprocessing pass requires every variable to be
  // declared before use. Disable it (the default-on behavior would force us
  // to call resize(num_vars) and pay a cost we don't need on tiny CNFs).
  solver.set("factor", 0);
  // Feed the CNF. CaDiCaL grows its var table on demand from add().
  for (int lit : enc.cnf()) solver.add(lit);

  // Build the propagator. It owns `eg` for the duration of the solve.
  UfPropagator prop(std::move(eg),
                    std::unordered_map<int, std::pair<Id, Id>>(enc.atom_endpoints()),
                    parallel);
  solver.connect_external_propagator(&prop);
  // Tell CaDiCaL which vars are theory atoms. Required by IPASIR-UP so it
  // notifies us only on those (skips aux Tseitin literal traffic).
  for (const auto& kv : enc.atom_endpoints()) {
    solver.add_observed_var(kv.first);
  }

  int sat = solver.solve();
  solver.disconnect_external_propagator();
  timings.merge_s = secs_since(merge_start);

  // ---- Result mapping ----
  // CaDiCaL: 10 = SAT, 20 = UNSAT.
  SolveResult result;
  if (sat == 10)      result = SolveResult::Sat;
  else if (sat == 20) result = SolveResult::Unsat;
  else throw std::runtime_error("CaDiCaL returned unknown status");

  timings.rebuild_s = 0.0;
  timings.check_s = 0.0;
  timings.total_s = secs_since(total_start);
  timings.solve_s = timings.build_s + timings.merge_s;
  return {result, timings};
}

}  // namespace pe
