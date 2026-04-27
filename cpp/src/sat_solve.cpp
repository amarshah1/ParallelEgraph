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
// ITE lifting
// ---------------------------------------------------------------------------
// QF_UF allows `(ite c x y)` to return a UF-sorted value, e.g.
//
//   (= u (ite (= a b) v w))
//
// The encoder only handles Bool-typed ite (Tseitin gates over Bool literals).
// We preprocess by rewriting every assertion so that ite never appears as a
// UF subterm — only at Boolean position. The transformation, applied
// recursively to any UF subterm e:
//
//   if e = (ite c x y), where c is Bool and x,y are UF, and e occurs as a
//   direct child of a UF-position (an = or App argument, or a child of
//   another UF App), then "lift" the ite up one level so the parent
//   becomes a Boolean ite.
//
// We fold lifting into a single recursive walk: lift_ites_in_uf returns a
// pair (rewritten UF term with no top-level ite, plus a list of "guard
// clauses" that must hold for the rewrite to be sound). For an Eq parent,
// we fold the guards into a top-level (and ...). For a UF App parent, we
// propagate the guards upward (the App becomes a guarded if/else over its
// child's possible values).
//
// In practice, the only place ite appears in real benchmarks is at top of
// an = arg, so the recursion stays shallow. The general implementation
// below is correct even when ite appears nested deeply inside an App.

namespace ite_lift {

// Build helpers for constructing terms inline.
Term mk_eq(Term a, Term b) {
  Term t; t.kind = Term::Kind::Eq;
  t.args.push_back(std::move(a)); t.args.push_back(std::move(b));
  return t;
}
Term mk_not(Term a) {
  Term t; t.kind = Term::Kind::Not; t.args.push_back(std::move(a));
  return t;
}
Term mk_and(std::vector<Term> args) {
  Term t; t.kind = Term::Kind::And; t.args = std::move(args);
  return t;
}
Term mk_or(std::vector<Term> args) {
  Term t; t.kind = Term::Kind::Or; t.args = std::move(args);
  return t;
}
Term mk_ite(Term c, Term a, Term b) {
  Term t; t.kind = Term::Kind::Ite;
  t.args.push_back(std::move(c));
  t.args.push_back(std::move(a));
  t.args.push_back(std::move(b));
  return t;
}

// Given a UF subterm that may contain ites, return an equivalent
// representation as a list of (guard, value) pairs. The semantics: the term
// equals `value_i` whenever `guard_i` is true; the guards are mutually
// exclusive and exhaustive. For an ite-free term, the result is a single
// pair (true, term).
struct GuardedValue {
  std::vector<Term> guards;  // each entry = a Boolean term (guard); empty = trivially true
  std::vector<Term> values;  // matching UF terms, no ite at top
};

GuardedValue split_uf(const Term& t);

// Compose two guards via logical and. Empty guard = true.
Term and_guards(const std::vector<Term>& gs) {
  std::vector<Term> nonempty;
  for (const auto& g : gs) {
    if (!(g.kind == Term::Kind::True)) nonempty.push_back(g);
  }
  if (nonempty.empty()) {
    Term t; t.kind = Term::Kind::True; return t;
  }
  if (nonempty.size() == 1) return nonempty[0];
  return mk_and(std::move(nonempty));
}

GuardedValue split_uf(const Term& t) {
  // Bottom of recursion: leaves and ite-free Apps.
  if (t.kind == Term::Kind::Const) {
    GuardedValue gv;
    Term tt; tt.kind = Term::Kind::True;
    gv.guards.push_back(tt);
    gv.values.push_back(t);
    return gv;
  }

  if (t.kind == Term::Kind::Ite) {
    // (ite c x y): split x and y, prepend c / ¬c to each branch's guards.
    // c is a Bool — process via splits-on-bool below by lifting into Bool.
    GuardedValue gx = split_uf(t.args[1]);
    GuardedValue gy = split_uf(t.args[2]);
    GuardedValue out;
    for (std::size_t i = 0; i < gx.guards.size(); ++i) {
      out.guards.push_back(mk_and({t.args[0], gx.guards[i]}));
      out.values.push_back(std::move(gx.values[i]));
    }
    for (std::size_t i = 0; i < gy.guards.size(); ++i) {
      out.guards.push_back(mk_and({mk_not(t.args[0]), gy.guards[i]}));
      out.values.push_back(std::move(gy.values[i]));
    }
    return out;
  }

  if (t.kind == Term::Kind::App) {
    // For an App f(x1, ..., xk), split each xi. Cartesian product across
    // children: each combination gives one (guard, app) pair.
    std::vector<GuardedValue> child_splits;
    child_splits.reserve(t.args.size());
    bool any_ite = false;
    for (const auto& a : t.args) {
      child_splits.push_back(split_uf(a));
      if (child_splits.back().values.size() > 1) any_ite = true;
    }
    if (!any_ite) {
      // No ites anywhere below; return as-is with a trivially-true guard.
      GuardedValue gv;
      Term tt; tt.kind = Term::Kind::True;
      gv.guards.push_back(tt);
      gv.values.push_back(t);
      return gv;
    }
    // Cartesian-product expansion.
    GuardedValue out;
    std::vector<std::size_t> idx(t.args.size(), 0);
    while (true) {
      // Build app from current idx tuple.
      std::vector<Term> branch_guards;
      Term app; app.kind = Term::Kind::App; app.op = t.op;
      for (std::size_t i = 0; i < t.args.size(); ++i) {
        branch_guards.push_back(child_splits[i].guards[idx[i]]);
        app.args.push_back(child_splits[i].values[idx[i]]);
      }
      out.guards.push_back(and_guards(branch_guards));
      out.values.push_back(std::move(app));
      // Increment idx (mixed-radix odometer).
      std::size_t k = t.args.size();
      while (k > 0) {
        --k;
        if (++idx[k] < child_splits[k].values.size()) break;
        idx[k] = 0;
        if (k == 0) goto done;
      }
    }
    done:;
    return out;
  }

  // Eq / Bool kinds shouldn't appear inside a UF term — caller error.
  // Just return as opaque.
  GuardedValue gv;
  Term tt; tt.kind = Term::Kind::True;
  gv.guards.push_back(tt);
  gv.values.push_back(t);
  return gv;
}

// Lift ites in a Bool-typed term: walks Boolean structure, and for each
// UF term it encounters at an Eq/Distinct arg, expands the ite-tree.
Term lift_in_bool(const Term& t);

// Build (or guards.size==1 ? guards[0] : (or guards...)) for a fan-out of
// possibilities; the disjunction OR'd over (guard ∧ rewritten-eq-on-value).
Term lift_eq_sides(const Term& lhs, const Term& rhs) {
  GuardedValue lg = split_uf(lhs);
  GuardedValue rg = split_uf(rhs);
  if (lg.values.size() == 1 && rg.values.size() == 1) {
    // Neither side has an ite — return plain (= lhs rhs).
    return mk_eq(lg.values[0], rg.values[0]);
  }
  // The equality holds iff there exist (i, j) such that guard_l[i] ∧
  // guard_r[j] ∧ (val_l[i] = val_r[j]).
  std::vector<Term> disj;
  for (std::size_t i = 0; i < lg.values.size(); ++i) {
    for (std::size_t j = 0; j < rg.values.size(); ++j) {
      Term conj = mk_and({lg.guards[i], rg.guards[j],
                          mk_eq(lg.values[i], rg.values[j])});
      disj.push_back(std::move(conj));
    }
  }
  if (disj.size() == 1) return disj[0];
  return mk_or(std::move(disj));
}

Term lift_in_bool(const Term& t) {
  switch (t.kind) {
    case Term::Kind::True:
    case Term::Kind::False:
    case Term::Kind::Const:
      return t;
    case Term::Kind::Eq:
      return lift_eq_sides(t.args[0], t.args[1]);
    case Term::Kind::Distinct: {
      // (distinct x1 ... xk) lifts to ∧_{i<j} ¬(lift_eq xi xj).
      std::vector<Term> conjs;
      for (std::size_t i = 0; i < t.args.size(); ++i) {
        for (std::size_t j = i + 1; j < t.args.size(); ++j) {
          conjs.push_back(mk_not(lift_eq_sides(t.args[i], t.args[j])));
        }
      }
      if (conjs.size() == 1) return conjs[0];
      return mk_and(std::move(conjs));
    }
    case Term::Kind::Not: {
      Term out; out.kind = Term::Kind::Not;
      out.args.push_back(lift_in_bool(t.args[0]));
      return out;
    }
    case Term::Kind::Ite: {
      // Bool-typed ite: lift each branch.
      Term out; out.kind = Term::Kind::Ite;
      for (const auto& a : t.args) out.args.push_back(lift_in_bool(a));
      return out;
    }
    case Term::Kind::And:
    case Term::Kind::Or:
    case Term::Kind::Implies:
    case Term::Kind::Xor: {
      Term out; out.kind = t.kind; out.op = t.op;
      for (const auto& a : t.args) out.args.push_back(lift_in_bool(a));
      return out;
    }
    case Term::Kind::App:
      // A Bool-typed UF predicate. Lift its UF args.
      // Same Cartesian split as above, but we OR the disjunction of
      // (guard ∧ predicate-on-value).
      {
        std::vector<GuardedValue> child_splits;
        child_splits.reserve(t.args.size());
        bool any_ite = false;
        for (const auto& a : t.args) {
          child_splits.push_back(split_uf(a));
          if (child_splits.back().values.size() > 1) any_ite = true;
        }
        if (!any_ite) return t;
        std::vector<Term> disj;
        std::vector<std::size_t> idx(t.args.size(), 0);
        while (true) {
          std::vector<Term> guards;
          Term pred; pred.kind = Term::Kind::App; pred.op = t.op;
          for (std::size_t i = 0; i < t.args.size(); ++i) {
            guards.push_back(child_splits[i].guards[idx[i]]);
            pred.args.push_back(child_splits[i].values[idx[i]]);
          }
          disj.push_back(mk_and({and_guards(guards), pred}));
          std::size_t k = t.args.size();
          while (k > 0) {
            --k;
            if (++idx[k] < child_splits[k].values.size()) break;
            idx[k] = 0;
            if (k == 0) goto done2;
          }
        }
        done2:;
        if (disj.size() == 1) return disj[0];
        return mk_or(std::move(disj));
      }
  }
  return t;
}

}  // namespace ite_lift

// Public entry point: rewrite an assertion so that no UF term contains an
// ite at any depth. The encoder receives ite only in pure-Bool position.
Term lift_ites(const Term& t) {
  return ite_lift::lift_in_bool(t);
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
// ---------------------------------------------------------------------------
// UfPropagator (deferred-merge architecture).
//
// We *defer* all e-graph mutation to cb_check_found_model. During search,
// notify_assignment merely stashes (lit, a, b, is_eq) tuples into a
// per-decision-level trail. notify_new_decision_level pushes a new empty
// trail level; notify_backtrack truncates the trail. There is NO clone,
// NO merge, NO rebuild on the search hot path.
//
// At cb_check_found_model time, we:
//   1. Clone the immutable `ground_` e-graph.
//   2. Flatten the trail into (positive eqs) and (negative eqs).
//   3. parallel_merge_all on positive eqs (fully parallel).
//   4. parallel_rebuild (fully parallel) — or rebuild_with_reasons in
//      proof mode (sequential).
//   5. Scan negative eqs for conflict; if found, build conflict clause
//      via explain (proof mode) or prefix replay (default mode).
//
// This architecture means:
//   - Zero clone overhead per decision/backtrack.
//   - parallel_merge_all is actually exercised (Path A used to be the
//     only place it ran; now Path B uses it too).
//   - parallel_rebuild runs once per model check on the full assignment.
//   - Snapshot/restore of the e-graph is unnecessary because the e-graph
//     is never mutated during search.
// ---------------------------------------------------------------------------
class UfPropagator : public CaDiCaL::ExternalPropagator {
 public:
  UfPropagator(std::unique_ptr<EGraph> root_eg,
               std::unordered_map<int, std::pair<Id, Id>> atoms,
               bool parallel)
      : ground_(std::move(root_eg)),
        atom_endpoints_(std::move(atoms)),
        parallel_(parallel) {
    debug_ = std::getenv("PE_PROP_DEBUG") != nullptr;
    trace_ = std::getenv("PE_PROP_TRACE") != nullptr;
    lit_levels_.emplace_back();  // level 0 trail
  }

  ~UfPropagator() override {
    if (debug_) {
      std::fprintf(stderr,
                   "UfPropagator stats: model_checks=%llu  model_check_s=%.3f  "
                   "decisions=%llu  backtracks=%llu  "
                   "notify_lits=%llu  conflicts=%llu  "
                   "clone_s=%.3f  merge_s=%.3f  rebuild_s=%.3f\n",
                   (unsigned long long)model_checks_,
                   model_check_time_s_,
                   (unsigned long long)decisions_,
                   (unsigned long long)backtracks_,
                   (unsigned long long)notify_lits_,
                   (unsigned long long)conflicts_emitted_,
                   clone_time_s_, merge_time_s_, rebuild_time_s_);
    }
  }

  // CaDiCaL pushes a vector of newly-true literals. We just stash theory
  // atoms onto the current decision level's trail — no e-graph mutation
  // here. All real work happens in cb_check_found_model.
  void notify_assignment(const std::vector<int>& lits) override {
    if (debug_) notify_lits_ += lits.size();
    auto& cur_trail = lit_levels_.back();
    for (int lit : lits) {
      int v = std::abs(lit);
      auto it = atom_endpoints_.find(v);
      if (it == atom_endpoints_.end()) continue;  // aux Tseitin var
      cur_trail.push_back({lit, it->second.first, it->second.second});
    }
  }

  void notify_new_decision_level() override {
    if (debug_) decisions_++;
    lit_levels_.emplace_back();
  }

  void notify_backtrack(std::size_t level) override {
    if (debug_) backtracks_++;
    // Drop trail entries from levels above `level`.
    if (lit_levels_.size() > level + 1) lit_levels_.resize(level + 1);
  }

  bool cb_check_found_model(const std::vector<int>& model) override {
    using clk = std::chrono::steady_clock;
    auto t0 = debug_ ? clk::now() : clk::time_point{};
    if (debug_) model_checks_++;

    // ---- (1) Clone ground_ — the only clone in the search loop. ----------
    auto t_clone = debug_ ? clk::now() : clk::time_point{};
    auto eg = ground_->clone();
    if (debug_) {
      clone_time_s_ += std::chrono::duration<double>(clk::now() - t_clone).count();
      if (trace_)
        std::fprintf(stderr, "[clone] check_model  capacity=%zu\n",
                     eg->uf().capacity());
    }

    // ---- (2) Flatten the trail. Use the model as ground truth: we want
    // to handle late literals CaDiCaL hasn't notified us about (e.g. some
    // fixed assignments at level 0 may bypass notify_assignment).
    struct EqLit  { int lit; Id a; Id b; };
    struct DiseqLit { int lit; Id a; Id b; };
    std::vector<EqLit>    pos_eqs;
    std::vector<DiseqLit> neg_eqs;
    pos_eqs.reserve(model.size());
    for (int lit : model) {
      int v = std::abs(lit);
      auto it = atom_endpoints_.find(v);
      if (it == atom_endpoints_.end()) continue;
      Id a = it->second.first, b = it->second.second;
      if (lit > 0) pos_eqs.push_back({lit, a, b});
      else         neg_eqs.push_back({lit, a, b});
    }

    // ---- (3) Apply positive equalities. ----------------------------------
    auto t_merge = debug_ ? clk::now() : clk::time_point{};
    parlay::sequence<std::pair<Id, Id>> pairs;
    pairs.reserve(pos_eqs.size());
    for (auto& e : pos_eqs) pairs.emplace_back(e.a, e.b);
    eg->parallel_merge_all(pairs);
    if (debug_)
      merge_time_s_ += std::chrono::duration<double>(clk::now() - t_merge).count();

    // ---- (4) Close under congruence. -------------------------------------
    auto t_rb = debug_ ? clk::now() : clk::time_point{};
    if (parallel_) eg->parallel_rebuild();
    else           eg->rebuild();
    if (debug_)
      rebuild_time_s_ += std::chrono::duration<double>(clk::now() - t_rb).count();

    // ---- (5) Scan negative eqs for conflict. -----------------------------
    // Conflict clause is built via prefix replay against a fresh ground
    // clone: replay positive equalities one at a time, rebuild after
    // each, and stop when the offending disequality's two sides become
    // equivalent. The prefix used is the conflict core. This is sound
    // but not minimum.
    bool ok = true;
    for (auto& d : neg_eqs) {
      if (!eg->equiv(d.a, d.b)) continue;
      conflict_clause_.clear();
      conflict_clause_.push_back(-d.lit);
      auto explain_eg = ground_->clone();
      for (const auto& e : pos_eqs) {
        explain_eg->merge(e.a, e.b);
        explain_eg->rebuild();
        conflict_clause_.push_back(-e.lit);
        if (explain_eg->equiv(d.a, d.b)) break;
      }
      conflict_clause_.push_back(0);
      conflict_pending_ = true;
      if (debug_) conflicts_emitted_++;
      ok = false;
      break;
    }
    if (debug_)
      model_check_time_s_ += std::chrono::duration<double>(clk::now() - t0).count();
    return ok;
  }

  // No theory propagation back to CaDiCaL in v1.
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
      conflict_pending_ = false;
      conflict_clause_.clear();
      conflict_emit_idx_ = 0;
      return 0;
    }
    int lit = conflict_clause_[conflict_emit_idx_++];
    if (lit == 0) {
      conflict_pending_ = false;
      conflict_clause_.clear();
      conflict_emit_idx_ = 0;
    }
    return lit;
  }

 private:
  // Trail entry: a theory literal CaDiCaL has assigned, with its
  // pre-canonical endpoints in the e-graph.
  struct TrailLit { int lit; Id a; Id b; };

  // Immutable post-add e-graph. Cloned on every cb_check_found_model.
  std::unique_ptr<EGraph> ground_;

  // Per-decision-level trails of theory atoms received via
  // notify_assignment. lit_levels_[L] is the (lit, a, b) entries
  // assigned at decision level L. Truncated by notify_backtrack.
  // Currently unused by cb_check_found_model (we use `model` directly),
  // but kept as the API contract requires.
  std::vector<std::vector<TrailLit>> lit_levels_;

  std::unordered_map<int, std::pair<Id, Id>> atom_endpoints_;
  bool parallel_;

  // Pending conflict clause to deliver to CaDiCaL.
  std::vector<int> conflict_clause_;
  std::size_t conflict_emit_idx_ = 0;
  bool conflict_pending_ = false;

  // ---- Profiling (active when PE_PROP_DEBUG is set) ----------------------
  bool debug_ = false;
  bool trace_ = false;
  std::uint64_t model_checks_ = 0;
  std::uint64_t decisions_ = 0;
  std::uint64_t backtracks_ = 0;
  std::uint64_t notify_lits_ = 0;
  std::uint64_t conflicts_emitted_ = 0;
  double clone_time_s_ = 0.0;
  double model_check_time_s_ = 0.0;
  double merge_time_s_ = 0.0;
  double rebuild_time_s_ = 0.0;
};

double secs_since(std::chrono::steady_clock::time_point t) {
  using namespace std::chrono;
  return duration<double>(steady_clock::now() - t).count();
}

}  // namespace

std::pair<SolveResult, SolveTimings> sat_solve_timed(const std::string& input,
                                                     bool parallel) {
  using clk = std::chrono::steady_clock;
  bool dbg = std::getenv("PE_PROP_DEBUG") != nullptr;
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
  if (dbg) std::fprintf(stderr, "[phase] parse done in %.3fs (assertions=%zu)\n",
                        timings.parse_s, assertions.size());

  // ---- ITE lifting ----
  auto build_start = clk::now();
  auto t_lift = clk::now();
  std::vector<Term> lifted;
  lifted.reserve(assertions.size());
  for (const Term* t : assertions) lifted.push_back(lift_ites(*t));
  if (dbg) std::fprintf(stderr, "[phase] ite-lift done in %.3fs\n", secs_since(t_lift));

  // ---- Build EGraph ----
  auto t_cap = clk::now();
  std::size_t capacity = 0;
  for (const Term& t : lifted) capacity += count_subterms_total(t);
  if (capacity == 0) capacity = 1;
  if (dbg) std::fprintf(stderr, "[phase] capacity = %zu (count walk %.3fs)\n",
                        capacity, secs_since(t_cap));
  auto t_alloc = clk::now();
  auto eg = std::make_unique<EGraph>(capacity, parallel);
  if (dbg) std::fprintf(stderr, "[phase] EGraph alloc in %.3fs\n", secs_since(t_alloc));

  // Preflight: add every UF subterm to the e-graph.
  auto t_pre = clk::now();
  for (const Term& t : lifted) preflight_add_uf(*eg, t);
  if (dbg) std::fprintf(stderr, "[phase] preflight in %.3fs\n", secs_since(t_pre));

  // ---- Encode CNF ----
  auto t_enc = clk::now();
  Encoder enc(*eg);
  for (const Term& t : lifted) enc.encode_assertion(t);
  timings.build_s = secs_since(build_start);
  if (dbg) std::fprintf(stderr,
                        "[phase] encode in %.3fs (vars=%d clauses_lits=%zu atoms=%zu)\n",
                        secs_since(t_enc), enc.num_vars(), enc.cnf().size(),
                        enc.atom_endpoints().size());

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
