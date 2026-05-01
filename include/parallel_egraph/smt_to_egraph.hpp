#pragma once
// Glue between pe::parse_smtlib and EGraph::bulk_init: walks a Term tree
// iteratively, hashcons-dedupes, and emits a flat DAG-ordered ENode
// sequence ready for bulk_init. Used by both the egraph-cc CLI
// (src/main.cpp) and the SMT-LIB benchmark (bench/smt_bench.cpp).
//
// The hashcons lives here, not in EGraph itself — EGraph has no
// incremental add()/hashcons; bulk_init expects a flat, unique ENode
// sequence, which this builder produces.

#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <ankerl/unordered_dense.h>
#include <parlay/sequence.h>

#include "parallel_egraph/egraph.hpp"
#include "parallel_egraph/fxhash.hpp"
#include "parallel_egraph/smtlib.hpp"

namespace pe {

struct ENodeHash {
  std::size_t operator()(const ENode& n) const noexcept {
    FxHasher h;
    h.write_str(n.op);
    for (Id c : n.children) h.write_u32(c);
    return static_cast<std::size_t>(h.finish());
  }
};

class SmtToEGraphBuilder {
 public:
  // Walk a Term tree iteratively, dedupe via hashcons, append unique
  // ENodes to nodes_ in DAG order. Returns the class id of the term's
  // root. After all add_term() calls, std::move(builder).take_nodes()
  // yields the flat ENode sequence to feed EGraph::bulk_init.
  Id add_term(const Term& term) {
    enum class WK { Process, Build };
    struct W {
      WK kind;
      const Term* term;
      std::string op;
      std::size_t nargs;
    };
    std::vector<W> stack;
    std::vector<Id> results;
    stack.push_back({WK::Process, &term, {}, 0});
    while (!stack.empty()) {
      W w = std::move(stack.back());
      stack.pop_back();
      if (w.kind == WK::Process) {
        const Term& t = *w.term;
        switch (t.kind) {
          case Term::Kind::Const:
            results.push_back(intern(ENode{t.op, {}}));
            break;
          case Term::Kind::App:
            stack.push_back({WK::Build, nullptr, t.op, t.args.size()});
            for (auto it = t.args.rbegin(); it != t.args.rend(); ++it) {
              stack.push_back({WK::Process, &*it, {}, 0});
            }
            break;
          case Term::Kind::Eq:
          case Term::Kind::Not:
            throw std::runtime_error(
                "= and not are not first-class terms; only allowed at "
                "the top of an assertion");
        }
      } else {
        std::size_t start = results.size() - w.nargs;
        std::vector<Id> children(results.begin() + start, results.end());
        results.erase(results.begin() + start, results.end());
        ENode node;
        node.op = std::move(w.op);
        node.children = std::move(children);
        results.push_back(intern(std::move(node)));
      }
    }
    return results.back();
  }

  parlay::sequence<ENode> take_nodes() && { return std::move(nodes_); }
  std::size_t size() const { return nodes_.size(); }

 private:
  Id intern(ENode node) {
    auto it = hashcons_.find(node);
    if (it != hashcons_.end()) return it->second;
    Id id = static_cast<Id>(nodes_.size());
    nodes_.push_back(node);
    hashcons_.emplace(std::move(node), id);
    return id;
  }

  parlay::sequence<ENode> nodes_;
  ankerl::unordered_dense::map<ENode, Id, ENodeHash> hashcons_;
};

}  // namespace pe
