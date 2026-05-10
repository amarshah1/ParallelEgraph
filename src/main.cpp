// egraph-cc — congruence closure on a QF_UF SMT-LIB 2 input.
//
// Parses the script, builds the e-graph (parser-side hashcons + bulk ctor)
// from the asserted equalities, runs parallel_close on the equality list,
// and reports sat / unsat: unsat iff any asserted disequality (a != b)
// collapses (find_root(a) == find_root(b)) after closure.
//
// Pass --timing for a final stderr line:
//   timing: read=...ms parse=...ms build=...ms close=...ms check=...ms dtor=...ms
// (consumed by run_all_benchmarks.py's egg suite). PE_TRACE=1 also still
// works and emits the per-round breakdowns.

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <parlay/sequence.h>

#include "parallel_egraph/egraph.hpp"
#include "parallel_egraph/smt_to_egraph.hpp"
#include "parallel_egraph/smtlib.hpp"

namespace {

int usage(const char* prog) {
  std::fprintf(stderr,
      "usage: %s [--timing] "
      "[--sequential[=nelson|topo|topo_iter|dst|simple]] <file.smt2>\n"
      "  --sequential            run sequential_close_nelson (default seq algo)\n"
      "  --sequential=nelson     same as --sequential\n"
      "  --sequential=topo       run sequential_close_topo\n"
      "  --sequential=topo_iter  run sequential_close_topo_iter\n"
      "  --sequential=dst        run sequential_close_dst\n"
      "  --sequential=simple     run sequential_close_simple (worklist + hashcons)\n"
      "Without --sequential, the parallel path is used; selector via env:\n"
      "  PE_USE_ASYNC=1       parallel_close_async_rounds (integer-sort)\n"
      "  PE_USE_ASYNC_GBK=1   parallel_close_async_rounds_groupby (group_by_key)\n"
      "  PE_USE_ASYNC_CONT=1  parallel_close_async_continuous (par_do)\n"
      "  PE_USE_NAIVE=1       parallel_close_naive_rounds (no dirty filter)\n"
      "  PE_USE_TOPO=1        parallel_close_topo\n"
      "  (none)               parallel_close (BSP)\n", prog);
  return 2;
}

std::string read_file(const std::string& path) {
  std::ifstream f(path);
  if (!f) throw std::runtime_error("cannot open " + path);
  std::stringstream buf;
  buf << f.rdbuf();
  return buf.str();
}

using clk = std::chrono::steady_clock;
double elapsed_ms(clk::time_point t0, clk::time_point t1) {
  return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

}  // namespace

int main(int argc, char** argv) {
  bool emit_timing = false;
  const char* path = nullptr;
  // --sequential family: None=parallel (default), or one of the
  // sequential algorithms.
  enum class SeqAlgo { None, Nelson, Topo, TopoIter, Dst, Simple };
  SeqAlgo seq_algo = SeqAlgo::None;
  for (int i = 1; i < argc; ++i) {
    const char* a = argv[i];
    if (std::strcmp(a, "--timing") == 0) {
      emit_timing = true;
    } else if (std::strcmp(a, "--sequential") == 0 ||
               std::strcmp(a, "--sequential=nelson") == 0) {
      seq_algo = SeqAlgo::Nelson;
    } else if (std::strcmp(a, "--sequential=topo") == 0) {
      seq_algo = SeqAlgo::Topo;
    } else if (std::strcmp(a, "--sequential=topo_iter") == 0) {
      seq_algo = SeqAlgo::TopoIter;
    } else if (std::strcmp(a, "--sequential=dst") == 0) {
      seq_algo = SeqAlgo::Dst;
    } else if (std::strcmp(a, "--sequential=simple") == 0) {
      seq_algo = SeqAlgo::Simple;
    } else if (path == nullptr) {
      path = a;
    } else {
      return usage(argv[0]);
    }
  }
  if (path == nullptr) return usage(argv[0]);

  auto t_start = clk::now();

  std::string input;
  try {
    input = read_file(path);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "%s\n", e.what());
    return 1;
  }
  auto t_read = clk::now();

  pe::Script script;
  try {
    script = pe::parse_smtlib(input);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "%s\n", e.what());
    return 1;
  }
  auto t_parse = clk::now();

  // Build phase: walk every assertion's terms through the parser-side
  // hashcons, collect (a, b) for equalities and disequalities.
  pe::SmtToEGraphBuilder builder;
  parlay::sequence<std::pair<pe::Id, pe::Id>> equalities;
  std::vector<std::pair<pe::Id, pe::Id>> disequalities;

  try {
    for (const auto& cmd : script.commands) {
      if (cmd.kind != pe::Command::Kind::Assert) continue;
      const pe::Term& t = cmd.term;
      if (t.kind == pe::Term::Kind::Eq) {
        pe::Id a = builder.add_term(t.args[0]);
        pe::Id b = builder.add_term(t.args[1]);
        equalities.emplace_back(a, b);
      } else if (t.kind == pe::Term::Kind::Not && t.args.size() == 1 &&
                 t.args[0].kind == pe::Term::Kind::Eq) {
        pe::Id a = builder.add_term(t.args[0].args[0]);
        pe::Id b = builder.add_term(t.args[0].args[1]);
        disequalities.emplace_back(a, b);
      } else if (t.kind == pe::Term::Kind::Distinct) {
        // (distinct t1 ... tn): assert all n*(n-1)/2 pairwise
        // disequalities. Intern each operand once and then enumerate
        // the upper-triangle pairs.
        std::vector<pe::Id> ids;
        ids.reserve(t.args.size());
        for (const auto& arg : t.args) {
          ids.push_back(builder.add_term(arg));
        }
        for (std::size_t i = 0; i < ids.size(); ++i) {
          for (std::size_t j = i + 1; j < ids.size(); ++j) {
            disequalities.emplace_back(ids[i], ids[j]);
          }
        }
      } else if (t.kind == pe::Term::Kind::Not && t.args.size() == 1 &&
                 t.args[0].kind == pe::Term::Kind::Distinct) {
        // (not (distinct ...)) means "at least two operands are
        // equal" — that's a disjunction, not expressible as a
        // conjunction of unit equalities. Refuse rather than
        // silently mis-handle.
        std::fprintf(stderr,
            "(not (distinct ...)) is not expressible as unit equalities; "
            "skipping is unsound, so we refuse this input\n");
        return 1;
      } else {
        std::fprintf(stderr,
            "unsupported assertion shape (only (= a b), (not (= a b)), "
            "and (distinct t1 ... tn))\n");
        return 1;
      }
    }
  } catch (const std::exception& e) {
    std::fprintf(stderr, "%s\n", e.what());
    return 1;
  }

  // Closure-algorithm selector.
  //   --sequential[=nelson|topo] → sequential path (single-threaded;
  //                                independent of PARLAY_NUM_THREADS).
  //   PE_USE_ASYNC=1      → parallel_close_async_rounds (mark-based
  //                         dirty filter, no parents_).
  //   PE_USE_ASYNC_CONT=1 → parallel_close_async_continuous (semisorter
  //                         + unioner via parlay::par_do, deque mailbox,
  //                         drain-gated BSP-with-overlap).
  //   PE_USE_NAIVE=1      → parallel_close_naive_rounds (semisort all
  //                         non-leaves every round; no dirty filter).
  //   PE_USE_TOPO=1       → parallel_close_topo (topological-sort-based;
  //                         matches what closure_compare_bench /
  //                         synthetic_bench / smt_bench tag as `par_topo`).
  // Default: parallel_close (BSP, parents_-driven).
  const bool use_async      = std::getenv("PE_USE_ASYNC")      != nullptr;
  const bool use_async_gbk  = std::getenv("PE_USE_ASYNC_GBK")  != nullptr;
  const bool use_async_cont = std::getenv("PE_USE_ASYNC_CONT") != nullptr;
  const bool use_naive      = std::getenv("PE_USE_NAIVE")      != nullptr;
  const bool use_topo       = std::getenv("PE_USE_TOPO")       != nullptr;
  if (int(use_async) + int(use_async_gbk) + int(use_async_cont) +
      int(use_naive) + int(use_topo) > 1) {
    std::fprintf(stderr,
                 "PE_USE_ASYNC, PE_USE_ASYNC_GBK, PE_USE_ASYNC_CONT, "
                 "PE_USE_NAIVE, and PE_USE_TOPO are mutually exclusive\n");
    return 2;
  }
  if (seq_algo != SeqAlgo::None &&
      (use_async || use_async_gbk || use_async_cont ||
       use_naive || use_topo)) {
    std::fprintf(stderr,
                 "--sequential and PE_USE_ASYNC*/"
                 "PE_USE_NAIVE/PE_USE_TOPO are mutually exclusive\n");
    return 2;
  }
  auto nodes = std::move(builder).take_nodes();

  bool unsat = false;
  clk::time_point t_build, t_close, t_check, t_dtor;

  if (seq_algo != SeqAlgo::None) {
    // Sequential path: EGraph<SequentialUnionFind>. Built unconditionally
    // with the default ctor (parents_); both sequential closures consume
    // it. Convert disequalities -> drop after verdict.
    auto eg = std::make_unique<pe::SequentialEGraph>(std::move(nodes));
    t_build = clk::now();
    if (seq_algo == SeqAlgo::Topo) {
      eg->sequential_close_topo(equalities);
    } else if (seq_algo == SeqAlgo::TopoIter) {
      eg->sequential_close_topo_iter(equalities);
    } else if (seq_algo == SeqAlgo::Dst) {
      eg->sequential_close_dst(equalities);
    } else if (seq_algo == SeqAlgo::Simple) {
      eg->sequential_close_simple(equalities);
    } else {
      eg->sequential_close_nelson(equalities);
    }
    t_close = clk::now();
    for (auto [a, b] : disequalities) {
      if (eg->equiv(a, b)) { unsat = true; break; }
    }
    t_check = clk::now();
    std::puts(unsat ? "unsat" : "sat");
    std::fflush(stdout);
    eg.reset();
    t_dtor = clk::now();
  } else {
    std::unique_ptr<pe::ConcurrentEGraph> eg;
    if (use_async || use_async_gbk || use_async_cont) {
      // All async-family variants share the async-flavor ctor.
      eg = std::make_unique<pe::ConcurrentEGraph>(std::move(nodes), pe::async);
    } else if (use_naive) {
      eg = std::make_unique<pe::ConcurrentEGraph>(std::move(nodes), pe::naive);
    } else if (use_topo) {
      eg = std::make_unique<pe::ConcurrentEGraph>(std::move(nodes), pe::topo);
    } else {
      eg = std::make_unique<pe::ConcurrentEGraph>(std::move(nodes));
    }
    t_build = clk::now();

    if (use_async) {
      eg->parallel_close_async_rounds(std::move(equalities));
    } else if (use_async_gbk) {
      eg->parallel_close_async_rounds_groupby(std::move(equalities));
    } else if (use_async_cont) {
      eg->parallel_close_async_continuous(std::move(equalities));
    } else if (use_naive) {
      eg->parallel_close_naive_rounds(std::move(equalities));
    } else if (use_topo) {
      eg->parallel_close_topo(std::move(equalities));
    } else {
      eg->parallel_close(std::move(equalities));
    }
    t_close = clk::now();

    for (auto [a, b] : disequalities) {
      if (eg->equiv(a, b)) { unsat = true; break; }
    }
    t_check = clk::now();

    std::puts(unsat ? "unsat" : "sat");
    std::fflush(stdout);

    // The EGraph dtor runs at function exit; capture its cost too.
    eg.reset();
    t_dtor = clk::now();
  }

  if (emit_timing) {
    std::fprintf(stderr,
                 "timing: read=%.2fms parse=%.2fms build=%.2fms "
                 "close=%.2fms check=%.2fms dtor=%.2fms\n",
                 elapsed_ms(t_start, t_read),
                 elapsed_ms(t_read,  t_parse),
                 elapsed_ms(t_parse, t_build),
                 elapsed_ms(t_build, t_close),
                 elapsed_ms(t_close, t_check),
                 elapsed_ms(t_check, t_dtor));
  }

  return 0;
}
