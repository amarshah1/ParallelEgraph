#pragma once
// Parser + builder for the .gates file format used by the
// miter-cc-benchmarks suite. See
// /Users/amarshah/Desktop/.../miter-cc-benchmarks/README.md for the
// format spec.
//
// File format (one gate per line):
//   p gates 0
//   AND <lhs> 2 <rhs1> <rhs2>
//   XOR <lhs> 2 <rhs1> <rhs2>
//   ITE <lhs> 3 <cond> <then> <else>
//   NOT <lhs> 1 <rhs>          # only in *_with_not/ variants
//   c done
//
// Literals are signed ints, 1-indexed, never zero. Negative means
// negated. The full multiset of candidate gates is included (duplicate
// rhs is intentional — CC's job is to discover them).
//
// Encoding into the e-graph
// =========================
//
// We treat each signed literal as its own class. For variable v ≥ 1:
//   * positive literal class: leaf ENode with op "v<v>"
//   * negative literal class: leaf ENode with op "v-<v>"
// For every negative literal that ever appears, we emit a unary
// `NOT` term whose child is the corresponding positive class, and
// initially-union it with the negative-literal leaf. This treats
// negation as a regular function symbol — congruences across negated
// literals propagate naturally (e.g., NOT(a) ≡ NOT(b) is sound iff
// a ≡ b, and CC handles that).
//
// Each gate becomes an ENode keyed on its operator with children =
// the literal class ids of its rhs. We initially-union the gate's
// class with the lhs literal class. Duplicate gates (same op + rhs)
// each get their own class id; CC discovers the duplicates by
// canonical-sig matching.
//
// DAG order in the emitted node sequence is:
//   [0,  L)        : L distinct literal leaves
//   [L,  L+M)      : M NOT terms (each child is a literal slot)
//   [L+M, total)   : G gate terms (each child is a literal slot)
// where L = #distinct literals seen, M = #distinct negative-literal
// classes (= #distinct NOTs), G = #gates.
//
// Construction is parallel where it matters:
//   1. Slurp + split into lines: parlay primitives over the mmap'd
//      char buffer.
//   2. Tokenize each line: parlay::map; output a `RawGate` struct.
//   3. Collect distinct literals: parlay::remove_duplicates over
//      every literal mentioned. Assign class ids by sort-and-pack.
//   4. Build the ENode sequence: parlay::tabulate over the (literals,
//      NOTs, gates) layout. Build the eqs sequence in parallel too.
// The hashcons step that the SMT builder needs is unnecessary here:
// literals are dedup'd by step 3, and gates intentionally aren't
// dedup'd.

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <ankerl/unordered_dense.h>
#include <parlay/parallel.h>
#include <parlay/primitives.h>
#include <parlay/sequence.h>

#include "parallel_egraph/egraph.hpp"

namespace pe {

// One parsed gate line. `lhs` and `rhs` are signed literal ints
// (1-indexed, never zero, sign = negation). The op tag mirrors the
// file's keyword. NOT has arity 1, AND/XOR arity 2, ITE arity 3.
struct RawGate {
  enum class Op : std::uint8_t { And, Xor, Ite, Not };
  Op op;
  std::int32_t lhs;
  // Up to 3 rhs slots. Unused tail entries are 0 (a sentinel,
  // distinguishable from any real literal which is non-zero).
  std::int32_t rhs[3];
  std::uint8_t arity;  // 1 (NOT), 2 (AND/XOR), or 3 (ITE)
};

// Output of load_gates_file: a flat ENode sequence in DAG order plus
// the initial-union list, ready to feed straight into EGraph<UF>'s
// ctor and any close method. Plus a few stats for tracing.
struct LoadedGates {
  parlay::sequence<ENode> nodes;
  parlay::sequence<std::pair<Id, Id>> eqs;
  std::size_t n_literals;
  std::size_t n_not_terms;
  std::size_t n_gates;
  // Parse / build phase wall-clock (seconds). Populated by the loader
  // so callers can report it without re-instrumenting.
  double read_s = 0.0;
  double parse_s = 0.0;
  double build_s = 0.0;
};

namespace gates_detail {

// Map "AND"/"XOR"/"ITE"/"NOT" prefix → Op. Caller has already
// confirmed the line is a gate line (not "p gates 0" or "c done").
inline RawGate::Op op_from_prefix(const char* p) {
  switch (p[0]) {
    case 'A': return RawGate::Op::And;
    case 'X': return RawGate::Op::Xor;
    case 'I': return RawGate::Op::Ite;
    case 'N': return RawGate::Op::Not;
  }
  throw std::runtime_error("unknown gate op prefix");
}

// Parse a single signed-int token starting at `*p`, advance `*p`
// past it. Whitespace-tolerant on the leading edge.
inline std::int32_t parse_int_advance(const char*& p, const char* end) {
  while (p < end && (*p == ' ' || *p == '\t')) ++p;
  bool neg = false;
  if (p < end && *p == '-') { neg = true; ++p; }
  std::int64_t v = 0;
  while (p < end && *p >= '0' && *p <= '9') {
    v = v * 10 + (*p - '0');
    ++p;
  }
  return static_cast<std::int32_t>(neg ? -v : v);
}

// Parse a single line into a RawGate. Returns false for non-gate
// lines (header, comments, blank). The line is the [start, end)
// half-open span; trailing newline is not part of the span.
inline bool parse_gate_line(const char* start, const char* end, RawGate& out) {
  // Skip leading whitespace.
  while (start < end && (*start == ' ' || *start == '\t')) ++start;
  if (start == end) return false;
  // Skip comments and the header.
  if (*start == 'c' || *start == 'p') return false;
  // Operator keyword.
  if (end - start < 3) return false;
  out.op = op_from_prefix(start);
  start += 3;
  out.lhs    = parse_int_advance(start, end);
  std::int32_t arity = parse_int_advance(start, end);
  if (arity < 1 || arity > 3) {
    throw std::runtime_error("gate arity out of range");
  }
  out.arity = static_cast<std::uint8_t>(arity);
  out.rhs[0] = parse_int_advance(start, end);
  out.rhs[1] = (arity >= 2) ? parse_int_advance(start, end) : 0;
  out.rhs[2] = (arity >= 3) ? parse_int_advance(start, end) : 0;
  return true;
}

inline const char* op_name(RawGate::Op op) {
  switch (op) {
    case RawGate::Op::And: return "AND";
    case RawGate::Op::Xor: return "XOR";
    case RawGate::Op::Ite: return "ITE";
    case RawGate::Op::Not: return "NOT";
  }
  return "?";
}

}  // namespace gates_detail

// Slurp the file into memory. Returns the buffer + size. Throws on
// I/O failure. Implementation reads the whole file in one syscall;
// gate files are at most tens of megabytes uncompressed.
inline std::string slurp(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot open " + path);
  f.seekg(0, std::ios::end);
  std::streamsize sz = f.tellg();
  f.seekg(0, std::ios::beg);
  std::string buf(static_cast<std::size_t>(sz), '\0');
  if (!f.read(buf.data(), sz)) {
    throw std::runtime_error("read failed: " + path);
  }
  return buf;
}

// Load a .gates file and produce a `LoadedGates` ready to hand to the
// EGraph ctor. Phases (each timed):
//   1. read   — slurp file into memory
//   2. parse  — line-split, tokenize each gate
//   3. build  — assign literal class ids, emit ENodes + initial unions
inline LoadedGates load_gates_file(const std::string& path) {
  using clk = std::chrono::steady_clock;
  auto sec_since = [](clk::time_point t) {
    return std::chrono::duration<double>(clk::now() - t).count();
  };

  LoadedGates out;

  // ---- 1. read --------------------------------------------------------
  auto t_read = clk::now();
  std::string buf = slurp(path);
  out.read_s = sec_since(t_read);

  // ---- 2. parse -------------------------------------------------------
  auto t_parse = clk::now();
  // Find newline offsets in parallel. starts[i] = offset of first byte
  // of line i; we synthesize an extra start at buf.size() so the last
  // line's length is starts[i+1] - starts[i].
  const char* base = buf.data();
  const std::size_t n_bytes = buf.size();
  // Pack newline positions into a sequence.
  auto newline_flag = parlay::tabulate(n_bytes, [&](std::size_t i) {
    return base[i] == '\n' ? 1 : 0;
  });
  auto newline_idx = parlay::pack_index<std::uint32_t>(newline_flag);
  // Line `i` spans [start_of_i, newline_idx[i]). start_of_0 = 0;
  // start_of_i (i > 0) = newline_idx[i-1] + 1.
  const std::size_t n_lines = newline_idx.size();
  // Parse each line in parallel into a RawGate; non-gate lines (header
  // / comments / blank) get a sentinel arity=0 we filter out.
  auto raw_lines = parlay::tabulate(n_lines, [&](std::size_t i) {
    const char* line_start = base + (i == 0 ? 0 : newline_idx[i - 1] + 1);
    const char* line_end   = base + newline_idx[i];
    RawGate g{};
    g.arity = 0;
    if (gates_detail::parse_gate_line(line_start, line_end, g)) {
      // arity already set inside parse_gate_line.
    }
    return g;
  });
  auto gates = parlay::filter(raw_lines, [](const RawGate& g) {
    return g.arity != 0;
  });
  out.n_gates = gates.size();
  out.parse_s = sec_since(t_parse);

  // ---- 3. build -------------------------------------------------------
  auto t_build = clk::now();

  // 3a. Collect every distinct signed literal that appears anywhere
  //     (as lhs or rhs of any gate). Sort-and-uniq is fast in parlay.
  //     Sequence size ≤ 4 × #gates; for the large miters that's ~30M
  //     ints, but that's still small (~120MB).
  auto all_lits = parlay::tabulate(gates.size() * 4, [&](std::size_t k) {
    std::size_t gi = k / 4;
    std::size_t slot = k % 4;
    const RawGate& g = gates[gi];
    if (slot == 0) return g.lhs;
    if (slot - 1 < g.arity) return g.rhs[slot - 1];
    return std::int32_t{0};   // sentinel, filtered below
  });
  auto lits_nonzero = parlay::filter(all_lits,
      [](std::int32_t x) { return x != 0; });
  // Sort + dedupe to get distinct literals in a canonical order.
  parlay::integer_sort_inplace(parlay::make_slice(lits_nonzero),
      [](std::int32_t x) -> std::uint32_t {
        // Map signed → unsigned for radix sort: shift to put negatives
        // in the lower half. Order doesn't matter for correctness; we
        // just need a stable assignment of class ids.
        return static_cast<std::uint32_t>(
            static_cast<std::int64_t>(x) + std::int64_t{2'147'483'648});
      });
  // Dedup adjacent duplicates after sort.
  auto unique_flag = parlay::tabulate(lits_nonzero.size(), [&](std::size_t i) {
    return (i == 0 || lits_nonzero[i] != lits_nonzero[i - 1]) ? 1 : 0;
  });
  auto unique_idx = parlay::pack_index<std::uint32_t>(unique_flag);
  parlay::sequence<std::int32_t> distinct_lits =
      parlay::map(unique_idx, [&](std::uint32_t i) { return lits_nonzero[i]; });
  const std::size_t L = distinct_lits.size();
  out.n_literals = L;

  // 3b. Build a hashmap: literal → class id (= index in distinct_lits).
  //     Ankerl flat hashmap, sequential build. This is the only
  //     genuinely-sequential phase; with ~10M-30M literals it's a few
  //     hundred ms. We could replace with a parallel hash table (like
  //     parlay's hashmap), but for now correctness > speed here.
  ankerl::unordered_dense::map<std::int32_t, Id> lit_to_class;
  lit_to_class.reserve(L);
  for (std::size_t i = 0; i < L; ++i) {
    lit_to_class.emplace(distinct_lits[i], static_cast<Id>(i));
  }

  // 3c. Identify negative literals; each gets one NOT term whose
  //     child is the positive-literal class. NOT term i lives at slot
  //     L + i in nodes_; we initially-union it with the negative
  //     literal's class so CC propagates equivalences across negation.
  //     Note: for files that explicitly carry NOT lines (the
  //     *_with_not/ variants), those NOT lines also become gate
  //     ENodes — we let them; they're redundant with the synthesized
  //     ones but harmless (CC will merge them).
  auto neg_lit_flag = parlay::tabulate(L, [&](std::size_t i) {
    return distinct_lits[i] < 0 ? 1 : 0;
  });
  auto neg_lit_idx = parlay::pack_index<std::uint32_t>(neg_lit_flag);
  const std::size_t M = neg_lit_idx.size();
  out.n_not_terms = M;

  // 3d. Compose the final ENode sequence.
  //     [0, L) literal leaves (in distinct_lits order)
  //     [L, L+M) NOT terms (one per negative literal)
  //     [L+M, total) gate terms
  const std::size_t G = gates.size();
  const std::size_t total = L + M + G;
  out.nodes = parlay::sequence<ENode>::uninitialized(total);

  // Literal leaves: op = "v<lit>".
  parlay::parallel_for(0, L, [&](std::size_t i) {
    new (&out.nodes[i]) ENode{
        std::string("v") + std::to_string(distinct_lits[i]),
        {}};
  });

  // NOT terms: child = positive-literal class of the corresponding
  // negative literal. We need lit_to_class for the positive twin;
  // hashmap lookup is fine here (parallel reads, no writes).
  parlay::parallel_for(0, M, [&](std::size_t i) {
    std::int32_t neg_lit = distinct_lits[neg_lit_idx[i]];
    std::int32_t pos_lit = -neg_lit;
    auto it = lit_to_class.find(pos_lit);
    Id pos_class = (it != lit_to_class.end())
        ? it->second
        : static_cast<Id>(neg_lit_idx[i]);  // fallback: self-loop
    new (&out.nodes[L + i]) ENode{std::string("NOT"),
                                   std::vector<Id>{pos_class}};
  });

  // Gate terms: children = literal class ids of the rhs entries.
  parlay::parallel_for(0, G, [&](std::size_t k) {
    const RawGate& g = gates[k];
    std::vector<Id> children(g.arity);
    for (std::uint8_t j = 0; j < g.arity; ++j) {
      children[j] = lit_to_class.at(g.rhs[j]);
    }
    new (&out.nodes[L + M + k]) ENode{
        std::string(gates_detail::op_name(g.op)),
        std::move(children)};
  });

  // 3e. Initial unions:
  //     - Each NOT term ≡ its negative-literal leaf class (M unions).
  //     - Each gate ≡ its lhs literal class (G unions).
  out.eqs = parlay::sequence<std::pair<Id, Id>>::uninitialized(M + G);
  parlay::parallel_for(0, M, [&](std::size_t i) {
    Id not_class = static_cast<Id>(L + i);
    Id neg_lit_class = static_cast<Id>(neg_lit_idx[i]);
    new (&out.eqs[i]) std::pair<Id, Id>{not_class, neg_lit_class};
  });
  parlay::parallel_for(0, G, [&](std::size_t k) {
    Id gate_class = static_cast<Id>(L + M + k);
    Id lhs_class  = lit_to_class.at(gates[k].lhs);
    new (&out.eqs[M + k]) std::pair<Id, Id>{gate_class, lhs_class};
  });

  out.build_s = sec_since(t_build);
  return out;
}

}  // namespace pe
