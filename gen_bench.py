#!/usr/bin/env python3
"""Generate scalable QF_UF benchmarks for stress-testing the e-graph.

Usage:
    python gen_bench.py <family> <n> [output_dir]
    python gen_bench.py all <n> [output_dir]       # generate all families at size n
    python gen_bench.py sweep <n1> <n2> <step> [output_dir]  # all families at n1, n1+step, ..., n2
    python gen_bench.py sweep:<family> <n1> <n2> <step> [output_dir]  # single family sweep

Families:
    chain — depth-n sequential congruence cascade                 O(n) congruences
    grid  — n merges, binary f on all pairs                       O(n²) congruences
    cube  — n merges, ternary f on all triples                    O(n³) congruences
    exp   — 1 merge cascading through n layers of 2 functions     O(2^n) congruences

Each benchmark produces a single UNSAT disequality that requires processing
the entire formula.  Balanced binary nesting keeps depth at O(log m) where
m is the number of leaf terms.
"""

import sys
import os


def nest_balanced(fn: str, elems: list[str]) -> str:
    """Build a balanced binary tree of function applications.

    nest_balanced("g", ["a", "b", "c", "d"]) => "(g (g a b) (g c d))"
    nest_balanced("g", ["a"]) => "a"

    Nesting depth is O(log n).
    """
    assert len(elems) >= 1
    if len(elems) == 1:
        return elems[0]
    if len(elems) == 2:
        return f"({fn} {elems[0]} {elems[1]})"
    mid = len(elems) // 2
    left = nest_balanced(fn, elems[:mid])
    right = nest_balanced(fn, elems[mid:])
    return f"({fn} {left} {right})"


def smt_header(comment_lines: list[str]) -> list[str]:
    lines = []
    for c in comment_lines:
        lines.append(f"; {c}")
    lines.append("(set-logic QF_UF)")
    lines.append("(declare-sort U 0)")
    return lines


# ---------------------------------------------------------------------------
# chain: O(n) sequential congruence cascade
# ---------------------------------------------------------------------------

def gen_chain(n: int) -> str:
    """a0=b0, a_{i+1}=f(a_i), b_{i+1}=f(b_i), assert a_n != b_n.

    Each congruence step depends on the previous one, so the cascade
    is inherently sequential.  Useful as a baseline showing the limit
    of parallelism.
    Merges: 2n+1.  Congruences: n (sequential).
    """
    lines = smt_header([
        f"Chain benchmark (n={n}): depth-{n} congruence cascade",
        f"UNSAT: requires {n} sequential congruence steps",
    ])
    lines.append("(declare-fun f (U) U)")
    for i in range(n + 1):
        lines.append(f"(declare-const a{i} U)")
        lines.append(f"(declare-const b{i} U)")
    lines.append("(assert (= a0 b0))")
    for i in range(n):
        lines.append(f"(assert (= a{i+1} (f a{i})))")
        lines.append(f"(assert (= b{i+1} (f b{i})))")
    lines.append(f"(assert (not (= a{n} b{n})))")
    lines.append("(check-sat)")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# grid: O(n²) congruences from n merges (binary f)
# ---------------------------------------------------------------------------

def gen_grid(n: int) -> str:
    """a_i = b_i  =>  f(a_i, a_j) = f(b_i, b_j) for all i,j.

    Inline f-terms in a balanced g-tree disequality.
    Merges: n.  Congruences: n².  Disequality depth: log(n²) + 1.
    """
    m = n * n
    lines = smt_header([
        f"Grid benchmark (n={n}): {n} merges, {m} congruences",
        f"UNSAT: a_i=b_i => f(a_i,a_j)=f(b_i,b_j) for all i,j",
    ])
    lines.append("(declare-fun f (U U) U)")
    if m >= 2:
        lines.append("(declare-fun g (U U) U)")

    for i in range(n):
        lines.append(f"(declare-const a{i} U)")
        lines.append(f"(declare-const b{i} U)")

    for i in range(n):
        lines.append(f"(assert (= a{i} b{i}))")

    a_elems = [f"(f a{i} a{j})" for i in range(n) for j in range(n)]
    b_elems = [f"(f b{i} b{j})" for i in range(n) for j in range(n)]
    nest_a = nest_balanced("g", a_elems)
    nest_b = nest_balanced("g", b_elems)
    lines.append(f"(assert (not (= {nest_a} {nest_b})))")
    lines.append("(check-sat)")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# cube: O(n³) congruences from n merges (ternary f)
# ---------------------------------------------------------------------------

def gen_cube(n: int) -> str:
    """a_i = b_i  =>  f(a_i, a_j, a_k) = f(b_i, b_j, b_k) for all i,j,k.

    Inline f-terms in a balanced g-tree disequality.
    Merges: n.  Congruences: n³.  Disequality depth: log(n³) + 1.
    """
    m = n ** 3
    lines = smt_header([
        f"Cube benchmark (n={n}): {n} merges, {m} congruences",
        f"UNSAT: a_i=b_i => f(a_i,a_j,a_k)=f(b_i,b_j,b_k) for all i,j,k",
    ])
    lines.append("(declare-fun f (U U U) U)")
    if m >= 2:
        lines.append("(declare-fun g (U U) U)")

    for i in range(n):
        lines.append(f"(declare-const a{i} U)")
        lines.append(f"(declare-const b{i} U)")

    for i in range(n):
        lines.append(f"(assert (= a{i} b{i}))")

    a_elems = [f"(f a{i} a{j} a{k})" for i in range(n) for j in range(n) for k in range(n)]
    b_elems = [f"(f b{i} b{j} b{k})" for i in range(n) for j in range(n) for k in range(n)]
    nest_a = nest_balanced("g", a_elems)
    nest_b = nest_balanced("g", b_elems)
    lines.append(f"(assert (not (= {nest_a} {nest_b})))")
    lines.append("(check-sat)")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# exp: O(2^n) congruences from 1 merge (exponential cascade)
# ---------------------------------------------------------------------------

def gen_exp(n: int) -> str:
    """Single merge a=b cascades through n layers, each with 2 unary functions.

    Layer i has 2^i named terms per side.  The merge a=b triggers layer-1
    congruences, which trigger layer-2, etc.  Named intermediates are
    essential here: each layer references the previous layer's constants.
    Total congruences: 2^{n+1}-2.  Disequality depth: n.
    """
    total_cong = 2 ** (n + 1) - 2 if n >= 1 else 0
    lines = smt_header([
        f"Exp benchmark (n={n}): 1 merge, {total_cong} congruences (exponential cascade)",
        f"UNSAT: a=b cascades through {n} layers of 2 functions each",
    ])

    for i in range(2 * n):
        lines.append(f"(declare-fun f{i} (U) U)")
    if n >= 1:
        lines.append("(declare-fun g (U U) U)")

    lines.append("(declare-const a U)")
    lines.append("(declare-const b U)")

    prev_a = ["a"]
    prev_b = ["b"]

    for layer in range(1, n + 1):
        fn0 = f"f{2 * (layer - 1)}"
        fn1 = f"f{2 * (layer - 1) + 1}"
        curr_a = []
        curr_b = []

        for fn in [fn0, fn1]:
            for parent in prev_a:
                name = f"ta{layer}_{len(curr_a)}"
                lines.append(f"(declare-const {name} U)")
                lines.append(f"(assert (= {name} ({fn} {parent})))")
                curr_a.append(name)
            for parent in prev_b:
                name = f"tb{layer}_{len(curr_b)}"
                lines.append(f"(declare-const {name} U)")
                lines.append(f"(assert (= {name} ({fn} {parent})))")
                curr_b.append(name)

        prev_a = curr_a
        prev_b = curr_b

    lines.append("(assert (= a b))")

    nest_a = nest_balanced("g", prev_a)
    nest_b = nest_balanced("g", prev_b)
    lines.append(f"(assert (not (= {nest_a} {nest_b})))")
    lines.append("(check-sat)")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------

FAMILIES = {
    "chain": gen_chain,
    "grid": gen_grid,
    "cube": gen_cube,
    "exp": gen_exp,
}


def write_bench(family: str, n: int, output_dir: str | None):
    smt = FAMILIES[family](n)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, f"{family}_n{n}_unsat.smt2")
        with open(path, "w") as f:
            f.write(smt)
        print(f"  {path}")
    else:
        print(smt, end="")


def usage():
    print(__doc__, file=sys.stderr)
    sys.exit(1)


def main():
    args = sys.argv[1:]
    if len(args) < 2:
        usage()

    cmd = args[0]

    if cmd == "all":
        n = int(args[1])
        out = args[2] if len(args) > 2 else "synthetic_benchmarks"
        for family in FAMILIES:
            write_bench(family, n, out)

    elif cmd.startswith("sweep"):
        if len(args) < 4:
            usage()
        n1, n2, step = int(args[1]), int(args[2]), int(args[3])
        out = args[4] if len(args) > 4 else "synthetic_benchmarks"
        if ":" in cmd:
            fam = cmd.split(":", 1)[1]
            if fam not in FAMILIES:
                print(f"Unknown family: {fam}", file=sys.stderr)
                usage()
            families = [fam]
        else:
            families = list(FAMILIES)
        for n in range(n1, n2 + 1, step):
            for family in families:
                write_bench(family, n, out)

    elif cmd in FAMILIES:
        n = int(args[1])
        out = args[2] if len(args) > 2 else None
        write_bench(cmd, n, out)

    else:
        print(f"Unknown command/family: {cmd}", file=sys.stderr)
        usage()


if __name__ == "__main__":
    main()
