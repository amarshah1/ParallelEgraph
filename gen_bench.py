#!/usr/bin/env python3
"""Generate scalable QF_UF benchmarks for stress-testing the e-graph.

Usage:
    python gen_bench.py <family> <n> [output_dir]
    python gen_bench.py all <n> [output_dir]       # generate all families at size n
    python gen_bench.py sweep <n1> <n2> <step> [output_dir]  # all families at n1, n1+step, ..., n2

Families:
    chain  — depth-n congruence cascade (sequential rebuild stress)
    wide   — n independent merges, single nested disequality (parallel merge stress)
    fanout — 1 merge triggering n congruences, single nested disequality (parallel rebuild stress)
    grid   — n merges with n^2 congruences, single nested disequality (combined stress)

Each benchmark produces a single UNSAT disequality that requires processing
the entire formula — no individual assertion pair gives an immediate contradiction.
Larger n produces deeper nesting and more function applications.
"""

import sys
import os


def nest_binary(fn: str, elems: list[str]) -> str:
    """Build a right-nested binary function application.

    nest_binary("f", ["a", "b", "c", "d"]) => "(f a (f b (f c d)))"
    nest_binary("f", ["a"]) => "a"
    """
    assert len(elems) >= 1
    result = elems[-1]
    for i in range(len(elems) - 2, -1, -1):
        result = f"({fn} {elems[i]} {result})"
    return result


def gen_chain(n: int) -> str:
    """Chain: a0=b0, a_{i+1}=f(a_i), b_{i+1}=f(b_i), a_n!=b_n.

    Single disequality at depth n requires all n sequential congruence steps.
    Merges: 2n+1. Congruence cascade depth: n.
    """
    lines = [
        f"; Chain benchmark (n={n}): depth-{n} congruence cascade",
        f"; UNSAT: single disequality requires {n} sequential congruence steps",
        "(set-logic QF_UF)",
        "(declare-sort U 0)",
        "(declare-fun f (U) U)",
    ]
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


def gen_wide(n: int) -> str:
    """Wide: n independent merges feeding a single deeply nested disequality.

    a_i = b_i for all i. A single disequality over right-nested f applications
    requires all n merges to resolve.  No individual merge immediately
    contradicts any assertion.
    Merges: n. Nesting depth: n-1.
    """
    lines = [
        f"; Wide benchmark (n={n}): {n} independent merges, single nested disequality",
        f"; UNSAT: requires all {n} merges to resolve nested congruence",
        "(set-logic QF_UF)",
        "(declare-sort U 0)",
        "(declare-fun f (U U) U)",
    ]
    for i in range(n):
        lines.append(f"(declare-const a{i} U)")
        lines.append(f"(declare-const b{i} U)")

    for i in range(n):
        lines.append(f"(assert (= a{i} b{i}))")

    a_elems = [f"a{i}" for i in range(n)]
    b_elems = [f"b{i}" for i in range(n)]
    nest_a = nest_binary("f", a_elems)
    nest_b = nest_binary("f", b_elems)
    lines.append(f"(assert (not (= {nest_a} {nest_b})))")

    lines.append("(check-sat)")
    return "\n".join(lines) + "\n"


def gen_fanout(n: int) -> str:
    """Fanout: single merge a=b triggers n congruences, one per function.

    Uses a nested binary function g to combine all n function results into
    a single disequality that requires all congruences to resolve.
    No individual f_i(a)!=f_i(b) assertion exists.
    Merges: 1. Congruences: n. Nesting depth: n.
    """
    lines = [
        f"; Fanout benchmark (n={n}): 1 merge, {n} congruences, single nested disequality",
        f"; UNSAT: a=b triggers f_i(a)=f_i(b) for all i, nested disequality requires all",
        "(set-logic QF_UF)",
        "(declare-sort U 0)",
    ]
    for i in range(n):
        lines.append(f"(declare-fun f{i} (U) U)")
    if n >= 2:
        lines.append("(declare-fun g (U U) U)")

    lines.append("(declare-const a U)")
    lines.append("(declare-const b U)")
    lines.append("(assert (= a b))")

    a_elems = [f"(f{i} a)" for i in range(n)]
    b_elems = [f"(f{i} b)" for i in range(n)]
    nest_a = nest_binary("g", a_elems)
    nest_b = nest_binary("g", b_elems)
    lines.append(f"(assert (not (= {nest_a} {nest_b})))")

    lines.append("(check-sat)")
    return "\n".join(lines) + "\n"


def gen_grid(n: int) -> str:
    """Grid: n constant pairs with binary f, single nested disequality.

    n base merges produce n^2 congruences on f(a_i, a_j) = f(b_i, b_j).
    A nested binary g collects all n^2 results into one disequality.
    Merges: n. Congruences: n^2. Nesting depth: n^2.
    """
    lines = [
        f"; Grid benchmark (n={n}): {n} merges, {n*n} congruences, single nested disequality",
        f"; UNSAT: a_i=b_i for all i, nested g over f(a_i,a_j) vs f(b_i,b_j)",
        "(set-logic QF_UF)",
        "(declare-sort U 0)",
        "(declare-fun f (U U) U)",
    ]
    if n >= 2:
        lines.append("(declare-fun g (U U) U)")

    for i in range(n):
        lines.append(f"(declare-const a{i} U)")
        lines.append(f"(declare-const b{i} U)")

    for i in range(n):
        lines.append(f"(assert (= a{i} b{i}))")

    a_elems = [f"(f a{i} a{j})" for i in range(n) for j in range(n)]
    b_elems = [f"(f b{i} b{j})" for i in range(n) for j in range(n)]
    nest_a = nest_binary("g", a_elems)
    nest_b = nest_binary("g", b_elems)
    lines.append(f"(assert (not (= {nest_a} {nest_b})))")

    lines.append("(check-sat)")
    return "\n".join(lines) + "\n"


FAMILIES = {
    "chain": gen_chain,
    "wide": gen_wide,
    "fanout": gen_fanout,
    "grid": gen_grid,
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

    elif cmd == "sweep":
        if len(args) < 4:
            usage()
        n1, n2, step = int(args[1]), int(args[2]), int(args[3])
        out = args[4] if len(args) > 4 else "synthetic_benchmarks"
        for n in range(n1, n2 + 1, step):
            for family in FAMILIES:
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
