pub mod process;
pub mod unionfind;
pub mod egraph;

use tikv_jemallocator::Jemalloc;

#[global_allocator]
static GLOBAL: Jemalloc = Jemalloc;

use std::time::Instant;

use yaspar_ir::ast::alg;
use yaspar_ir::ast::{Context, Repr, Term, Typecheck};
use yaspar_ir::untyped::UntypedAst;

use egraph::EGraph;
use process::Assertion;

/// Result of solving an SMT-LIB QF_UF instance.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SolveResult {
    Sat,
    Unsat,
}

/// Detailed timing breakdown for each solver phase (in seconds).
#[derive(Debug, Clone)]
pub struct SolveTimings {
    pub parse_s: f64,
    pub build_s: f64,   // adding terms to the e-graph
    pub merge_s: f64,   // union-find merges (parallel or sequential)
    pub rebuild_s: f64, // congruence closure
    pub check_s: f64,   // disequality checking
    pub total_s: f64,
    /// build + merge + rebuild + check (excludes parsing)
    pub solve_s: f64,
}

impl std::fmt::Display for SolveTimings {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "timing: parse={:.6} build={:.6} merge={:.6} rebuild={:.6} check={:.6} solve={:.6} total={:.6}",
            self.parse_s, self.build_s, self.merge_s, self.rebuild_s, self.check_s, self.solve_s, self.total_s,
        )
    }
}

/// Parse and solve an SMT-LIB QF_UF string.
pub fn solve_with_mode(input: &str, parallel: bool) -> Result<SolveResult, String> {
    let (result, _) = solve_timed(input, parallel)?;
    Ok(result)
}

/// Emit a profiling marker. When profiling with samply/flamegraph, this
/// function shows up in the trace so you can filter to only the solve phase.
/// Call `begin_solve_region` / `end_solve_region` around the hot path.
#[inline(never)]
pub fn begin_solve_region() {}

#[inline(never)]
pub fn end_solve_region() {}

/// Parse and solve, returning both the result and a detailed timing breakdown.
pub fn solve_timed(input: &str, parallel: bool) -> Result<(SolveResult, SolveTimings), String> {
    let total_start = Instant::now();

    // --- Parse ---
    let parse_start = Instant::now();
    let commands = UntypedAst
        .parse_script_str(input)
        .map_err(|e| format!("Parse error: {e}"))?;

    let mut context = Context::new();
    let typed_commands = commands
        .type_check(&mut context)
        .map_err(|e| format!("Type-check error: {e}"))?;

    let assertions: Vec<Term> = typed_commands
        .iter()
        .filter_map(|c| {
            if let alg::Command::Assert(t) = c.repr() {
                Some(t.clone())
            } else {
                None
            }
        })
        .collect();
    let parse_s = parse_start.elapsed().as_secs_f64();

    // --- Solve (build + merge + rebuild + check) ---
    begin_solve_region();
    let mut eg = if parallel { EGraph::new_parallel() } else { EGraph::new() };
    let mut disequalities: Vec<(u32, u32)> = Vec::new();

    let build_start = Instant::now();
    if parallel {
        let mut equalities: Vec<(u32, u32)> = Vec::new();
        for assertion in &assertions {
            match process::process_assertion(&mut eg, assertion) {
                Assertion::Equality(a, b) => equalities.push((a, b)),
                Assertion::Disequality(a, b) => disequalities.push((a, b)),
            }
        }
        let build_s = build_start.elapsed().as_secs_f64();

        let merge_start = Instant::now();
        eg.parallel_merge_all(&equalities);
        let merge_s = merge_start.elapsed().as_secs_f64();

        // --- Rebuild ---
        let rebuild_start = Instant::now();
        eg.rebuild();
        let rebuild_s = rebuild_start.elapsed().as_secs_f64();

        // --- Check disequalities ---
        let check_start = Instant::now();
        let mut result = SolveResult::Sat;
        for (a, b) in &disequalities {
            if eg.equiv(*a, *b) {
                result = SolveResult::Unsat;
                break;
            }
        }
        let check_s = check_start.elapsed().as_secs_f64();

        end_solve_region();
        let total_s = total_start.elapsed().as_secs_f64();
        let solve_s = build_s + merge_s + rebuild_s + check_s;

        Ok((result, SolveTimings { parse_s, build_s, merge_s, rebuild_s, check_s, total_s, solve_s }))
    } else {
        // Sequential: build and merge are interleaved
        for assertion in &assertions {
            match process::process_assertion(&mut eg, assertion) {
                Assertion::Equality(a, b) => {
                    eg.merge(a, b);
                }
                Assertion::Disequality(a, b) => {
                    disequalities.push((a, b));
                }
            }
        }
        let build_s = 0.0; // interleaved, can't separate
        let merge_s = build_start.elapsed().as_secs_f64(); // build+merge combined

        // --- Rebuild ---
        let rebuild_start = Instant::now();
        eg.rebuild();
        let rebuild_s = rebuild_start.elapsed().as_secs_f64();

        // --- Check disequalities ---
        let check_start = Instant::now();
        let mut result = SolveResult::Sat;
        for (a, b) in &disequalities {
            if eg.equiv(*a, *b) {
                result = SolveResult::Unsat;
                break;
            }
        }
        let check_s = check_start.elapsed().as_secs_f64();

        end_solve_region();
        let total_s = total_start.elapsed().as_secs_f64();
        let solve_s = merge_s + rebuild_s + check_s;

        Ok((result, SolveTimings { parse_s, build_s, merge_s, rebuild_s, check_s, total_s, solve_s }))
    }
}
