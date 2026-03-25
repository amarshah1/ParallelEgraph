pub mod process;
pub mod unionfind;
pub mod egraph;

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

/// Parse and solve an SMT-LIB QF_UF string. Returns Sat or Unsat.
pub fn solve(input: &str) -> Result<SolveResult, String> {
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

    let mut eg = EGraph::new();
    let mut disequalities: Vec<(u32, u32)> = Vec::new();

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

    eg.rebuild();

    for (a, b) in &disequalities {
        if eg.equiv(*a, *b) {
            return Ok(SolveResult::Unsat);
        }
    }

    Ok(SolveResult::Sat)
}