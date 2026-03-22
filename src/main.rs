use std::fs;
use std::path::PathBuf;
use yaspar_ir::ast::alg;
use yaspar_ir::ast::{Context, Repr, Term, Typecheck};
use yaspar_ir::untyped::UntypedAst;

use parallel_egraph::egraph::EGraph;
use parallel_egraph::process::{self, Assertion};

fn main() -> Result<(), String> {
    let input_file: PathBuf = std::env::args()
        .nth(1)
        .ok_or("Usage: parallel-egraph <smt2-file>")?
        .into();

    let input = fs::read_to_string(&input_file)
        .map_err(|e| format!("Error reading file {}: {}", input_file.display(), e))?;

    let commands = UntypedAst
        .parse_script_str(&input)
        .map_err(|e| format!("Error parsing SMT file: {e}"))?;

    let mut context = Context::new();

    let typed_commands = commands
        .type_check(&mut context)
        .map_err(|e| format!("Error checking typed commands: {e}"))?;

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

    // Phase 1: add all terms and collect merges
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

    // Phase 2: congruence closure
    eg.rebuild();

    // Phase 3: check disequalities
    let mut sat = true;
    for (a, b) in &disequalities {
        if eg.equiv(*a, *b) {
            println!("UNSAT: disequality violated between classes {} and {}", a, b);
            sat = false;
        }
    }

    if sat {
        println!("All disequalities satisfied.");
    }

    println!();
    eg.print();

    Ok(())
}
