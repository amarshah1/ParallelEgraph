use yaspar_ir::ast::{Repr, Term};
use yaspar_ir::ast::alg;
use yaspar_ir::traits::Contains;

use crate::egraph::{EGraph, ENode, Id};

/// Iteratively add a yaspar-ir Term into the e-graph, returning its e-class id.
/// Uses an explicit stack to avoid stack overflow on deeply nested terms.
pub fn add_term(eg: &mut EGraph, term: &Term) -> Id {
    enum Work<'a> {
        Process(&'a Term),
        Build(String, usize), // op, num_children
    }

    let mut stack: Vec<Work> = vec![Work::Process(term)];
    let mut results: Vec<Id> = Vec::new();

    while let Some(work) = stack.pop() {
        match work {
            Work::Process(t) => match t.repr() {
                alg::Term::Constant(c, _) => {
                    let op = format!("{}", c);
                    results.push(eg.add(ENode::leaf(op)));
                }
                alg::Term::Global(qid, _) => {
                    let name = qid.id_str().inner().clone();
                    results.push(eg.add(ENode::leaf(name)));
                }
                alg::Term::App(qid, args, _) => {
                    let op = qid.id_str().inner().clone();
                    stack.push(Work::Build(op, args.len()));
                    // Push children in reverse so the first child is processed first
                    for arg in args.iter().rev() {
                        stack.push(Work::Process(arg));
                    }
                }
                _ => panic!(
                    "Unsupported term variant: only constants, globals, and function applications are supported"
                ),
            },
            Work::Build(op, n) => {
                let start = results.len() - n;
                let children: Vec<Id> = results.drain(start..).collect();
                results.push(eg.add(ENode::new(op, children)));
            }
        }
    }

    results.pop().unwrap()
}

/// An assertion is either an equality (merge) or a disequality (no merge).
pub enum Assertion {
    Equality(Id, Id),
    Disequality(Id, Id),
}

/// Process an assertion. Supports:
/// - `(= a b)`: equality (merge)
/// - `(not (= a b))`: disequality
/// Panics on anything else.
pub fn process_assertion(eg: &mut EGraph, term: &Term) -> Assertion {
    match term.repr() {
        alg::Term::Eq(a, b) => {
            let id_a = add_term(eg, a);
            let id_b = add_term(eg, b);
            Assertion::Equality(id_a, id_b)
        }
        alg::Term::Not(inner) => {
            match inner.repr() {
                alg::Term::Eq(a, b) => {
                    let id_a = add_term(eg, a);
                    let id_b = add_term(eg, b);
                    Assertion::Disequality(id_a, id_b)
                }
                _ => panic!(
                    "Unsupported negated assertion: only (not (= ...)) is supported"
                ),
            }
        }
        _ => panic!(
            "Unsupported assertion: only (= ...) and (not (= ...)) are supported"
        ),
    }
}
