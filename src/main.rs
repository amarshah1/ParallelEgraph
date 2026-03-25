use std::fs;
use std::path::PathBuf;

use parallel_egraph::{solve, SolveResult};

fn main() -> Result<(), String> {
    let input_file: PathBuf = std::env::args()
        .nth(1)
        .ok_or("Usage: parallel-egraph <smt2-file>")?
        .into();

    let input = fs::read_to_string(&input_file)
        .map_err(|e| format!("Error reading file {}: {}", input_file.display(), e))?;

    match solve(&input)? {
        SolveResult::Sat => println!("sat"),
        SolveResult::Unsat => println!("unsat"),
    }

    Ok(())
}
