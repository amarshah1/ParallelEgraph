use std::fs;
use std::path::PathBuf;

use parallel_egraph::{solve, solve_parallel, SolveResult};

fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();

    let mut parallel = false;
    let mut file_arg = None;

    for arg in &args[1..] {
        if arg == "--parallel" || arg == "-p" {
            parallel = true;
        } else if file_arg.is_none() {
            file_arg = Some(arg.clone());
        } else {
            return Err(format!("Unexpected argument: {arg}"));
        }
    }

    let input_file: PathBuf = file_arg
        .ok_or("Usage: parallel-egraph [--parallel] <smt2-file>")?
        .into();

    let input = fs::read_to_string(&input_file)
        .map_err(|e| format!("Error reading file {}: {}", input_file.display(), e))?;

    let result = if parallel {
        solve_parallel(&input)?
    } else {
        solve(&input)?
    };

    match result {
        SolveResult::Sat => println!("sat"),
        SolveResult::Unsat => println!("unsat"),
    }

    Ok(())
}
