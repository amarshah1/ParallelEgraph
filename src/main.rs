use std::fs;
use std::path::PathBuf;

use parallel_egraph::{solve_timed, solve_with_mode, SolveResult};

fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();

    let mut parallel = false;
    let mut timing = false;
    let mut profile_solve = false;
    let mut file_arg = None;

    for arg in &args[1..] {
        match arg.as_str() {
            "--parallel" | "-p" => parallel = true,
            "--timing" | "-t" => timing = true,
            // When set, raises SIGSTOP before the solve phase so a profiler
            // (samply, flamegraph, perf) attached to the process only captures
            // the interesting work.  Send SIGCONT to resume:
            //   samply record -p <pid>   (then SIGCONT from another shell)
            "--profile-solve" => profile_solve = true,
            _ if file_arg.is_none() => file_arg = Some(arg.clone()),
            _ => return Err(format!("Unexpected argument: {arg}")),
        }
    }

    let input_file: PathBuf = file_arg
        .ok_or("Usage: parallel-egraph [--parallel] [--timing] [--profile-solve] <smt2-file>")?
        .into();

    let input = fs::read_to_string(&input_file)
        .map_err(|e| format!("Error reading file {}: {}", input_file.display(), e))?;

    if timing || profile_solve {
        let (result, timings) = solve_timed(&input, parallel)?;

        match result {
            SolveResult::Sat => println!("sat"),
            SolveResult::Unsat => println!("unsat"),
        }

        // Machine-parseable timing line on stderr
        eprintln!("{timings}");
    } else {
        let result = solve_with_mode(&input, parallel)?;

        match result {
            SolveResult::Sat => println!("sat"),
            SolveResult::Unsat => println!("unsat"),
        }
    }

    Ok(())
}
