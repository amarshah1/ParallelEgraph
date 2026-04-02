use std::fs;
use std::path::Path;

use parallel_egraph::{solve_with_mode, SolveResult};

fn expected_result(filename: &str) -> SolveResult {
    if filename.contains("_unsat") {
        SolveResult::Unsat
    } else if filename.contains("_sat") {
        SolveResult::Sat
    } else {
        panic!("Cannot determine expected result from filename: {filename}");
    }
}

fn run_smt2(name: &str) {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join(name);
    let input = fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("failed to read {}: {e}", path.display()));
    let expected = expected_result(name);
    let actual =
        solve_with_mode(&input, false).unwrap_or_else(|e| panic!("{name}: solve failed: {e}"));
    assert_eq!(actual, expected, "{name}: expected {expected:?}, got {actual:?}");
}

macro_rules! smt2_test {
    ($func_name:ident, $file:expr) => {
        #[test]
        fn $func_name() {
            run_smt2($file);
        }
    };
}

smt2_test!(t01_trivial_sat,              "01_trivial_sat.smt2");
smt2_test!(t02_trivial_unsat,            "02_trivial_unsat.smt2");
smt2_test!(t03_congruence_unsat,         "03_congruence_unsat.smt2");
smt2_test!(t04_different_ops_sat,        "04_different_ops_sat.smt2");
smt2_test!(t05_cascade_unsat,            "05_cascade_unsat.smt2");
smt2_test!(t06_multiarg_unsat,           "06_multiarg_unsat.smt2");
smt2_test!(t07_partial_args_sat,         "07_partial_args_sat.smt2");
smt2_test!(t08_transitivity_unsat,       "08_transitivity_unsat.smt2");
smt2_test!(t09_no_reverse_congruence_sat,"09_no_reverse_congruence_sat.smt2");
smt2_test!(t10_deep_nesting_unsat,       "10_deep_nesting_unsat.smt2");
smt2_test!(t11_multi_diseq_sat,          "11_multi_diseq_sat.smt2");
smt2_test!(t12_hidden_congruence_unsat,  "12_hidden_congruence_unsat.smt2");
smt2_test!(t13_diamond_unsat,            "13_diamond_unsat.smt2");
smt2_test!(t14_diamond_sat,              "14_diamond_sat.smt2");
smt2_test!(t15_stress_unsat,             "15_stress_unsat.smt2");
smt2_test!(t16_stress_multiarg_sat,      "16_stress_multiarg_sat.smt2");
