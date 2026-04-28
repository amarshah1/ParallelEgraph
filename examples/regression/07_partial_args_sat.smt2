; a = c but b and d are NOT asserted equal.
; f(a, b) != f(c, d) is satisfiable because the second argument may differ.
; Expected: SAT
(set-logic QF_UF)
(declare-sort U 0)
(declare-fun f (U U) U)
(declare-const a U)
(declare-const b U)
(declare-const c U)
(declare-const d U)
(assert (= a c))
(assert (not (= (f a b) (f c d))))
(check-sat)
