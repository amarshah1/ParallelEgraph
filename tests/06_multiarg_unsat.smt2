; a = c, b = d  =>  f(a, b) = f(c, d) by congruence on both arguments.
; Expected: UNSAT
(set-logic QF_UF)
(declare-sort U 0)
(declare-fun f (U U) U)
(declare-const a U)
(declare-const b U)
(declare-const c U)
(declare-const d U)
(assert (= a c))
(assert (= b d))
(assert (not (= (f a b) (f c d))))
(check-sat)
