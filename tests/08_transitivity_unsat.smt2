; a = b, b = c, c = d  =>  a = d by transitivity  =>  f(a) = f(d) by congruence.
; Expected: UNSAT
(set-logic QF_UF)
(declare-sort U 0)
(declare-fun f (U) U)
(declare-const a U)
(declare-const b U)
(declare-const c U)
(declare-const d U)
(assert (= a b))
(assert (= b c))
(assert (= c d))
(assert (not (= (f a) (f d))))
(check-sat)
