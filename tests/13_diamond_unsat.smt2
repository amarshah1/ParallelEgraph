; Diamond-shaped congruence:
;   a = b
;   f(a) = c,  f(b) = d   =>  c = d  (because f(a) = f(b) by congruence)
;   g(c) and g(d) must be equal by congruence on c = d
; Expected: UNSAT
(set-logic QF_UF)
(declare-sort U 0)
(declare-fun f (U) U)
(declare-fun g (U) U)
(declare-const a U)
(declare-const b U)
(declare-const c U)
(declare-const d U)
(assert (= a b))
(assert (= (f a) c))
(assert (= (f b) d))
(assert (not (= (g c) (g d))))
(check-sat)
