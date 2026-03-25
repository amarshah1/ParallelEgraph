; a = b, but f(a) != g(b) is fine because f and g are different symbols.
; Expected: SAT
(set-logic QF_UF)
(declare-sort U 0)
(declare-fun f (U) U)
(declare-fun g (U) U)
(declare-const a U)
(declare-const b U)
(assert (= a b))
(assert (not (= (f a) (g b))))
(check-sat)
