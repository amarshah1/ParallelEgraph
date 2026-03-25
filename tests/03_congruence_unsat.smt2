; a = b implies f(a) = f(b) by congruence.
; The disequality f(a) != f(b) is violated.
; Expected: UNSAT
(set-logic QF_UF)
(declare-sort U 0)
(declare-fun f (U) U)
(declare-const a U)
(declare-const b U)
(assert (= a b))
(assert (not (= (f a) (f b))))
(check-sat)
