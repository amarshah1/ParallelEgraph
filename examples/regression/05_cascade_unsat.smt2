; a = b  =>  f(a) = f(b)  =>  g(f(a)) = g(f(b)) by cascading congruence.
; Expected: UNSAT
(set-logic QF_UF)
(declare-sort U 0)
(declare-fun f (U) U)
(declare-fun g (U) U)
(declare-const a U)
(declare-const b U)
(assert (= a b))
(assert (not (= (g (f a)) (g (f b)))))
(check-sat)
