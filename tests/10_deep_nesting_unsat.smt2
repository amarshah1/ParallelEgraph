; a = b propagates through 4 levels of nesting:
;   f(a) = f(b), g(f(a)) = g(f(b)), h(g(f(a))) = h(g(f(b))),
;   k(h(g(f(a)))) = k(h(g(f(b))))
; Expected: UNSAT
(set-logic QF_UF)
(declare-sort U 0)
(declare-fun f (U) U)
(declare-fun g (U) U)
(declare-fun h (U) U)
(declare-fun k (U) U)
(declare-const a U)
(declare-const b U)
(assert (= a b))
(assert (not (= (k (h (g (f a)))) (k (h (g (f b)))))))
(check-sat)
