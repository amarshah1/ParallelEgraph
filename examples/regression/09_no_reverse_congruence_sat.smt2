; f(a) = f(b) does NOT imply a = b (f is not injective).
; The disequality a != b is satisfiable.
; Expected: SAT
(set-logic QF_UF)
(declare-sort U 0)
(declare-fun f (U) U)
(declare-const a U)
(declare-const b U)
(assert (= (f a) (f b)))
(assert (not (= a b)))
(check-sat)
