; Multiple equalities and multiple disequalities, all satisfiable.
; a = b, c = d, but {a,b} and {c,d} are in separate equivalence classes.
; Expected: SAT
(set-logic QF_UF)
(declare-sort U 0)
(declare-fun f (U) U)
(declare-fun g (U) U)
(declare-const a U)
(declare-const b U)
(declare-const c U)
(declare-const d U)
(assert (= a b))
(assert (= c d))
(assert (not (= a c)))
(assert (not (= (f a) (f c))))
(assert (not (= (g a) (g c))))
(check-sat)
