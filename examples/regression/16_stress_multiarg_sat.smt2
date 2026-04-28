; Stress test: many equalities but all disequalities are between
; genuinely different equivalence classes.
; Expected: SAT
(set-logic QF_UF)
(declare-sort U 0)
(declare-fun f (U U) U)
(declare-fun g (U) U)
(declare-const a0 U)
(declare-const a1 U)
(declare-const a2 U)
(declare-const a3 U)
(declare-const b0 U)
(declare-const b1 U)
(declare-const b2 U)
(declare-const b3 U)
; group A: a0 = a1 = a2 = a3
(assert (= a0 a1))
(assert (= a1 a2))
(assert (= a2 a3))
; group B: b0 = b1 = b2 = b3
(assert (= b0 b1))
(assert (= b1 b2))
(assert (= b2 b3))
; A and B are separate, so these disequalities hold:
(assert (not (= a0 b0)))
(assert (not (= (g a0) (g b0))))
(assert (not (= (f a0 b0) (f b0 a0))))
(check-sat)
