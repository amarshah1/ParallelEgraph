; ite buried inside a UF App
(set-logic QF_UF)
(declare-sort U 0)
(declare-fun f (U) U)
(declare-const a U) (declare-const b U) (declare-const v U) (declare-const w U)
(assert (= (f (ite (= a b) v w)) (f v)))
(check-sat)
