; Multiple equalities that together cause a hidden congruence violation.
; a = c, b = d  =>  f(a, b) = f(c, d)
; f(a, b) = x, f(c, d) = y  =>  x = y
; not(x = y) is violated.
; Expected: UNSAT
(set-logic QF_UF)
(declare-sort U 0)
(declare-fun f (U U) U)
(declare-const a U)
(declare-const b U)
(declare-const c U)
(declare-const d U)
(declare-const x U)
(declare-const y U)
(assert (= a c))
(assert (= b d))
(assert (= (f a b) x))
(assert (= (f c d) y))
(assert (not (= x y)))
(check-sat)
