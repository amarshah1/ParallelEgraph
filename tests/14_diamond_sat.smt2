; Similar structure but the chain is broken.
;   a = b
;   f(a) = c  (but NO f(b) = d assertion)
;   g(c) != g(d) is satisfiable because c and d are independent.
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
(assert (= (f a) c))
(assert (not (= (g c) (g d))))
(check-sat)
