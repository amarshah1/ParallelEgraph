; Named alias bound to a function term, then used as a child of a further
; function. Exercises parent_index[non-root] migration during congruence
; closure: after (= t (f a)), the rank tiebreak can make f(a)'s class the
; root, orphaning parent_index[t_class] — which is where g(t) is registered.
;   a = b  =>  f(a) = f(b)  =>  t = f(b)  =>  g(t) = g(f(b))
; Expected: UNSAT
(set-logic QF_UF)
(declare-sort U 0)
(declare-fun f (U) U)
(declare-fun g (U) U)
(declare-const a U)
(declare-const b U)
(declare-const t U)
(assert (= t (f a)))
(assert (= a b))
(assert (not (= (g t) (g (f b)))))
(check-sat)
