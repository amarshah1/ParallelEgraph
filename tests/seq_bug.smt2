; Sequential-closure bug repro: 0=2 and 4=1 should imply g(4) ≡ g(5).
; Construction:
;   constants a, b, c
;   f, g unary
;   t40 = f(a)
;   t41 = f(c)        ; 4 = a, 5 = c → 4≡5 by congruence
;   t6  = g(f(a))
;   t7  = g(b)        ; 4 = b → 6≡7 directly (after f(a) = b)
;   t8  = g(f(c))     ; 8 = g(5) — should ≡ 6 ≡ 7
;
; Asserted: a = c, f(a) = b. The disequality (g(f(a)) != g(f(c))) is
; UNSAT under correct congruence closure, since:
;   a = c → f(a) ≡ f(c) → g(f(a)) ≡ g(f(c)).
;
; If we change the disequality to (g(b) != g(f(c))), it's still UNSAT:
;   a = c → f(a) ≡ f(c). f(a) = b → b ≡ f(c). g(b) ≡ g(f(c)).
;
; This is the case the buggy sequential closure misses because it
; processes g(f(a)) before discovering f(a)≡f(c) is implied, leaving a
; stale sig_table entry for g(f(a)) under the old (pre-merge) hash.

(set-logic QF_UF)
(declare-sort U 0)
(declare-fun f (U) U)
(declare-fun g (U) U)
(declare-const a U)
(declare-const b U)
(declare-const c U)
(assert (= a c))
(assert (= (f a) b))
(assert (not (= (g b) (g (f c)))))
(check-sat)
