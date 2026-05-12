#!/usr/bin/env bash
# Aggregator + claim-verification for the FMCAD 2025 artifact.
#
# Reads the four CSVs in $1 (default bench/results/artifact/) and prints
# a summary that maps each paper claim to the relevant numerical result.
# Algorithm tags follow the bench CSVs:
#   sequential: nelson_seq, nelson_topo (unsound illustrative single-pass),
#               nelson_topo_iter (sound iterated; headline sequential),
#               nelson_dst, nelson_simple*
#   parallel:   par_parents (BSP with parents_), par_topo_iter (sound
#               depth-stratified iterated), par_filter, par_filter_min_id,
#               par_filter_gbk, par_naive (sound async/filter family)
# All claims are ratios; tolerate ±20% across machines.

set -euo pipefail
OUT=${1:-bench/results/artifact}

claim() { printf "\n=== %s ===\n" "$1"; }

# median of $column for a (workload, algorithm) pair in $csv.
# Workload is in column $w_col (1 for closure_compare, 1 for synthetic with $2 as n).
median_cc() {
  local csv=$1 workload=$2 algo=$3
  awk -F, -v w="$workload" -v a="$algo" '
    NR > 1 && $1 == w && $7 == a { print $11 }
  ' "$csv" | sort -n | awk '
    { v[NR]=$1 } END { if (NR > 0) print v[int(NR/2)+(NR%2==0?0:1)] }
  '
}
median_sy() {
  local csv=$1 family=$2 n=$3 algo=$4
  awk -F, -v f="$family" -v n="$n" -v a="$algo" '
    NR > 1 && $1 == f && $2 == n && $6 == a { print $10 }
  ' "$csv" | sort -n | awk '
    { v[NR]=$1 } END { if (NR > 0) print v[int(NR/2)+(NR%2==0?0:1)] }
  '
}
ratio() {
  awk -v n="$1" -v d="$2" 'BEGIN { if (d+0 > 0) printf "%6.2fx", n/d; else printf "  n/a" }'
}

T1=$OUT/T1_closure_compare.csv
T2=$OUT/T2_synthetic.csv
T3=$OUT/T3_eggcc.csv
T4=$OUT/T4_scaling.csv

# ---- T1: closure_compare ----
if [ -s "$T1" ]; then
  claim "T1 closure_compare (synthetic uniform DAGs)"
  threads=$(awk -F, 'NR > 1 { print $9; exit }' "$T1")
  echo "par_threads = $threads"
  printf "%-8s | %10s %12s %10s | %12s %12s | %s\n" \
    "workload" "nelson_seq" "nel_topo_iter" "par_filter" "topo_iter/seq" "par_filt/iter" "par_filter_min"
  for w in small medium large deep-s deep-m deep-l; do
    nel=$(median_cc "$T1" "$w" nelson_seq)
    iter=$(median_cc "$T1" "$w" nelson_topo_iter)
    pf=$(median_cc "$T1" "$w" par_filter)
    pfm=$(median_cc "$T1" "$w" par_filter_min_id)
    printf "%-8s | %9.2fms %11.2fms %9.2fms | %s %s | %.2fms\n" \
      "$w" "${nel:-0}" "${iter:-0}" "${pf:-0}" \
      "$(ratio "${nel:-0}" "${iter:-0}")" \
      "$(ratio "${iter:-0}" "${pf:-0}")" \
      "${pfm:-0}"
  done
  echo "(claim: nelson_topo_iter ≥ 2× nelson_seq across the board; par_filter beats nelson_topo_iter on large workloads at high T)"
fi

# ---- T2: synthetic ----
if [ -s "$T2" ]; then
  claim "T2 synthetic_bench (chain / grid / cube / quartic / quintic)"
  printf "%-12s | %10s %12s %10s | %s\n" "family-n" "nelson_seq" "nel_topo_iter" "par_filter" "par_filter/iter"
  awk -F, 'NR > 1 { print $1"-"$2 }' "$T2" | sort -uV | while read fn; do
    fam=${fn%-*}; n=${fn##*-}
    nel=$(median_sy "$T2" "$fam" "$n" nelson_seq)
    iter=$(median_sy "$T2" "$fam" "$n" nelson_topo_iter)
    pf=$(median_sy "$T2" "$fam" "$n" par_filter)
    printf "%-12s | %9.2fms %11.2fms %9.2fms | %s\n" \
      "$fn" "${nel:-0}" "${iter:-0}" "${pf:-0}" \
      "$(ratio "${iter:-0}" "${pf:-0}")"
  done
  echo "(claim: par_filter ≥ 3× nelson_topo_iter on the largest synthetic workloads — quintic-20, quartic-20)"
fi

# ---- T3: eggcc ----
if [ -s "$T3" ]; then
  claim "T3 eggcc (507 ground SMT-LIB files)"
  awk -F, '
    NR == 1 { next }
    { ms[$1, $6] = $9; files[$1] = 1; classes[$1] = $4 + 0 }
    END {
      for (f in files) {
        # Require sound headline algorithms present.
        if (!((f, "nelson_seq") in ms && (f, "nelson_topo_iter") in ms &&
              (f, "par_filter") in ms)) continue
        n++
        sum_nel += ms[f, "nelson_seq"]
        sum_iter += ms[f, "nelson_topo_iter"]
        sum_pf  += ms[f, "par_filter"]
        if (ms[f, "par_filter"] < ms[f, "nelson_topo_iter"]) wins_pf++
        if (classes[f] < 1000)        b="<1K"
        else if (classes[f] < 10000)  b="1-10K"
        else if (classes[f] < 100000) b="10-100K"
        else                          b=">=100K"
        bn[b]++
        bs_iter[b]+=ms[f,"nelson_topo_iter"]
        bs_pf[b]  +=ms[f,"par_filter"]
        if (ms[f,"par_filter"] < ms[f,"nelson_topo_iter"]) bw_pf[b]++
      }
      printf "files paired:               %d / 507\n", n
      printf "Σ nelson_seq:               %.1f ms\n", sum_nel
      printf "Σ nelson_topo_iter:         %.1f ms\n", sum_iter
      printf "Σ par_filter:               %.1f ms\n", sum_pf
      printf "par_filter vs nelson_seq:   %.2fx\n", sum_nel/sum_pf
      printf "par_filter vs topo_iter:    %.2fx\n", sum_iter/sum_pf
      printf "win rate vs topo_iter:      %d/%d (%.1f%%)\n\n", wins_pf, n, 100*wins_pf/n
      printf "%-10s %5s | %12s %12s %8s | %5s\n", "bucket", "files", "Σtopo_iter", "Σpar_filter", "ratio", "wins"
      order="<1K 1-10K 10-100K >=100K"; split(order, bs, " ")
      for (i = 1; i <= 4; i++) {
        b = bs[i]
        if (bn[b] > 0)
          printf "%-10s %5d | %10.1fms %10.1fms %7.2fx | %d/%d\n",
                 b, bn[b], bs_iter[b], bs_pf[b], bs_iter[b]/bs_pf[b], bw_pf[b], bn[b]
      }
    }' "$T3"
  echo "(claim: par_filter aggregate ≥ 1.2× nelson_topo_iter; on the ≥100K-class subset ≥ 1.4× with > 80% win rate)"
fi

# ---- T4: scaling ----
if [ -s "$T4" ]; then
  claim "T4 strong scaling for par_filter (median ms per T)"
  printf "%-8s |" "workload"
  for T in 1 2 4 8 16 32 64 96 144 192; do printf " %7s" "T=$T"; done
  printf "\n"
  for w in small medium large deep-s deep-m deep-l; do
    row=$(printf "%-8s |" "$w")
    for T in 1 2 4 8 16 32 64 96 144 192; do
      val=$(awk -F, -v w="$w" -v T="$T" -v a="par_filter" '
        NR > 1 && $1 == w && $9 == T && $7 == a { print $11 }' "$T4" | sort -n | awk '{v[NR]=$1} END {if(NR>0) print v[int(NR/2)+1]}')
      if [ -n "${val:-}" ]; then row="$row $(printf '%7.1f' "$val")"
      else                       row="$row $(printf '%7s' '-')"
      fi
    done
    echo "$row"
  done
  echo "(claim: par_filter strong-scales monotonically until atomic-UF/memory-bandwidth saturation; expect plateau at T = 64-128 on multi-socket NUMA)"
fi

echo
echo "==="
echo "Note: nelson_topo (without _iter suffix) is the unsound single-pass"
echo "illustrative variant; it appears in the CSVs but is NOT the sequential"
echo "champion. nelson_topo_iter is. Likewise par_topo (no _iter) is illustrative;"
echo "par_topo_iter / par_filter / par_filter_min_id are the sound parallel variants."
echo
echo "Tolerance: claim ratios are within ±20% on commodity hardware. Absolute"
echo "milliseconds vary up to 2× across machines; the ratios are stable."
