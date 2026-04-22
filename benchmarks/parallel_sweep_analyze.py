#!/usr/bin/env python3
"""
Analyze parallel_sweep.c CSV output.

For each (op, N):
  - baseline = simd-only time (1 thread, no parallel region)
  - best = fastest (parallel_variant, thread_count) time
  - speedup = baseline / best

Two stats reported side-by-side: MIN (peak capability) and MEDIAN (typical).

Emits:
  1. Per-op speedup grid (median) — rows are N, columns are variant/threads.
  2. "Capped variant" report — how cap4 / cap8 compare with simd baseline.
  3. Threshold-per-op summary plus tier assignment.

Usage:
  python3 benchmarks/parallel_sweep_analyze.py /tmp/parallel_sweep_results.csv
"""

import csv
import sys
from collections import defaultdict

SPEEDUP_THRESHOLD = 1.2  # minimum speedup to count as "worth it"


def load(path: str):
    # rows[op][N][(variant, threads)] = (min_ns, med_ns)
    rows: dict[str, dict[int, dict[tuple[str, int], tuple[float, float]]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    with open(path) as f:
        reader = csv.DictReader(f)
        for r in reader:
            op = r["op"]
            variant = r["variant"]
            threads = int(r["threads"])
            N = int(r["N"])
            t_min = float(r["time_ns_min"])
            t_med = float(r["time_ns_median"])
            rows[op][N][(variant, threads)] = (t_min, t_med)
    return rows


def grid(rows, op, stat_idx: int, label: str) -> None:
    print(f"\n=== {op}  ({label}; baseline = simd/t1; * = speedup >= 1.2x) ===")
    per_N = rows[op]
    Ns = sorted(per_N.keys())
    cfg_set = set()
    for N in Ns:
        for cfg in per_N[N]:
            if cfg == ("simd", 1):
                continue
            cfg_set.add(cfg)
    cfgs = sorted(cfg_set)
    header = f"{'N':>10} " + " ".join(f"{v[:8]}/t{t:<2}".rjust(12) for v, t in cfgs)
    print(header)
    print("-" * len(header))
    for N in Ns:
        base = per_N[N].get(("simd", 1))
        if base is None:
            continue
        base_v = base[stat_idx]
        cells = []
        for cfg in cfgs:
            v = per_N[N].get(cfg)
            if v is None:
                cells.append(f"{'.':>12}")
            else:
                s = base_v / v[stat_idx]
                marker = "*" if s >= SPEEDUP_THRESHOLD else " "
                cells.append(f"{s:>11.2f}{marker}")
        print(f"{N:>10} " + " ".join(cells))


def capped_report(rows) -> None:
    """How does the fixed-thread cap4/cap8 policy perform across all ops?"""
    print("\n\n=== Capped-variant report (median speedup vs simd/t1) ===")
    ops = sorted(rows.keys())
    Ns = sorted({n for op in rows for n in rows[op]})
    header = f"{'op':<10} {'N':>10} {'cap4':>10} {'cap8':>10} {'best_sweep':>14}"
    print(header)
    print("-" * len(header))
    for op in ops:
        for N in Ns:
            r = rows[op].get(N, {})
            base = r.get(("simd", 1))
            if not base:
                continue
            base_med = base[1]
            cap4 = r.get(("par_cap4", 4))
            cap8 = r.get(("par_cap8", 8))
            c4 = f"{base_med / cap4[1]:>10.2f}" if cap4 else f"{'-':>10}"
            c8 = f"{base_med / cap8[1]:>10.2f}" if cap8 else f"{'-':>10}"
            best_speedup = 0.0
            best_cfg = None
            for cfg, (tmi, tme) in r.items():
                if cfg == ("simd", 1):
                    continue
                if cfg[0].startswith("par_cap"):
                    continue
                s = base_med / tme
                if s > best_speedup:
                    best_speedup = s
                    best_cfg = cfg
            bs = f"{best_speedup:>6.2f}x {best_cfg[0][:6]}/t{best_cfg[1]}" if best_cfg else "-"
            print(f"{op:<10} {N:>10} {c4} {c8} {bs:>14}")
        print()


def thresholds(rows) -> None:
    print("\n\n=== Thresholds + tier assignment (using MEDIAN) ===")
    print(f"{'op':<10} {'N_cross':>10} {'speedup':>10} {'best':<18} "
          f"{'cap4@maxN':>10} {'cap8@maxN':>10}")
    print("-" * 80)
    for op in sorted(rows.keys()):
        per_N = rows[op]
        Ns = sorted(per_N.keys())
        cross_N = None
        cross_speedup = 0.0
        cross_cfg = None
        for N in Ns:
            base = per_N[N].get(("simd", 1))
            if base is None:
                continue
            best_speedup = 0.0
            best_cfg = None
            for cfg, (tmi, tme) in per_N[N].items():
                if cfg == ("simd", 1):
                    continue
                s = base[1] / tme
                if s > best_speedup:
                    best_speedup = s
                    best_cfg = cfg
            if cross_N is None and best_speedup >= SPEEDUP_THRESHOLD:
                cross_N = N
                cross_speedup = best_speedup
                cross_cfg = best_cfg
        max_N = Ns[-1]
        base_max = per_N[max_N].get(("simd", 1))
        cap4_max = per_N[max_N].get(("par_cap4", 4))
        cap8_max = per_N[max_N].get(("par_cap8", 8))
        c4_str = f"{base_max[1] / cap4_max[1]:.2f}x" if (base_max and cap4_max) else "-"
        c8_str = f"{base_max[1] / cap8_max[1]:.2f}x" if (base_max and cap8_max) else "-"
        if cross_N is not None:
            cfg_str = f"{cross_cfg[0][:10]}/t{cross_cfg[1]}"
            print(f"{op:<10} {cross_N:>10} {cross_speedup:>9.2f}x {cfg_str:<18} "
                  f"{c4_str:>10} {c8_str:>10}")
        else:
            print(f"{op:<10} {'never':>10} {'-':>10} {'-':<18} "
                  f"{c4_str:>10} {c8_str:>10}")


def main(path: str, op_filter: str | None) -> None:
    rows = load(path)
    # Per-op median grid only (min grid is noisy, we saw in v1)
    if op_filter:
        grid(rows, op_filter, stat_idx=1, label="median")
    else:
        for op in sorted(rows.keys()):
            grid(rows, op, stat_idx=1, label="median")
    capped_report(rows)
    thresholds(rows)


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/parallel_sweep_results.csv"
    opf = sys.argv[2] if len(sys.argv) > 2 else None
    main(path, opf)
