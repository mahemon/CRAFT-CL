#!/usr/bin/env python3
"""
analyze_merge_ablation.py — compares merge rules and runs the recency probe.

Usage:
  python analyze_merge_ablation.py \
      --runs overwrite=OUT1 ema=OUT2 fisher=OUT3 prototype=OUT4 \
      [--csv merge_ablation_summary.csv]

Each OUT* is a run output dir containing cl_results.json
(written by train.py at the end of the stream).

Reports, per rule:
  OP, BWT, LA (mean diagonal), n_groups
Recency probe, per rule and per group with >= 2 members:
  for each member task, in arrival order: a_tt, a_Tt, drop (a_tt - a_Tt),
  join index within group; Spearman rho of drop vs join index
  (positive rho = earlier joiners forget more = recency bias).
"""

import argparse
import csv
import json
import os
import sys


def spearman(x, y):
    n = len(x)
    if n < 3:
        return float("nan")
    def ranks(v):
        order = sorted(range(n), key=lambda i: v[i])
        r = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    rx, ry = ranks(x), ranks(y)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
    dx = sum((rx[i] - mx) ** 2 for i in range(n)) ** 0.5
    dy = sum((ry[i] - my) ** 2 for i in range(n)) ** 0.5
    return num / (dx * dy) if dx > 0 and dy > 0 else float("nan")


def load_run(path):
    f = os.path.join(path, "cl_results.json")
    if not os.path.isfile(f):
        # fall back: search one level of subdirs (timestamped output dirs)
        hits = [os.path.join(path, d, "cl_results.json")
                for d in os.listdir(path)
                if os.path.isfile(os.path.join(path, d, "cl_results.json"))]
        if not hits:
            sys.exit(f"cl_results.json not found under {path}")
        f = sorted(hits)[-1]
    with open(f) as fh:
        return json.load(fh), f


def analyze(rule, res):
    pm = res["perf_matrix"]
    order = list(res["diagonal"].keys())
    n = len(order)
    diag = [pm[j][j] for j in range(n)]
    last = [pm[n - 1][j] for j in range(n)]
    valid = [j for j in range(n) if diag[j] is not None and last[j] is not None]
    la = sum(diag[j] for j in valid) / len(valid)
    op = res.get("overall_performance",
                 sum(last[j] for j in valid) / len(valid))
    bwt = res.get("backward_transfer",
                  sum(diag[j] - last[j] for j in valid) / len(valid))

    # group membership in arrival order
    t2g = res["task_to_group"]
    groups = {}
    for j, t in enumerate(order):
        gid = t2g.get(t)
        groups.setdefault(gid, []).append((j, t))

    probe_rows = []
    for gid, members in sorted(groups.items()):
        if len(members) < 2:
            continue
        for join_idx, (j, t) in enumerate(members):
            if diag[j] is None or last[j] is None:
                continue
            probe_rows.append({
                "rule": rule, "gid": gid, "task": t,
                "arrival_pos": j + 1, "join_idx": join_idx,
                "a_tt": round(diag[j], 2), "a_Tt": round(last[j], 2),
                "drop": round(diag[j] - last[j], 2),
                "group_size": len(members),
            })
    return {"rule": rule, "OP": round(op, 2), "BWT": round(bwt, 2),
            "LA": round(la, 2), "n_groups": res.get("n_groups",
                                                    len(groups))}, probe_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True,
                    help="rule=output_dir pairs")
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()

    summaries, all_probe = [], []
    for spec in args.runs:
        rule, path = spec.split("=", 1)
        res, f = load_run(path)
        print(f"[load] {rule}: {f}")
        s, probe = analyze(rule, res)
        summaries.append(s)
        all_probe.extend(probe)

    print("\n=== MERGE-RULE COMPARISON " + "=" * 40)
    print(f"{'rule':<12}{'OP':>8}{'BWT':>8}{'LA':>8}{'groups':>8}")
    for s in summaries:
        print(f"{s['rule']:<12}{s['OP']:>8.2f}{s['BWT']:>8.2f}"
              f"{s['LA']:>8.2f}{s['n_groups']:>8}")

    print("\n=== RECENCY PROBE (groups with >= 2 members) " + "=" * 20)
    print(f"{'rule':<12}{'gid':>4}{'task':<15}{'pos':>5}{'join':>6}"
          f"{'a_tt':>8}{'a_Tt':>8}{'drop':>8}")
    for r in all_probe:
        print(f"{r['rule']:<12}{r['gid']:>4}{r['task']:<15}"
              f"{r['arrival_pos']:>5}{r['join_idx']:>6}"
              f"{r['a_tt']:>8.2f}{r['a_Tt']:>8.2f}{r['drop']:>8.2f}")

    print("\n=== RECENCY CORRELATION (Spearman rho, drop vs join index) ===")
    print("positive rho: earlier joiners forget more (last-writer-wins bias)")
    rules = sorted({r["rule"] for r in all_probe})
    for rule in rules:
        rows = [r for r in all_probe if r["rule"] == rule]
        gids = sorted({r["gid"] for r in rows})
        pooled_x, pooled_y = [], []
        for gid in gids:
            g = [r for r in rows if r["gid"] == gid]
            if len(g) < 3:
                continue
            x = [r["join_idx"] for r in g]
            y = [r["drop"] for r in g]
            rho = spearman(x, y)
            # invert join_idx -> drop: early join (low idx) high drop => rho<0
            # report as recency bias = -rho for readability
            print(f"  {rule:<12} G{gid} (n={len(g)}): "
                  f"recency-bias rho = {-rho:+.3f}")
            pooled_x.extend(x)
            pooled_y.extend(y)
        if len(pooled_x) >= 3:
            print(f"  {rule:<12} pooled (n={len(pooled_x)}): "
                  f"recency-bias rho = {-spearman(pooled_x, pooled_y):+.3f}")

    if args.csv:
        with open(args.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(all_probe[0].keys()))
            w.writeheader()
            w.writerows(all_probe)
        print(f"\n[csv] probe rows written to {args.csv}")


if __name__ == "__main__":
    main()
