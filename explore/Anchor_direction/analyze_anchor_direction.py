#!/usr/bin/env python3
"""
analyze_anchor_direction.py — compares KL penalty direction arms
(forward / reverse / symmetric) on OP, BWT, LA, and in-group forgetting.

Usage:
  python analyze_anchor_direction.py \
      --runs forward=DIR1 reverse=DIR2 symmetric=DIR3 \
      [--reference forward] [--csv anchor_direction.csv]
"""

import argparse
import csv
import json
import os
import sys


def find_results(path):
    cand = os.path.join(path, "cl_results.json")
    if os.path.isfile(cand):
        return cand
    if os.path.isfile(path) and path.endswith(".json"):
        return path
    hits = []
    if os.path.isdir(path):
        for d in sorted(os.listdir(path)):
            c = os.path.join(path, d, "cl_results.json")
            if os.path.isfile(c):
                hits.append(c)
    if not hits:
        sys.exit(f"[error] cl_results.json not found under: {path}")
    if len(hits) > 1:
        print(f"[warn] {len(hits)} results under {path}; using latest")
    return hits[-1]


def load(path):
    f = find_results(path)
    with open(f) as fh:
        res = json.load(fh)
    order = list(res["diagonal"].keys())
    pm = res["perf_matrix"]
    n = len(order)
    diag = {t: pm[j][j] for j, t in enumerate(order)}
    last = {t: pm[n - 1][j] for j, t in enumerate(order)}
    valid = [t for t in order if diag[t] is not None and last[t] is not None]
    la = sum(diag[t] for t in valid) / len(valid)
    op = res.get("overall_performance",
                 sum(last[t] for t in valid) / len(valid))
    bwt = res.get("backward_transfer",
                  sum(diag[t] - last[t] for t in valid) / len(valid))
    t2g = res.get("task_to_group", {})
    return {"file": f, "order": order, "pm": pm, "diag": diag,
            "op": op, "bwt": bwt, "la": la,
            "n_groups": res.get("n_groups"), "t2g": t2g}


def per_task_ingroup_forgetting(res):
    """Mean immediate in-group drop caused by each task, for a rough
    per-arm forgetting-severity comparison beyond aggregate BWT."""
    order, pm, t2g = res["order"], res["pm"], res["t2g"]
    n = len(order)
    per_task = {}
    for idx, t in enumerate(order):
        prior_same = [j for j in range(idx)
                      if t2g.get(order[j]) == t2g.get(t)
                      and pm[idx][j] is not None
                      and pm[idx - 1][j] is not None]
        if prior_same:
            per_task[t] = sum(pm[idx - 1][j] - pm[idx][j]
                              for j in prior_same) / len(prior_same)
    return per_task


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--reference", default="forward")
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()

    arms = {}
    for spec in args.runs:
        tag, path = spec.split("=", 1)
        arms[tag] = load(path)
        print(f"[load] {tag}: {arms[tag]['file']}")

    if args.reference not in arms:
        sys.exit(f"[error] --reference '{args.reference}' not among "
                 f"loaded arms: {list(arms)}")

    print(f"\n{'='*72}\n  ANCHOR DIRECTION SUMMARY  (reference: {args.reference})")
    print(f"    {'arm':<12}{'OP':>8}{'BWT':>8}{'LA':>8}{'groups':>8}")
    for tag, a in arms.items():
        print(f"    {tag:<12}{a['op']:>8.2f}{a['bwt']:>8.2f}{a['la']:>8.2f}"
              f"{str(a['n_groups']):>8}")

    print(f"\n  Verdict guide: forward winning on BWT at comparable OP "
          f"validates the current (mode-covering) choice. Reverse winning "
          f"indicates mode-seeking retention works better in intervention "
          f"space and should change the shipped configuration.")

    print(f"\n{'='*72}\n  PER-TASK IN-GROUP FORGETTING (mean drop on prior "
          f"same-group tasks)")
    per_arm_forget = {tag: per_task_ingroup_forgetting(a)
                      for tag, a in arms.items()}
    all_tasks = sorted({t for d in per_arm_forget.values() for t in d})
    header = f"    {'task':<14}" + "".join(f"{tag:>12}" for tag in arms)
    print(header)
    rows = []
    for t in all_tasks:
        line = f"    {t:<14}"
        row = {"task": t}
        for tag in arms:
            v = per_arm_forget[tag].get(t)
            row[tag] = v
            line += f"{v:>12.2f}" if v is not None else f"{'--':>12}"
        print(line)
        rows.append(row)

    if args.csv:
        fieldnames = ["task"] + list(arms.keys())
        with open(args.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)
        print(f"\n[csv] per-task forgetting by arm -> {args.csv}")


if __name__ == "__main__":
    main()
