#!/usr/bin/env python3
"""
analyze_anchor_identity.py — compares anchor-identity arms
(group / pi0 / prev_task) on OP, BWT, LA, and per-task drops.

Usage:
  python analyze_anchor_identity.py \
      --runs group=DIR1 pi0=DIR2 prev_task=DIR3 \
      [--reference group] [--csv anchor_identity.csv]
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
    return {"file": f, "order": order, "diag": diag, "op": op, "bwt": bwt,
            "la": la, "n_groups": res.get("n_groups")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--reference", default="group")
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
    ref = arms[args.reference]

    print(f"\n{'='*72}\n  ANCHOR IDENTITY SUMMARY  (reference: {args.reference})")
    print(f"    {'arm':<12}{'OP':>8}{'BWT':>8}{'LA':>8}{'groups':>8}"
          f"{'ΔBWT vs ref':>14}")
    for tag, a in arms.items():
        d_bwt = a["bwt"] - ref["bwt"] if tag != args.reference else 0.0
        print(f"    {tag:<12}{a['op']:>8.2f}{a['bwt']:>8.2f}{a['la']:>8.2f}"
              f"{str(a['n_groups']):>8}{d_bwt:>14.2f}")

    print(f"\n  Verdict guide: 'group' winning on BWT at comparable OP "
          f"validates the anchoring argument in the method section. "
          f"'pi0' tying (or beating) 'group' weakens the anchoring claim "
          f"-- the KL term would be better described as generic "
          f"output-space regularization rather than anchoring to what "
          f"the group has learned. 'prev_task' isolates whether the "
          f"GROUP aggregate specifically matters, or whether anchoring "
          f"to any recently-trained state is sufficient.")

    print(f"\n  PER-TASK a_tt DELTA vs {args.reference}")
    order = ref["order"]
    header = f"    {'task':<14}" + "".join(f"{t:>12}" for t in arms
                                           if t != args.reference)
    print(header)
    rows = []
    for task in order:
        line = f"    {task:<14}"
        row = {"task": task, "ref_a_tt": ref["diag"].get(task)}
        for tag, a in arms.items():
            if tag == args.reference:
                continue
            v, r = a["diag"].get(task), ref["diag"].get(task)
            d = (v - r) if (v is not None and r is not None) else None
            row[f"delta_{tag}"] = d
            line += f"{d:>12.2f}" if d is not None else f"{'--':>12}"
        print(line)
        rows.append(row)

    if args.csv:
        fieldnames = ["task", "ref_a_tt"] + [f"delta_{t}" for t in arms
                                             if t != args.reference]
        with open(args.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)
        print(f"\n[csv] per-task deltas -> {args.csv}")


if __name__ == "__main__":
    main()
