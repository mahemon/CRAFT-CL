#!/usr/bin/env python3
"""
analyze_routing_gate.py — compares kl_routing vs learned_gate.

Usage:
  python analyze_routing_gate.py \
      --runs kl_routing=DIR1 learned_gate=DIR2 [--csv routing_gate.csv]
"""

import argparse
import csv
import json
import os
import sys
from itertools import combinations


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
    return {"file": f, "order": order, "op": op, "bwt": bwt, "la": la,
            "n_groups": res.get("n_groups"), "t2g": t2g}


def rand_index(t2g_a, t2g_b, tasks):
    """Unadjusted Rand index over task-pair co-membership agreement:
    fraction of task pairs that agree on 'same group / different group'
    between the two partitions. Range [0,1]; 1.0 = identical partition
    structure (up to group-id relabeling)."""
    pairs = list(combinations(tasks, 2))
    if not pairs:
        return float("nan")
    agree = 0
    for a, b in pairs:
        same_a = t2g_a.get(a) == t2g_a.get(b)
        same_b = t2g_b.get(a) == t2g_b.get(b)
        if same_a == same_b:
            agree += 1
    return agree / len(pairs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()

    arms = {}
    for spec in args.runs:
        tag, path = spec.split("=", 1)
        arms[tag] = load(path)
        print(f"[load] {tag}: {arms[tag]['file']}")

    print(f"\n{'='*72}\n  ROUTING VS LEARNED GATE — SUMMARY")
    print(f"    {'arm':<14}{'OP':>8}{'BWT':>8}{'LA':>8}{'groups':>8}")
    for tag, a in arms.items():
        print(f"    {tag:<14}{a['op']:>8.2f}{a['bwt']:>8.2f}{a['la']:>8.2f}"
              f"{str(a['n_groups']):>8}")

    if "kl_routing" in arms and "learned_gate" in arms:
        kr, lg = arms["kl_routing"], arms["learned_gate"]
        common = [t for t in kr["order"] if t in lg["t2g"]]
        ri = rand_index(kr["t2g"], lg["t2g"], common)
        print(f"\n  PARTITION AGREEMENT (unadjusted Rand index, "
              f"co-membership over {len(common)} tasks): {ri:.3f}")
        print(f"  (1.0 = identical group structure up to relabeling; "
              f"0.5 = chance-level agreement on a random binary split)")

        d_op = lg["op"] - kr["op"]
        d_bwt = lg["bwt"] - kr["bwt"]
        print(f"\n  ΔOP  (learned_gate - kl_routing): {d_op:+.2f}")
        print(f"  ΔBWT (learned_gate - kl_routing): {d_bwt:+.2f}")
        print(f"\n  Verdict guide: kl_routing matching or beating "
              f"learned_gate on OP/BWT at ZERO trainable routing "
              f"parameters is the strongest sentence available for the "
              f"contributions list. learned_gate winning bounds the "
              f"claim to 'parameter-free routing at a stated cost' -- "
              f"report the cost as ΔOP/ΔBWT above, and the gate's "
              f"parameter count from the run logs "
              f"('gate_param_count' in the routing debug entries).")

    print(f"\n  PER-TASK GROUP ASSIGNMENT (raw, for manual inspection)")
    all_tasks = sorted({t for a in arms.values() for t in a["order"]})
    header = f"    {'task':<14}" + "".join(f"{tag:>14}" for tag in arms)
    print(header)
    rows = []
    for t in all_tasks:
        line = f"    {t:<14}"
        row = {"task": t}
        for tag, a in arms.items():
            g = a["t2g"].get(t, "--")
            row[tag] = g
            line += f"{str(g):>14}"
        print(line)
        rows.append(row)

    if args.csv:
        fieldnames = ["task"] + list(arms.keys())
        with open(args.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)
        print(f"\n[csv] per-task group assignment -> {args.csv}")


if __name__ == "__main__":
    main()
