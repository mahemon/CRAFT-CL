#!/usr/bin/env python3
"""
analyze_truncation.py — routing-truncation sweep analysis.

Two modes, auto-detected from --runs contents:

1. Sweep mode (TRACE-8, 6+ tasks): reports the discovered partition
   (task_to_group) per arm, K (number of groups) per arm, and pairwise
   partition agreement (Rand index) against the k64_p16 reference arm.
   Partition INVARIANCE across k/probe_size is the desired outcome --
   it means the paper's k=64 default is not an under-truncated artifact.

2. Duplicate-unit-test mode (2-task ScienceQA-A/ScienceQA-B runs):
   reports whether A and B were routed into the SAME group at each arm.
   This is the direct falsification test for the top-k saturation
   failure mode described in Limitations.

Usage:
  python analyze_truncation.py \
      --runs k64_p16=DIR1 k128_p16=DIR2 k256_p16=DIR3 k512_p16=DIR4 \
             k64_p8=DIR5 k64_p64=DIR6 \
      [--reference k64_p16] [--csv truncation.csv]
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
    t2g = res.get("task_to_group", {})
    if not t2g:
        # routing_diagnostic_only writes task_to_group into memory only;
        # rebuild it from routing_log if the top-level field is empty.
        t2g = {}
        for e in res.get("routing_log", []):
            if "gid" in e:
                t2g[e["task"]] = e["gid"]
    n_groups = res.get("n_groups", len(set(t2g.values())) if t2g else None)
    return {"file": f, "t2g": t2g, "n_groups": n_groups,
            "routing_log": res.get("routing_log", [])}


def rand_index(t2g_a, t2g_b, tasks):
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
    ap.add_argument("--reference", default=None,
                    help="defaults to the first arm containing 'k64' and "
                         "'p16', or just the first arm if none match")
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()

    arms = {}
    for spec in args.runs:
        tag, path = spec.split("=", 1)
        arms[tag] = load(path)
        print(f"[load] {tag}: {arms[tag]['file']}  "
              f"(K={arms[tag]['n_groups']})")

    all_tasks = sorted({t for a in arms.values() for t in a["t2g"]})

    # ---- duplicate-unit-test mode ----
    if set(all_tasks) == {"ScienceQA-A", "ScienceQA-B"}:
        print(f"\n{'='*72}\n  DUPLICATE-TASK UNIT TEST (ScienceQA-A vs ScienceQA-B)")
        print(f"    {'arm':<12}{'gid(A)':>8}{'gid(B)':>8}{'merged?':>10}")
        for tag, a in arms.items():
            ga = a["t2g"].get("ScienceQA-A", "?")
            gb = a["t2g"].get("ScienceQA-B", "?")
            merged = "YES" if ga == gb and ga != "?" else "no"
            print(f"    {tag:<12}{str(ga):>8}{str(gb):>8}{merged:>10}")
        print(f"\n  Reading guide: at the paper's default (k64_p16), "
              f"expect 'no' (the documented saturation failure -- both "
              f"halves route to fresh clusters despite being near-"
              f"identical data). If a larger-k arm flips to 'YES', that "
              f"is the fix: the failure is a truncation artifact, not a "
              f"property of the routing signal itself, and the "
              f"Limitations section should state the fix (raise "
              f"kl_routing_capture_top_k / kl_routing_top_k) rather than "
              f"leaving it as an open problem.")
        if args.csv:
            with open(args.csv, "w", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(["arm", "gid_A", "gid_B", "merged"])
                for tag, a in arms.items():
                    ga = a["t2g"].get("ScienceQA-A", "?")
                    gb = a["t2g"].get("ScienceQA-B", "?")
                    w.writerow([tag, ga, gb, ga == gb and ga != "?"])
            print(f"\n[csv] -> {args.csv}")
        return

    # ---- sweep mode ----
    ref_tag = args.reference
    if ref_tag is None:
        ref_tag = next((t for t in arms if "k64" in t and "p16" in t),
                       next(iter(arms)))
    if ref_tag not in arms:
        sys.exit(f"[error] reference arm '{ref_tag}' not among "
                 f"{list(arms)}")
    ref = arms[ref_tag]

    print(f"\n{'='*72}\n  TRUNCATION SWEEP — PARTITION SUMMARY  "
          f"(reference: {ref_tag})")
    print(f"    {'arm':<12}{'K':>5}{'RandIdx vs ref':>18}")
    for tag, a in arms.items():
        ri = (1.0 if tag == ref_tag else
              rand_index(ref["t2g"], a["t2g"], all_tasks))
        print(f"    {tag:<12}{str(a['n_groups']):>5}{ri:>18.3f}")

    print(f"\n  Reading guide: Rand index 1.0 across all arms = partition "
          f"is invariant to top_k / probe_size in this range -- the "
          f"paper's k=64 default is not under-truncated, and the "
          f"Limitations paragraph on saturation can be scoped to the "
          f"documented near-duplicate case only. Any arm below ~0.9 "
          f"indicates the partition IS sensitive to truncation in the "
          f"tested range and the routing method section needs a "
          f"sensitivity caveat.")

    print(f"\n  PER-TASK GROUP ASSIGNMENT")
    header = f"    {'task':<14}" + "".join(f"{tag:>10}" for tag in arms)
    print(header)
    rows = []
    for t in all_tasks:
        line = f"    {t:<14}"
        row = {"task": t}
        for tag, a in arms.items():
            g = a["t2g"].get(t, "--")
            row[tag] = g
            line += f"{str(g):>10}"
        print(line)
        rows.append(row)

    if args.csv:
        fieldnames = ["task"] + list(arms.keys())
        with open(args.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)
        print(f"\n[csv] -> {args.csv}")


if __name__ == "__main__":
    main()
