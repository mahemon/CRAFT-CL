#!/usr/bin/env python3
"""
analyze_stream_ablation.py — stream ablation comparison (jvrh W1).

Primary analysis (always runs, needs only cl_results.json):
  OP / BWT / LA / n_groups per arm, plus per-task diagonal (a_tt) deltas
  against the f15l15 reference arm, so which tasks are most sensitive to
  losing a stream is visible at a glance.

Secondary analysis (optional, --benchmark_dir): a prompt-length PROXY for
the length-interaction question. This is NOT per-example stratified eval
(the current pipeline logs aggregate per-task scores only, not
per-example predictions) — it correlates each task's median raw prompt
token length against that task's f-only vs l-only OP delta. Treat this
as a coarse first look, not a substitute for real per-example
stratification (see README for the patch point if that's needed later).

Usage:
  python analyze_stream_ablation.py \
      --runs f15l15=DIR1 f30=DIR2 l30=DIR3 f5l25=DIR4 \
      [--reference f15l15] [--csv stream_ablation.csv] \
      [--benchmark_dir /path/to/LLM-CL-Benchmark_5000] [--model MODEL_PATH]
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
    ap.add_argument("--runs", nargs="+", required=True,
                    help="arm=dir pairs, e.g. f15l15=outputs/..._f15l15_...")
    ap.add_argument("--reference", default="f15l15",
                    help="arm tag used as the baseline for per-task deltas")
    ap.add_argument("--csv", default=None)
    ap.add_argument("--benchmark_dir", default=None,
                    help="optional: enables the prompt-length proxy "
                         "(needs the raw TRACE task JSON files)")
    ap.add_argument("--model", default=None,
                    help="tokenizer path/id for the length proxy "
                         "(required if --benchmark_dir is given)")
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

    print(f"\n{'='*72}\n  STREAM ABLATION SUMMARY  (reference: {args.reference})")
    print(f"    {'arm':<10}{'OP':>8}{'BWT':>8}{'LA':>8}{'groups':>8}"
          f"{'ΔOP vs ref':>12}")
    for tag, a in arms.items():
        d_op = a["op"] - ref["op"] if tag != args.reference else 0.0
        print(f"    {tag:<10}{a['op']:>8.2f}{a['bwt']:>8.2f}{a['la']:>8.2f}"
              f"{str(a['n_groups']):>8}{d_op:>12.2f}")

    print(f"\n  PER-TASK a_tt DELTA vs {args.reference} "
          f"(negative = hurt by losing a stream)")
    order = ref["order"]
    header = f"    {'task':<14}" + "".join(f"{t:>10}" for t in arms
                                           if t != args.reference)
    print(header)
    rows = []
    for task in order:
        line = f"    {task:<14}"
        row = {"task": task, "ref_a_tt": ref["diag"].get(task)}
        for tag, a in arms.items():
            if tag == args.reference:
                continue
            v = a["diag"].get(task)
            r = ref["diag"].get(task)
            delta = (v - r) if (v is not None and r is not None) else None
            row[f"delta_{tag}"] = delta
            line += f"{delta:>10.2f}" if delta is not None else f"{'--':>10}"
        print(line)
        rows.append(row)

    worst = sorted(
        [r for r in rows if any(v is not None for k, v in r.items()
                                if k.startswith("delta_"))],
        key=lambda r: min(v for k, v in r.items()
                          if k.startswith("delta_") and v is not None))
    if worst:
        print(f"\n  Most stream-sensitive task: {worst[0]['task']} "
              f"(largest negative delta in at least one arm)")

    if args.csv:
        fieldnames = ["task", "ref_a_tt"] + [f"delta_{t}" for t in arms
                                             if t != args.reference]
        with open(args.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)
        print(f"\n[csv] per-task deltas -> {args.csv}")

    if args.benchmark_dir:
        run_length_proxy(args, arms, order)


def run_length_proxy(args, arms, order):
    print(f"\n{'='*72}\n  PROMPT-LENGTH PROXY (best-effort; see README)")
    if not args.model:
        print("  [skip] --model required for the length proxy "
              "(tokenizer needed to measure prompt length)")
        return
    try:
        from transformers import AutoTokenizer
    except ImportError:
        print("  [skip] transformers not available in this environment")
        return
    try:
        tok = AutoTokenizer.from_pretrained(args.model, use_fast=False)
    except Exception as e:
        print(f"  [skip] could not load tokenizer '{args.model}': {e}")
        return

    field_candidates = ["prompt", "input", "instruction", "question", "text"]
    med_len = {}
    for task in order:
        path = os.path.join(args.benchmark_dir, task, "eval.json")
        if not os.path.isfile(path):
            path = os.path.join(args.benchmark_dir, task, "test.json")
        if not os.path.isfile(path):
            print(f"  [skip] {task}: no eval/test.json under "
                  f"{args.benchmark_dir}/{task}/")
            continue
        try:
            with open(path) as fh:
                data = json.load(fh)
        except Exception as e:
            print(f"  [skip] {task}: could not parse {path} ({e})")
            continue
        if isinstance(data, dict):
            data = data.get("data", data.get("examples", []))
        if not isinstance(data, list) or not data:
            print(f"  [skip] {task}: unrecognized JSON shape at {path}")
            continue
        field = next((f for f in field_candidates if f in data[0]), None)
        if field is None:
            print(f"  [skip] {task}: none of {field_candidates} found in "
                  f"example keys {list(data[0].keys())[:6]} — edit "
                  f"field_candidates in this script")
            continue
        lens = [len(tok.encode(ex[field])) for ex in data[:200]
                if isinstance(ex.get(field), str)]
        if not lens:
            continue
        lens.sort()
        med_len[task] = lens[len(lens) // 2]

    if not med_len:
        print("  [skip] no task yielded a usable prompt-length sample; "
              "check the benchmark directory layout / field names above")
        return

    print(f"    {'task':<14}{'median_len':>12}", end="")
    for tag in arms:
        if tag != args.reference:
            print(f"{('Δ_' + tag):>12}", end="")
    print()
    for task, mlen in sorted(med_len.items(), key=lambda kv: kv[1]):
        line = f"    {task:<14}{mlen:>12}"
        for tag, a in arms.items():
            if tag == args.reference:
                continue
            v, r = a["diag"].get(task), arms[args.reference]["diag"].get(task)
            d = (v - r) if (v is not None and r is not None) else None
            line += f"{d:>12.2f}" if d is not None else f"{'--':>12}"
        print(line)
    print("\n  Read qualitatively: sort by median_len and look for a trend "
          "in the f-only / l-only columns. This is a task-level proxy "
          "(median prompt length vs whole-task OP delta), not per-example "
          "stratification within a task.")


if __name__ == "__main__":
    main()
