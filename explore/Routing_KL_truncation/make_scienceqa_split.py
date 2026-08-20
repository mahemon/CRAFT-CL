#!/usr/bin/env python3
"""
make_scienceqa_split.py — builds the ScienceQA-A / ScienceQA-B duplicate-
task unit test described in the paper's Limitations section: two random
halves of the same dataset, presented as separate task directories, used
to check whether the routing distance's top-k truncation causes spurious
NEW-group decisions on near-identical data (the documented saturation
failure mode).

Standalone: no train.py imports, no GPU. Reads the existing ScienceQA
train/eval JSON, shuffles with a fixed seed, splits into two disjoint
halves, and writes them as two new task directories with the SAME file
names and SAME JSON structure as the source (list-of-examples, optionally
wrapped under a top-level key such as "data" or "examples" -- both forms
are detected and preserved). No example fields are inspected or renamed,
so this works regardless of the benchmark's exact schema.

Usage:
  python make_scienceqa_split.py \
      --benchmark_dir /path/to/LLM-CL-Benchmark_5000 \
      --out_dir /path/to/LLM-CL-Benchmark_ScienceQA_split \
      [--seed 42] [--eval_frac 0.5]

Then point a run at:
  --cl_benchmark_dir /path/to/LLM-CL-Benchmark_ScienceQA_split
  --cl_tasks "ScienceQA-A,ScienceQA-B"
"""

import argparse
import json
import os
import random
import sys


def load_examples(path):
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, list):
        return data, None
    if isinstance(data, dict):
        for key in ("data", "examples", "instances"):
            if key in data and isinstance(data[key], list):
                return data[key], key
        sys.exit(f"[error] {path}: dict JSON but no list found under "
                 f"'data'/'examples'/'instances'. Top-level keys: "
                 f"{list(data.keys())}")
    sys.exit(f"[error] {path}: unrecognized JSON top-level type "
             f"{type(data).__name__}")


def save_examples(path, examples, wrap_key):
    payload = examples if wrap_key is None else {wrap_key: examples}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def split_file(src_path, dst_a, dst_b, seed, eval_frac=None):
    if not os.path.isfile(src_path):
        print(f"  [skip] not found: {src_path}")
        return 0, 0
    examples, wrap_key = load_examples(src_path)
    rng = random.Random(seed)
    idx = list(range(len(examples)))
    rng.shuffle(idx)
    mid = len(idx) // 2
    idx_a, idx_b = idx[:mid], idx[mid:]
    ex_a = [examples[i] for i in idx_a]
    ex_b = [examples[i] for i in idx_b]
    save_examples(dst_a, ex_a, wrap_key)
    save_examples(dst_b, ex_b, wrap_key)
    print(f"  [split] {src_path}: {len(examples)} -> "
         f"A={len(ex_a)}, B={len(ex_b)}")
    return len(ex_a), len(ex_b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark_dir", required=True,
                    help="TRACE benchmark root containing a ScienceQA/ dir")
    ap.add_argument("--out_dir", required=True,
                    help="destination benchmark root; will contain "
                         "ScienceQA-A/ and ScienceQA-B/")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    src_dir = os.path.join(args.benchmark_dir, "ScienceQA")
    if not os.path.isdir(src_dir):
        sys.exit(f"[error] {src_dir} not found. Check --benchmark_dir "
                 f"points at the TRACE benchmark root (the directory "
                 f"that directly contains task subdirectories).")

    print(f"[make_scienceqa_split] source: {src_dir}")
    print(f"[make_scienceqa_split] dest:   {args.out_dir}")
    print(f"[make_scienceqa_split] seed:   {args.seed}")

    any_split = False
    for fname in ("train.json", "eval.json", "test.json"):
        src = os.path.join(src_dir, fname)
        if not os.path.isfile(src):
            continue
        dst_a = os.path.join(args.out_dir, "ScienceQA-A", fname)
        dst_b = os.path.join(args.out_dir, "ScienceQA-B", fname)
        n_a, n_b = split_file(src, dst_a, dst_b, args.seed)
        any_split = any_split or (n_a > 0 or n_b > 0)

    if not any_split:
        sys.exit(f"[error] no train.json/eval.json/test.json found under "
                 f"{src_dir}; nothing split.")

    print(f"\n[make_scienceqa_split] done. Run with:")
    print(f"  --cl_benchmark_dir {args.out_dir}")
    print(f"  --cl_tasks \"ScienceQA-A,ScienceQA-B\"")
    print(f"  --per_task_epochs \"4,4\"")


if __name__ == "__main__":
    main()
