#!/usr/bin/env python
"""
collect_bwt.py -- recompute BWT for finished CRAFT runs in three ways.

Read-only: it never writes into run folders and does not change train.py,
collect_results.py or any log. It reads what train.py already saves:

    <run_dir>/cl_results.json           diagonal + final_row (+ perf_matrix)
    <run_dir>/train_args.json           configuration (train runs)
    <run_dir>/eval_*/cl_results.json    eval-only runs (eval_args.json next to it)

and, optionally, the PERFORMANCE MATRIX block printed at the end of a log
(--logs), for runs whose cl_results.json is not available.

Notation: R[i][i] = score on task i right after learning it (diagonal),
          R[T][i] = score on task i after the last task (final row),
          d_i = R[T][i] - R[i][i]   (negative = forgetting).

    LA            = (1/T)     * sum_{i=1..T}   R[i][i]
    OP            = (1/T)     * sum_{i=1..T}   R[T][i]
    BWT_GEM       = (1/(T-1)) * sum_{i=1..T-1} d_i            Lopez-Paz & Ranzato 2017, Eq. 3
                                                               (= the value train.py logs)
    BWT_TRACE     = (1/T)     * sum_{i=1..T-1} d_i  = OP - LA  Wang et al. 2023 (TRACE), Eq. 3
                                                               (= current tables / collect_results.py, sign flipped)
    BWT_TreeLoRAcode  = (1/T) * sum_{i=1..T} min(d_i, 0)   TreeLoRA repo code, inference/collect_results.py
                                                               (the TreeLoRA paper equation equals BWT_TRACE, sign flipped)

All three use negative = forgetting. Scores are reported in percent.

Usage
    python scripts/collect_bwt.py --roots output_log [more roots ...] --out results_bwt
    python scripts/collect_bwt.py --roots output_log --logs 'output_log/logs_*/*.txt' --out results_bwt
    python scripts/collect_bwt.py --roots output_log --include llama3_2_1b --split test

Outputs (in --out)
    bwt_runs.csv      one row per run, all three BWT values and the checks
    bwt_summary.md    mean +- std over seeds per configuration, and how the three rules differ
"""

import argparse
import csv
import glob
import json
import math
import os
import re
import statistics
from collections import OrderedDict, defaultdict

# ------------------------------------------------------------------ streams
TRACE8 = ["C-STANCE", "FOMC", "MeetingBank", "Py150", "ScienceQA",
          "NumGLUE-cm", "NumGLUE-ds", "20Minuten"]
TRACE15 = TRACE8 + ["dbpedia", "amazon", "yahoo", "agnews", "yelp", "BoolQA", "QQP"]
MATH = ["ScienceQA", "NumGLUE-cm", "NumGLUE-ds"]
OLORA = {  # task orders of O-LoRA (Wang et al., 2023a)
    "O-LoRA order 1": ["dbpedia", "amazon", "yahoo", "agnews"],
    "O-LoRA order 2": ["dbpedia", "amazon", "agnews", "yahoo"],
    "O-LoRA order 3": ["yahoo", "amazon", "agnews", "dbpedia"],
    "O-LoRA order 4": ["MNLI", "CB", "WiC", "COPA", "QQP", "BoolQA", "RTE", "IMDB",
                       "yelp", "amazon", "SST-2", "dbpedia", "agnews", "MultiRC", "yahoo"],
    "O-LoRA order 5": ["MultiRC", "BoolQA", "WiC", "MNLI", "CB", "COPA", "QQP", "RTE",
                       "IMDB", "SST-2", "dbpedia", "agnews", "yelp", "amazon", "yahoo"],
    "O-LoRA order 6": ["yelp", "amazon", "MNLI", "CB", "COPA", "QQP", "RTE", "IMDB",
                       "SST-2", "dbpedia", "agnews", "yahoo", "MultiRC", "BoolQA", "WiC"],
}

MODEL_KEYS = ["r1distill_qwen_1_5b", "r1qwen1p5b8", "r1distill_llama_8b", "llama3_2_1b",
              "llama2_7b", "gemma_2b", "mistral_7b", "mistral8", "qwen2_5_3b", "qwen2_5_7b",
              "qwen3_4b"]


def stream_name(tasks):
    if tasks == TRACE8:
        return "TRACE-8"
    if tasks == TRACE15:
        return "TRACE-15"
    if tasks == MATH:
        return "Math-LLM"
    for name, order in OLORA.items():
        if tasks == order:
            return name
    return f"custom-{len(tasks)} ({tasks[0]}..{tasks[-1]})" if tasks else "unknown"


# ------------------------------------------------------------------ metrics
def bwt_three_ways(diag, final):
    """diag, final: lists in task order (percent; None = missing).
    Returns a dict with LA, OP, the three BWT values and checks."""
    T = len(diag)
    pairs = [(d, f) for d, f in zip(diag, final) if d is not None and f is not None]
    if T < 2 or len(pairs) < T:
        return None                                          # incomplete matrix
    d = [f - g for g, f in zip(diag, final)]                 # d_i = R[T][i] - R[i][i]
    la = sum(diag) / T
    op = sum(final) / T
    earlier = d[:-1]                                         # tasks 1..T-1
    gem = sum(earlier) / (T - 1)
    trace = sum(earlier) / T
    treelora = sum(min(x, 0.0) for x in d) / T
    gains = [x for x in earlier if x > 0]
    return OrderedDict(
        T=T, LA=la, OP=op,
        BWT_GEM=gem, BWT_TRACE=trace, BWT_TreeLoRAcode=treelora,
        OP_minus_LA=op - la,
        last_task_gap=d[-1],                                 # should be 0 (same cell)
        n_improved=len(gains), sum_gains=sum(gains),
        per_task_change=" ".join(f"{x:+.2f}" for x in d),
    )


def to_percent(values):
    vals = [v for v in values if v is not None]
    scale = 100.0 if vals and max(abs(v) for v in vals) <= 1.0 + 1e-9 else 1.0
    return [None if v is None else v * scale for v in values], scale


# ------------------------------------------------------------------ run metadata
def load_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def parse_name(name):
    """RTR_<arm>_<model>_s<seed>_lr... -> (arm, model, seed). Best effort."""
    base = re.sub(r"^EVAL_", "", name)
    m = re.match(r"^(?:RTR|CLR)_(.+?)_s(\d+)_lr", base)
    arm, model, seed = None, None, None
    if m:
        head, seed = m.group(1), int(m.group(2))
        for key in MODEL_KEYS:
            if head.endswith("_" + key) or head == key:
                model, arm = key, head[: -len(key) - 1] if head != key else ""
                break
        if model is None:
            arm = head
    return arm, model, seed


def model_short(hf_name):
    if not hf_name:
        return None
    s = os.path.basename(str(hf_name).rstrip("/")).lower()
    table = [("llama-3.2-1b", "llama3_2_1b"), ("llama-2-7b", "llama2_7b"),
             ("gemma-2b", "gemma_2b"), ("mistral-7b", "mistral_7b"),
             ("r1-distill-qwen-1.5b", "r1distill_qwen_1_5b"),
             ("r1-distill-llama-8b", "r1distill_llama_8b")]
    for k, v in table:
        if k in s:
            return v
    return s


def run_record_from_json(res_path):
    res = load_json(res_path)
    if not res or "diagonal" not in res or "final_row" not in res:
        return None
    run_dir = os.path.dirname(res_path)
    is_eval = os.path.basename(run_dir).startswith("eval_")
    train_dir = os.path.dirname(run_dir) if is_eval else run_dir
    args = (load_json(os.path.join(run_dir, "eval_args.json")) if is_eval else None) \
        or load_json(os.path.join(train_dir, "train_args.json")) or {}

    tasks = list(res["diagonal"].keys())                     # CL order (json keeps insertion order)
    diag = [res["diagonal"][t] for t in tasks]
    final = [res["final_row"].get(t) for t in tasks]
    diag, scale = to_percent(diag)
    final, _ = to_percent(final)

    name = os.path.basename(train_dir)
    arm, model_from_name, seed_from_name = parse_name(name)
    split_file = args.get("cl_eval_file") or res.get("eval_file")
    split = {"eval.json": "val", "test.json": "test"}.get(str(split_file),
                                                         "val" if "_val_" in name else "test")
    logged = res.get("backward_transfer")
    return OrderedDict(
        source="json", path=res_path, run=name + ("/" + os.path.basename(run_dir) if is_eval else ""),
        stream=stream_name(tasks), model=model_short(args.get("model")) or model_from_name,
        arm=arm, split=split,
        seed=args.get("seed", seed_from_name),
        C=args.get("group_capacity"), delta=args.get("kl_routing_threshold"),
        beta=args.get("kl_beta"), h=args.get("merge_kl_halflife"),
        router=res.get("inference_router"),
        logged_BWT=None if logged is None else logged * scale,
        mtime=os.path.getmtime(res_path),
        diag=diag, final=final,
    )


MATRIX_HDR = re.compile(r"^\s*PERFORMANCE MATRIX\s*$")


def run_record_from_log(log_path):
    """Parse the last PERFORMANCE MATRIX block printed by train.py."""
    with open(log_path, errors="replace") as f:
        lines = f.read().splitlines()
    starts = [i for i, l in enumerate(lines) if MATRIX_HDR.match(l)]
    if not starts:
        return None
    i = starts[-1] + 1
    while i < len(lines) and not lines[i].strip().startswith("Trained"):
        i += 1
    i += 2                                                   # skip header and dashes
    rows = []
    while i < len(lines) and lines[i].strip():
        parts = lines[i].split()
        vals = [None if p == "---" else float(p) for p in parts[1:]]
        rows.append((parts[0], vals))
        i += 1
    if not rows:
        return None
    tasks = [r[0] for r in rows]
    n = len(tasks)
    if any(len(v) != n for _, v in rows):
        return None
    diag = [rows[j][1][j] for j in range(n)]
    final = rows[-1][1]
    diag, scale = to_percent(diag)
    final, _ = to_percent(final)
    logged = None
    for l in lines[i:i + 10]:
        m = re.search(r"Backward Transfer \(BWT\)\s+([+-]?\d*\.\d+)", l)
        if m:
            logged = float(m.group(1)) * scale
    name = re.sub(r"_\d+\.txt$", "", os.path.basename(log_path)).replace(".txt", "")
    arm, model, seed = parse_name(name)
    return OrderedDict(
        source="log", path=log_path, run=name, stream=stream_name(tasks), model=model,
        arm=arm, split="val" if "_val_" in name else "test", seed=seed,
        C=None, delta=None, beta=None, h=None, router=None,
        logged_BWT=logged, mtime=os.path.getmtime(log_path), diag=diag, final=final,
    )


# ------------------------------------------------------------------ main
def fmt(x, nd=2):
    return "" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def mean_std(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return ""
    if len(xs) == 1:
        return f"{xs[0]:.2f}"
    return f"{statistics.mean(xs):.2f} ± {statistics.stdev(xs):.2f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--roots", nargs="*", default=[], help="folders searched for cl_results.json")
    ap.add_argument("--logs", nargs="*", default=[], help="log globs; used only for runs without cl_results.json")
    ap.add_argument("--include", default=None, help="regex on run name or path")
    ap.add_argument("--split", choices=["val", "test"], default=None)
    ap.add_argument("--all_runs", action="store_true",
                    help="keep every run; default keeps the newest run per configuration and seed")
    ap.add_argument("--out", default="results_bwt")
    a = ap.parse_args()

    records, seen = [], set()
    for root in a.roots:
        for p in glob.glob(os.path.join(root, "**", "cl_results.json"), recursive=True):
            r = run_record_from_json(p)
            if r:
                records.append(r)
                seen.add(r["run"].split("/")[0])
    for pattern in a.logs:
        for p in glob.glob(pattern):
            r = run_record_from_log(p)
            # eval-only logs are named EVAL_<train run>_<val|test>_<job>.txt
            if r and re.sub(r"_(val|test)$", "", r["run"].replace("EVAL_", "")) not in seen:
                records.append(r)

    rows, skipped = [], []
    for r in records:
        if a.include and not re.search(a.include, r["run"] + " " + r["path"]):
            continue
        if a.split and r["split"] != a.split:
            continue
        m = bwt_three_ways(r["diag"], r["final"])
        if m is None:
            skipped.append(r["run"])
            continue
        row = OrderedDict((k, v) for k, v in r.items() if k not in ("diag", "final"))
        row.update(m)
        row["logged_vs_GEM"] = None if r["logged_BWT"] is None else r["logged_BWT"] - m["BWT_GEM"]
        rows.append(row)

    def cfg_key(r):
        return (r["stream"], r["model"], r["arm"], r["split"], r["C"], r["delta"], r["beta"], r["h"])

    if not a.all_runs:                                       # newest run per configuration + seed
        best = {}
        for r in rows:
            k = cfg_key(r) + (r["seed"],)
            if k not in best or r["mtime"] > best[k]["mtime"]:
                best[k] = r
        rows = list(best.values())
    rows.sort(key=lambda r: tuple(str(x) for x in cfg_key(r)) + (str(r["seed"]),))

    os.makedirs(a.out, exist_ok=True)
    cols = ["run", "stream", "model", "arm", "split", "C", "delta", "beta", "h", "seed", "router",
            "T", "LA", "OP", "BWT_GEM", "BWT_TRACE", "BWT_TreeLoRAcode", "OP_minus_LA",
            "logged_BWT", "logged_vs_GEM", "last_task_gap", "n_improved", "sum_gains",
            "per_task_change", "source", "path"]
    with open(os.path.join(a.out, "bwt_runs.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in rows:
            w.writerow([fmt(r.get(c), 4) if isinstance(r.get(c), float) else fmt(r.get(c)) for c in cols])

    groups = defaultdict(list)
    for r in rows:
        groups[cfg_key(r)].append(r)

    md = ["# BWT three ways (CRAFT runs)", "",
          "Negative = forgetting, percent. d_i = final score - score right after learning task i.", "",
          "- **GEM**: (1/(T-1)) sum_{i<T} d_i  (Lopez-Paz & Ranzato 2017; what train.py logs)",
          "- **TRACE**: (1/T) sum_{i<T} d_i = OP - LA  (Wang et al. 2023; current tables)",
          "- **TreeLoRA code**: (1/T) sum_i min(d_i, 0)  (TreeLoRA repo, inference/collect_results.py; gains clipped to 0). The TreeLoRA *paper* equation equals TRACE with the sign flipped.", "",
          "## Per configuration (mean ± std over seeds)", "",
          "| Stream | Model | Arm | Split | C | δ | β | h | Seeds | LA | OP | BWT GEM | BWT TRACE | BWT TreeLoRA code |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for k, rs in groups.items():
        stream, model, arm, split, C, delta, beta, h = k
        seeds = ",".join(str(r["seed"]) for r in rs)
        md.append("| " + " | ".join([
            str(stream), str(model), str(arm), str(split), fmt(C), fmt(delta), fmt(beta), fmt(h), seeds,
            mean_std([r["LA"] for r in rs]), mean_std([r["OP"] for r in rs]),
            mean_std([r["BWT_GEM"] for r in rs]), mean_std([r["BWT_TRACE"] for r in rs]),
            mean_std([r["BWT_TreeLoRAcode"] for r in rs])]) + " |")

    md += ["", "## How the three rules differ, run by run", "",
           "- TRACE / GEM = (T-1)/T always (same sum, different divisor).",
           "- TreeLoRA code - TRACE = -(sum of gains)/T: zero unless some earlier task improved.", "",
           "| Run | Seed | T | per-task change d_i | GEM | TRACE | TreeLoRA code | TRACE/GEM | gains clipped | logged - GEM |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        ratio = r["BWT_TRACE"] / r["BWT_GEM"] if abs(r["BWT_GEM"]) > 1e-12 else float("nan")
        ratio_s = "" if math.isnan(ratio) else f"{ratio:.3f}"
        lvg = r["logged_vs_GEM"]
        lvg_s = "" if lvg is None else f"{lvg:+.2f}"
        md.append(f"| {r['run']} | {r['seed']} | {r['T']} | {r['per_task_change']} | "
                  f"{r['BWT_GEM']:.2f} | {r['BWT_TRACE']:.2f} | {r['BWT_TreeLoRAcode']:.2f} | "
                  f"{ratio_s} | {r['n_improved']} ({r['sum_gains']:+.2f}) | {lvg_s} |")

    warn = [r for r in rows if abs(r["last_task_gap"]) > 1e-6]
    warn_log = [r for r in rows if r["logged_vs_GEM"] is not None and abs(r["logged_vs_GEM"]) > 0.01]
    md += ["", "## Checks", "",
           f"- Runs found: {len(records)}; used: {len(rows)}; skipped (incomplete matrix): {len(skipped)}.",
           f"- Last task: diagonal differs from final row in {len(warn)} run(s)"
           + (": " + ", ".join(r["run"] for r in warn) if warn else " (OP - LA = TRACE BWT holds in every run)."),
           f"- train.py logged BWT differs from GEM by more than 0.01 in {len(warn_log)} run(s)"
           + (": " + ", ".join(r["run"] for r in warn_log) if warn_log else ".")]
    if skipped:
        md += ["", "Skipped (diagonal or final row incomplete):", ""] + [f"- {s}" for s in skipped]

    with open(os.path.join(a.out, "bwt_summary.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md))
    print(f"\nWrote {os.path.join(a.out, 'bwt_runs.csv')} and {os.path.join(a.out, 'bwt_summary.md')}")


if __name__ == "__main__":
    main()

    # python scripts/collect_bwt.py --roots output_log $R2 --out results_bwt