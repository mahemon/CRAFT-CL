#!/usr/bin/env python3
"""
collect_results_bwt.py -- every finished CRAFT run, with BWT computed three ways,
per seed and averaged over seeds, for each backbone, each setting and each code
location (output_log folder).

Read-only. It reuses the log parser of scripts/collect_results.py (same run
detection, same current/legacy rule, same join of EVAL_ logs to their training
run) and does not change that script, train.py or any log.

For each run it takes the score of every task right after its training
(diagonal, R[i][i]) and at the end of the stream (final row, R[T][i]), with
d_i = R[T][i] - R[i][i] (negative = forgetting), and reports

    LA                 (1/T)     sum_{i=1..T}   R[i][i]
    OP                 (1/T)     sum_{i=1..T}   R[T][i]
    BWT_GEM            (1/(T-1)) sum_{i=1..T-1} d_i            Lopez-Paz & Ranzato 2017 (what train.py logs)
    BWT_TRACE          (1/T)     sum_{i=1..T-1} d_i = OP - LA   Wang et al. 2023 (current tables; TreeLoRA
                                                                paper equation with the sign flipped)
    BWT_TreeLoRAcode   (1/T)     sum_{i=1..T}   min(d_i, 0)     TreeLoRA repo inference/collect_results.py

All three: negative = forgetting, higher is better, percent.

Each run also gets its start date (from the log name, else the "Started" line)
and its hyperparameters (from the argument dump at the top of the log). Runs
from different output_log folders ("code" = the folder holding output_log,
e.g. CRAFT or Tuned_router) are kept apart, since they come from different
versions of train.py.

Seed mean: one run per seed (the most recent one when a seed was run more than
once; every run is still listed in the per-seed file), mean +- std over seeds.

Usage (from the repository root):
    R2=<another output_log folder>
    python scripts/collect_results_bwt.py --roots output_log $R2 --out results_bwt
    python scripts/collect_results_bwt.py --roots output_log $R2 --split test
    python scripts/collect_results_bwt.py --roots output_log $R2 --backbone llama3_2_1b gemma_2b

Writes
    <out>_by_seed.csv     one row per finished current-method run (every seed, every repeat)
    <out>_by_config.csv   one row per backbone + setting + code folder: seeds used, mean and std
    <out>_summary.md      per backbone and setting: the hyperparameters, each seed, then the seed mean
"""
import argparse
import csv
import glob
import os
import re
import statistics
import sys
from collections import defaultdict, OrderedDict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import collect_results as cr          # noqa: E402  (scripts/collect_results.py, next to this file)

# extra arguments read from the log's argument dump (not parsed by collect_results.py)
EXTRA_ARGS = {"max_n_eval_example": "eval_cap",
              "max_n_train_example": "train_cap", "kl_routing_direction": "route_dir",
              "merge_kl_source": "merge_src", "inference_router": "router",
              "greedy_decoding": "greedy", "dtype": "dtype"}
SETTING_KEYS = ["code", "stream", "split", "arm", "delta", "cap", "beta", "h", "merge",
                "rank", "positions", "warmup", "lr", "bs_x_gas",
                "eval_cap", "train_cap", "route_dir", "router", "greedy"]
METRICS = ["LA", "OP", "BWT_GEM", "BWT_TRACE", "BWT_TreeLoRAcode"]
# read per run and shown, but not used to group runs: an EVAL_ log may not print the
# per-task training lines, and it must still average with its training-and-eval siblings
RUN_INFO = ["epochs", "pos_per_task", "max_len"]
TS = re.compile(r"(20\d\d)[_-]?(\d\d)[_-]?(\d\d)[_-](\d\d)[_-]?(\d\d)[_-]?(\d\d)")


def bwt_three(diag, final):
    """diag, final in percent, task order. None if the matrix is incomplete."""
    T = len(diag)
    if T < 2 or any(x is None for x in diag) or any(x is None for x in final):
        return None
    d = [f - g for g, f in zip(diag, final)]
    la, op = sum(diag) / T, sum(final) / T
    gains = [x for x in d[:-1] if x > 0]
    return OrderedDict(
        T=T, LA=la, OP=op,
        BWT_GEM=sum(d[:-1]) / (T - 1),
        BWT_TRACE=sum(d[:-1]) / T,
        BWT_TreeLoRAcode=sum(min(x, 0.0) for x in d) / T,
        n_gains=len(gains), sum_gains=sum(gains),
        last_task_gap=d[-1],
        per_task_change=" ".join(f"{x:+.2f}" for x in d))


def log_extras(path):
    """Start date and extra hyperparameters from a job log."""
    s = open(path, errors="ignore").read()
    cut = s.find("PERFORMANCE MATRIX")
    head = s if cut < 0 else s[:cut]
    out = {v: None for v in EXTRA_ARGS.values()}
    out.update(epochs=None, pos_per_task=None, max_len=None)
    for mm in re.finditer(r"^  (\w+)\s{2,}(\S.*?)\s*$", head, re.M):
        k = mm.group(1)
        if k in EXTRA_ARGS and out[EXTRA_ARGS[k]] is None:
            v = mm.group(2).strip()
            out[EXTRA_ARGS[k]] = None if v in ("None", "") else v
    # per-task epochs, positions and length, from the "CL step k/T: task [epochs=.. pos=.. max_len=..]" lines
    steps = OrderedDict()
    for mm in re.finditer(r"CL step (\d+)/\d+:\s+(\S+)\s+\[epochs=(\d+)\s+pos=(\S+)\s+max_len=(\d+)\]", s):
        steps.setdefault(int(mm.group(1)), (mm.group(3), mm.group(4), mm.group(5)))
    out["epochs"] = ",".join(v[0] for v in steps.values()) or None
    out["pos_per_task"] = ",".join(v[1] for v in steps.values()) or None
    out["max_len"] = ",".join(sorted(set(v[2] for v in steps.values()), key=int)) or None
    m = TS.search(os.path.basename(path))
    if m:
        out["date"] = "%s-%s-%s %s:%s" % m.groups()[:5]
    else:
        st = re.search(r"Started\s*:\s*(.+)", s)
        out["date"] = st.group(1).strip() if st else None
    return out


def code_label(root):
    """Name of the folder that holds output_log, e.g. CRAFT or Tuned_router."""
    p = os.path.abspath(root.rstrip("/"))
    return os.path.basename(os.path.dirname(p)) if os.path.basename(p) == "output_log" else os.path.basename(p)


def fmt(x, nd=2):
    if x is None:
        return "—"
    return f"{x:.{nd}f}" if isinstance(x, float) else str(x)


def ms(vals, nd=2):
    v = [x for x in vals if x is not None]
    if not v:
        return "—"
    if len(v) == 1:
        return f"{v[0]:.{nd}f}"
    return f"{statistics.mean(v):.{nd}f} ± {statistics.stdev(v):.{nd}f}"


def merged(rs, k):
    vals = sorted({r[k] for r in rs if r.get(k)})
    return "—" if not vals else (vals[0] if len(vals) == 1 else "differs by run, see _by_seed.csv")


def hp_line(r, rs):
    h = fmt(r["h"]) if r["merge"] == "kl" and r["h"] is not None else (r["merge"] or "—")
    lr = "—" if r["lr"] is None else f"{r['lr']:g}"
    return (f"code={r['code']} · δ={fmt(r['delta'], 1)} · C={fmt(r['cap'])} · β={fmt(r['beta'])} · h={h} · "
            f"rank={fmt(r['rank'])} · positions={r['positions'] or '—'} · warm-up={fmt(r['warmup'])} · "
            f"lr={lr} · batch={r['bs_x_gas']} · epochs per task={merged(rs, 'epochs')} · "
            f"positions per task={merged(rs, 'pos_per_task')} · max_len={merged(rs, 'max_len')} · "
            f"eval_cap={r['eval_cap'] or 'all'} · train_cap={r['train_cap'] or 'all'} · "
            f"routing={r['route_dir'] or '—'} · router={r['router'] or '—'} · greedy={r['greedy'] or '—'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--roots", nargs="+", required=True, help="output_log folder(s)")
    ap.add_argument("--out", default="results_bwt")
    ap.add_argument("--split", default=None, choices=["eval", "test", "auto"])
    ap.add_argument("--backbone", nargs="*", default=None, help="keep only these backbone tags")
    a = ap.parse_args()

    runs = []
    n_seen = n_legacy = n_unfinished = n_incomplete = 0
    for root in a.roots:
        code = code_label(root)
        for d in sorted(glob.glob(os.path.join(root, "logs_*_router_*"))):
            tag = os.path.basename(d).split("_router_", 1)[1]
            for p in sorted(glob.glob(os.path.join(d, "RTR_*.txt")) + glob.glob(os.path.join(d, "EVAL_RTR_*.txt"))):
                try:
                    r, task_rows = cr.parse_log(p, tag)
                    ex = log_extras(p)
                except Exception as e:
                    print(f"skip {p}: {type(e).__name__}: {e}")
                    continue
                n_seen += 1
                if a.split and r["split"] != a.split:
                    continue
                if a.backbone and r["backbone"] not in a.backbone:
                    continue
                if r["status"] != "finished":
                    n_unfinished += 1
                    continue
                if r["method"] != "current":
                    n_legacy += 1
                    continue
                diag = [t["right_after"] for t in task_rows]
                final = [t["final"] for t in task_rows]
                m = bwt_three(diag, final)
                if m is None:
                    n_incomplete += 1
                    continue
                r = dict(r)
                r.update(ex)
                r["code"] = code
                row = OrderedDict((k, r.get(k)) for k in ["backbone"] + SETTING_KEYS + RUN_INFO + ["seed", "date", "K", "max_group"])
                row.update(m)
                row["log"] = os.path.basename(r["log"])
                row["path"] = r["log"]
                runs.append(row)
    if not runs:
        raise SystemExit("No finished current-method runs found.")

    def cfg(r):
        return tuple([r["backbone"]] + [r[k] for k in SETTING_KEYS])

    def when(r):
        return (r["date"] or "", r["log"])

    runs.sort(key=lambda r: tuple(str(x) for x in cfg(r)) + (str(r["seed"]),) + when(r))

    cols = list(runs[0].keys())
    with open(a.out + "_by_seed.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in runs:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})

    groups = defaultdict(list)
    for r in runs:
        groups[cfg(r)].append(r)
    cfg_rows = []
    md = ["# CRAFT runs: BWT three ways", "",
          "Percent. Negative BWT = forgetting, higher is better. d_i = final score - score right after training task i.", "",
          "- **GEM**: (1/(T-1)) sum over the first T-1 tasks of d_i (Lopez-Paz & Ranzato 2017; what train.py logs)",
          "- **TRACE**: (1/T) sum over the first T-1 tasks of d_i = OP - LA (Wang et al. 2023; current tables)",
          "- **TreeLoRA code**: (1/T) sum over all tasks of min(d_i, 0) (TreeLoRA repo; gains clipped to 0)", "",
          "One block per setting. Seed mean: one run per seed (the most recent if a seed was run twice); "
          "every run is in the _by_seed.csv file. `code` = the folder holding output_log (different train.py versions).", ""]
    by_bb = defaultdict(list)
    for key, rs in groups.items():
        by_bb[key[0]].append((key, rs))

    for bb in sorted(by_bb):
        md += [f"## {bb}", ""]
        for key, rs in sorted(by_bb[bb], key=lambda kv: tuple(str(x) for x in kv[0])):
            latest = {}
            for r in sorted(rs, key=when):
                latest[r["seed"]] = r
            used = [latest[s] for s in sorted(latest, key=lambda s: (s is None, s))]
            first = rs[0]
            md += [f"**{first['stream']} · {first['split']} · {first['arm']}**  ",
                   hp_line(first, rs), "",
                   "| Seed | Date | K | LA | OP | BWT GEM | BWT TRACE | BWT TreeLoRA code | Log |",
                   "|---|---|---|---|---|---|---|---|---|"]
            for r in sorted(rs, key=lambda x: (str(x["seed"]),) + when(x)):
                mark = "" if r is latest.get(r["seed"]) else " (older, not averaged)"
                md.append(f"| {r['seed']}{mark} | {r['date'] or '—'} | {fmt(r['K'])} | {fmt(r['LA'])} | {fmt(r['OP'])} | "
                          f"{fmt(r['BWT_GEM'])} | {fmt(r['BWT_TRACE'])} | {fmt(r['BWT_TreeLoRAcode'])} | {r['log']} |")
            if len(used) > 1:
                md.append(f"| **mean ({len(used)} seeds)** | | {ms([r['K'] for r in used], 1)} | "
                          f"{ms([r['LA'] for r in used])} | {ms([r['OP'] for r in used])} | "
                          f"{ms([r['BWT_GEM'] for r in used])} | {ms([r['BWT_TRACE'] for r in used])} | "
                          f"{ms([r['BWT_TreeLoRAcode'] for r in used])} | |")
            md.append("")
            row = OrderedDict(zip(["backbone"] + SETTING_KEYS, key))
            row["seeds"] = ",".join(str(r["seed"]) for r in used)
            row["dates"] = ",".join(str(r["date"]) for r in used)
            row["n_seeds"] = len(used)
            row["n_older_runs_dropped"] = len(rs) - len(used)
            row["T"] = first["T"]
            for mname in ["K"] + METRICS:
                vals = [r[mname] for r in used if r[mname] is not None]
                row[mname + "_mean"] = round(statistics.mean(vals), 4) if vals else None
                row[mname + "_std"] = round(statistics.stdev(vals), 4) if len(vals) > 1 else None
            cfg_rows.append(row)

    with open(a.out + "_by_config.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(cfg_rows[0].keys()))
        w.writeheader()
        w.writerows(cfg_rows)

    gap = [r for r in runs if abs(r["last_task_gap"]) > 1e-6]
    md += ["## Checks", "",
           f"- Logs read: {n_seen}. Used: {len(runs)} finished current-method runs. "
           f"Skipped: {n_unfinished} unfinished, {n_legacy} legacy, {n_incomplete} with an incomplete matrix.",
           f"- Code folders: " + ", ".join(sorted({r['code'] for r in runs})) + ".",
           f"- Last task: right-after score differs from final score in {len(gap)} run(s)"
           + (": " + ", ".join(r["log"] for r in gap) if gap else " (so BWT_TRACE = OP - LA in every run)."),
           f"- Runs where clipping changes BWT (some earlier task improved): "
           f"{sum(1 for r in runs if r['n_gains'] > 0)} of {len(runs)}."]
    open(a.out + "_summary.md", "w").write("\n".join(md) + "\n")
    print("\n".join(md))
    print(f"\nWrote {a.out}_by_seed.csv ({len(runs)} runs), {a.out}_by_config.csv ({len(cfg_rows)} settings), "
          f"{a.out}_summary.md")


if __name__ == "__main__":
    main()





# cd <repository root>
# R2=<another output_log folder>
# python scripts/collect_results_bwt.py --roots output_log $R2 --out results_bwt