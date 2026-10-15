#!/usr/bin/env python3
"""
make_tables.py -- turn collect_results.py output into the paper's tables.

  python scripts/collect_results.py --roots output_log [OLD_ROOT/output_log] --out results
  python scripts/make_tables.py --runs results_runs.csv --tasks results_tasks.csv --out paper_tables.tex
  ... --baselines      also write the comparison tables with the baselines of Qian et al. (off by default:
                       the main and stream results then go into one CRAFT-only table, tab:craft)

Writes
  <out>                 LaTeX tables (booktabs), one per experiment family
  <out>.missing.txt     every run the plan needs that is not finished yet
  <out>.kl_pairs.csv    (terminal KL, in-group drop caused) per joining task, for a scatter plot

Conventions
  * OP / BWT / LA in percent; BWT = OP - LA = mean over all T tasks of (final - right after training),
    negative = forgetting, higher is better. Qian et al. (2025) report the opposite sign (LA - OP), so
    their BWT values are negated once below (BWT_SIGN block); the magnitudes compare directly.
    collect_results.py still writes LA - OP to results_runs.csv; load() negates it.
  * CRAFT entries are mean +- std over the seeds found (n shown in the caption).
  * Main and stream tables use the test split; ablation and sensitivity tables use the
    validation split (eval.json) and seed 42.
  * "task-agnostic" = the learned task head picks the task at test time (TreeLoRA's setting);
    its BWT is measured against the oracle diagonal.
Baseline numbers: Qian et al. (2025), TreeLoRA, ICML -- Tables 3, 7, 9, 10, 12.
Standard library only.
"""
import argparse, csv, math, statistics
from collections import defaultdict

# --------------------------------------------------------------------------
# Final CRAFT configuration per backbone (selection: Llama-1B validation runs;
# beta and h follow the KL-scale rule). Change here if a value changes.
# --------------------------------------------------------------------------
FINAL = {   # backbone tag: (delta, capacity, beta, h)
    "llama3_2_1b":         (0.7, 2, 0.3, 0.6),
    "gemma_2b":            (0.7, 2, 0.2, 1.0),
    "llama2_7b":           (0.7, 2, 0.3, 0.6),
    "mistral_7b":          (0.7, 2, 0.3, 0.6),
    "r1distill_qwen_1_5b": (0.7, 2, 0.3, 0.6),
    "r1distill_llama_8b":  (0.7, 2, 0.3, 0.6),
}
FINAL_LR = {"llama3_2_1b": 2e-4, "gemma_2b": 2e-4, "llama2_7b": 1e-4, "mistral_7b": 5e-4,
            "r1distill_qwen_1_5b": 2e-4, "r1distill_llama_8b": 2e-4, "llama2_13b": 1e-4}
FINAL["llama2_13b"] = (0.7, 2, 0.3, 0.6)
DEF_RANK, DEF_POS, DEF_WU = 8, "f15+l15", 100
PRETTY = {"llama2_13b": "Llama-2-13B-Chat", "llama3_2_1b": "Llama-3.2-1B-Instruct", "gemma_2b": "Gemma-2B-it", "llama2_7b": "Llama-2-7B-Chat",
          "mistral_7b": "Mistral-7B-Instruct-v0.3", "r1distill_qwen_1_5b": "R1-Distill-Qwen-1.5B",
          "r1distill_llama_8b": "R1-Distill-Llama-8B"}
LAYERS = {"llama2_13b": 40, "llama3_2_1b": 16, "gemma_2b": 18, "llama2_7b": 32, "mistral_7b": 32,
          "r1distill_qwen_1_5b": 28, "r1distill_llama_8b": 32}
PRIVATE_FRAC = 0.38          # share of layers in the private deep l-stream

# --------------------------------------------------------------------------
# Baselines: Qian et al. (2025). (OP, OP std, BWT, BWT std); None = not reported.
# --------------------------------------------------------------------------
BASE_COLS = ["FIX (ICL)", "SeqLoRA", "OGD", "GEM", "EWC", "L2P", "DualPrompt", "HiDeLoRA", "O-LoRA", "TreeLoRA"]
TRACE8 = {  # Table 3
    "mistral_7b": [(45.04, .3, None, None), (46.94, 1.2, 11.41, .6), (45.44, 1.6, 12.75, .7), (52.32, 1.2, 6.01, .3),
                   (52.45, 1.3, 5.98, .8), (49.32, .8, 5.34, .6), (51.14, 1.2, 6.13, .5), (51.81, .9, 6.25, .3),
                   (52.02, .8, 8.13, .6), (54.77, 1.1, 3.77, .4)],
    "llama2_7b":  [(38.94, .3, None, None), (34.30, 1.2, 18.50, .8), (42.09, 1.6, 8.06, 1.2), (40.08, 1.6, 6.77, 1.2),
                   (42.36, 1.2, 5.97, .8), (36.23, .8, 8.25, .8), (37.69, 1.2, 8.03, .8), (41.60, .8, 7.12, .4),
                   (42.78, .8, 7.16, .4), (43.52, 1.0, 3.46, .4)],
    "gemma_2b":   [(32.30, .2, None, None), (31.89, .8, 15.28, .4), (32.85, 1.4, 12.27, .9), (26.48, 1.5, 18.25, .9),
                   (28.35, 1.6, 16.96, 1.2), (31.14, 1.2, 15.77, .7), (32.42, 1.0, 14.25, .5), (33.25, .9, 13.66, .5),
                   (33.73, .8, 12.36, .4), (33.41, .9, 8.50, .5)],
    "llama3_2_1b": [(31.16, .4, None, None), (29.73, 1.6, 17.03, 1.2), (30.12, 2.0, 15.20, 1.6), (32.19, 2.0, 10.74, 1.6),
                    (31.96, 1.6, 11.62, 1.2), (29.38, 1.2, 13.57, .8), (30.76, 1.2, 11.34, .8), (33.73, 1.2, 12.36, .8),
                    (32.94, .8, 12.89, 1.2), (36.14, .7, 7.36, .8)],
}
MATHLLM = [(36.92, .5, None, None), (40.24, 1.0, 9.12, .8), (42.09, 1.4, 8.06, .5), (40.75, 1.4, 16.25, .5),
           (37.25, 1.1, 17.50, .6), (31.67, 1.0, 2.54, .4), (37.72, 1.0, 9.54, .7), (42.83, .7, 11.92, .5),
           (42.00, 1.0, 11.67, .6), (45.59, 1.2, 2.32, .6)]                         # Table 7 (Llama-2-7B)
TRACE15 = [(41.32, None, None, None), (40.71, None, 15.72, None), (32.52, None, 21.32, None), (35.48, None, 18.33, None),
           (31.46, None, 22.22, None), (41.05, None, 14.92, None), (41.29, None, 15.58, None), (42.38, None, 11.23, None),
           (44.02, None, 10.99, None), (45.68, None, 6.41, None)]                    # Table 9 (Llama-3.2-1B)
L2_13B = {"FIX (ICL)": (40.15, None), "SeqLoRA": (39.16, 15.58), "OGD": (42.32, 9.72), "GEM": (43.77, 8.42),
          "EWC": (41.23, 10.12), "HiDeLoRA": (43.27, 11.27), "O-LoRA": (44.32, 5.19), "TreeLoRA": (47.13, 3.42)}  # Table 13
RECENT_L2 = {"TreeLoRA": (43.52, 3.46), "SAPT": (42.93, 5.49), "TASL": (43.19, 4.58)}  # Table 12
ORDER_COLS = ["FIX (ICL)", "OGD", "GEM", "SeqLoRA", "HiDeLoRA", "O-LoRA", "TreeLoRA"]
ORDERS = {  # Table 10, (OP, BWT)
    1: [(48.75, None), (54.16, 10.82), (54.10, 10.25), (55.71, 8.63), (56.32, 2.75), (59.50, 2.51), (59.73, 2.22)],
    2: [(48.75, None), (46.82, 21.50), (46.70, 20.93), (45.26, 7.70), (53.41, 5.58), (52.53, 5.65), (53.78, 5.74)],
    3: [(48.75, None), (43.79, 27.31), (51.12, 16.24), (49.03, 19.05), (61.25, 3.12), (63.82, 2.03), (62.76, 2.23)],
    4: [(52.13, None), None, None, None, (59.44, 4.33), (59.89, 4.67), (58.45, 4.98)],
    5: [(52.13, None), None, None, None, (54.49, 7.52), (57.05, 4.42), (58.12, 3.31)],
    6: [(52.13, None), None, None, None, (57.26, 6.98), (58.02, 4.73), (59.00, 4.12)],
}

# BWT_SIGN: the numbers above are as printed by Qian et al. (positive = forgetting).
# The paper uses BWT = OP - LA (negative = forgetting), so negate their BWT once here.
def _neg(x):
    return -x if x is not None else None
TRACE8 = {k: [(o, os_, _neg(b), bs) for o, os_, b, bs in v] for k, v in TRACE8.items()}
MATHLLM = [(o, os_, _neg(b), bs) for o, os_, b, bs in MATHLLM]
TRACE15 = [(o, os_, _neg(b), bs) for o, os_, b, bs in TRACE15]
L2_13B = {k: (o, _neg(b)) for k, (o, b) in L2_13B.items()}
RECENT_L2 = {k: (o, _neg(b)) for k, (o, b) in RECENT_L2.items()}
ORDERS = {k: [(v[0], _neg(v[1])) if v else None for v in vs] for k, vs in ORDERS.items()}

ABLATIONS = [  # (block, arm, label, what differs from the reference)
    ("KL anchor", "noreg", "no regularizer", None),
    ("KL anchor", "agem", "A-GEM instead of KL", None),
    ("Merge", "merge_overwrite", "overwrite instead of KL merge", None),
    ("Merge", "merge_h2.0", "KL merge, $h = 2.0$", None),
    ("Routing", "dir_forward", "distance on the arriving task's data", None),
    ("Routing", "private_fixed", "fixed task-family groups (no routing)", None),
    ("Storage", "shared", "no private deep l-stream", None),
    ("Capacity", "cap3", "capacity $C = 3$", "cap"),
]
ARM_TIER = {"cap3": "P1"}            # everything else in ABLATIONS is P0 on Llama-1B
GEMMA_ARMS = {"noreg", "merge_overwrite", "shared"}   # p1_gemma_ablate runs only these on Gemma
BETAS = [0.1, 0.3, 1.0]
REQUIRED_SEEDS = [42, 1234]          # P0
THIRD_SEED = 2024                     # P1: matches the baselines' three runs


def f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def load(path):
    rows = list(csv.DictReader(open(path)))
    for r in rows:
        for k in ("seed", "cap", "K", "n_tasks", "bench", "rank", "warmup", "max_group"):
            r[k] = int(float(r[k])) if r.get(k) not in (None, "") else None
        for k in ("delta", "beta", "h", "LA", "OP", "BWT", "BWT_in_groups", "head_OP", "train_hours", "wall_hours", "lr"):
            r[k] = f(r.get(k))
        for k in ("BWT", "BWT_in_groups"):     # collect_results.py writes LA - OP; the paper uses OP - LA
            if r[k] is not None:
                r[k] = -r[k]
    return [r for r in rows if r["status"] == "finished" and r["method"] == "current"]


def close(a, b):
    return a is not None and b is not None and abs(a - b) < 1e-6


def pick(runs, backbone, stream, split, arm="v20260828", delta=None, cap=None, beta=None, h=None,
         rank=DEF_RANK, positions=DEF_POS, warmup=DEF_WU, lr="final"):
    if lr == "final":
        lr = FINAL_LR.get(backbone)
    out = []
    for r in runs:
        if rank is not None and r.get("rank") not in (None, rank):
            continue
        if positions is not None and r.get("positions") not in (None, "", positions):
            continue
        if warmup is not None and r.get("warmup") not in (None, warmup):
            continue
        if lr is not None and r.get("lr") is not None and not close(r["lr"], lr):
            continue
        if r["backbone"] != backbone or r["stream"] != stream or r["split"] != split or r["arm"] != arm:
            continue
        if delta is not None and not close(r["delta"], delta):
            continue
        if cap is not None and r["cap"] != cap:
            # capacity never reached: a run at a larger capacity whose groups all
            # stayed at <= cap tasks routed exactly like a capacity-`cap` run
            if not (r["cap"] is not None and r["cap"] > cap and r.get("max_group") is not None
                    and r["max_group"] <= cap):
                continue
        if beta is not None and not close(r["beta"], beta):
            continue
        if h is not None and not close(r["h"], h):
            continue
        out.append(r)
    # one run per seed: keep the latest log name (timestamps sort)
    best = {}
    for r in sorted(out, key=lambda x: x["log"]):
        best[r["seed"]] = r
    return [best[s] for s in sorted(best)]


def final_runs(runs, bb, stream="TRACE-8@5000", split="test", arm="v20260828"):
    d, c, b, h = FINAL[bb]
    return pick(runs, bb, stream, split, arm, d, c, b, h)


def agg(vals):
    v = [x for x in vals if x is not None]
    if not v:
        return None, None, 0
    return statistics.mean(v), (statistics.stdev(v) if len(v) > 1 else None), len(v)


def cell(m, s=None, bold=False, digits=2):
    if m is None:
        return "--"
    if round(m, digits) == 0:
        m = 0.0
    txt = f"{m:.{digits}f}" + (f"$\\pm${s:.2f}" if s is not None else "")
    return f"\\textbf{{{txt}}}" if bold else txt


def craft_stats(rs, head=False):
    if head:
        op = [r["head_OP"] for r in rs]
        bwt = [(r["head_OP"] - r["LA"]) if r["LA"] is not None and r["head_OP"] is not None else None for r in rs]
    else:
        op = [r["OP"] for r in rs]
        bwt = [r["BWT"] for r in rs]
    return agg(op), agg(bwt)


def params_ratio(bb, K, T):
    L = LAYERS.get(bb, 16)
    Ld = round(PRIVATE_FRAC * L)
    return (2 * L * T) / (K * (2 * L - Ld) + T * Ld) if K else None


# --------------------------------------------------------------------------
def table_main(runs, missing):
    order = ["mistral_7b", "llama2_7b", "gemma_2b", "llama3_2_1b"]
    cols = BASE_COLS + ["CRAFT (task-agn.)", "CRAFT (oracle)"]
    L = ["\\begin{table*}[t]", "\\centering", "\\scriptsize", "\\setlength{\\tabcolsep}{3pt}",
         "\\caption{TRACE-8, test split. OP (\\%, $\\uparrow$) and BWT $=$ OP $-$ LA (\\%, $\\uparrow$, negative = forgetting; baseline BWT sign-flipped to this convention). "
         "Baselines from \\citet{qian2025treelora}, mean$\\pm$std over three runs. CRAFT: mean$\\pm$std over the seeds shown in the last row. "
         "Task-agnostic: the learned task head selects the task at test time, the setting of the baselines. Best per row in bold.}",
         "\\label{tab:main}", "\\resizebox{\\textwidth}{!}{%",
         "\\begin{tabular}{l l " + "c" * len(cols) + "}", "\\toprule",
         " & & " + " & ".join(cols) + " \\\\", "\\midrule"]
    seeds_note = []
    for bb in order:
        rs = final_runs(runs, bb)
        for s in REQUIRED_SEEDS + [THIRD_SEED]:
            if s not in [r["seed"] for r in rs]:
                missing.append(f"{'P0' if s in REQUIRED_SEEDS else 'P1'} main TRACE-8 test | {bb} | seed {s}")
        (op_h, bwt_h), (op_o, bwt_o) = craft_stats(rs, head=True), craft_stats(rs)
        base = TRACE8[bb]
        ops = [b[0] for b in base] + [op_h[0], op_o[0]]
        bwts = [b[2] for b in base] + [bwt_h[0], bwt_o[0]]
        best_op = max(x for x in ops if x is not None)
        best_bwt = max(x for x in bwts if x is not None)
        row_op = [cell(b[0], b[1], close(b[0], best_op)) for b in base] + \
                 [cell(op_h[0], op_h[1], close(op_h[0], best_op)), cell(op_o[0], op_o[1], close(op_o[0], best_op))]
        row_bwt = [cell(b[2], b[3], close(b[2], best_bwt)) for b in base] + \
                  [cell(bwt_h[0], bwt_h[1], close(bwt_h[0], best_bwt)), cell(bwt_o[0], bwt_o[1], close(bwt_o[0], best_bwt))]
        L += [f"\\multirow{{2}}{{*}}{{{PRETTY[bb]}}} & OP & " + " & ".join(row_op) + " \\\\",
              " & BWT & " + " & ".join(row_bwt) + " \\\\", "\\midrule"]
        seeds_note.append(f"{PRETTY[bb]}: {','.join(str(r['seed']) for r in rs) or 'none'}")
    L[-1] = "\\bottomrule"
    L += ["\\end{tabular}}", f"% CRAFT seeds -- {'; '.join(seeds_note)}", "\\end{table*}", ""]
    return "\n".join(L)


def table_recent(runs):
    rs = final_runs(runs, "llama2_7b")
    (op, bwt) = craft_stats(rs)
    cols = list(RECENT_L2) + ["CRAFT"]
    ops = [v[0] for v in RECENT_L2.values()] + [op[0]]
    bw = [v[1] for v in RECENT_L2.values()] + [bwt[0]]
    bo, bb_ = max(x for x in ops if x is not None), max(x for x in bw if x is not None)
    L = ["\\begin{table}[t]", "\\centering", "\\small",
         "\\caption{Recent baselines on Llama-2-7B-Chat, TRACE-8 test split. Baselines from \\citet{qian2025treelora} (SAPT without generative replay).}",
         "\\label{tab:recent}", "\\begin{tabular}{l " + "c" * len(cols) + "}", "\\toprule",
         " & " + " & ".join(cols) + " \\\\", "\\midrule",
         "OP $\\uparrow$ & " + " & ".join(cell(x, None, close(x, bo)) for x in ops) + " \\\\",
         "BWT $\\uparrow$ & " + " & ".join(cell(x, None, close(x, bb_)) for x in bw) + " \\\\",
         "\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    return "\n".join(L)


def table_stream(runs, missing, stream, bb, base, label, caption, need_seeds):
    rs = final_runs(runs, bb, stream=stream)
    for s in need_seeds + [THIRD_SEED]:
        if s not in [r["seed"] for r in rs]:
            missing.append(f"{'P0' if s in need_seeds else 'P1'} {label} | {bb} | seed {s}")
    (op, bwt) = craft_stats(rs, head=True)
    ops = [b[0] for b in base] + [op[0]]
    bws = [b[2] for b in base] + [bwt[0]]
    bo = max(x for x in ops if x is not None)
    bb_ = max(x for x in bws if x is not None)
    cols = BASE_COLS + ["CRAFT"]
    L = ["\\begin{table*}[t]", "\\centering", "\\scriptsize", "\\setlength{\\tabcolsep}{3pt}",
         f"\\caption{{{caption} CRAFT seeds: {','.join(str(r['seed']) for r in rs) or 'none'} (task-agnostic).}}",
         f"\\label{{tab:{label}}}", "\\begin{tabular}{l " + "c" * len(cols) + "}", "\\toprule",
         " & " + " & ".join(cols) + " \\\\", "\\midrule",
         "OP $\\uparrow$ & " + " & ".join([cell(b[0], b[1], close(b[0], bo)) for b in base] + [cell(op[0], op[1], close(op[0], bo))]) + " \\\\",
         "BWT $\\uparrow$ & " + " & ".join([cell(b[2], b[3], close(b[2], bb_)) for b in base] + [cell(bwt[0], bwt[1], close(bwt[0], bb_))]) + " \\\\",
         "\\bottomrule", "\\end{tabular}", "\\end{table*}", ""]
    return "\n".join(L)


def table_orders(runs, missing):
    cols = ORDER_COLS + ["CRAFT"]
    L = ["\\begin{table*}[t]", "\\centering", "\\scriptsize",
         "\\caption{O-LoRA task orders on Llama-3.2-1B-Instruct, OP / BWT (\\%). Orders 1--3: four classification tasks; "
         "orders 4--6: fifteen tasks. Baselines from \\citet{qian2025treelora}. CRAFT: seed 42, task-agnostic.}",
         "\\label{tab:orders}", "\\begin{tabular}{l " + "c" * len(cols) + "}", "\\toprule",
         "Order & " + " & ".join(cols) + " \\\\", "\\midrule"]
    for k in range(1, 7):
        rs = final_runs(runs, "llama3_2_1b", stream=f"O-LoRA order {k}@500")
        if not rs:
            missing.append(f"P0 O-LoRA order {k} | llama3_2_1b | seed 42")
        elif 1234 not in [r["seed"] for r in rs]:
            missing.append(f"P1 O-LoRA order {k} | llama3_2_1b | seed 1234")
        (op, bwt) = craft_stats(rs, head=True)
        vals = ORDERS[k] + [(op[0], bwt[0])]
        best = max(v[0] for v in vals if v and v[0] is not None)
        cells = []
        for v in vals:
            if not v or v[0] is None:
                cells.append("--")
            else:
                o = cell(v[0], None, close(v[0], best))
                cells.append(o + " / " + cell(v[1]))
        L.append(f"{k} & " + " & ".join(cells) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular}", "\\end{table*}", ""]
    return "\n".join(L)


def table_sharing(runs, missing):
    bb = "llama3_2_1b"
    d, c, b, h = FINAL[bb]
    arms = [("Task-wise (no sharing)", pick(runs, bb, "TRACE-8@5000", "test", "taskwise")),
            ("CRAFT", final_runs(runs, bb)),
            ("All-in-one, KL anchor", pick(runs, bb, "TRACE-8@5000", "test", "allinone_KL")),
            ("All-in-one, no regularizer", pick(runs, bb, "TRACE-8@5000", "test", "allinone_noreg"))]
    L = ["\\begin{table}[t]", "\\centering", "\\small", "\\setlength{\\tabcolsep}{4pt}",
         "\\caption{Sharing structure on Llama-3.2-1B-Instruct, TRACE-8 test split. Stored interventions relative to "
         "task-wise storage, and training hours on one A100.}",
         "\\label{tab:sharing}", "\\begin{tabular}{l c c c c c c}", "\\toprule",
         "Storage & $K$ & Fewer params & LA $\\uparrow$ & OP $\\uparrow$ & BWT $\\uparrow$ & Train h \\\\", "\\midrule"]
    for name, rs in arms:
        if not rs:
            missing.append(f"P0 sharing | {bb} | {name}")
        elif 1234 not in [r["seed"] for r in rs]:
            missing.append(f"P1 sharing | {bb} | {name} | seed 1234")
        K = agg([r["K"] for r in rs])[0]
        ratio = params_ratio(bb, round(K), 8) if K else None
        ratio_txt = ("%.2f$\\times$" % ratio) if ratio else "--"     # outside the f-string (Python < 3.12)
        L.append(f"{name} & {('%.1f' % K) if K else '--'} & {ratio_txt} & "
                 f"{cell(*agg([r['LA'] for r in rs])[:2])} & {cell(*agg([r['OP'] for r in rs])[:2])} & "
                 f"{cell(*agg([r['BWT'] for r in rs])[:2])} & {cell(agg([r['train_hours'] for r in rs])[0], None, False, 1)} \\\\")
    L += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    return "\n".join(L)


def table_craft(runs):
    """CRAFT only, no baselines: main TRACE-8 and every stream, one row per backbone / stream."""
    rows = [("TRACE-8", bb, "TRACE-8@5000") for bb in
            ["mistral_7b", "llama2_7b", "gemma_2b", "llama3_2_1b", "llama2_13b", "r1distill_llama_8b", "r1distill_qwen_1_5b"]]
    rows += [("TRACE-15", "llama3_2_1b", "TRACE-15@500")]
    rows += [(f"O-LoRA order {k}", "llama3_2_1b", f"O-LoRA order {k}@500") for k in range(1, 7)]
    rows += [("Math-LLM", "llama2_7b", "Math-LLM@5000")]
    L = ["\\begin{table*}[t]", "\\centering", "\\small", "\\setlength{\\tabcolsep}{5pt}",
         "\\caption{CRAFT results, test split. LA, OP and BWT in \\%; BWT $=$ OP $-$ LA (negative = forgetting). "
         "Oracle: the task label is given at test time; task-agnostic: the learned task head selects the task. "
         "$K$: number of groups. Mean$\\pm$std over the seeds listed.}",
         "\\label{tab:craft}", "\\begin{tabular}{l l l c c c c c}", "\\toprule",
         "Benchmark & Backbone & Seeds & $K$ & LA $\\uparrow$ & OP $\\uparrow$ & BWT $\\uparrow$ & OP task-agn. $\\uparrow$ \\\\",
         "\\midrule"]
    last = None
    for bench, bb, stream in rows:
        rs = final_runs(runs, bb, stream=stream)
        if not rs and bb in ("llama2_13b", "r1distill_llama_8b", "r1distill_qwen_1_5b"):
            continue                                   # extra backbones: only when run
        group = bench.split(" order")[0]
        if last is not None and group != last:
            L.append("\\midrule")
        last = group
        K = agg([r["K"] for r in rs])
        L.append(f"{bench} & {PRETTY[bb]} & {','.join(str(r['seed']) for r in rs) or '--'} & "
                 f"{cell(K[0], K[1], False, 1)} & {cell(*agg([r['LA'] for r in rs])[:2])} & "
                 f"{cell(*agg([r['OP'] for r in rs])[:2])} & {cell(*agg([r['BWT'] for r in rs])[:2])} & "
                 f"{cell(*agg([r['head_OP'] for r in rs])[:2])} \\\\")
    L += ["\\bottomrule", "\\end{tabular}", "\\end{table*}", ""]
    return "\n".join(L)


ABL_BACKBONES = ["llama3_2_1b", "gemma_2b"]


def _ref_runs(runs, bb):
    d, c, b, h = FINAL[bb]
    return pick(runs, bb, "TRACE-8@5000", "eval", "v20260828", d, c, b, h) or \
        pick(runs, bb, "TRACE-8@5000", "eval", "cap_2", d, c, b, h)


def _paired_delta(rs, ref):
    """Mean over seeds present in both of (variant - reference) OP."""
    by = {r["seed"]: r for r in ref}
    d = [r["OP"] - by[r["seed"]]["OP"] for r in rs if r["seed"] in by and r["OP"] is not None and by[r["seed"]]["OP"] is not None]
    return (statistics.mean(d), len(d)) if d else (None, 0)


def table_ablation(runs, missing):
    hdr = " & ".join(f"\\multicolumn{{3}}{{c}}{{{PRETTY[bb]}}}" for bb in ABL_BACKBONES)
    L = ["\\begin{table*}[t]", "\\centering", "\\small", "\\setlength{\\tabcolsep}{4pt}",
         "\\caption{Component ablations, TRACE-8 validation split. One change at a time from CRAFT's final configuration. "
         "Mean over the seeds found (n in brackets); $\\Delta$OP is paired by seed against the CRAFT reference run. "
         "Differences below about 1.5 OP (3.5 when the number of groups differs) are within run-to-run noise.}",
         "\\label{tab:ablation}", "\\begin{tabular}{l l " + "ccr" * len(ABL_BACKBONES) + "}", "\\toprule",
         f" & & {hdr} \\\\",
         " & ".join(["", ""] + ["OP", "BWT", "$\\Delta$OP"] * len(ABL_BACKBONES)) + " \\\\", "\\midrule"]
    refs = {bb: _ref_runs(runs, bb) for bb in ABL_BACKBONES}
    for bb in ABL_BACKBONES:
        for s_ in (REQUIRED_SEEDS if bb == "llama3_2_1b" else [42]):
            if s_ not in [r["seed"] for r in refs[bb]]:
                tier = "P0" if (bb == "llama3_2_1b" and s_ == 42) else "P1"
                missing.append(f"{tier} ablation reference v20260828 (validation) | {bb} | seed {s_}")
    row = ["", "CRAFT (reference)"]
    for bb in ABL_BACKBONES:
        rs = refs[bb]
        op, bw = agg([r["OP"] for r in rs]), agg([r["BWT"] for r in rs])
        row += [cell(op[0]) + (f" ({op[2]})" if op[2] else ""), cell(bw[0]), "--"]
    L.append(" & ".join(row) + " \\\\")
    last = None
    for block, arm, label, varies in ABLATIONS:
        row = [block if block != last else "", label]
        sep = "\\midrule\n" if block != last else ""
        last = block
        for bb in ABL_BACKBONES:
            d, c, b, h = FINAL[bb]
            if arm == "cap3":
                rs = pick(runs, bb, "TRACE-8@5000", "eval", "cap_3", d, 3, b, h) or \
                     pick(runs, bb, "TRACE-8@5000", "eval", "v20260828", d, 3, b, h) or \
                     pick(runs, bb, "TRACE-8@5000", "eval", "delta_0.7", d, 3, b, h)
            else:
                # fixed task-family groups have no capacity (logged as 0), so do not filter on it
                rs = pick(runs, bb, "TRACE-8@5000", "eval", arm, None, None if arm == "private_fixed" else c, None, None)
            if not rs and (bb == "llama3_2_1b" or arm in GEMMA_ARMS):
                tier = ARM_TIER.get(arm, "P0") if bb == "llama3_2_1b" else "P1"
                missing.append(f"{tier} ablation {arm} (validation) | {bb} | seed 42")
            elif bb == "llama3_2_1b" and arm != "cap3" and 1234 not in [r["seed"] for r in rs]:
                missing.append(f"P1 ablation {arm} (validation) | {bb} | seed 1234")
            op, bw = agg([r["OP"] for r in rs]), agg([r["BWT"] for r in rs])
            dop, n = _paired_delta(rs, refs[bb])
            row += [cell(op[0]) + (f" ({op[2]})" if op[2] else ""), cell(bw[0]), ("%+.2f" % dop) if dop is not None else "--"]
        L.append(sep + " & ".join(row) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular}", "\\end{table*}", ""]
    return "\n".join(L)


SENS = [  # (label, list of (value, arm, filter overrides, tier of the run that fills it))
    ("Routing threshold $\\delta$", [(v, "v20260828" if v == 0.7 else f"delta_{v}", {"delta": v},
                                     "P1" if v in (0.3, 1.5) else "P0") for v in (0.3, 0.5, 0.7, 1.0, 1.5, 2.0)]),
    ("Group capacity $C$", [(v, "v20260828" if v == 2 else f"cap_{v}", {"cap": v}, "P1") for v in (2, 3)]),
    ("Anchor strength $\\beta$", [(v, "v20260828" if v == 0.3 else f"beta_{v}", {"beta": v}, "P0") for v in (0.1, 0.3, 1.0)]),
    ("Merge half-life $h$", [(v, "v20260828" if v == 0.6 else f"merge_h{v}", {"h": v},
                             "P0" if v in (0.6, 2.0) else "P1") for v in (0.2, 0.6, 2.0, 5.0)]
     + [("overwrite", "merge_overwrite", {"h": None}, "P0")]),
    ("Intervention rank $r$", [(v, "v20260828", {"rank": v}, "P1") for v in (4, 8, 16)]),
    ("Positions per stream", [(v, "v20260828" if v == 15 else f"pos_{v}", {"positions": f"f{v}+l{v}"}, "P2") for v in (9, 15, 21)]),
    ("Warm-up steps $S_{wu}$", [(v, "v20260828" if v == 100 else f"wu_{v}", {"warmup": v}, "P2") for v in (50, 100, 400)]),
]


def table_sensitivity(runs, missing):
    bb = "llama3_2_1b"
    d, c, b, h = FINAL[bb]
    L = ["\\begin{table}[t]", "\\centering", "\\small", "\\setlength{\\tabcolsep}{5pt}",
         f"\\caption{{Hyperparameter sensitivity, Llama-3.2-1B-Instruct, TRACE-8 validation split. One value changes at a time "
         f"from the final configuration ($\\delta={d}$, $C={c}$, $\\beta={b}$, $h={h}$, $r=8$, 15 positions, 100 warm-up steps); "
         "mean over the seeds found (n). Final values marked $\\star$.}",
         "\\label{tab:sensitivity}", "\\begin{tabular}{l l c c c}", "\\toprule",
         "Hyperparameter & Value & $K$ & OP $\\uparrow$ & BWT $\\uparrow$ \\\\", "\\midrule"]
    for label, values in SENS:
        first = True
        for v, arm, over, tier in values:
            kw = dict(delta=d, cap=c, beta=b, h=h, rank=DEF_RANK, positions=DEF_POS, warmup=DEF_WU, lr="final")
            kw.update(over)
            if arm == "merge_overwrite":
                kw["h"] = None
            is_final = (arm == "v20260828" and all(
                (k in ("delta", "cap", "beta", "h") and close(kv, FINAL[bb][("delta", "cap", "beta", "h").index(k)])) or
                (k == "rank" and kv == DEF_RANK) or (k == "positions" and kv == DEF_POS) or
                (k == "warmup" and kv == DEF_WU) or (k == "lr" and close(kv, FINAL_LR[bb]))
                for k, kv in over.items()))
            rs = pick(runs, bb, "TRACE-8@5000", "eval", arm, **kw)
            if not rs and is_final:             # the earlier capacity-2 reference run
                rs = pick(runs, bb, "TRACE-8@5000", "eval", "cap_2", d, c, b, h)
            if not rs and "cap" in over:          # earlier capacity runs carry other arm names
                for alt in ("v20260828", "delta_0.7"):
                    rs = rs or pick(runs, bb, "TRACE-8@5000", "eval", alt, **kw)
            if not rs:
                missing.append(f"{tier} sensitivity {label.split('$')[0].strip()} = {v} (validation) | {bb} | seed 42")
            K = agg([r["K"] for r in rs])[0]
            op, bw = agg([r["OP"] for r in rs]), agg([r["BWT"] for r in rs])
            vtxt = (("%.0e" % v if v < 0.01 else f"{v:g}") if isinstance(v, float) else str(v)) + (" $\\star$" if is_final else "")
            L.append(f"{label if first else ''} & {vtxt} & {('%.1f' % K) if K else '--'} & "
                     f"{cell(op[0]) + (f' ({op[2]})' if op[2] else '')} & {cell(bw[0])} \\\\")
            first = False
        L.append("\\midrule")
    L[-1] = "\\bottomrule"
    L += ["\\end{tabular}", "\\end{table}", ""]
    return "\n".join(L)


def table_13b(runs, missing):
    rs = final_runs(runs, "llama2_13b")
    if not rs:
        return "% Llama-2-13B table: no runs (extra_13b is beyond the budget)\n"
    (op, bwt) = craft_stats(rs, head=True)
    cols = list(L2_13B) + ["CRAFT"]
    ops = [v[0] for v in L2_13B.values()] + [op[0]]
    bws = [v[1] for v in L2_13B.values()] + [bwt[0]]
    bo, bb_ = max(x for x in ops if x is not None), max(x for x in bws if x is not None)
    L = ["\\begin{table*}[t]", "\\centering", "\\small",
         f"\\caption{{Llama-2-13B-Chat, TRACE-8 test split. Baselines from \\citet{{qian2025treelora}}, Table 13. "
         f"CRAFT seeds: {','.join(str(r['seed']) for r in rs) or 'none'} (task-agnostic).}}",
         "\\label{tab:13b}", "\\begin{tabular}{l " + "c" * len(cols) + "}", "\\toprule",
         " & " + " & ".join(cols) + " \\\\", "\\midrule",
         "OP $\\uparrow$ & " + " & ".join(cell(x, None, close(x, bo)) for x in ops) + " \\\\",
         "BWT $\\uparrow$ & " + " & ".join(cell(x, None, close(x, bb_)) for x in bws) + " \\\\",
         "\\bottomrule", "\\end{tabular}", "\\end{table*}", ""]
    return "\n".join(L)


def kl_analysis(tasks_csv, pairs_out):
    if not tasks_csv:
        return ""
    rows = [r for r in csv.DictReader(open(tasks_csv))
            if r.get("status") == "finished" and r.get("method") == "current"
            and r.get("decision") == "JOIN" and r.get("terminal_kl") not in (None, "")
            and r.get("in_group_drop_caused") not in (None, "")]
    with open(pairs_out, "w", newline="") as fo:
        w = csv.writer(fo)
        w.writerow(["backbone", "stream", "seed", "task", "terminal_kl", "in_group_drop_caused"])
        for r in rows:
            w.writerow([r["backbone"], r["stream"], r["seed"], r["task"], r["terminal_kl"], r["in_group_drop_caused"]])
    x = [float(r["terminal_kl"]) for r in rows]
    y = [float(r["in_group_drop_caused"]) for r in rows]

    def pearson(a, b):
        if len(a) < 3:
            return None
        ma, mb = statistics.mean(a), statistics.mean(b)
        sa = math.sqrt(sum((u - ma) ** 2 for u in a)); sb = math.sqrt(sum((v - mb) ** 2 for v in b))
        return sum((u - ma) * (v - mb) for u, v in zip(a, b)) / (sa * sb) if sa and sb else None

    def ranks(a):
        o = sorted(range(len(a)), key=lambda i: a[i]); rk = [0.0] * len(a); i = 0
        while i < len(o):
            j = i
            while j + 1 < len(o) and a[o[j + 1]] == a[o[i]]:
                j += 1
            for k in range(i, j + 1):
                rk[o[k]] = (i + j) / 2 + 1
            i = j + 1
        return rk

    pr = pearson(x, y)
    sr = pearson(ranks(x), ranks(y)) if len(x) >= 3 else None
    return ("% KL vs in-group forgetting (joining tasks only; needs runs with --per_task_bwd_eval)\n"
            f"% n = {len(x)} joining tasks, Pearson r = {pr if pr is None else round(pr, 3)}, "
            f"Spearman rho = {sr if sr is None else round(sr, 3)}; pairs in {pairs_out}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--tasks", default=None)
    ap.add_argument("--out", default="paper_tables.tex")
    ap.add_argument("--baselines", action="store_true", help="also write the tables that compare with Qian et al.")
    a = ap.parse_args()
    runs = load(a.runs)
    pending = [r for r in csv.DictReader(open(a.runs)) if r.get("status") == "trained, eval pending"]
    missing = []
    # the comparison tables are always built, so the missing-run list stays complete
    compare = [table_main(runs, missing), table_recent(runs),
             table_stream(runs, missing, "TRACE-15@500", "llama3_2_1b", TRACE15, "trace15",
                          "Fifteen-task stream (TRACE-8 + seven classification tasks) on Llama-3.2-1B-Instruct, test split. "
                          "Baselines from \\citet{qian2025treelora}.", REQUIRED_SEEDS),
             table_orders(runs, missing),
             table_stream(runs, missing, "Math-LLM@5000", "llama2_7b", MATHLLM, "mathllm",
                          "Math-LLM (ScienceQA, NumGLUE-cm, NumGLUE-ds) on Llama-2-7B-Chat, test split. "
                          "Baselines from \\citet{qian2025treelora}, mean$\\pm$std over three runs.", REQUIRED_SEEDS),
             table_13b(runs, missing)]
    parts = ["% Generated by make_tables.py -- needs \\usepackage{booktabs,multirow,graphicx}", ""] + \
            (compare if a.baselines else [table_craft(runs)]) + [
             table_sharing(runs, missing), table_ablation(runs, missing), table_sensitivity(runs, missing),
             kl_analysis(a.tasks, a.out + ".kl_pairs.csv") if a.tasks else ""]
    open(a.out, "w").write("\n".join(parts))
    missing = sorted(set(missing), key=lambda m: (m[:2], m))
    open(a.out + ".missing.txt", "w").write("\n".join(missing) + ("\n" if missing else ""))
    print(f"Wrote {a.out}  ({len(runs)} finished current-method runs used)")
    for tier in ("P0", "P1", "P2"):
        t = [m for m in missing if m.startswith(tier)]
        print(f"Missing {tier}: {len(t)}")
        for m in t:
            print("  - " + m)
    print(f"Full list: {a.out}.missing.txt")
    if pending:
        print(f"\n{len(pending)} trained run(s) are waiting for evaluation; they fill cells above once evaluated:")
        print("  bash scripts/eval_later.sh --pending")


if __name__ == "__main__":
    main()