#!/usr/bin/env python3
"""
analyze_beta_dose.py — the beta dose-response analysis (metareview
consensus item 3: does KL predict AND control forgetting?).

For each beta arm and each task t that JOINED an existing group:
  D_term_t : raw forward KL(anchor || live), token-mean, on t's own data,
             measured at save time (kl_per_task[t].kl_final_val_diagnostic
             in cl_results.json — logged by the unmodified pipeline).
  F_t      : immediate in-group forgetting caused by t = mean over prior
             same-group tasks j of (pm[t-1][j] - pm[t][j]); positive =
             forgetting, negative = positive backward transfer.

The claim "KL predicts and controls forgetting" decomposes into two
testable statements, and this script tests both:

  PREDICTS : pooled rank correlation of D_term vs F_t > 0, computed WITH
             negative-F_t points included (reported with and without, so
             the with-negatives number is the headline and the exclusion
             criticized by ido5 is never repeated).
  CONTROLS : as beta increases, per-arm mean D_term decreases
             monotonically, and mean F_t decreases with it
             (the dose-response of the knob).

Usage:
  python analyze_beta_dose.py \
      --runs 0.05=DIR 0.1=DIR 0.3=DIR 0.5=DIR 1.0=DIR \
      [--csv beta_dose.csv] [--latex beta_dose.tex] [--fig beta_dose.png]

Each DIR contains cl_results.json (searched one level deep). Requires
--per_task_bwd_eval runs. beta is parsed from the run tag.
"""

import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict


# ────────────────────────── stats (no scipy needed) ──────────────────────

def _ranks(v):
    n = len(v)
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


def pearson(x, y):
    n = len(x)
    if n < 3:
        return float("nan")
    mx, my = sum(x) / n, sum(y) / n
    num = sum((x[i] - mx) * (y[i] - my) for i in range(n))
    dx = math.sqrt(sum((v - mx) ** 2 for v in x))
    dy = math.sqrt(sum((v - my) ** 2 for v in y))
    return num / (dx * dy) if dx > 0 and dy > 0 else float("nan")


def spearman(x, y):
    if len(x) < 3:
        return float("nan")
    return pearson(_ranks(x), _ranks(y))


# ─────────────────────────────── loading ────────────────────────────────

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


def extract_points(beta, res):
    """One point per JOIN task: (beta, task, gid, D_term, F_t)."""
    order = list(res["diagonal"].keys())
    pm = res["perf_matrix"]
    t2g = res["task_to_group"]
    n = len(order)

    # terminal KL per task from kl_per_task; fall back to group kl_history
    kl_by_task = {}
    for e in res.get("kl_per_task", []):
        v = e.get("kl_final_val_diagnostic")
        if v is None:
            v = e.get("kl_running_mean_final")
        kl_by_task[e["task_name"]] = v

    # which tasks joined (vs opened) — from routing_log if present,
    # else infer: a task joined iff another task with the same gid
    # precedes it in stream order.
    joined = {}
    rlog = res.get("routing_log")
    if rlog:
        for e in rlog:
            joined[e["task"]] = not e.get("is_new_group", True)
    else:
        seen_gids = set()
        for t in order:
            g = t2g[t]
            joined[t] = g in seen_gids
            seen_gids.add(g)

    pts, skipped = [], []
    for idx, t in enumerate(order):
        if not joined.get(t, False):
            continue
        d_term = kl_by_task.get(t)
        if d_term is None or (isinstance(d_term, float)
                              and math.isnan(d_term)):
            skipped.append((t, "no terminal KL logged"))
            continue
        prior_same = [j for j in range(idx)
                      if t2g[order[j]] == t2g[t]
                      and pm[idx][j] is not None
                      and pm[idx - 1][j] is not None]
        if not prior_same:
            skipped.append((t, "no measurable prior in-group task"))
            continue
        f_t = sum(pm[idx - 1][j] - pm[idx][j] for j in prior_same) \
            / len(prior_same)
        pts.append({"beta": beta, "task": t, "gid": t2g[t],
                    "d_term": float(d_term), "f_t": round(f_t, 4),
                    "n_prior_ingroup": len(prior_same)})
    return pts, skipped, res


# ─────────────────────────────── reporting ──────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True,
                    help="beta=dir pairs, e.g. 0.3=outputs/..._beta0p3_...")
    ap.add_argument("--csv", default=None)
    ap.add_argument("--latex", default=None)
    ap.add_argument("--fig", default=None,
                    help="scatter figure path (needs matplotlib)")
    args = ap.parse_args()

    all_pts, per_beta_meta = [], []
    for spec in args.runs:
        btag, path = spec.split("=", 1)
        beta = float(btag)
        f = find_results(path)
        with open(f) as fh:
            res = json.load(fh)
        pts, skipped, res = extract_points(beta, res)
        print(f"[load] beta={beta}: {f}  ->  {len(pts)} join-task points"
              + (f"  (skipped: {skipped})" if skipped else ""))
        all_pts.extend(pts)
        per_beta_meta.append({
            "beta": beta, "n": len(pts),
            "op": res.get("overall_performance"),
            "bwt": res.get("backward_transfer"),
            "n_groups": res.get("n_groups"),
            "mean_dterm": (sum(p["d_term"] for p in pts) / len(pts)
                           if pts else float("nan")),
            "mean_ft": (sum(p["f_t"] for p in pts) / len(pts)
                        if pts else float("nan")),
        })

    if not all_pts:
        sys.exit("[error] no usable points across arms")

    # ── PREDICTS: pooled correlation, negatives included ──
    x = [p["d_term"] for p in all_pts]
    y = [p["f_t"] for p in all_pts]
    neg = [p for p in all_pts if p["f_t"] < 0]
    pos_only = [p for p in all_pts if p["f_t"] >= 0]
    print(f"\n{'='*72}\n  PREDICTS: terminal KL vs in-group forgetting "
          f"(n={len(all_pts)}, negatives INCLUDED)")
    print(f"    Pearson r  = {pearson(x, y):+.3f}")
    print(f"    Spearman ρ = {spearman(x, y):+.3f}")
    print(f"    negative-F_t points (positive transfer): {len(neg)}"
          + ("".join(f"\n      {p['task']} (beta={p['beta']}, G{p['gid']}): "
                     f"F_t={p['f_t']:+.2f}, D_term={p['d_term']:.4f}"
             for p in neg) if neg else ""))
    if neg and len(pos_only) >= 3:
        xp = [p["d_term"] for p in pos_only]
        yp = [p["f_t"] for p in pos_only]
        print(f"    [secondary] excluding negatives (n={len(pos_only)}): "
              f"Pearson {pearson(xp, yp):+.3f}, "
              f"Spearman {spearman(xp, yp):+.3f}")

    per_beta = defaultdict(list)
    for p in all_pts:
        per_beta[p["beta"]].append(p)
    print(f"\n  per-beta correlations (small n — read qualitatively):")
    for b in sorted(per_beta):
        pts = per_beta[b]
        xb = [p["d_term"] for p in pts]
        yb = [p["f_t"] for p in pts]
        print(f"    beta={b:<5} n={len(pts):>2}  "
              f"Spearman {spearman(xb, yb):+.3f}")

    # ── CONTROLS: monotonicity of the knob ──
    per_beta_meta.sort(key=lambda m: m["beta"])
    print(f"\n{'='*72}\n  CONTROLS: dose-response of beta")
    print(f"    {'beta':>6}{'n':>4}{'mean D_term':>13}{'mean F_t':>10}"
          f"{'OP':>8}{'BWT':>8}{'K':>4}")
    for m in per_beta_meta:
        op = f"{m['op']:.2f}" if m["op"] is not None else "--"
        bw = f"{m['bwt']:.2f}" if m["bwt"] is not None else "--"
        print(f"    {m['beta']:>6}{m['n']:>4}{m['mean_dterm']:>13.4f}"
              f"{m['mean_ft']:>10.3f}{op:>8}{bw:>8}"
              f"{m['n_groups'] if m['n_groups'] is not None else '--':>4}")
    betas = [m["beta"] for m in per_beta_meta]
    dterms = [m["mean_dterm"] for m in per_beta_meta]
    fts = [m["mean_ft"] for m in per_beta_meta]
    mono_d = all(dterms[i] >= dterms[i + 1] - 1e-9
                 for i in range(len(dterms) - 1))
    print(f"    mean D_term monotone nonincreasing in beta: "
          f"{'YES' if mono_d else 'NO'}")
    print(f"    Spearman(beta, mean D_term) = "
          f"{spearman(betas, dterms):+.3f}   "
          f"Spearman(beta, mean F_t) = {spearman(betas, fts):+.3f}")
    print(f"\n  verdict guide: PREDICTS holds if pooled Spearman > 0 with "
          f"negatives included;\n  CONTROLS holds if Spearman(beta, mean "
          f"D_term) is clearly negative and mean F_t moves with it.")

    # routing-partition drift warning
    ks = {m["beta"]: m["n_groups"] for m in per_beta_meta
          if m["n_groups"] is not None}
    if len(set(ks.values())) > 1:
        print(f"\n  [warn] arms discovered different partitions {ks}: "
              f"beta affects warm-up-anchored routing indirectly. Compare "
              f"pooled correlations, not per-group detail, across arms.")

    if args.csv:
        with open(args.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(all_pts[0].keys()))
            w.writeheader()
            w.writerows(all_pts)
        print(f"\n[csv] {len(all_pts)} points -> {args.csv}")

    if args.latex:
        lines = [
            r"% auto-generated by analyze_beta_dose.py",
            r"\begin{table}[htbp]\centering\small",
            r"\caption{Dose-response of the KL anchor coefficient "
            r"$\beta$ on the 15-task stream. D$^{\mathrm{term}}$ is the "
            r"terminal forward KL to the anchor on the new task; $F$ is "
            r"immediate in-group forgetting. Negative-$F$ points are "
            r"included in all correlations.}",
            r"\label{tab:beta_dose}",
            r"\begin{tabular}{r r r r r r}\toprule",
            r"$\beta$ & $n$ & mean D$^{\mathrm{term}}$ & mean $F$ & "
            r"OP & BWT \\ \midrule",
        ]
        for m in per_beta_meta:
            lines.append(
                f"{m['beta']} & {m['n']} & {m['mean_dterm']:.3f} & "
                f"{m['mean_ft']:.2f} & "
                f"{m['op'] if m['op'] is not None else '--'} & "
                f"{m['bwt'] if m['bwt'] is not None else '--'} \\\\")
        lines += [r"\bottomrule\end{tabular}\end{table}"]
        with open(args.latex, "w") as fh:
            fh.write("\n".join(lines) + "\n")
        print(f"[latex] table -> {args.latex}")

    if args.fig:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("[fig] matplotlib unavailable; skipped "
                  "(CSV has all points)")
            return
        fig, ax = plt.subplots(figsize=(5.2, 3.6))
        cmap = plt.get_cmap("viridis")
        bs = sorted({p["beta"] for p in all_pts})
        for i, b in enumerate(bs):
            pts = [p for p in all_pts if p["beta"] == b]
            ax.scatter([p["d_term"] for p in pts],
                       [p["f_t"] for p in pts],
                       color=cmap(i / max(len(bs) - 1, 1)),
                       label=f"β={b}", s=28, edgecolors="k",
                       linewidths=0.4)
        ax.axhline(0.0, lw=0.6, color="grey")
        ax.set_xlabel(r"terminal KL to anchor  $D^{\mathrm{term}}_t$")
        ax.set_ylabel(r"in-group forgetting  $F_t$")
        ax.legend(fontsize=7, frameon=False)
        fig.tight_layout()
        fig.savefig(args.fig, dpi=200)
        print(f"[fig] scatter -> {args.fig}")


if __name__ == "__main__":
    main()
