
import argparse
import json
import os
from math import sqrt
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Iterable

import numpy as np

import matplotlib
matplotlib.use("Agg")       
import matplotlib.pyplot as plt


def _load_results(run_dir_or_file: str) -> Tuple[dict, Path]:
    """Accept either a directory containing cl_results.json or the file itself."""
    p = Path(run_dir_or_file)
    if p.is_dir():
        results_path = p / "cl_results.json"
    else:
        results_path = p
    if not results_path.exists():
        raise FileNotFoundError(f"cl_results.json not found at {results_path}")
    with open(results_path, "r") as f:
        results = json.load(f)
    for k in ("perf_matrix", "kl_per_task", "task_to_group"):
        if k not in results:
            raise KeyError(
                f"cl_results.json is missing '{k}'.  "
                f"Did the training run finish, and were the E4/E7 logging "
                f"edits applied to finetune_cl?"
            )
    return results, results_path.parent


def _bwt_pairs(perf_matrix: List[List[Optional[float]]],
               kl_per_task: List[dict],
               task_to_group: Dict[str, object],
               tasks: List[str]) -> Iterable[dict]:

    n = len(tasks)
    for t in range(1, n):
        kl_drift = (kl_per_task[t].get("raw_integrated_kl")
                    or kl_per_task[t].get("raw_terminal_kl")
                    or kl_per_task[t].get("raw_mean_kl")
                    or kl_per_task[t].get("kl_running_mean_final"))
        is_new_t = bool(kl_per_task[t].get("is_new_group", False))
        task_t = tasks[t]
        group_t = task_to_group.get(task_t)
        for k in range(t):
            if perf_matrix[t][k] is None or perf_matrix[k][k] is None:
                continue
            task_k = tasks[k]
            group_k = task_to_group.get(task_k)
            yield {
                "t_idx":         t,
                "k_idx":         k,
                "task_t":        task_t,
                "task_k":        task_k,
                "kl_drift_t":    kl_drift,
                "bwt_tk":        float(perf_matrix[t][k] - perf_matrix[k][k]),
                "group_t":       group_t,
                "group_k":       group_k,
                "is_new_group_t": is_new_t,
                "same_group":    (group_t is not None and group_t == group_k),
            }

def _pearson(x: List[float], y: List[float]) -> Tuple[float, float, float]:
    n = len(x)
    if n < 3:
        return float("nan"), float("nan"), float("nan")
    xa = np.array(x, dtype=np.float64)
    ya = np.array(y, dtype=np.float64)
    mx, my = xa.mean(), ya.mean()
    num = ((xa - mx) * (ya - my)).sum()
    den = sqrt(((xa - mx) ** 2).sum() * ((ya - my) ** 2).sum())
    if den < 1e-12:
        return float("nan"), float("nan"), float(n - 2)
    r = float(num / den)
    if abs(r) >= 1.0 - 1e-12:
        return r, float("inf") if r > 0 else float("-inf"), float(n - 2)
    t = r * sqrt((n - 2) / (1.0 - r * r))
    return r, float(t), float(n - 2)


def _summary_stats(values: List[float]) -> Dict[str, Optional[float]]:
    if not values:
        return {"n": 0, "mean": None, "std": None,
                "median": None, "min": None, "max": None}
    v = np.array(values, dtype=np.float64)
    return {
        "n":      int(v.size),
        "mean":   float(v.mean()),
        "std":    float(v.std(ddof=0)),
        "median": float(np.median(v)),
        "min":    float(v.min()),
        "max":    float(v.max()),
    }

def e4_analysis(results: dict, out_dir: Path) -> dict:
    perf_matrix   = results["perf_matrix"]
    kl_per_task   = results["kl_per_task"]
    task_to_group = results["task_to_group"]
    tasks         = [e["task_name"] for e in kl_per_task]

    rows = list(_bwt_pairs(perf_matrix, kl_per_task, task_to_group, tasks))

    csv_path = out_dir / "e4_kl_vs_bwt.csv"
    with open(csv_path, "w") as f:
        f.write("t_idx,k_idx,task_t,task_k,kl_drift_t,bwt_tk,"
                "group_t,group_k,is_new_group_t,same_group\n")
        for r in rows:
            kl = "" if r["kl_drift_t"] is None else f"{r['kl_drift_t']:.6f}"
            f.write(
                f"{r['t_idx']},{r['k_idx']},{r['task_t']},{r['task_k']},"
                f"{kl},{r['bwt_tk']:.6f},"
                f"{r['group_t']},{r['group_k']},"
                f"{r['is_new_group_t']},{r['same_group']}\n"
            )
    pts = [(r["kl_drift_t"], r["bwt_tk"], r["same_group"])
           for r in rows if r["kl_drift_t"] is not None]
    n_excluded = len(rows) - len(pts)

    if not pts:
        return {
            "n_points":         0,
            "n_excluded_no_kl": n_excluded,
            "pearson_r":        None,
            "csv_path":         str(csv_path),
            "png_path":         None,
            "note": ("No (t, k) pair has a recorded KL drift.  Either no task "
                     "was trained under reg_mode in {kl, rkl}, or every task "
                     "was assigned to a new group (no anchor → no KL controller)."),
        }

    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    same = [p[2] for p in pts]
    r, t_stat, df = _pearson(xs, ys)

    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    same_x = [x for x, s in zip(xs, same) if s]
    same_y = [y for y, s in zip(ys, same) if s]
    diff_x = [x for x, s in zip(xs, same) if not s]
    diff_y = [y for y, s in zip(ys, same) if not s]
    if diff_x:
        ax.scatter(diff_x, diff_y, alpha=0.55, s=42, marker="o",
                   color="#888888", edgecolors="none",
                   label=f"between-group  (n={len(diff_x)})")
    if same_x:
        ax.scatter(same_x, same_y, alpha=0.85, s=52, marker="^",
                   color="#d62728", edgecolors="none",
                   label=f"within-group  (n={len(same_x)})")
    ax.axhline(0, ls="--", lw=0.8, color="#555")
    ax.set_xlabel("KL drift during training of task $t$  (raw, integrated over steps)")
    ax.set_ylabel(r"BWT$[t,k]$  =  acc$_t(k)$ − acc$_k(k)$")
    title = (f"E4: KL drift vs backward transfer   "
             f"n={len(xs)},  Pearson r = {r:+.3f}")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=9, frameon=False)
    ax.grid(True, lw=0.3, alpha=0.4)
    fig.tight_layout()
    png_path = out_dir / "e4_kl_vs_bwt.png"
    fig.savefig(png_path, dpi=150)
    plt.close(fig)

    return {
        "n_points":            len(xs),
        "n_excluded_no_kl":    n_excluded,
        "n_within_group_pts":  len(same_x),
        "n_between_group_pts": len(diff_x),
        "pearson_r":           r,
        "t_stat":              t_stat,
        "df":                  df,
        "csv_path":            str(csv_path),
        "png_path":            str(png_path),
    }

def e7_analysis(results: dict, out_dir: Path) -> dict:
    perf_matrix   = results["perf_matrix"]
    kl_per_task   = results["kl_per_task"]
    task_to_group = results["task_to_group"]
    tasks         = [e["task_name"] for e in kl_per_task]

    within: List[dict] = []
    between: List[dict] = []

    for r in _bwt_pairs(perf_matrix, kl_per_task, task_to_group, tasks):
        gt, gk = r["group_t"], r["group_k"]
        if gt is None or gk is None:
            continue
        if r["same_group"]:
            within.append({"t": r["task_t"], "k": r["task_k"],
                           "group": gt, "bwt": r["bwt_tk"]})
        else:
            between.append({"t": r["task_t"], "k": r["task_k"],
                            "g_t": gt, "g_k": gk, "bwt": r["bwt_tk"]})

    table = {
        "n_groups": len(set(task_to_group.values())) if task_to_group else 0,
        "n_tasks":  len(tasks),
        "within_group": {
            "stats": _summary_stats([d["bwt"] for d in within]),
            "pairs": within,
        },
        "between_group": {
            "stats": _summary_stats([d["bwt"] for d in between]),
            "pairs": between,
        },
    }

    json_path = out_dir / "e7_bwt_decomposition.json"
    with open(json_path, "w") as f:
        json.dump(table, f, indent=2)

    w = table["within_group"]["stats"]
    b = table["between_group"]["stats"]
    if (w["n"] > 0) or (b["n"] > 0):
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        names = ["between-group", "within-group"]
        means = [b["mean"] if b["mean"] is not None else 0.0,
                 w["mean"] if w["mean"] is not None else 0.0]
        stds  = [b["std"]  if b["std"]  is not None else 0.0,
                 w["std"]  if w["std"]  is not None else 0.0]
        ns    = [b["n"], w["n"]]
        colors = ["#888888", "#d62728"]
        bars = ax.bar(names, means, yerr=stds, capsize=5,
                      color=colors, edgecolor="none", alpha=0.85)
        for bar, n in zip(bars, ns):
            ax.text(bar.get_x() + bar.get_width() / 2.0,
                    bar.get_height(),
                    f"  n={n}", ha="center",
                    va="bottom" if bar.get_height() >= 0 else "top",
                    fontsize=9, color="#333")
        ax.axhline(0, ls="--", lw=0.8, color="#555")
        ax.set_ylabel("Mean BWT  (acc_t(k) − acc_k(k))")
        ax.set_title("E7: BWT decomposition")
        ax.grid(True, axis="y", lw=0.3, alpha=0.4)
        fig.tight_layout()
        bar_path = out_dir / "e7_bwt_decomposition.png"
        fig.savefig(bar_path, dpi=150)
        plt.close(fig)
        table["png_path"] = str(bar_path)

    table["json_path"] = str(json_path)
    return table


def _format_summary(e4: dict, e7: dict, results: dict) -> str:
    n_tasks = len(results["kl_per_task"])
    L = []
    L.append("=" * 72)
    L.append("  CRAFT-CL post-hoc analysis  (E4 + E7)")
    L.append("=" * 72)
    L.append("")
    L.append(f"  Tasks:              {n_tasks}")
    L.append(f"  Groups created:     {e7['n_groups']}")
    L.append("")

    L.append("─── E4: KL drift vs backward-transfer ─────────────────────────")
    if e4["n_points"] == 0:
        L.append(f"  No usable pairs.  {e4.get('note', '')}")
    else:
        L.append(f"  Pairs plotted:           {e4['n_points']}")
        L.append(f"  Pairs excluded (no KL):  {e4['n_excluded_no_kl']}")
        L.append(f"    within-group:          {e4['n_within_group_pts']}")
        L.append(f"    between-group:         {e4['n_between_group_pts']}")
        r = e4["pearson_r"]
        L.append(f"  Pearson r:               {r:+.4f}   (t={e4['t_stat']:+.3f}, df={e4['df']:.0f})")
        if r is not None:
            if r < -0.5:
                interp = "strongly negative — KL drift predicts forgetting (claim supported)"
            elif r < -0.2:
                interp = "weakly negative — direction matches claim, magnitude small"
            elif r < 0.2:
                interp = "near zero — KL drift does NOT predict forgetting"
            else:
                interp = "positive — claim contradicted by data"
            L.append(f"  Interpretation:          {interp}")
        L.append(f"  Plot:  {e4['png_path']}")
        L.append(f"  CSV:   {e4['csv_path']}")
    L.append("")

    L.append("─── E7: BWT decomposition (within-group vs between-group) ────")
    w = e7["within_group"]["stats"]
    b = e7["between_group"]["stats"]
    if w["n"] == 0 and b["n"] == 0:
        L.append("  No (t,k) pairs with defined group labels for both tasks.")
    else:
        if b["n"] > 0:
            L.append(f"  between-group  (n={b['n']:>3d}): "
                     f"mean BWT = {b['mean']:+.4f}  ± {b['std']:.4f}    "
                     f"(median={b['median']:+.4f}, range=[{b['min']:+.4f}, {b['max']:+.4f}])")
        else:
            L.append("  between-group  (n=  0): —")
        if w["n"] > 0:
            L.append(f"  within-group   (n={w['n']:>3d}): "
                     f"mean BWT = {w['mean']:+.4f}  ± {w['std']:.4f}    "
                     f"(median={w['median']:+.4f}, range=[{w['min']:+.4f}, {w['max']:+.4f}])")
        else:
            L.append("  within-group   (n=  0): —")
        L.append("")
        L.append("  Interpretation:")
        L.append("    between-group BWT should be ≈ 0 (groups don't share W).")
        L.append("    A non-zero between-group mean indicates either eval")
        L.append("    nondeterminism or a contamination bug in intervention")
        L.append("    state loading at evaluation time.")
        L.append("    within-group BWT measures overwrite-induced interference,")
        L.append("    i.e. forgetting introduced when a later in-group task")
        L.append("    overwrites the shared cluster state that earlier")
        L.append("    in-group tasks contributed to.  This is the controlled")
        L.append("    drift bounded by the KL-anchored fine-tuning step.")
        L.append(f"  JSON:  {e7['json_path']}")
        if "png_path" in e7:
            L.append(f"  Plot:  {e7['png_path']}")
    L.append("")
    return "\n".join(L)


def write_summary_txt(out_dir: Path, e4: dict, e7: dict, results: dict) -> str:
    text = _format_summary(e4, e7, results)
    txt_path = out_dir / "analysis_summary.txt"
    with open(txt_path, "w") as f:
        f.write(text)
    print(text)
    return str(txt_path)


def run_analysis(run_dir_or_file: str,
                 out_dir: Optional[str] = None) -> dict:
    """Read cl_results.json from `run_dir_or_file` (a directory or the JSON
    path), produce E4 + E7 outputs in `out_dir` (default: same directory),
    return a dict summarising the results."""
    results, parent_dir = _load_results(run_dir_or_file)
    out = Path(out_dir) if out_dir else parent_dir
    out.mkdir(parents=True, exist_ok=True)
    e4 = e4_analysis(results, out)
    e7 = e7_analysis(results, out)
    summary_path = write_summary_txt(out, e4, e7, results)
    return {"e4": e4, "e7": e7, "summary_txt": summary_path,
            "out_dir": str(out)}


def main():
    p = argparse.ArgumentParser(
        description=("Post-hoc analysis of CRAFT-CL runs: "
                     "E4 (KL drift vs BWT) and E7 (within/between-group BWT)."))
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--run_dir", type=str,help="Directory containing cl_results.json")
    src.add_argument("--input", type=str,help="Path to cl_results.json directly")
    p.add_argument("--out_dir", type=str, default=None,help="Where to write outputs (default: alongside input).")
    args = p.parse_args()
    src_path = args.run_dir if args.run_dir else args.input
    run_analysis(src_path, args.out_dir)

if __name__ == "__main__":
    main()
