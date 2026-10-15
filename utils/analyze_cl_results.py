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


def _raw_kl(entry: dict) -> Optional[float]:
    """Raw (un-normalised) KL drift recorded while this task trained."""
    return (entry.get("raw_integrated_kl")
            or entry.get("raw_terminal_kl")
            or entry.get("raw_mean_kl")
            or entry.get("kl_running_mean_final"))


def _overwriters(tasks: List[str],
                 task_to_group: Dict[str, object],
                 k: int,
                 t: int) -> List[int]:
    """Indices j in (k, t] whose task was merged into task k's group.

    These are the tasks that overwrote the shared state task k depends on,
    and therefore the only tasks that can have damaged task k.  If this list
    is empty, task k's group was untouched between k and t, so BWT[t,k] must
    be exactly 0 — any deviation is an evaluation bug, not forgetting.
    """
    group_k = task_to_group.get(tasks[k])
    if group_k is None:
        return []
    return [j for j in range(k + 1, t + 1)
            if task_to_group.get(tasks[j]) == group_k]


def _bwt_pairs(perf_matrix: List[List[Optional[float]]],
               kl_per_task: List[dict],
               task_to_group: Dict[str, object],
               tasks: List[str]) -> Iterable[dict]:
    """Yield one record per (t, k) pair with t > k and both scores available.

    NOTE on `same_group`: a pair is "within-group" when at least one task
    trained after k was merged into k's group.  It is NOT "task t and task k
    happen to share a group".  With a diagonal+final-row perf_matrix the only
    available t is the last task, and the old definition therefore asked
    "does k share a group with the LAST task", which mislabels every task
    whose group was overwritten by some intermediate task.
    """
    n = len(tasks)
    for t in range(1, n):
        task_t = tasks[t]
        group_t = task_to_group.get(task_t)
        is_new_t = bool(kl_per_task[t].get("is_new_group", False))
        for k in range(t):
            if perf_matrix[t][k] is None or perf_matrix[k][k] is None:
                continue
            task_k = tasks[k]
            group_k = task_to_group.get(task_k)

            # Tasks that overwrote k's group at or before step t.
            ow = _overwriters(tasks, task_to_group, k, t)

            # KL drift attributable to THIS pair: the drift absorbed by k's
            # group after k finished.  Distinguish "no overwriter at all"
            # (a real zero) from "overwriters exist but their KL was never
            # recorded" (unknown -> excluded from E4).
            if not ow:
                kl_drift: Optional[float] = 0.0
            else:
                vals = [_raw_kl(kl_per_task[j]) for j in ow]
                vals = [v for v in vals if v is not None]
                kl_drift = float(sum(vals)) if vals else None

            yield {
                "t_idx":          t,
                "k_idx":          k,
                "task_t":         task_t,
                "task_k":         task_k,
                "kl_drift_t":     kl_drift,
                "bwt_tk":         float(perf_matrix[t][k] - perf_matrix[k][k]),
                "group_t":        group_t,
                "group_k":        group_k,
                "is_new_group_t": is_new_t,
                "n_overwriters":  len(ow),
                "overwriters":    ";".join(tasks[j] for j in ow),
                "same_group":     bool(ow),
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


def _matrix_coverage(perf_matrix: List[List[Optional[float]]]) -> Dict[str, int]:
    """How much of the lower triangle of perf_matrix is actually filled."""
    n = len(perf_matrix)
    possible = n * (n - 1) // 2
    filled = sum(1 for t in range(1, n) for k in range(t)
                 if perf_matrix[t][k] is not None
                 and perf_matrix[k][k] is not None)
    rows = sorted({t for t in range(1, n) for k in range(t)
                   if perf_matrix[t][k] is not None})
    return {"possible": possible, "filled": filled, "n_rows_evaluated": len(rows)}


def e4_analysis(results: dict, out_dir: Path) -> dict:
    perf_matrix   = results["perf_matrix"]
    kl_per_task   = results["kl_per_task"]
    task_to_group = results["task_to_group"]
    tasks         = [e["task_name"] for e in kl_per_task]

    rows = list(_bwt_pairs(perf_matrix, kl_per_task, task_to_group, tasks))

    csv_path = out_dir / "e4_kl_vs_bwt.csv"
    with open(csv_path, "w") as f:
        f.write("t_idx,k_idx,task_t,task_k,kl_drift_after_k,bwt_tk,"
                "group_t,group_k,is_new_group_t,n_overwriters,overwriters,"
                "same_group\n")
        for r in rows:
            kl = "" if r["kl_drift_t"] is None else f"{r['kl_drift_t']:.6f}"
            f.write(
                f"{r['t_idx']},{r['k_idx']},{r['task_t']},{r['task_k']},"
                f"{kl},{r['bwt_tk']:.6f},"
                f"{r['group_t']},{r['group_k']},"
                f"{r['is_new_group_t']},{r['n_overwriters']},"
                f"{r['overwriters']},{r['same_group']}\n"
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
            "coverage":         _matrix_coverage(perf_matrix),
            "note": ("No (t, k) pair has a usable KL drift.  Either no task "
                     "was trained under reg_mode in {kl, rkl}, or every task "
                     "was assigned to a new group (no anchor -> no KL controller)."),
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
    ax.set_xlabel("KL drift absorbed by task $k$'s group after $k$ finished  (raw)")
    ax.set_ylabel(r"BWT$[t,k]$  =  acc$_t(k)$ - acc$_k(k)$")
    r_txt = f"{r:+.3f}" if np.isfinite(r) else "undefined"
    ax.set_title(f"E4: KL drift vs backward transfer   "
                 f"n={len(xs)},  Pearson r = {r_txt}")
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
        "n_distinct_x":        int(len(set(np.round(xs, 9)))),
        "pearson_r":           r,
        "t_stat":              t_stat,
        "df":                  df,
        "csv_path":            str(csv_path),
        "png_path":            str(png_path),
        "coverage":            _matrix_coverage(perf_matrix),
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
                           "group": gk, "bwt": r["bwt_tk"],
                           "overwriters": r["overwriters"]})
        else:
            between.append({"t": r["task_t"], "k": r["task_k"],
                            "g_t": gt, "g_k": gk, "bwt": r["bwt_tk"]})

    # Sanity check: a task whose group was never overwritten must score
    # exactly what it scored on its own diagonal.  A non-zero value here is
    # the only signature of a genuine evaluation / state-loading bug.
    violations = [d for d in between if abs(d["bwt"]) > 1e-9]

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
        "isolation_violations": violations,
        "coverage": _matrix_coverage(perf_matrix),
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
        ax.set_ylabel("Mean BWT  (acc_t(k) - acc_k(k))")
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
    cov = e7.get("coverage", {})
    L = []
    L.append("=" * 72)
    L.append("  CRAFT-CL post-hoc analysis  (E4 + E7)")
    L.append("=" * 72)
    L.append("")
    L.append(f"  Tasks:              {n_tasks}")
    L.append(f"  Groups created:     {e7['n_groups']}")
    if cov:
        L.append(f"  BWT pairs usable:   {cov['filled']} of {cov['possible']} "
                 f"({cov['n_rows_evaluated']} backward row(s) evaluated)")
        if cov["filled"] < cov["possible"]:
            L.append("                      (perf_matrix is sparse - run full "
                     "backward eval for per-step attribution)")
    L.append("")

    L.append("--- E4: KL drift vs backward-transfer -------------------------")
    if e4["n_points"] == 0:
        L.append(f"  No usable pairs.  {e4.get('note', '')}")
    else:
        L.append(f"  Pairs plotted:           {e4['n_points']}")
        L.append(f"  Pairs excluded (no KL):  {e4['n_excluded_no_kl']}")
        L.append(f"    within-group:          {e4['n_within_group_pts']}")
        L.append(f"    between-group:         {e4['n_between_group_pts']}")
        L.append(f"  Distinct KL values:      {e4['n_distinct_x']}")
        r = e4["pearson_r"]
        if r is None or not np.isfinite(r):
            L.append("  Pearson r:               undefined")
            if e4["n_distinct_x"] <= 1:
                L.append("  Interpretation:          KL drift is constant across "
                         "pairs - correlation is not defined.")
            elif e4["n_points"] < 3:
                L.append("  Interpretation:          too few pairs (<3) for a "
                         "correlation.")
            else:
                L.append("  Interpretation:          BWT is constant across "
                         "pairs - correlation is not defined.")
        else:
            L.append(f"  Pearson r:               {r:+.4f}   "
                     f"(t={e4['t_stat']:+.3f}, df={e4['df']:.0f})")
            if r < -0.5:
                interp = "strongly negative - KL drift predicts forgetting (claim supported)"
            elif r < -0.2:
                interp = "weakly negative - direction matches claim, magnitude small"
            elif r < 0.2:
                interp = "near zero - KL drift does NOT predict forgetting"
            else:
                interp = "positive - claim contradicted by data"
            L.append(f"  Interpretation:          {interp}")
        L.append(f"  Plot:  {e4['png_path']}")
        L.append(f"  CSV:   {e4['csv_path']}")
    L.append("")

    L.append("--- E7: BWT decomposition (within-group vs between-group) ----")
    w = e7["within_group"]["stats"]
    b = e7["between_group"]["stats"]
    if w["n"] == 0 and b["n"] == 0:
        L.append("  No (t,k) pairs with defined group labels for both tasks.")
    else:
        if b["n"] > 0:
            L.append(f"  between-group  (n={b['n']:>3d}): "
                     f"mean BWT = {b['mean']:+.4f}  +/- {b['std']:.4f}    "
                     f"(median={b['median']:+.4f}, range=[{b['min']:+.4f}, {b['max']:+.4f}])")
        else:
            L.append("  between-group  (n=  0): -")
        if w["n"] > 0:
            L.append(f"  within-group   (n={w['n']:>3d}): "
                     f"mean BWT = {w['mean']:+.4f}  +/- {w['std']:.4f}    "
                     f"(median={w['median']:+.4f}, range=[{w['min']:+.4f}, {w['max']:+.4f}])")
        else:
            L.append("  within-group   (n=  0): -")
        L.append("")

        viol = e7.get("isolation_violations", [])
        if viol:
            L.append("  [WARN] group isolation violated:")
            for d in viol:
                L.append(f"    {d['k']}: no later task joined its group "
                         f"({d['g_k']}), but BWT = {d['bwt']:+.4f}")
            L.append("    -> eval nondeterminism or a contamination bug in "
                     "intervention state loading.")
        else:
            L.append("  Group isolation: OK - every task whose group was never "
                     "overwritten scored exactly its diagonal value.")
        L.append("")
        L.append("  Interpretation:")
        L.append("    A pair is within-group when at least one task trained after")
        L.append("    k was merged into k's group, i.e. k's shared state was")
        L.append("    overwritten.  Within-group BWT is the overwrite-induced")
        L.append("    interference the KL-anchored step is meant to bound.")
        L.append("    Between-group pairs share no weights, so their BWT is 0 by")
        L.append("    construction; it is a correctness check, not a result.")
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
    print("running from the run_analysis today Sept 17.")
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
    src.add_argument("--run_dir", type=str, help="Directory containing cl_results.json")
    src.add_argument("--input", type=str, help="Path to cl_results.json directly")
    p.add_argument("--out_dir", type=str, default=None, help="Where to write outputs (default: alongside input).")
    args = p.parse_args()
    src_path = args.run_dir if args.run_dir else args.input
    run_analysis(src_path, args.out_dir)


if __name__ == "__main__":
    main()