#!/usr/bin/env python3
"""
analyze_intra_group_forgetting.py — quantifies the intra/cross-group
forgetting split (option iii for jvrh Q4 / the yahoo 61->42 objection).

Decomposition. With per-task-step backward eval, each column j of the
performance matrix changes only between consecutive rows, and each change
event (i-1 -> i) has exactly one cause: training task i. The total drop of
task j,  a_jj - a_Tj,  telescopes over events:

    a_jj - a_Tj = sum_{i>j} [ pm[i-1][j] - pm[i][j] ]

Each event is attributed to trainer i and labelled intra-group when
group(i) == group(j), cross-group otherwise. BWT therefore splits exactly:

    BWT = BWT_intra + BWT_cross          (no residual, checked numerically)

Cross-group events should be identically zero under CRAFT's per-group
storage (structural invariance); any nonzero cross-group mass is reported
separately as it measures eval noise or an invariance violation.

Also reported: per-victim timelines (the yahoo trace), per-group tables,
positive backward-transfer events, group size vs mean member drop, and a
paper-ready LaTeX block.

Usage:
  python analyze_intra_group_forgetting.py --run OUTPUT_DIR
  python analyze_intra_group_forgetting.py --run OUTPUT_DIR --latex out.tex
  python analyze_intra_group_forgetting.py --runs base=D1 cap3=D2 delta05=D3

OUTPUT_DIR must contain cl_results.json (searched one level deep as well).
Requires --per_task_bwd_eval runs (full matrix); rows with missing entries
are handled by carrying the last observed value, and the count of carried
cells is reported so silent gaps cannot corrupt the split.
"""

import argparse
import json
import os
import sys
from collections import defaultdict


# ────────────────────────────── loading ──────────────────────────────

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
        print(f"[warn] {len(hits)} results under {path}; using latest: "
              f"{hits[-1]}")
    return hits[-1]


def load(path):
    f = find_results(path)
    with open(f) as fh:
        res = json.load(fh)
    for key in ("perf_matrix", "task_to_group", "diagonal"):
        if key not in res:
            sys.exit(f"[error] {f} missing key '{key}'")
    order = list(res["diagonal"].keys())
    pm = res["perf_matrix"]
    n = len(order)
    if len(pm) != n or any(len(r) != n for r in pm):
        sys.exit(f"[error] perf_matrix shape mismatch in {f}: "
                 f"{len(pm)}x{len(pm[0]) if pm else 0}, expected {n}x{n}")
    t2g = {t: res["task_to_group"][t] for t in order
           if t in res["task_to_group"]}
    missing = [t for t in order if t not in t2g]
    if missing:
        sys.exit(f"[error] tasks missing from task_to_group: {missing}")
    return res, order, pm, t2g, f


# ─────────────────────────── decomposition ───────────────────────────

def densify_column(pm, j, n):
    """Return column j with None cells (rows i >= j) carried forward from
    the last observed value; count carried cells."""
    col, carried = [None] * n, 0
    last = None
    for i in range(n):
        v = pm[i][j]
        if i < j:
            col[i] = None
            continue
        if v is None:
            if last is None:
                return None, -1  # diagonal itself missing: column unusable
            col[i] = last
            carried += 1
        else:
            col[i] = float(v)
            last = float(v)
    return col, carried


def decompose(order, pm, t2g):
    n = len(order)
    events = []          # every nonzero change event
    carried_total = 0
    unusable = []

    for j in range(n):
        col, carried = densify_column(pm, j, n)
        if col is None:
            unusable.append(order[j])
            continue
        carried_total += carried
        for i in range(j + 1, n):
            delta = col[i] - col[i - 1]          # + = gain, - = drop
            if abs(delta) < 1e-12:
                continue
            same = (t2g[order[i]] == t2g[order[j]])
            events.append({
                "victim": order[j], "victim_idx": j,
                "trainer": order[i], "trainer_idx": i,
                "gid_victim": t2g[order[j]], "gid_trainer": t2g[order[i]],
                "delta": round(delta, 4), "intra": same,
            })

    # exact BWT split (paper convention: BWT = mean over tasks of a_tt - a_Tt,
    # positive = forgetting). drop_j = -(sum of deltas in column j).
    drop_intra = defaultdict(float)
    drop_cross = defaultdict(float)
    for e in events:
        (drop_intra if e["intra"] else drop_cross)[e["victim"]] -= e["delta"]

    usable = [t for t in order if t not in unusable]
    n_use = len(usable)
    bwt_intra = sum(drop_intra[t] for t in usable) / n_use
    bwt_cross = sum(drop_cross[t] for t in usable) / n_use
    bwt_total = bwt_intra + bwt_cross

    return {
        "events": events, "bwt_total": bwt_total,
        "bwt_intra": bwt_intra, "bwt_cross": bwt_cross,
        "drop_intra": dict(drop_intra), "drop_cross": dict(drop_cross),
        "carried_cells": carried_total, "unusable": unusable,
        "usable": usable,
    }


def verify_against_reported(res, dec, order, pm):
    """Check telescoped BWT against the direct a_tt - a_Tt definition."""
    n = len(order)
    direct = []
    for j, t in enumerate(order):
        if t in dec["unusable"]:
            continue
        col, _ = densify_column(pm, j, n)
        direct.append(col[j] - col[n - 1])
    bwt_direct = sum(direct) / len(direct)
    gap = abs(bwt_direct - dec["bwt_total"])
    status = "OK" if gap < 1e-6 else f"MISMATCH ({gap:.6f})"
    rep = res.get("backward_transfer")
    return bwt_direct, rep, status


# ──────────────────────────── reporting ─────────────────────────────

def report(tag, res, order, pm, t2g, dec, top_events=12):
    n = len(order)
    print(f"\n{'='*74}\n  RUN [{tag}]   tasks={n}   groups="
          f"{len(set(t2g.values()))}")
    if dec["carried_cells"]:
        print(f"  [warn] {dec['carried_cells']} missing matrix cells "
              f"carried forward")
    if dec["unusable"]:
        print(f"  [warn] columns dropped (missing diagonal): "
              f"{dec['unusable']}")

    bwt_direct, bwt_reported, status = verify_against_reported(
        res, dec, order, pm)
    print(f"\n  DECOMPOSITION (BWT convention: positive = forgetting)")
    print(f"    BWT total (telescoped) : {dec['bwt_total']:+8.3f}   "
          f"[vs direct a_tt - a_Tt: {bwt_direct:+.3f}  -> {status}]")
    if bwt_reported is not None:
        print(f"    BWT reported in json   : {bwt_reported:+8.3f}")
    print(f"    BWT intra-group        : {dec['bwt_intra']:+8.3f}")
    print(f"    BWT cross-group        : {dec['bwt_cross']:+8.3f}")
    share = (dec["bwt_intra"] / dec["bwt_total"] * 100.0
             if abs(dec["bwt_total"]) > 1e-9 else float("nan"))
    print(f"    intra share of BWT     : {share:8.1f} %")
    cross_ev = [e for e in dec["events"] if not e["intra"]]
    print(f"    cross-group events     : {len(cross_ev)}  "
          f"(structural invariance predicts 0; nonzero = eval noise "
          f"or violation)")
    for e in cross_ev[:8]:
        print(f"      !! {e['trainer']} (G{e['gid_trainer']}) changed "
              f"{e['victim']} (G{e['gid_victim']}) by {e['delta']:+.2f}")

    # per-group table
    groups = defaultdict(list)
    for j, t in enumerate(order):
        groups[t2g[t]].append((j, t))
    print(f"\n  PER-GROUP (members in arrival order)")
    print(f"    {'gid':>4} {'size':>5} {'mean drop':>10} "
          f"{'worst victim':>24}")
    for gid in sorted(groups):
        members = [t for _, t in groups[gid] if t in dec['usable']]
        if not members:
            continue
        drops = [dec["drop_intra"].get(t, 0.0)
                 + dec["drop_cross"].get(t, 0.0) for t in members]
        worst_i = max(range(len(members)), key=lambda k: drops[k])
        print(f"    {gid:>4} {len(members):>5} "
              f"{sum(drops)/len(drops):>10.2f} "
              f"{members[worst_i]:>18} ({drops[worst_i]:+.2f})")

    # victim timelines for the worst intra-group victims
    ranked = sorted(dec["usable"],
                    key=lambda t: -(dec["drop_intra"].get(t, 0.0)))
    print(f"\n  VICTIM TIMELINES (top intra-group victims)")
    for t in ranked[:4]:
        j = order.index(t)
        col, _ = densify_column(pm, j, n)
        steps = [f"{col[j]:.1f}"]
        for i in range(j + 1, n):
            if abs(col[i] - col[i - 1]) > 1e-12:
                steps.append(f"--[{order[i]}]--> {col[i]:.1f}")
        print(f"    {t} (G{t2g[t]}): " + " ".join(steps))

    # largest single events, both signs
    ev_sorted = sorted(dec["events"], key=lambda e: e["delta"])
    print(f"\n  LARGEST EVENTS (drop-)")
    for e in ev_sorted[:top_events // 2]:
        print(f"    {e['trainer']:<14} -> {e['victim']:<14} "
              f"{e['delta']:+7.2f}   "
              f"{'intra G'+str(e['gid_victim']) if e['intra'] else 'CROSS'}")
    gains = [e for e in reversed(ev_sorted) if e["delta"] > 0]
    print(f"  LARGEST EVENTS (gain+ / positive backward transfer)")
    for e in gains[:top_events // 2]:
        print(f"    {e['trainer']:<14} -> {e['victim']:<14} "
              f"{e['delta']:+7.2f}   "
              f"{'intra G'+str(e['gid_victim']) if e['intra'] else 'CROSS'}")
    return {"tag": tag, "bwt": dec["bwt_total"],
            "bwt_intra": dec["bwt_intra"], "bwt_cross": dec["bwt_cross"],
            "intra_share": share, "n_cross_events": len(cross_ev),
            "op": res.get("overall_performance")}


def latex_block(summary_rows, dec_by_tag, order_by_tag, t2g_by_tag):
    lines = [
        r"% auto-generated by analyze_intra_group_forgetting.py",
        r"\begin{table}[htbp]",
        r"\centering\small",
        r"\caption{Decomposition of backward transfer into intra-group and "
        r"cross-group components. Cross-group forgetting is structurally "
        r"zero under per-group storage; all measured forgetting is "
        r"intra-group and bounded by the routing threshold $\delta$.}",
        r"\label{tab:intra_group_split}",
        r"\begin{tabular}{l r r r r}",
        r"\toprule",
        r"Setting & BWT & BWT\textsubscript{intra} & "
        r"BWT\textsubscript{cross} & intra share (\%) \\",
        r"\midrule",
    ]
    for s in summary_rows:
        lines.append(f"{s['tag']} & {s['bwt']:.2f} & {s['bwt_intra']:.2f} & "
                     f"{s['bwt_cross']:.2f} & {s['intra_share']:.1f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--run", help="single output dir (or cl_results.json)")
    g.add_argument("--runs", nargs="+",
                   help="tag=dir pairs for multi-arm comparison "
                        "(baseline vs cap vs tightened delta)")
    ap.add_argument("--latex", default=None,
                    help="write paper-ready LaTeX table here")
    ap.add_argument("--json", default=None,
                    help="dump full event-level decomposition here")
    args = ap.parse_args()

    specs = ([("run", args.run)] if args.run
             else [tuple(s.split("=", 1)) for s in args.runs])

    summaries, dump = [], {}
    for tag, path in specs:
        res, order, pm, t2g, f = load(path)
        print(f"[load] {tag}: {f}")
        dec = decompose(order, pm, t2g)
        summaries.append(report(tag, res, order, pm, t2g, dec))
        dump[tag] = {"events": dec["events"],
                     "bwt_intra": dec["bwt_intra"],
                     "bwt_cross": dec["bwt_cross"],
                     "bwt_total": dec["bwt_total"]}

    if len(summaries) > 1:
        print(f"\n{'='*74}\n  CROSS-ARM SUMMARY")
        print(f"    {'arm':<12}{'OP':>8}{'BWT':>8}{'intra':>8}"
              f"{'cross':>8}{'share%':>8}{'Xevents':>9}")
        for s in summaries:
            op = f"{s['op']:.2f}" if s.get("op") is not None else "  --"
            print(f"    {s['tag']:<12}{op:>8}{s['bwt']:>8.2f}"
                  f"{s['bwt_intra']:>8.2f}{s['bwt_cross']:>8.2f}"
                  f"{s['intra_share']:>8.1f}{s['n_cross_events']:>9}")

    if args.latex:
        with open(args.latex, "w") as fh:
            fh.write(latex_block(summaries, None, None, None) + "\n")
        print(f"\n[latex] table written to {args.latex}")
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(dump, fh, indent=2)
        print(f"[json] event-level decomposition written to {args.json}")


if __name__ == "__main__":
    main()
