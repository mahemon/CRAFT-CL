#!/usr/bin/env python3
"""
litar_summary.py - one line per CRAFT job log that ran LITAR, plus its routing decisions.

  python scripts/litar_summary.py                      # logs changed since yesterday 00:00
  python scripts/litar_summary.py --since 2026-10-14   # logs changed since that date
  python scripts/litar_summary.py --details            # also every LITAR decision per task

Reads only the job logs (output_log/logs_*/*.txt). A run that is still going shows
status "running"; a run whose LITAR row has not been reached yet shows no LITAR numbers.
"""
import argparse
import datetime as dt
import glob
import os
import re

ROW = re.compile(r"^\s+(oracle|litar|learned_head|group_oracle|\w+)\s+(-?\d\.\d{4})\s+(-?[+-]?\d\.\d{4})\s*$")
DEC = re.compile(r"\[litar\] (\S+) -> (\S+) \(G(\d+), oracle G(\S+)\)\s+conf=([\d.]+)\s+"
                 r"per-prompt group acc=(\S+)\s+m=(\d+)\s+K=(\d+)")
FALLBACK = re.compile(r"\[litar\] (cannot route|this memory has no LITAR state)")


def parse(path):
    r = {"log": os.path.basename(path), "dir": os.path.basename(os.path.dirname(path)),
         "start": None, "end": None, "exit": None, "cap": None, "rows": {}, "dec": [],
         "fallback": 0, "warn": 0}
    in_table = False
    with open(path, errors="replace") as f:
        for line in f:
            if line.startswith("  Started  :"):
                r["start"] = line.split(":", 1)[1].strip()
            elif "Finished at" in line and "exit code" in line:
                m = re.search(r"Finished at (.*) with exit code (\d+)", line)
                if m:
                    r["end"], r["exit"] = m.group(1).strip(), int(m.group(2))
            elif line.strip().startswith("group_cap  :"):
                r["cap"] = line.split(":", 1)[1].strip()
            elif "Final row by router" in line:
                in_table = True
            elif in_table:
                m = ROW.match(line)
                if m:
                    r["rows"][m.group(1)] = (float(m.group(2)), float(m.group(3)))
                elif line.strip() and not line.strip().startswith("Router"):
                    in_table = False
            m = DEC.search(line)
            if m:
                r["dec"].append(dict(task=m.group(1), pred=m.group(2), g=m.group(3),
                                     og=m.group(4), conf=float(m.group(5)),
                                     pp=m.group(6), m=int(m.group(7)), K=int(m.group(8))))
            if FALLBACK.search(line):
                r["fallback"] += 1
            if "[litar] WARNING" in line:
                r["warn"] += 1
    return r


def hours(a, b):
    fmt = "%a %b %d %I:%M:%S %p %Z %Y"
    try:
        x = dt.datetime.strptime(a, fmt); y = dt.datetime.strptime(b, fmt)
        return f"{(y - x).total_seconds() / 3600:.1f}"
    except Exception:
        return "?"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="output_log")
    p.add_argument("--since", default=None, help="YYYY-MM-DD (default: yesterday)")
    p.add_argument("--details", action="store_true")
    a = p.parse_args()
    since = (dt.datetime.strptime(a.since, "%Y-%m-%d") if a.since
             else dt.datetime.combine(dt.date.today() - dt.timedelta(days=1), dt.time()))
    logs = [f for f in glob.glob(os.path.join(a.root, "logs_*", "*.txt"))
            if dt.datetime.fromtimestamp(os.path.getmtime(f)) >= since]
    runs = [parse(f) for f in sorted(logs, key=os.path.getmtime)]
    runs = [r for r in runs if r["dec"] or "litar" in r["rows"] or r["fallback"]
            or "litar" in open(os.path.join(a.root, r["dir"], r["log"]), errors="replace").read(200000)]
    if not runs:
        print(f"no log since {since:%Y-%m-%d} mentions LITAR"); return
    print(f"LITAR runs, logs changed since {since:%Y-%m-%d %H:%M}\n")
    hdr = f"{'status':<8} {'h':>5} {'cap':>5}  {'oracle OP':>9} {'litar OP':>9} {'head OP':>8}  {'task hits':>9} {'group hits':>10} {'prompt acc':>10}  log"
    print(hdr); print("-" * len(hdr))
    for r in runs:
        st = ("running" if r["exit"] is None else "ok" if r["exit"] == 0 else f"exit{r['exit']}")
        h = hours(r["start"], r["end"]) if r["start"] and r["end"] else "-"
        op = lambda k: f"{r['rows'][k][0]:.4f}" if k in r["rows"] else "-"
        d = r["dec"]
        th = f"{sum(x['task'] == x['pred'] for x in d)}/{len(d)}" if d else "-"
        gh = f"{sum(x['g'] == x['og'] for x in d)}/{len(d)}" if d else "-"
        pa = (f"{sum(float(x['pp']) * x['m'] for x in d if x['pp'] != 'None') / sum(x['m'] for x in d):.3f}"
              if d else "-")
        extra = (f"  [{r['fallback']} oracle fallback(s)]" if r["fallback"] else "") + \
                (f"  [{r['warn']} LITAR warning(s)]" if r["warn"] else "")
        print(f"{st:<8} {h:>5} {str(r['cap'] or '-'):>5}  {op('oracle'):>9} {op('litar'):>9} {op('learned_head'):>8}"
              f"  {th:>9} {gh:>10} {pa:>10}  {r['dir']}/{r['log']}{extra}")
        if a.details and d:
            for x in d:
                flag = "" if x["task"] == x["pred"] else "   <-- wrong task"
                print(f"           {x['task']:<15} -> {x['pred']:<15} G{x['g']} (oracle G{x['og']})"
                      f"  conf={x['conf']:.3f}  per-prompt={x['pp']}  K={x['K']}{flag}")
    print("\nh = run time in hours (Started -> Finished). prompt acc = share of single prompts whose own "
          "decision picked the oracle's group. With the old litar_router.py, the LITAR OP of sampled "
          "tasks differs from oracle by sampling noise even when every task is a hit.")


if __name__ == "__main__":
    main()
