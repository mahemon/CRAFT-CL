#!/usr/bin/env python3
"""
analyze_eviction.py — eviction demonstration or deletion (C3).

Eviction firing is NOT reliably serialized to cl_results.json (checked:
routing_log / kl_per_task do not carry a stable "evicted" field across
all code paths). The ground truth is the stdout log, which the
KLRazorController prints unconditionally at both trigger sites:

  "[KL] EVICT at first-epoch boundary (step N): first-epoch mean KL
   = X > absolute threshold Y. Task will retrain as own new group."

  "[KL] EVICT at step N: rolling mean (last W) = X > threshold Y
   (severity=Zx, steps-since-epoch1=S). Task will ..."

and, whether or not eviction fired, the routing decision line is always
printed by _print_routing_debug / the kl_routing phase log, letting this
script also confirm the forced merge actually happened as intended
(ScienceQA JOINing FOMC's group) before asking whether eviction rescued
it.

Usage:
  python analyze_eviction.py --logs no_evict=LOG1.txt evict_default=LOG2.txt \
      evict_sens=LOG3.txt evict_vsens=LOG4.txt
  (LOG* are the .txt stdout files under explore/Eviction_demonstration/logs/,
   or a directory containing exactly one .txt file to auto-select.)
"""

import argparse
import glob
import os
import re
import sys


EVICT_ABS_RE = re.compile(
    r"\[KL\] EVICT at first-epoch boundary \(step (\d+)\): "
    r"first-epoch mean KL = (\d+\.\d+) > absolute threshold (\d+\.\d+)")
EVICT_SEV_RE = re.compile(
    r"\[KL\] EVICT at step (\d+): rolling mean \(last \d+\) = (\d+\.\d+) "
    r"> threshold (\d+\.\d+)\s+\(severity=(\d+\.\d+)")
FIRST_EPOCH_RE = re.compile(
    r"\[KL\] First-epoch done at step (\d+): KL mean=(\d+\.\d+) "
    r"std=(\d+\.\d+)")
JOIN_RE = re.compile(
    r"\[kl_routing\] closest=G(\d+)\s+distance=(\d+\.\d+).*?→\s+"
    r"(JOIN G(\d+)|NEW GROUP)")
TASK_STEP_RE = re.compile(r"CL step \d+/\d+:\s+(\S+)")


def find_log(path):
    if os.path.isfile(path):
        return path
    if os.path.isdir(path):
        hits = sorted(glob.glob(os.path.join(path, "**", "*.txt"),
                                recursive=True))
        hits = [h for h in hits if not h.endswith(".err.txt")]
        if not hits:
            sys.exit(f"[error] no .txt log found under {path}")
        if len(hits) > 1:
            print(f"[warn] {len(hits)} logs under {path}; using latest: "
                 f"{hits[-1]}")
        return hits[-1]
    sys.exit(f"[error] not found: {path}")


def parse_log(path):
    with open(path, errors="replace") as f:
        text = f.read()

    joins = JOIN_RE.findall(text)
    first_epochs = FIRST_EPOCH_RE.findall(text)
    evict_abs = EVICT_ABS_RE.findall(text)
    evict_sev = EVICT_SEV_RE.findall(text)

    # Which task each routing decision belongs to: walk lines in order,
    # tracking the most recent "CL step k/n: TASK" header.
    events = []
    current_task = None
    for line in text.splitlines():
        m = TASK_STEP_RE.search(line)
        if m:
            current_task = m.group(1)
            continue
        if "[kl_routing]" in line and ("JOIN" in line or "NEW GROUP" in line):
            events.append(("routing", current_task, line.strip()))
        if "[KL] EVICT" in line:
            events.append(("evict", current_task, line.strip()))
        if "[KL] First-epoch done" in line:
            events.append(("first_epoch", current_task, line.strip()))

    return {
        "path": path,
        "joins": joins,
        "first_epochs": first_epochs,
        "evict_abs": evict_abs,
        "evict_sev": evict_sev,
        "n_evictions": len(evict_abs) + len(evict_sev),
        "events": events,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", nargs="+", required=True,
                    help="arm=path.txt (or arm=dir) pairs")
    args = ap.parse_args()

    arms = {}
    for spec in args.logs:
        tag, path = spec.split("=", 1)
        p = find_log(path)
        arms[tag] = parse_log(p)
        print(f"[load] {tag}: {p}")

    print(f"\n{'='*72}\n  FORCED-MERGE CONFIRMATION (did ScienceQA actually "
          f"JOIN FOMC's group?)")
    for tag, a in arms.items():
        print(f"\n  [{tag}]")
        if not a["events"]:
            print(f"    no routing/eviction lines found -- check the log "
                  f"path / job actually ran to completion")
            continue
        for kind, task, line in a["events"]:
            tag_str = {"routing": "ROUTE ", "evict": "EVICT ",
                      "first_epoch": "EPOCH1"}[kind]
            print(f"    [{tag_str}] {task or '?':<12} {line}")

    print(f"\n{'='*72}\n  EVICTION SUMMARY")
    print(f"    {'arm':<16}{'n_evictions':>13}{'first_epoch_mean_KL':>22}")
    for tag, a in arms.items():
        fe_str = (f"{a['first_epochs'][-1][1]}"
                  if a["first_epochs"] else "--")
        print(f"    {tag:<16}{a['n_evictions']:>13}{fe_str:>22}")

    print(f"\n  Verdict guide:")
    print(f"    - no_evict must show the JOIN decision with eviction "
          f"absent by construction (sanity check on the forcing setup).")
    print(f"    - If ANY evict_* arm shows n_evictions > 0: eviction "
          f"works under intentionally forced mis-routing. Keep it in "
          f"C3, cite the specific (severity, abs_threshold) that fired "
          f"as the demonstration, and report whether the post-eviction "
          f"outcome improved over no_evict on ScienceQA's own accuracy.")
    print(f"    - If evict_vsens (severity=1.05, abs=0.1 -- maximally "
          f"sensitive short of trivial) shows n_evictions == 0: the "
          f"training-time KL trajectory never diverges enough to "
          f"trigger eviction even when routing is deliberately wrong. "
          f"This is the deletion-worthy finding -- remove eviction from "
          f"the contributions list and Eq. 5, and note in the method "
          f"section that the trained intervention adapts to accommodate "
          f"a forced merge rather than triggering reassignment.")


if __name__ == "__main__":
    main()
