#!/usr/bin/env python3
"""
kl_scale.py -- KL scale s of a backbone relative to Llama-3.2-1B, and the
beta and h that follow from it (beta = 0.3 / s, h = 0.6 * s).

It reads, from each job log, the terminal KL of every task that joined a
group ("Save-time KL_final on train split = ..."). That KL is measured on
the training split, so no evaluation data is used.

  s = median over tasks that joined a group in both runs of
      KL(backbone) / KL(Llama-1B)

Usage
  python scripts/kl_scale.py --ref  <Llama-1B job log .txt> \
                             --logs <new backbone job log(s) .txt>
Several --logs (e.g. seeds) are averaged per task first.
"""
import argparse, re, statistics
from collections import defaultdict


def joined_task_kl(path):
    kl, cur = {}, None
    for line in open(path, errors="ignore"):
        m = re.search(r"\[Phase 1 complete\] kl_routing for '([^']+)'", line)
        if m:
            cur = m.group(1)
            continue
        m = re.search(r"Save-time KL_final on \w+ split = ([0-9.]+)", line)
        if m and cur:
            kl[cur] = float(m.group(1))
    return kl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", nargs="+", required=True, help="Llama-1B job log(s)")
    ap.add_argument("--logs", nargs="+", required=True, help="job log(s) of the new backbone")
    ap.add_argument("--beta_ref", type=float, default=0.3)
    ap.add_argument("--h_ref", type=float, default=0.6)
    a = ap.parse_args()

    def pooled(paths):
        acc = defaultdict(list)
        for p in paths:
            for t, v in joined_task_kl(p).items():
                acc[t].append(v)
        return {t: statistics.mean(v) for t, v in acc.items()}

    ref, new = pooled(a.ref), pooled(a.logs)
    common = sorted(set(ref) & set(new))
    if not common:
        raise SystemExit("No task joined a group in both runs. Run the new backbone once more "
                         "(any seed); its KL values are needed.")
    print(f"{'task':<14} {'KL Llama-1B':>12} {'KL new':>10} {'ratio':>7}")
    ratios = []
    for t in common:
        r = new[t] / ref[t]
        ratios.append(r)
        print(f"{t:<14} {ref[t]:>12.3f} {new[t]:>10.3f} {r:>7.2f}")
    s = statistics.median(ratios)
    print(f"\nKL scale s = {s:.2f}  (median over {len(common)} task(s))")
    print(f"beta = {a.beta_ref} / s = {a.beta_ref / s:.2f}     h = {a.h_ref} x s = {a.h_ref * s:.2f}")
    if 0.8 <= s <= 1.2:
        print("s is close to 1: keep beta 0.3 and h 0.6.")
    if len(common) < 2:
        print("Only one task in common: treat s as rough.")


if __name__ == "__main__":
    main()
