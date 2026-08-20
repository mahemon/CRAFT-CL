# Intra-group forgetting (jvrh Q4: yahoo 61.0 → 42.0)

Implements all three response options:

- **(iii) quantify the split** — `analyze_intra_group_forgetting.py`, runs on
  any existing `cl_results.json`, **no GPU needed**. This is the primary
  deliverable and already produces paper-ready numbers from Table 17
  (see below).
- **(i) tighten routing** — δ ∈ {0.6, 0.5} arms in the SLURM script.
- **(ii) per-group capacity cap** — new `--group_capacity` flag in the
  patched `train.py`; cap ∈ {4, 3} arms in the SLURM script.

## Contents

```
Intra_group_forgetting/
├── train.py                            # patched: --group_capacity (3 surgical patches)
├── analyze_intra_group_forgetting.py   # option (iii): exact BWT decomposition
├── demo_paper_table17.py               # runs (iii) on the paper's Table 17 data
├── submit_intra_group_llama15.sh       # options (i)+(ii): 5 arms, 15-task, Llama-1B
├── demo_run/                           # demo output (cl_results.json, .tex, events.json)
├── logs/  outputs/                     # created by the SLURM script
```

## Option (iii) — results available right now

```bash
python demo_paper_table17.py
```

reconstructs `cl_results.json` from the paper's Table 17 + Table 8b and runs
the decomposition. Verified output:

```
BWT total       +2.607   (matches paper's reported 2.60; telescoping check OK)
BWT intra-group +2.607
BWT cross-group +0.000
intra share      100.0 %
cross-group events: 0
```

Per-group: G0 (7 members) mean drop 4.37, worst yahoo +19.0;
G1 mean 2.00 (FOMC +6.0); G2 0.00; G3 0.62.
Timelines: `yahoo 61.0 --[agnews]--> 45.0 --[yelp]--> 44.0 --[BoolQA]--> 42.0`.
Positive backward transfer events also extracted (amazon→Py150 +6.2,
BoolQA→dbpedia +5.0, QQP→ScienceQA +3.0, ...).

The decomposition is exact: with per-task backward eval, every column change
telescopes into events each attributable to exactly one trainer task, so
BWT = BWT_intra + BWT_cross with zero residual (checked numerically against
the direct a_tt − a_Tt definition). Cross-group events are reported
separately as an invariance audit — nonzero mass there would indicate eval
noise or a storage-isolation violation.

**Paper paragraph this licenses** (adapt to house style, no "we"):

> CRAFT does not eliminate forgetting; it localizes it. Because groups are
> stored independently, cross-group backward transfer is structurally zero,
> a property the evaluation matrix confirms exactly: decomposing BWT on the
> 15-task stream attributes 100% of the 2.61-point backward transfer to
> intra-group events and 0.00 points to cross-group events. The routing
> threshold δ is therefore an explicit knob on where interference may occur.
> The largest intra-group drops concentrate in the largest discovered group
> (G0, seven tasks; worst member −19.0), while the same sharing also
> produces positive backward transfer (+6.2 on Py150 from a later in-group
> task). Appendix table reports the full event-level attribution.

Run the analyzer on your real output dirs (not the reconstruction) before the
numbers go in the paper; Table 17 is single-seed.

## Options (i) and (ii) — the runnable arms

`train.py` patches (verified, `--group_capacity 0` reproduces the original
code path exactly):
1. argparse: `--group_capacity` (0 = unlimited, paper default)
2. `run_kl_routing_phase`: groups at capacity are excluded from routing
   candidates before the closest-group decision (the task can still join the
   second-closest eligible group); if no eligible group remains, a new group
   opens. Exclusions are logged and recorded in the routing debug dict
   (`capacity_excluded`).
3. Call site passes `args.group_capacity` through.

Under `ratio` normalization, excluded groups still participate in the
d_max denominator, so distances remain comparable to the no-cap run.

```bash
# on Delta, from ROOT_DIR:
export HF_TOKEN=hf_...
bash Intra_group_forgetting/submit_intra_group_llama15.sh          # all 5 arms
ARMS="cap3" bash Intra_group_forgetting/submit_intra_group_llama15.sh   # one arm
```

Arms (flat submission, one GPU each): `base` δ=0.7/no-cap,
`delta06`, `delta05` (option i), `cap4`, `cap3` (option ii).
Verify `BENCH_DIR_REL` and the 15-task epoch schedule before submitting
(same two items as the merge ablation).

## Analyze the arms

```bash
python analyze_intra_group_forgetting.py \
  --runs base=outputs/CRAFT_CL_intragrp_base_llama32_1b_15task_<ts> \
         delta06=outputs/CRAFT_CL_intragrp_delta06_llama32_1b_15task_<ts> \
         delta05=outputs/CRAFT_CL_intragrp_delta05_llama32_1b_15task_<ts> \
         cap4=outputs/CRAFT_CL_intragrp_cap4_llama32_1b_15task_<ts> \
         cap3=outputs/CRAFT_CL_intragrp_cap3_llama32_1b_15task_<ts> \
  --latex intra_split_table.tex --json intra_events.json
```

## Analysis rules

1. **The (iii) claim stands on the base arm alone**: intra share ≈ 100%,
   cross events = 0. If any arm shows nonzero cross-group events, investigate
   before writing anything — under all-groupwise storage that indicates eval
   nondeterminism (check greedy decoding) or a load-path bug, not a method
   property.
2. **Interpret (i)/(ii) as a frontier, not a fix**: tightening δ or capping
   groups trades transfer for isolation. Compare arms on (OP, BWT, LA,
   n_groups, parameter count = n_groups × per-group cost). Expected shape:
   caps reduce the worst-victim drop (yahoo) and raise n_groups; if OP falls
   as caps tighten, that is the transfer being surrendered — report it,
   because it is the empirical argument for why the paper's operating point
   tolerates intra-group forgetting.
3. **Watch positive transfer under caps**: if cap3 splits G0 before BoolQA
   joins, the BoolQA→dbpedia +5.0 and BoolQA→C-STANCE +3.0 gains disappear.
   Gains lost to isolation belong in the same table as drops prevented.
4. **Routing shifts confound per-group comparisons**: arms discover
   different partitions by design. Compare stream-level aggregates across
   arms; compare per-group detail only within an arm.
5. **Eviction interaction**: capacity exclusions can push a task to open a
   new group even when its nearest group is compatible; check
   `capacity_excluded` in the routing log to count how often the cap, rather
   than the KL distance, made the decision.
6. **Seeds**: arms are seed-42 selection runs; the arm that enters the paper
   needs the 3-seed treatment.
