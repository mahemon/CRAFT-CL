# Merge-rule ablation (ido5-W3 / Q2)

Answers: "Why is direct overwrite the right merging rule? What happens with
interpolation or maintaining multiple group prototypes?" plus the recency-bias
probe. Everything except `--merge_rule` is pinned to the R3 config of the main
runs (ratio routing thr=0.7 floor=0.01, KL anchor β=0.3 normalized, eviction
enabled, all-groupwise, per-task backward eval, oracle inference, seed 42).

## Contents

```
Merge_rule_ablation/
├── train.py                            # patched copy (5 surgical patches, listed below)
├── merge_rules.py                      # NEW: 4 merge rules + diagonal Fisher estimator
├── submit_merge_ablation_llama15.sh    # Llama-1B, 15-task stream (recency probe primary)
├── submit_merge_ablation_gemma_trace8.sh  # Gemma-2B-it, TRACE-8 (small-group check)
├── analyze_merge_ablation.py           # comparison table + recency probe
├── logs/                               # SLURM stdout/err (created by scripts)
└── outputs/                            # run outputs incl. cl_results.json
```

## The four rules

| rule      | group state after task t joins                  | tests |
|-----------|--------------------------------------------------|-------|
| overwrite | trained state (paper default, unchanged)         | baseline |
| ema       | (1-α)·old + α·trained, α=0.5                     | parameter-space interpolation vs KL-only restraint |
| fisher    | per-param (F_old·old + F_new·new)/(F_old+F_new), Fisher accumulated across merges | importance-weighted interpolation |
| prototype | uniform mean over one stored prototype per member task | task-balanced merge; directly removes last-writer-wins |

Design notes: all rules interpolate the raw (unconstrained) state-dict
tensors, matching the existing overwrite transfer path — the Stiefel
parametrization on R is reapplied on the next forward, exactly as for
overwrite. Fisher = mean over 8 batches of grad² of the task CE loss w.r.t.
intervention parameters, estimated on current-task data only (no CL-discipline
violation). Eviction under `prototype` also removes the task's prototype and
recomputes the mean.

## train.py patches (vs the uploaded original)

1. Import block: `from merge_rules import ...`
2. argparse: `--merge_rule {overwrite,ema,fisher,prototype}`,
   `--merge_ema_alpha`, `--merge_fisher_batches`
3. Memory config: `memory.merge_rule`, `memory.merge_ema_alpha` set next to
   `memory.all_groupwise`
4. `WeightDistanceMemory.save()`: `fisher_new` kwarg; new-group branch seeds
   prototypes/Fisher; join branch dispatches on the rule; eviction cleans up
   prototypes; join log label shows the active rule
5. Call site in `finetune_cl`: Fisher estimated right before `memory.save()`
   when rule=fisher

`overwrite` reproduces the original code path bit-for-bit (the else-branch is
the original assignment).

## Run

```bash
# on a Delta login node, from ROOT_DIR (/projects/bfwa/mhossen1/CRAFT-CL):
export HF_TOKEN=hf_...          # rotate before submission as planned
bash Merge_rule_ablation/submit_merge_ablation_llama15.sh       # 4 jobs, 15-task
bash Merge_rule_ablation/submit_merge_ablation_gemma_trace8.sh  # 4 jobs, TRACE-8

squeue -u $USER
```

Flat submission, one GPU per job, 8 jobs total. Scripts export
`PYTHONPATH=$ROOT_DIR` so `utils/` and `agnostic_router` resolve while the
entry point lives in this folder.

**Verify before submitting (2 items):**
- `BENCH_DIR_REL` in the llama script points at `LLM-CL-Benchmark_500`; confirm
  that dir contains all 15 tasks and matches whatever the paper's 15-task runs
  used, else edit the variable.
- The 15-task epoch schedule uses the canonical `5,4,7,5,4,5,5,7` + seven 5s.
  Your uploaded TRACE-8 Delta llama script used `4,4,5,5,4,5,5,5` — the
  ablation must match whichever schedule produced the paper's 15-task table.

## Analyze

```bash
cd $ROOT_DIR/Merge_rule_ablation
python analyze_merge_ablation.py \
  --runs overwrite=outputs/CRAFT_CL_merge_overwrite_llama32_1b_15task_<ts> \
         ema=outputs/CRAFT_CL_merge_ema_llama32_1b_15task_<ts> \
         fisher=outputs/CRAFT_CL_merge_fisher_llama32_1b_15task_<ts> \
         prototype=outputs/CRAFT_CL_merge_prototype_llama32_1b_15task_<ts> \
  --csv merge_ablation_llama15.csv
```

## Analysis rules (what goes in the paper, and how to read the numbers)

1. **Headline table**: OP / BWT / LA per rule per model. Overwrite is
   vindicated iff it is within noise of the best rule on OP while not worst on
   BWT. If prototype wins BWT by a wide margin at similar OP, the honest
   conclusion is that overwrite is *adequate under the KL anchor*, not optimal
   — write it that way.
2. **Recency probe** (the paper-ready evidence for or against overwrite): for
   each group with ≥3 members on the 15-task stream, drop (a_tt − a_Tt) vs
   join index. Report the pooled recency-bias ρ per rule. Expected pattern if
   ido5-W3's concern is real: overwrite shows ρ clearly > 0 (early joiners
   forget more — yahoo 61→42 is exactly this); prototype should flatten ρ
   toward 0. If overwrite's ρ is already near 0, the KL anchor alone controls
   recency bias and the ablation *closes* the objection.
3. **Routing invariance check**: confirm `n_groups` and `task_to_group` are
   identical across the four runs. Routing happens before merging within each
   task, but from task 2 onward the *group states being routed against* differ
   by rule, so partitions can drift. If they drift, per-group comparisons are
   confounded — report OP/BWT/LA only, state the partition difference, and do
   not compare groups across rules.
4. **Positive transfer**: check whether amazon→Py150-style negative drops
   (backward *gains*) survive under prototype averaging. Uniform averaging can
   dilute them; if it does, that is an argument *for* overwrite worth one
   sentence.
5. **Cost note for the paper**: fisher adds ~8 forward+backward batches per
   task; prototype adds P× storage per group (one state per member task) —
   which erodes the O(K·L·h·r) footprint argument of Table 21. If prototype
   wins, that storage cost must be disclosed next to the win.
6. **Seeds**: these scripts run seed 42 only (arm-selection pass). Whichever
   rule enters the paper table needs the standard 3-seed treatment before the
   number is citable.
