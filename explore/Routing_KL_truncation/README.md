# Routing KL truncation (Limitations: is saturation a truncation artifact?)

Answers: the routing distance restricts symmetric KL to the union of
each side's top-64 tokens. Two questions: does the discovered TRACE-8
partition depend on this truncation level, and is the documented
ScienceQA-A/ScienceQA-B saturation failure (two random halves of the
same data route to fresh clusters instead of merging) a consequence of
the truncation specifically, fixable by raising k?

## Cost design: `--routing_diagnostic_only`

Routing-only reruns should cost warm-up + probe forwards, not full
training. `train.py` gains a flag that does exactly that: run warm-up
and the routing decision for every task as normal, then **skip Phase 2
training and per-task evaluation entirely**, registering the warm-up
state itself as the group's stored representative so later tasks still
have something to route against. `cl_results.json` is still written,
but its `overall_performance`/`backward_transfer` are meaningless (no
task was ever trained) — only `task_to_group` / `routing_log` /
`n_groups` are meaningful, and the analyzer reads only those.

## A correctness finding that reshaped this study's scope

Read through `_kl_logits_distance` before designing the sweep, because
the plan called for a k ∈ {64, 256, 1024, full} range. **A literal
full-vocabulary arm is infeasible**: the union-support match step
(`match = union_idx.unsqueeze(-1) == top_idx_X.unsqueeze(-2)`) is
**O(k²) per token position** — its tensor shape is `[..., 2k, k]`. At
k=64 that's ~4K entries per token; at k=16384 it would be ~268M entries
per token, multiple GB of temporaries per probe batch. The sweep below
tops out at **k=512** (an 8× increase over the paper's default), which
is large enough to test truncation sensitivity without attempting
anything resembling full-vocabulary KL. **Do not describe the k512 arm
as "full vocabulary" in the paper** — call it a large-k arm and cite
this cost constraint if a reviewer asks why full-vocab wasn't tested.

## train.py patches (3 independent additions, on top of the earlier
routing-rule fixes; `--routing_diagnostic_only` unset and
`--kl_routing_capture_top_k 256` reproduce the original pipeline
exactly)

1. **`--routing_diagnostic_only`** (new flag): inserted right before
   "Phase 2: Full training" in the main task loop. When set, registers
   the just-computed warm-up state into `memory.groups[assigned_gid]`
   (new-group or join, same bookkeeping `memory.save()` would do),
   updates `memory.task_to_group`, appends to `routing_log`, and
   `continue`s to the next task — no trainer, no eval.
2. **`run_kl_routing_phase` now always returns the warm-up `iv_state`**
   as a 4th return value (`warm_iv_state`), on both its return paths
   (the "no existing groups yet" early exit and the main path). This
   required moving the warm-up call to happen *before* the
   `if not memory.groups` check (previously warm-up was skipped
   entirely for the very first task) — a deliberate, harmless behavior
   change scoped to this copy of `train.py` only, needed so
   `routing_diagnostic_only` has a state to register even for the task
   that opens the first group.
3. **`--kl_routing_capture_top_k`** (new flag, default 256, matching
   the previously-hardcoded default): controls the top-k size CAPTURED
   at probe-forward time, threaded through both `_probe_forward_logits`
   calls. `--kl_routing_top_k` (pre-existing) can only ever *subselect*
   from within what was captured — `_kl_logits_distance` silently
   clamps `top_k` to the captured size. Both flags are set to the same
   value in the sweep script so each arm fully uses what it captures.

## Contents

```
explore/Routing_KL_truncation/
├── train.py                              # patched (3 additions above)
├── make_scienceqa_split.py               # standalone data-prep, no GPU
├── submit_truncation_sweep_trace8.sh     # 6 arms, TRACE-8, diagnostic-only
├── submit_duplicate_unit_test.sh         # same 6 arms, ScienceQA-A/B split
├── analyze_truncation.py                 # auto-detects sweep vs dup-test mode
├── logs/  outputs/
```

## Run

**Step 1 — build the ScienceQA-A/B split** (once, no GPU needed):

```bash
python explore/Routing_KL_truncation/make_scienceqa_split.py \
  --benchmark_dir "$ROOT_DIR/dataset/TRACE_Benchmark/LLM-CL-Benchmark_5000" \
  --out_dir "$ROOT_DIR/dataset/TRACE_Benchmark/LLM-CL-Benchmark_ScienceQA_split" \
  --seed 42
```

**Step 2 — submit both sweeps:**

```bash
export HF_TOKEN=hf_...
bash explore/Routing_KL_truncation/submit_truncation_sweep_trace8.sh   # 6 jobs, ~3h cap each
bash explore/Routing_KL_truncation/submit_duplicate_unit_test.sh       # 6 jobs, ~1h cap each
```

12 jobs total, all diagnostic-only (warm-up + probes, no training) —
cheap relative to every other study in `explore/`. Single arm:
`ARMS="k512_p16" bash explore/Routing_KL_truncation/submit_truncation_sweep_trace8.sh`.

Arms (`name : capture_top_k : kl_routing_top_k : probe_size`):

| arm | capture | top_k | probe_size |
|---|---|---|---|
| k64_p16 (reference, paper default) | 64 | 64 | 16 |
| k128_p16 | 128 | 128 | 16 |
| k256_p16 | 256 | 256 | 16 |
| k512_p16 (largest feasible) | 512 | 512 | 16 |
| k64_p8 | 64 | 64 | 8 |
| k64_p64 | 64 | 64 | 64 |

## Analyze

```bash
# sweep (TRACE-8 partition invariance)
python explore/Routing_KL_truncation/analyze_truncation.py \
  --runs k64_p16=outputs/CRAFT_CL_trunc_k64_p16_llama32_1b_trace8_<ts> \
         k128_p16=outputs/CRAFT_CL_trunc_k128_p16_llama32_1b_trace8_<ts> \
         k256_p16=outputs/CRAFT_CL_trunc_k256_p16_llama32_1b_trace8_<ts> \
         k512_p16=outputs/CRAFT_CL_trunc_k512_p16_llama32_1b_trace8_<ts> \
         k64_p8=outputs/CRAFT_CL_trunc_k64_p8_llama32_1b_trace8_<ts> \
         k64_p64=outputs/CRAFT_CL_trunc_k64_p64_llama32_1b_trace8_<ts> \
  --csv truncation_sweep.csv

# duplicate unit test (same command shape, dup-mode auto-detected from
# the task set being exactly {ScienceQA-A, ScienceQA-B})
python explore/Routing_KL_truncation/analyze_truncation.py \
  --runs k64_p16=outputs/CRAFT_CL_trunc_dup_k64_p16_llama32_1b_<ts> \
         k512_p16=outputs/CRAFT_CL_trunc_dup_k512_p16_llama32_1b_<ts> \
  --csv duplicate_test.csv
```

## Analysis rules

1. **Sweep mode, headline test**: Rand index 1.0 against the reference
   across all six arms means the discovered TRACE-8 partition is not an
   artifact of truncation in the tested range — write one sentence
   citing this table and move on. Any arm below ~0.9 means the method
   section needs a sensitivity caveat on `k`.
2. **Duplicate-test mode is the direct fix-or-not answer for
   Limitations.** If ScienceQA-A/B merge ("YES") at some k in the
   sweep, the failure mode has a one-hyperparameter fix and the
   Limitations paragraph should say so explicitly rather than leaving
   it as unresolved future work. If they never merge even at k=512,
   report that too — it means the saturation is not purely a
   support-truncation effect and may need a different fix (e.g. seeding
   warm-up from the closest existing cluster, as the paper's own
   Limitations section already speculates).
3. **probe_size arms (k64_p8, k64_p64) test a different failure mode**:
   too few probe examples could make routing noisy/non-representative
   rather than systematically biased. If `k64_p8`'s partition disagrees
   with `k64_p16` while `k64_p64` agrees, that's evidence the paper's
   probe_size=16 is on the low side of a stability plateau — worth a
   sentence in the appendix regardless of which arm is adopted.
4. **routing_diagnostic_only's behavior change** (warm-up now runs for
   the very first task too) is local to this copy of `train.py`. It
   does not affect any other study's results or any full-pipeline run
   using a different copy.
5. **Seeds**: warm-up is stochastic per task even at fixed
   `fixed_intervention_seed` if data order/shuffling differs run to
   run; these are single-seed diagnostic passes. If the truncation
   result is going in the paper, rerun the reference and the flipped
   arm (if any) at 2-3 seeds to confirm the partition change is not
   noise.
