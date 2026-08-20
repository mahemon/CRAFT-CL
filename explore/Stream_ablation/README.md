# Stream ablation (jvrh W1)

Answers: "what does each prompt stream (f-stream / l-stream) contribute,
and does the position budget interact with prompt length?" No ablation
for f-only, l-only, or asymmetric $t_{pos}$ currently exists in the paper
despite §B.2 implying one.

## Design

`train.py` builds the pyreft position string as `f{t_pos}+l{t_pos}`,
always symmetric — there was no way to zero out one stream or use
different sizes for each. Four arms, total intervention-position budget
held fixed at 30 (matching the paper's default `f15+l15`) so parameter
count and total supervised-position count stay constant across arms and
only the *split* varies:

| arm     | f  | l  | tests |
|---------|----|----|-------|
| f15l15  | 15 | 15 | reference (paper default) |
| f30     | 30 | 0  | prefix (instruction-context) stream alone |
| l30     | 0  | 30 | suffix (answer-adjacent) stream alone |
| f5l25   | 5  | 25 | asymmetric budget |

## train.py patch

Two edits, both isolated to position-string construction; nothing else
in the pipeline changes.

1. New CLI args `--f_positions` / `--l_positions` (int, default `None`).
   When unset, both default to `--n_positions` — **the original
   symmetric behavior is preserved exactly** when the new flags are not
   passed, so every other script in this repo is unaffected.
2. `task_pos` construction now branches on which counts are nonzero:
   `f{a}+l{b}` when both > 0, `f{a}` when only the prefix stream is
   used, `l{b}` when only the suffix stream is used. pyreft's position
   parser natively supports single-sided position strings, so no
   downstream changes are needed — `LoReftTRACEDataset` and the
   intervention builder receive `task_pos` unmodified either way.
   `question_offset` (the f-stream's token start position, computed from
   the chat-template header) is only consumed when an f-component is
   present, so it is inert and harmless on `l30`.

Validated: `f_positions=0` and `l_positions=0` simultaneously raises a
clear error rather than silently building an empty position string.

## Contents

```
explore/Stream_ablation/
├── train.py                            # patched: --f_positions / --l_positions
├── submit_stream_ablation_trace8.sh    # 4 arms, TRACE-8, Llama-1B
├── analyze_stream_ablation.py          # OP/BWT/LA table + per-task deltas + optional length proxy
├── logs/  outputs/
```

## Run

```bash
export HF_TOKEN=hf_...
bash explore/Stream_ablation/submit_stream_ablation_trace8.sh        # 4 jobs
ARMS="l30" bash explore/Stream_ablation/submit_stream_ablation_trace8.sh   # 1 arm
```

Flat submission, one A100 per job, 23h walltime each.

**One open decision, flagged rather than silently resolved**: the
script uses `lr=2e-4`. Your `submit_craft_delta.sh` hardcodes `1e-4`,
but a cluster log from this exact configuration
(`R4_kl_floor_ratio_thr0_7_floor0_05...txt`) recorded `lr=0.0002`
(2e-4) for the identical epoch schedule and model. If `1e-4` is the
value that actually produced your reported TRACE-8 llama numbers,
change `submit_job "2e-4"` to `"1e-4"` before submitting, so this
ablation's reference arm (f15l15) is comparable to the paper table
rather than to a different learning rate.

## Analyze

```bash
python explore/Stream_ablation/analyze_stream_ablation.py \
  --runs f15l15=outputs/CRAFT_CL_stream_f15l15_llama32_1b_trace8_<ts> \
         f30=outputs/CRAFT_CL_stream_f30_llama32_1b_trace8_<ts> \
         l30=outputs/CRAFT_CL_stream_l30_llama32_1b_trace8_<ts> \
         f5l25=outputs/CRAFT_CL_stream_f5l25_llama32_1b_trace8_<ts> \
  --csv stream_ablation.csv
```

Prints an OP/BWT/LA/n\_groups table against the f15l15 reference, and a
per-task diagonal-accuracy (`a_tt`) delta table so which tasks are most
sensitive to losing a stream is visible per task, not just in the
aggregate.

### Optional: prompt-length proxy

```bash
python explore/Stream_ablation/analyze_stream_ablation.py \
  --runs f15l15=... f30=... l30=... f5l25=... \
  --benchmark_dir "$ROOT_DIR/dataset/TRACE_Benchmark/LLM-CL-Benchmark_5000" \
  --model /projects/bdpm/mhossen1/models/Llama-3.2-1B-Instruct
```

**This is a best-effort proxy, not real per-example stratification.**
`cl_results.json` only stores one aggregate score per task, so the
pipeline has no per-example predictions to stratify by length within a
task. What this flag does instead: tokenizes each task's raw eval
prompts, takes the median length, and correlates *median prompt length
per task* against that task's *whole-task* f-only / l-only OP delta.
It will tell you whether long-prompt tasks (e.g. MeetingBank) are more
or less stream-sensitive than short-prompt tasks (e.g. C-STANCE) — a
task-level signal, not a within-task length effect.

If real per-example stratification is needed later, the patch point is
the evaluation loop in `finetune_cl` (`cl_eval_sets` / per-example
generation): logging `(example_id, prompt_len, correct)` triples to
`cl_results.json` per task would let this analyzer bucket by length
quartile *within* a task. That is a separate, larger patch and is not
included here to avoid modifying the eval loop under time pressure —
flagging it rather than guessing at the eval code's exact structure.

The benchmark-JSON field name is guessed from a short candidate list
(`prompt`, `input`, `instruction`, `question`, `text`); if none match
your TRACE JSON schema, the script prints the actual example keys it
found and skips that task, so it fails loud and specific rather than
silently mis-measuring.

## Analysis rules

1. **Headline test**: if `l30`'s OP is within 1-2 points of `f15l15`
   while `f30` drops well below both, the suffix stream carries the
   task-specific signal and the prefix stream is prunable — this
   directly rewrites the intro's stream-role claim and shrinks the
   parameter-count table (halving intervention sites per layer).
2. **If both single-stream arms drop substantially**, both streams are
   load-bearing and the current symmetric default is justified rather
   than assumed; report both drops together as the ablation the method
   section currently lacks.
3. **`f5l25` is the tie-breaker**, not a redundant arm: if it lands
   close to `f15l15` (better than `f30`, comparable to `l30`), that
   argues the *total* budget matters more than the *split*, and 15/15
   is not a load-bearing design choice, only a convenient default.
4. **Per-task deltas before aggregate conclusions.** A flat aggregate
   OP drop can hide a bimodal pattern (some tasks unaffected, others
   collapse) that a single mean obscures — read the per-task table
   before writing the one-sentence summary.
5. **Routing partition drift**: arms with a different position budget
   train different warm-up representations, so `n_groups` and group
   membership can differ across arms. Compare OP/BWT/LA at the stream
   level; do not compare per-group structure across arms without first
   checking `task_to_group` agreement.
6. **Seeds**: single-seed (42) arm-selection runs. Whichever
   configuration is reported in the paper needs the standard 3-seed
   treatment before the number is citable.
