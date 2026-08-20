# Routing vs learned gate (C2 as a result, not an assertion)

Answers: the paper claims gate-free KL probe routing avoids the
trainable-router overhead of MoE-LoRA-style methods, but never measures
this against an actual learned gate at matched probe data. This study
builds a minimal one and compares.

## Design

Two arms on the same TRACE-8 stream, differing only in `--router_method`:

| arm | mechanism | trainable params |
|---|---|---|
| kl_routing | symmetric KL between warmed-up output distributions (paper default) | 0 |
| learned_gate | linear classifier over mean-pooled **frozen-backbone** hidden states of the same probe batch | `hidden_size × n_groups` |

Both arms see **identical probe data** (`--kl_routing_probe_size 16`),
so the comparison isolates "trainable gate" vs "training-free KL
distance" and nothing else — the same probe batch, same size, same
tasks in the same order.

## `learned_gate.py` (new module)

- `pooled_base_features(base_model, probe_batch)`: mean-pools the frozen
  backbone's **final hidden layer** (no intervention applied) over the
  probe batch, averaged across examples to one task-level feature
  vector. This is intentionally simpler than kl_routing's mechanism — a
  classic MoE-style gate conditions on the input representation, not on
  a warmed-up intervention's output distribution — and needs no warm-up
  training at all for the gate itself.
- `LearnedGateRouter`: maintains a growing history of
  `(feature, group_id)` pairs (one per task routed so far) and refits a
  small `nn.Linear` from scratch via full-batch Adam whenever the number
  of classes changes. Task counts are small (8-15), so refitting a tiny
  logistic regression on a handful of vectors costs milliseconds, not a
  training expense worth amortizing incrementally. Routes by
  argmax-confidence against `--gate_confidence_threshold` (default 0.5);
  below threshold, opens a new group exactly as kl_routing's `δ`
  threshold does.

## train.py patches (4 edits; `--router_method kl_routing` reproduces
the original path exactly — the new branch is purely additive)

1. `from learned_gate import LearnedGateRouter, pooled_base_features`
2. `--router_method` gains `learned_gate` as a third choice, plus
   `--gate_confidence_threshold` / `--gate_lr` / `--gate_epochs`.
3. `gate_router = LearnedGateRouter(...)` instantiated once before the
   main task loop (mirrors how `memory` is instantiated), only when
   `router_method == "learned_gate"`.
4. New `elif router_method == "learned_gate":` branch in the routing
   dispatch chain, structurally parallel to the `kl_routing` branch:
   builds the same probe batch via the existing
   `_build_current_task_probe` helper, extracts the pooled feature,
   calls `gate_router.route(...)`, registers the outcome, and produces
   the same `(assigned_gid, is_new_group, routing_debug)` triple every
   other router_method produces — so nothing downstream (Phase 2
   training, group storage, eval) needs to know which router ran.

## Contents

```
explore/Routing_vs_learned_gate/
├── train.py                          # patched (4 edits above)
├── learned_gate.py                   # NEW: pooled features + LearnedGateRouter
├── submit_routing_gate_trace8.sh     # 2 arms, TRACE-8, Llama-1B
├── analyze_routing_gate.py
├── logs/  outputs/
```

## Run

```bash
export HF_TOKEN=hf_...
bash explore/Routing_vs_learned_gate/submit_routing_gate_trace8.sh          # 2 jobs
ARMS="learned_gate" bash explore/Routing_vs_learned_gate/submit_routing_gate_trace8.sh  # 1 arm
```

Flat submission, one A100 per job, 23h walltime (full Phase 2 training
runs in both arms — only the routing decision differs, not the training
cost). Same pre-submission checks as the other TRACE-8 studies.

## Analyze

```bash
python explore/Routing_vs_learned_gate/analyze_routing_gate.py \
  --runs kl_routing=outputs/CRAFT_CL_routergate_kl_routing_llama32_1b_trace8_<ts> \
         learned_gate=outputs/CRAFT_CL_routergate_learned_gate_llama32_1b_trace8_<ts> \
  --csv routing_gate.csv
```

Prints OP/BWT/LA/n_groups per arm, an **unadjusted Rand index** over
task-pair co-membership (1.0 = identical partition structure up to
group-id relabeling, 0.5 = chance-level agreement), ΔOP/ΔBWT
(learned_gate − kl_routing), and the raw per-task group assignment table
for manual inspection.

## Analysis rules

1. **kl_routing matching or beating learned_gate on OP/BWT at zero
   trainable routing parameters** is the strongest sentence available
   for the contributions list — write it with this table as the
   citation. This is also the outcome that most directly answers C2 as
   asked by jvrh and ido5 (is gate-free routing actually good, or just
   parameter-free).
2. **learned_gate winning** bounds the claim to "parameter-free routing
   at a stated cost": report the ΔOP/ΔBWT gap and the gate's parameter
   count (`gate_param_count` in each task's routing debug entry, printed
   in the stdout log and derivable as `hidden_size × n_groups_at_that_point`)
   side by side, so the trade is quantified rather than asserted.
3. **Partition agreement (Rand index) matters independently of OP/BWT.**
   Two routers can produce similar downstream performance while
   disagreeing substantially on *which* tasks share a group — if Rand
   index is well below 1.0 but OP/BWT are close, that's worth a
   sentence: it means CRAFT's downstream performance is not sharply
   sensitive to the exact partition, which is itself informative about
   how much the specific KL-routing mechanism actually matters versus
   "any reasonable grouping works."
4. **The gate's feature is deliberately not the same object kl_routing
   compares.** kl_routing measures divergence in output-distribution
   space after a brief warm-up; the gate measures similarity in
   input-representation space with no warm-up at all. If the gate wins,
   the natural follow-up (not implemented here) is a gate that pools
   the *warmed-up* model's hidden states instead of the frozen
   backbone's, to isolate whether the win comes from "trainable" or
   from "warm-up-free."
5. **Seeds**: seed-42 selection runs. Whichever router is reported in
   the paper needs the standard 3-seed treatment; note that
   `LearnedGateRouter`'s refits are themselves stochastic (Adam init),
   so re-running the same seed twice with a different `gate_lr`/`gate_epochs`
   is a reasonable robustness check before committing to the gate's
   numbers if it turns out to be the more favorable arm.
