# Anchor direction (ido5 W2)

Answers: the training loss uses KL(anchor‖live) — mode-covering, LwF-style
— but the cited forgetting mechanism (Shenfeld et al., RL's Razor)
concerns reverse-KL bias under on-policy sampling, a mechanism that
doesn't operate under SFT-with-penalty. Which direction actually retains
better under CRAFT's anchor objective, empirically, rather than assumed
by citation.

## Design

Three arms differing only in the training-time KL penalty direction:

| arm       | penalty | reading |
|-----------|---------|---------|
| forward   | KL(anchor‖live) | mode-covering: live is penalized for missing mass the anchor assigns (paper default) |
| reverse   | KL(live‖anchor) | mode-seeking: live is penalized for placing mass where the anchor assigns little |
| symmetric | mean of both | agnostic middle ground |

## train.py patch (4 edits, `--kl_direction forward` reproduces the original path exactly)

1. `KLRazorController.__init__` gains a `kl_direction` parameter
   (validated against `{forward, reverse, symmetric}`), stored as
   `self.kl_direction`.
2. `compute_step_kl` — the function that produces the actual training
   loss term — branches on `self.kl_direction`. This is the only
   function changed; it is called once per training step.
3. New CLI arg `--kl_direction` (default `forward`).
4. `kl_direction=str(getattr(args, "kl_direction", "forward"))` passed
   through at the `KLRazorController(...)` construction call site.

**Deliberately unchanged**: `evaluate_save_time_kl`, the terminal-KL
diagnostic used by the β dose-response study (`kl_per_task[*].kl_final_val_diagnostic`
in `cl_results.json`), stays forward-only and unconditional. This keeps
terminal-KL readings comparably defined across all three direction arms
— the diagnostic measures "how far did training end up from the anchor,"
which should use one fixed yardstick regardless of what penalty drove
training there.

## Contents

```
explore/Anchor_direction/
├── train.py                             # patched: --kl_direction
├── submit_anchor_direction_llama15.sh   # 3 arms, 15-task, Llama-1B
├── analyze_anchor_direction.py
├── logs/  outputs/
```

## Run

```bash
export HF_TOKEN=hf_...
bash explore/Anchor_direction/submit_anchor_direction_llama15.sh        # 3 jobs
ARMS="reverse" bash explore/Anchor_direction/submit_anchor_direction_llama15.sh  # 1 arm
```

Flat submission, one A100 per job, 24h walltime. Same two pre-submission
checks as the other 15-task studies: `BENCH_DIR_REL` and the epoch
schedule. Script uses `lr=2e-4` (matching the R4 Nova log recovered
earlier); change to `1e-4` if that turns out to be the value behind your
reported numbers.

## Analyze

```bash
python explore/Anchor_direction/analyze_anchor_direction.py \
  --runs forward=outputs/CRAFT_CL_direction_forward_llama32_1b_15task_<ts> \
         reverse=outputs/CRAFT_CL_direction_reverse_llama32_1b_15task_<ts> \
         symmetric=outputs/CRAFT_CL_direction_symmetric_llama32_1b_15task_<ts> \
  --csv anchor_direction.csv
```

Prints OP/BWT/LA/n_groups per arm and a per-task in-group-forgetting
table (mean immediate drop each task causes on prior same-group tasks),
which is a finer-grained read than aggregate BWT alone.

## Analysis rules

1. **Forward winning on BWT at comparable OP** validates the current
   choice with evidence; write one sentence in the method section citing
   this table rather than only the RLHF-literature analogy.
2. **Reverse winning** is a finding, not noise: mode-seeking retention
   in intervention space would mean the anchor should concentrate live's
   probability mass on what the anchor is confident about, rather than
   covering everywhere the anchor puts mass. Change the shipped default
   before any camera-ready if this holds up under 3 seeds.
3. **Symmetric as a sanity check**: if it lands between forward and
   reverse, both directions are contributing independent restraint. If
   it matches whichever single direction is better, the other direction
   is redundant, not harmful.
4. **Routing partition should NOT drift across arms** here, unlike the
   stream/merge-rule studies: `kl_direction` only affects the *training*
   loss after a task is already routed, not the warm-up/routing phase
   itself. If `n_groups` or `task_to_group` differs across arms, that
   indicates order-of-operations leakage (e.g. training-time drift
   feeding back into a later task's warm-up) worth investigating before
   trusting the comparison.
5. **Seeds**: arms are seed-42 selection runs. The direction that enters
   the paper needs the standard 3-seed treatment.
