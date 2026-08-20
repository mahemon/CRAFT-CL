# Beta dose-response (metareview consensus item 3)

Tests the paper's central claim — "KL divergence measures and controls
catastrophic forgetting" — as two falsifiable statements instead of an
n = 11 correlation:

- **PREDICTS**: pooled over all arms, terminal KL to the anchor
  (D_term_t, measured on the new task's own data) rank-correlates with
  the immediate in-group forgetting F_t that task caused — with
  negative-F_t (positive transfer) points **included**, closing ido5-W1's
  selection-on-the-dependent-variable objection.
- **CONTROLS**: as β increases, per-arm mean D_term decreases
  monotonically and mean F_t decreases with it. This is the dose-response
  of the knob: β demonstrably *sets* the KL, and KL demonstrably tracks
  forgetting.

If both hold, the paper may claim an empirical KL–forgetting relationship
under CRAFT's anchor objective (never a "law"). If PREDICTS holds but
CONTROLS is flat, β is redundant with early stopping — report that
honestly. If neither holds, the KL framing must be downgraded to "KL as
drift restraint," dropping the forgetting-prediction language.

## Why train.py is UNMODIFIED

The pipeline already logs everything the analysis needs:

- `evaluate_save_time_kl` computes the **raw** token-mean forward
  KL(anchor‖live) over ≤50 batches of the task's own data at save time —
  raw, not the running-mean-normalized value used in the loss — and it
  lands in `cl_results.json → kl_per_task[*].kl_final_val_diagnostic`.
- F_t is computed from `perf_matrix` + `task_to_group` (per-task backward
  eval), as the mean immediate drop on prior same-group tasks.

So the study is: run the unchanged pipeline at 5 β values; only
`--kl_beta` varies. The copy of `train.py` here is verbatim, for
self-containment only.

## Contents

```
explore/Beta_dose_response/
├── train.py                       # verbatim copy, zero patches
├── submit_beta_dose_llama15.sh    # 5 arms: β ∈ {0.05, 0.1, 0.3, 0.5, 1.0}
├── analyze_beta_dose.py           # PREDICTS + CONTROLS analysis
├── logs/  outputs/
```

## Run

```bash
# on Delta, from ROOT_DIR:
export HF_TOKEN=hf_...
bash explore/Beta_dose_response/submit_beta_dose_llama15.sh        # 5 jobs

# β=0.3 duplicates the paper config — if the Intra_group_forgetting
# `base` arm already ran on this stream, reuse it and submit 4:
BETAS="0.05 0.1 0.5 1.0" bash explore/Beta_dose_response/submit_beta_dose_llama15.sh
```

Flat submission, one A100 per job, 24h walltime each. Same two
pre-submission checks as the other studies: `BENCH_DIR_REL` and the
15-task epoch schedule.

## Analyze

```bash
python explore/Beta_dose_response/analyze_beta_dose.py \
  --runs 0.05=outputs/CRAFT_CL_betadose_0p05_llama32_1b_15task_<ts> \
         0.1=outputs/CRAFT_CL_betadose_0p1_llama32_1b_15task_<ts> \
         0.3=outputs/CRAFT_CL_betadose_0p3_llama32_1b_15task_<ts> \
         0.5=outputs/CRAFT_CL_betadose_0p5_llama32_1b_15task_<ts> \
         1.0=outputs/CRAFT_CL_betadose_1p0_llama32_1b_15task_<ts> \
  --csv beta_dose.csv --latex beta_dose.tex --fig beta_dose.png
```

Prints pooled Pearson/Spearman (negatives included, with the
excluding-negatives number as a labelled secondary), per-β correlations,
the CONTROLS monotonicity table (mean D_term, mean F_t, OP, BWT, K per β),
and writes the paper figure (scatter, β as color) plus a LaTeX table.

## Analysis rules

1. **The headline number is the with-negatives pooled Spearman.** The
   figure replaces the current Figure 4; its caption must say negatives
   are included and interpret them as positive transfer, not as points
   "the conjecture does not address."
2. **CONTROLS needs the monotone D_term column.** If mean D_term does not
   fall with β, the anchor coefficient is not doing what §3.3 says —
   check whether `kl_normalize_kl` (loss uses KL/running-mean) is
   flattening the effective pressure; if so, that interaction belongs in
   the paper, not hidden.
3. **Routing drift across arms**: β affects training, and group states
   feed the next task's routing, so arms can discover different
   partitions (the analyzer warns when K differs). If partitions differ,
   pooled correlations remain valid (each point is measured within its
   own arm's partition); per-group comparisons across arms are not.
4. **Sample size**: ~11 join-points per arm × 5 arms ≈ 55 pooled points —
   enough for a rank correlation but state n in the caption. The Math-LLM
   conjecture-verification runs across multiple seeds extend this pool if
   more power is needed.
5. **Wording ceiling**: this experiment licenses "β provides monotone
   control of terminal KL, and terminal KL tracks in-group forgetting
   (Spearman ρ = X, n = Y, negatives included)". It does not license
   "KL predicts forgetting" as a law, and Corollary-1-style Pareto claims
   stay deleted regardless of outcome.
6. **Seeds**: arms are seed-42. If the resulting figure becomes the
   paper's primary evidence for the KL claim (it will), the 3-seed
   extension of at least β ∈ {0.1, 0.3, 0.5} is required before
   submission; pool seeds as additional points, colored by β only.
