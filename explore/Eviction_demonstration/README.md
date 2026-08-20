# Eviction demonstration or deletion (C3: dead code in every reported run)

Answers: Eq. 5's eviction mechanism reports "Evicted? no" on all 23
routing rows in Tables 15/16 — it has never fired under normal routing.
Either it works under the conditions it was designed for and simply
never encounters them in TRACE-8/15-task, or it doesn't work at all.
This study forces the condition it was designed for and checks.

## No train.py patch needed

Eviction is fully implemented already
(`--kl_eviction_enabled`, `--kl_eviction_severity_threshold`,
`--kl_evict_absolute_threshold`, `--kl_eviction_early_steps` all exist
and are wired into `KLRazorController`). This study only varies existing
CLI flags. `train.py` here is a **verbatim copy** — verified byte-
identical to the uploaded original.

## Design: reproduce the paper's own adversarial test, forced

The paper's Limitations/robustness section already documents a spurious
merge: at routing threshold δ ≥ 1.0, FOMC and ScienceQA — a
characterized incompatible pair — merge into the same group ("JOIN G0"),
outside the operating plateau. That table used an isolated two-task
stream (FOMC then ScienceQA only), not the full 8-task order, since
other tasks between them in the standard order would occupy different
groups and not reproduce the exact adjacency.

This study reproduces that isolated setup exactly: `--cl_tasks
"FOMC,ScienceQA"`, `--kl_routing_threshold 1.0`. This is the paper's own
worst-case routing decision, forced on purpose — the correct place to
ask whether eviction ever rescues a bad merge, since it is not a
contrived pathological case but the paper's own documented failure
point.

## Four arms (only eviction sensitivity varies)

| arm | eviction | severity | abs_threshold | purpose |
|---|---|---|---|---|
| no_evict | off | — | — | control: confirms the forced merge happens and stays merged |
| evict_default | on | 1.3 | 1.0 | the paper's shipped thresholds — per Tables 15/16 these never fire under normal routing; checks whether they fire even under forced mis-routing |
| evict_sens | on | 1.1 | 0.3 | more sensitive |
| evict_vsens | on | 1.05 | 0.1 | maximally sensitive short of trivial — if eviction never fires even here, that is itself the finding |

Two independent trigger paths exist in `KLRazorController` and both are
exercised by this sweep:
- **Absolute**: at the first-epoch boundary, `mean(first-epoch KL) >
  eviction_absolute_threshold`.
- **Severity**: within `eviction_early_steps` of the first-epoch
  boundary, `rolling_mean / threshold > eviction_severity_threshold`
  (`threshold` = first-epoch mean + std).

## Contents

```
explore/Eviction_demonstration/
├── train.py                  # verbatim copy, zero patches
├── submit_eviction_demo.sh   # 4 arms, forced FOMC->ScienceQA, δ=1.0
├── analyze_eviction.py       # parses STDOUT LOGS (not cl_results.json)
├── logs/  outputs/
```

## Why the analyzer reads stdout logs, not `cl_results.json`

Eviction firing is not reliably serialized into the JSON results across
all code paths in this pipeline — the ground truth is the unconditional
`print()` at both trigger sites in `KLRazorController`
(`set_first_epoch_boundary` and the rolling-window check). Rather than
guess at a JSON schema that might not carry this field, the analyzer
regexes the `.txt` stdout log directly. This was verified against the
exact f-string formats in `train.py` (not assumed) — an earlier draft of
the JOIN-decision regex only matched the floor-triggered NEW GROUP
variant and silently missed the plain JOIN/NEW GROUP lines; it was
caught by testing against the real three print formats before shipping,
not by testing against a hand-written mock alone.

## Run

```bash
export HF_TOKEN=hf_...
bash explore/Eviction_demonstration/submit_eviction_demo.sh          # 4 jobs
ARMS="evict_vsens" bash explore/Eviction_demonstration/submit_eviction_demo.sh  # 1 arm
```

Flat submission, one A100 per job, 4h walltime cap (only 2 tasks, cheap
relative to every other study here).

## Analyze

```bash
python explore/Eviction_demonstration/analyze_eviction.py \
  --logs no_evict=explore/Eviction_demonstration/logs/fomc_scienceqa/llama32_1b_no_evict_<...>.txt \
         evict_default=explore/Eviction_demonstration/logs/fomc_scienceqa/llama32_1b_evict_default_<...>.txt \
         evict_sens=explore/Eviction_demonstration/logs/fomc_scienceqa/llama32_1b_evict_sens_<...>.txt \
         evict_vsens=explore/Eviction_demonstration/logs/fomc_scienceqa/llama32_1b_evict_vsens_<...>.txt
```

(A directory also works in place of an exact filename — the analyzer
auto-selects the newest `.txt` under it.)

Prints, per arm: the routing/eviction/first-epoch-boundary lines in
order (confirms the forced merge happened before asking about eviction),
and a summary table of eviction-event counts and the observed
first-epoch mean KL.

## Analysis rules

1. **Sanity check first**: `no_evict` must show `JOIN G0` for
   ScienceQA. If it doesn't, δ=1.0 didn't force the merge as intended —
   check whether `kl_routing_normalize=ratio` changes what δ=1.0 means
   relative to the paper's adversarial table (which may have used raw
   distance); if so, rerun with `--kl_routing_normalize raw` instead.
2. **Any eviction firing keeps the mechanism**: if `evict_default`,
   `evict_sens`, or `evict_vsens` shows `n_evictions > 0`, eviction is
   real, not dead code — cite the specific (severity, abs_threshold)
   pair that fired as the concrete demonstration in the appendix, and
   additionally report ScienceQA's own downstream accuracy under that
   arm versus `no_evict` (did eviction actually help, or just fire and
   still hurt).
3. **No firing even at evict_vsens is the deletion case**: this means
   the training-time KL trajectory does not diverge enough to trip
   either trigger even when routing is deliberately wrong — the trained
   intervention quietly adapts to accommodate the bad merge rather than
   triggering reassignment. In that case, remove eviction from the
   contributions list, delete or heavily caveat Eq. 5, and add one
   sentence noting that forced mis-routing was tested and did not
   surface eviction as a working safety net.
4. **Report the actual first-epoch mean KL observed** (printed in the
   summary table) regardless of outcome — it tells you how far below
   `evict_vsens`'s abs_threshold=0.1 the true trajectory sits, which is
   the number a reviewer would want if they ask "how close did it come."
5. **Single seed.** This is a threshold-discovery pass, not a citable
   result by itself. If eviction fires and enters the paper, rerun the
   firing configuration at 2-3 seeds to confirm it isn't a one-off
   artifact of this particular warm-up.
