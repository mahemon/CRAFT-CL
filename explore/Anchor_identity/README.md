# Anchor identity (tests the §3.3 "choice of anchor" assertion)

Answers: §3.3 argues the routed group's state is the correct KL anchor
("the anchor must encode what the system already knows about S_k") but
never measures the alternatives. This study measures them.

## Design

Three arms, varying only which state is loaded into the frozen anchor
model used by the KL penalty:

| arm       | anchor state | tests |
|-----------|--------------|-------|
| group     | the routed group's current stored state (paper default) | reference |
| pi0       | the no-intervention frozen backbone | is anchoring to accumulated group knowledge necessary, or does anchoring to *anything stable* (even nothing) work as well? |
| prev_task | the immediately preceding task's own trained state, regardless of its group | does the specific *group aggregate* matter, or is anchoring to any recently-trained state sufficient? |

**The set of tasks that receive a KL anchor at all is identical across
all three arms** — still gated on "task joins an existing group,"
exactly as in the default pipeline. Only which state fills the anchor
changes. This keeps the comparison controlled: a difference in outcome
is attributable to anchor identity, not to a different subset of tasks
being anchored.

## train.py patch

**New file `anchor_variants.py`** (imported, not copied into `train.py`):
- `make_identity_shadow(shadow_reft)` — for the `pi0` arm. Zeroing a
  LoReFT intervention's `W` and `b` does **not** give a true no-op:
  `Φ(h) = h + Rᵀ(Wh + b − Rh)`, so `W=0, b=0` collapses to `h − RᵀRh`,
  which is not `h` unless `RᵀR = 0` — false for an orthonormal `R`. A
  genuine pi0 anchor requires bypassing the intervention's forward
  entirely. This function overrides each intervention module's
  `.forward` to the identity function via a per-instance method bind
  (standard technique; does not touch the pyreft/pyvene library, only
  this one throwaway shadow copy). After this call the shadow model's
  forward pass reproduces the frozen backbone exactly, regardless of
  whatever untrained parameters its modules hold.
- `snapshot_group_state(memory, gid)` — for the `prev_task` arm. Deep
  copies a group's `iv_state` right after that group was just saved.
  Under the default `overwrite` merge rule this is **exact**: overwrite
  replaces the group state with the trained state, so the snapshot taken
  immediately after task *t*'s save equals task *t*'s own trained state
  precisely. **This arm should be run with `merge_rule=overwrite`** (the
  paper default) — with any other merge rule the snapshot would reflect
  a blended state rather than the pure prior task, which is a different
  ablation.

**Five edits to `train.py`**, all additive (`--anchor_identity group`
reproduces the original code path exactly):
1. `from anchor_variants import make_identity_shadow, snapshot_group_state`
2. New CLI arg `--anchor_identity` (default `group`, choices
   `group`/`pi0`/`prev_task`).
3. `prev_task_anchor_state: Dict = {}` and `prev_task_name: Optional[str] = None`
   initialized before the main task loop.
4. The anchor-construction call site (previously hardcoded to
   `memory.groups[assigned_gid]["iv_state"]`) now branches on
   `args.anchor_identity`: `group`/`pi0` both build from the group state
   (pi0 then overrides forward to identity, so the loaded params are
   inert either way); `prev_task` builds from `prev_task_anchor_state`,
   falling back to the group state with a printed warning if no
   previous-task snapshot exists yet.
5. Right after every `memory.save(...)` call:
   `prev_task_anchor_state = snapshot_group_state(memory, final_gid)`,
   `prev_task_name = task_name` — feeds the *next* task's `prev_task` arm.

`groupwise_iv_keys` (which parameters the live model trains and later
merges into its group) is unchanged by any of this — it always reflects
the task's *assigned* group, independent of what the anchor is built
from, since after training the task still merges into that group
regardless of what it was anchored against during training.

## Contents

```
explore/Anchor_identity/
├── train.py                            # patched: --anchor_identity (5 edits)
├── anchor_variants.py                  # NEW: identity-shadow + snapshot helpers
├── submit_anchor_identity_llama15.sh   # 3 arms, 15-task, Llama-1B
├── analyze_anchor_identity.py
├── logs/  outputs/
```

## Run

```bash
export HF_TOKEN=hf_...
bash explore/Anchor_identity/submit_anchor_identity_llama15.sh          # 3 jobs
ARMS="pi0" bash explore/Anchor_identity/submit_anchor_identity_llama15.sh   # 1 arm
```

Flat submission, one A100 per job, 24h walltime. Same two
pre-submission checks as the other 15-task studies (`BENCH_DIR_REL`,
epoch schedule). `merge_rule` is not passed explicitly, so it uses
`train.py`'s built-in default (overwrite) — required for `prev_task`'s
exactness guarantee; do not combine this study with the merge-rule
ablation's non-overwrite arms.

## Analyze

```bash
python explore/Anchor_identity/analyze_anchor_identity.py \
  --runs group=outputs/CRAFT_CL_anchor_group_llama32_1b_15task_<ts> \
         pi0=outputs/CRAFT_CL_anchor_pi0_llama32_1b_15task_<ts> \
         prev_task=outputs/CRAFT_CL_anchor_prev_task_llama32_1b_15task_<ts> \
  --csv anchor_identity.csv
```

Prints OP/BWT/LA/n_groups per arm and per-task diagonal-accuracy deltas
against the `group` reference.

## Analysis rules

1. **`group` beating both alternatives on BWT at comparable OP** is the
   clean win: it validates §3.3's argument with a table instead of a
   sentence, and is exactly the evidence ido5's review implicitly asked
   for by never having it shown.
2. **`pi0` tying `group`** is the most damaging possible outcome for the
   method section: it means the KL term restrains drift in general, not
   specifically drift from what the group has learned, so "preserve the
   prior tasks routed here" (the stated semantics in §3.3) is not what
   the mechanism actually does. If this happens, the honest rewrite
   describes the KL term as output-space drift regularization rather
   than group-knowledge anchoring, and softens the "correct
   continual-learning objective" framing.
3. **`prev_task` between `group` and `pi0`** would suggest partial
   credit: anchoring to *some* accumulated adaptation helps over no
   anchor, but the specific *group aggregate* (which pools potentially
   several prior tasks) is doing additional work beyond what any single
   recent task's state would provide. This is the outcome that most
   cleanly supports keeping group-level storage over simpler
   alternatives (e.g. always anchor to the last task, no grouping).
4. **Per-task deltas matter more here than in other studies**: anchor
   identity plausibly has a bigger effect on later-arriving members of
   large groups (e.g. G0's seven members on the 15-task stream) than on
   early or singleton-group tasks, since later members are the ones
   whose anchor has accumulated the most (or, for pi0, whose anchor has
   accumulated nothing regardless of position). Check whether the
   `group` vs `pi0` gap widens with join position — this connects
   directly to the Intra_group_forgetting study's recency analysis.
5. **Seeds**: seed-42 selection runs; whichever anchor identity is
   reported needs the 3-seed treatment before the number is citable.
