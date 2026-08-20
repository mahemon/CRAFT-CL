I'll read the code first — several reviewer complaints are answerable (or confirmable) only from the implementation.Read both files. Three findings in `train.py` change the diagnosis materially, so I'll lead with the structural story and fold the code evidence in.

---

## 1. Why it was actually rejected

The scores tell the story: **four "not good" ratings from jvrh and Pwt7, and ido5 gave Significance 2 despite Quality/Clarity/Originality 3.** Nobody disputed the numbers. Everyone disputed whether the numbers mean what you say they mean.

The rejection is not "weak results." It is three compounding failures:

**(a) The core premise is asserted, never tested.** The paper's whole identity is "representation space is the right place to do CL." Pwt7 punctured this in two sentences: the function is still being modified, so forgetting is still possible; and the routing + KL-anchor + merge machinery is *architecture-agnostic* — it would work on LoRA experts unchanged. Once a reviewer sees that, LoReFT becomes an implementation detail, not a contribution, and the paper collapses into "Learning-without-Forgetting distillation + a novel routing rule." That is one contribution, not four. **CRAFT-LoRA is the single missing experiment that caused the most damage.**

**(b) Your own Table 1 argues against your method.** Task-wise isolated interventions: OP 50.91, zero forgetting by construction. CRAFT: OP 44.17, BWT 0.87. You lose 6.7 OP points and gain nothing a reviewer can see, because the compensating advantage (4.20M vs 15.74M params, Table 20) is stated in a different appendix, never plotted against OP, and never framed as a trade-off curve. ido5's Q3 is exactly this. As currently written, the paper's own ablation is a recommendation not to use CRAFT.

**(c) The stated method is not the implemented method.** This is the part reviewers only smelled; a fourth reviewer with the anonymous repo open would have made it fatal. Details below.

Underneath all three: **honesty flags.** jvrh checked checklist items 7 and 8 and found both false (per-seed values are not in Appendix A; compute resources are not in §4 — and indeed the string "A100" appears nowhere in the LaTeX despite the checklist claiming it does). Once a reviewer catches two false checklist answers, every unverifiable claim gets discounted. That single paragraph in jvrh's review probably cost more than the missing ablations.

---

## 2. Code-vs-paper mismatches (highest severity, not yet caught)

These are blockers regardless of venue.

**M1 — Equation 4 is not what the code computes.**
Paper: `d(t,k) = D_KG / max(min(D_K, D_G), ε)`.
Code (`run_kl_routing_phase`, lines ~1360–1440): `raw_kl[gid] = D_KG`, then `normalize="raw"` → `distance = D_KG`. `D_K` and `D_G` are never computed. There is no `ε` denominator. Instead there is `min_dkg=0.01`: if `closest_raw_dkg < min_dkg` → open a new group.

Consequence: Appendix Table 15's narrative ("NEW decisions for MeetingBank and Py150 are triggered by the ε floor… the warm-up did not move enough from baseline") describes a *denominator* condition on `D_K`/`D_G` that the code does not evaluate. The actual trigger is that the task-to-group distance itself was tiny (0.0012, 0.0003 — both below `min_dkg=0.01`). Those two tasks were near-*identical* to G0, and the code opened new groups anyway. That is a defensible design choice, but it is the opposite of the explanation printed in the paper. A reviewer who diffs Eq. 4 against the linked repo has grounds for a fraud concern, not just a clarity concern.

**M2 — A per-task private parameter tier exists and is undisclosed.**
`--all_groupwise` defaults to `False`. With the default, `_split_iv_state_by_5tier` routes the deep-layer l-stream into bucket `t3_l`, and `save()` stores it as `self.t3l_tasks[task_name]` — keyed by **task**, not group. `load(task_name)` applies group state, then overwrites with that task's private `t3_l`.

If the reported runs used the default, then: Eq. 3 (`φ^(t) = φ_{S_k}`) is false; Table 20's 4.20M is an undercount (true cost is `K·shared + T·t3_l`); Table 21's `O(K L h r)` complexity claim is false (it grows in `T`); the intro's claim that the l-stream "carries a shared intervention parameter for cross-task transfer" is backwards; and the near-perfect cross-group invariance in Table 17 is partly explained by private parameters rather than by routing. **Verify which flag produced every reported number before anything else.**

**M3 — Eviction (Eq. 5) is dead code in the reported results.**
`eviction_enabled: bool = False` by default; `eviction_absolute_threshold: float = 0.0` gates the μ₁ path. Tables 15 and 16 report "Evicted? no" for all 23 rows. A mechanism presented as one half of contribution C3 never fires in any experiment. Either enable and demonstrate it, or delete it from the contribution list.

**M4 — The routing KL is an approximation, and that's why the sanity test fails.**
`_kl_logits_distance` uses `temperature=2.0` and `top_k=64`, gathers the union of both top-k sets, sets non-overlapping entries to `-inf`, and renormalizes. When two warmed-up models have disjoint top-64 supports, the renormalized distributions are near-orthogonal and symmetric KL saturates — precisely the ScienceQA-A/ScienceQA-B failure you already report in Limitations. Right now that reads as a mysterious defect. It is a straightforward consequence of truncated-support KL, and saying so converts a red flag into a controlled design analysis.

---

## 3. Prioritized work plan

### P0 — Without these, resubmission fails again

| # | Task | Answers |
|---|---|---|
| 1 | **Reconcile code and paper.** Rewrite §3.2 to state the implemented rule exactly: symmetric KL on label-position logits, T=2.0, top-64 union support, mean over unmasked tokens, `min_dkg` no-signal floor. Fix the Table 15 ε narrative. Confirm the `all_groupwise` setting and correct Eq. 3 / Table 20 / Table 21 accordingly. | M1, M2, M3, ido5-W2 |
| 2 | **CRAFT-LoRA ablation.** Same routing, same KL anchor, same overwrite, LoRA experts instead of LoReFT. Report OP/BWT/params on TRACE-8 across all three backbones. | Pwt7's central question |
| 3 | **Plasticity table.** Report `a_{t,t}` (diagonal) per method, not just OP/BWT. Add "Learning Accuracy" LA = mean of diagonal, and plot LA vs BWT for all baselines. Must include TreeLoRA and O-LoRA rerun in your harness. | jvrh Q3, the "does it just not learn?" suspicion |
| 4 | **Rerun at least TreeLoRA and O-LoRA yourself.** Every baseline is currently borrowed from Qian et al. Your O-LoRA Order-1 result is 77.75 vs their 59.73 — an 18-point gap on 4-task classification that no reviewer will read as method quality. Either the prompt format, eval sample count, or metric differs. Find it and say so. | The single largest credibility hole |
| 5 | **Inference protocol, stated honestly.** Add a subsection: CRAFT uses a task-ID lookup, exactly as O-LoRA/HiDeLoRA/TreeLoRA do under their reported protocols. Then run the task-agnostic variant you already have commented out in §3.4 and report the OP drop and the per-prompt cost. Move task-ID dependence into Limitations. | All three reviewers |

### P1 — Needed for ICLR/NAACL acceptance, not merely for survival

| # | Task | Answers |
|---|---|---|
| 6 | **Merge-rule ablation.** Overwrite vs EMA vs Fisher-weighted vs multiple prototypes per group, plus a recency-bias probe: after training the full 15-task stream, measure per-group performance as a function of arrival position. | ido5-W3, Q2 |
| 7 | **Intra-group forgetting.** jvrh's yahoo 61.0 → 42.0 example is real and unaddressed. Either (i) tighten routing so G0 doesn't absorb six tasks, (ii) add a per-group capacity cap, or (iii) state plainly that CRAFT converts cross-group forgetting to intra-group forgetting and quantify the split. Option (iii) is honest and cheap. | jvrh Q4 |
| 8 | **Ablations you claim to have run.** §B.2 lists sweeps over `t_pos ∈ {9,13,15,17}`, `r ∈ {4,8,16}`, and KL window `∈ {10,20,30,50}` but reports none of them. Report the tables. Also f-only vs l-only vs f+l. | jvrh's explicit complaint |
| 9 | **Overhead accounting.** Warm-up is `S_wu` steps plus `K` model builds plus `K+1` probe forwards per task. Report wall-clock as a fraction of task training, and how it scales at K=4 vs K=8. | ido5 Q4 |
| 10 | **Statistical significance.** On Llama-2-7B: 43.69±0.39 vs TreeLoRA 43.52±1.0. That is not a win. Either run paired seeds and a test, or delete "improves overall performance" for the 7B and Gemma settings and rewrite the abstract around forgetting reduction, which *is* large and robust. | Overclaiming |

### P2 — Polish

11. Delete §4 and Appendix A (already your plan). Theorem 1 is a Lagrangian-duality restatement, and Corollary 1's "no setting incurs less forgetting" is not licensed by an `R²` on n=11. ido5-W1 is aimed at this.
12. Cite Li & Hoiem (LwF) explicitly and position §3.3 as LwF-in-representation-space. Pwt7 named it; not citing it looks worse than citing it.
13. Cite HiDePrompt or drop it from §5.3. Add OGD to the bibliography (used in Table 3, never cited).
14. Fix the checklist. Add a real compute paragraph (GPU type, hours, per-run wall time) and either put per-seed values in the appendix or answer item 7 differently.
15. Add MoSEs and the MoE-LoRA family to related work with an explicit differentiation sentence.

---

## 4. Venue calibration

**ICLR** rewards mechanistic clarity and punishes unsupported framing. Your headline claim ("representation space is structurally better for CL") is exactly the kind of claim ICLR reviewers will demand be isolated experimentally. P0-2 (CRAFT-LoRA) is non-negotiable there. If CRAFT-LoRA performs comparably, do not hide it — reframe the paper as *"KL-probe routing + group anchoring, instantiated on two PEFT families"*, which is a stronger and more honest paper than the current one.

**NAACL/ACL/EMNLP** weight benchmark rigor and reproduced baselines more heavily than conceptual novelty. P0-4 (rerun baselines) matters most there, and the 18-point O-LoRA-order discrepancy would be caught immediately.

**CoLM** is the best fit for the current shape: LLM-specific, tolerant of empirical-over-theoretical papers, and receptive to "here is a mechanism and here is honest evidence about when it fails."

Realistic read: with P0 complete you have a borderline ICLR paper. With P0 + P1 you have a solid one. Without P0-2 and P0-4, any venue at this tier rejects again.

---

## 5. The one reframe that changes the most

Stop claiming representation space eliminates forgetting. It doesn't, and Table 17 shows it doesn't. Claim instead:

> Forgetting can be *localized*. Freezing the backbone confines all interference to a low-dimensional intervention subspace, where it becomes measurable in output space and controllable by a single scalar. CRAFT converts unbounded cross-task interference into bounded intra-group interference, and the routing threshold δ is the explicit knob that sets that boundary.

That claim is supported by evidence you already have (Table 17's block structure, Figure 3's δ plateau, Figure 4's KL–forgetting relation), survives Pwt7's objection, absorbs jvrh's yahoo counterexample as a *predicted* phenomenon instead of a defect, and does not require the theory section to hold.

---

Your last line — "also do it:" — got cut off. If you meant run the pre-submission audit prompt from the third document against the current LaTeX, say so and I'll do that as a separate pass; it needs a different output format (triaged BLOCKERS/SHOULD-FIX/NITS with grep anchors) than what's above.