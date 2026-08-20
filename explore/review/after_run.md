Two separate questions; taking them in order.

## 1. What is actually required vs what wins points

Not all eight P0 items carry equal rejection risk. Sorted by what happens if you skip it:

**Non-negotiable (skipping = rejected again):**

- **LA / plasticity reporting (P0-4).** Yes, this is a must, and it's also the cheapest item on the whole list. jvrh asked "is forgetting low because the model barely learns?" — a question derivable from your own tables (LA = OP + BWT) and partially true (−8.4 vs O-LoRA on Gemma). A resubmission that still hides the plasticity column invites the same reviewer to derive it themselves, and a reviewer who computes your hidden cost variable is done with the paper. Cost: near zero for CRAFT; the baseline diagonals come free with P0-2.
- **CRAFT-LoRA (P0-3).** Pwt7's entire review reduces to this experiment. Without it, the substrate claim remains unfalsifiable and Pwt7-type reviewers at the next venue rescore it 2/2/2/2.
- **Same-harness baselines, at minimum TreeLoRA + O-LoRA (P0-2).** The 77-vs-59 order-table anomaly is visible to any experienced reviewer without reading the code. Cross-harness headline numbers will not survive a second round anywhere.
- **Code–paper reconciliation (P0-1, P0-6).** With an anonymous repo linked in the abstract, Eq. 4 vs the implemented rule is a one-diff discovery, and it reads as fabrication rather than drift. Also the checklist fixes — jvrh already demonstrated reviewers check these.

**Required to address, but "address" ≠ "solve" — honest scoping suffices:**

- **Task-agnostic inference (P0-5).** All three reviewers asked, so silence is not an option. But acceptance does not require CRAFT to *win* task-agnostically — it requires the protocol to be stated honestly (task-aware, matching TreeLoRA; easier than O-LoRA/HiDe) and the task-agnostic gap to be *measured*, whatever it is. A result like "predicted routing recovers 92% of oracle OP at K forward passes per prompt" closes the objection even if imperfect. And here's the thing: **you've already built this.** Your Delta scripts carry `INFERENCE_ROUTER ∈ {oracle, subspace_energy, perplexity, prototype}` and `agnostic_router.py`. The experiment all three reviewers demanded is one env var away from running. Given that, skipping it would be indefensible.
- **Intra-group forgetting (was P1-9).** Now done — the decomposition folder gives you the 100%/0% split from Table 17 data. Costs a paragraph and a table.

So the honest answer: **1–8 are sufficient only if "task-agnostic inference" means the measured-gap version, and LA is inside that set, not optional.** What moves the paper from "survives review" to "accepted" is P1-7 (β dose-response — because the KL claim is the paper's scientific identity and the metareview named it) plus significance tests on the 7B/Gemma margins. The merge ablation and capacity arms you've now built are supporting material, not gatekeepers.

## 2. The novelty objection — how to actually tackle it

You cannot add novelty in a revision; you can only reposition what exists. The objection decomposes into three specific complaints, each with a different correct response:

**(a) "CRAFT is methodologically identical to LoReFT" (jvrh).** The response is a *design-decisions* subsection answering jvrh's literal question — "what was considered in order to apply LoReFT to continual learning?" Content you already have, scattered: LoReFT is single-task; extending it to a stream required deciding the unit of sharing (per-task / global / grouped — Table 1 is this ablation), the routing signal (output-distribution KL vs parameter-space similarity — Table 4's L2-SP/EWC comparison is this), the anchor (group state vs π₀ — §3.3's "Choice of anchor" paragraph is this), and the Stiefel-parameter transfer note (currently buried in an appendix implementation note). Assembled into one place, "identical to LoReFT" stops being writable, because every one of those decisions has an ablation showing the alternative fails.

**(b) "Distillation regularization = LwF; table lookup isn't novel" (Pwt7).** Concede and cite. The move that works with reviewers is claim-precision, not claim-inflation: state that the KL penalty is the LwF primitive, and locate the contribution exactly where it is — the *anchor selection* (an evolving, routed group state rather than LwF's fixed previous model), the *gate-free probe routing* (Eq. 4's replacement — the one component both jvrh and ido5 explicitly called novel/interesting), and the *single-signal unification*. A paper that cites LwF and says "we change the anchor and show why that choice matters (Table 4, anchor ablation)" is more credible than one that doesn't mention LwF and gets caught.

**(c) "Why LoReFT over LoRA?" (Pwt7, and the metareview's consensus item #1).** This one cannot be argued — only measured. CRAFT-LoRA is the answer regardless of outcome, and note the asymmetry: **every outcome improves the paper.** LoReFT wins → the substrate claim is finally earned and the current framing stands with evidence. Parity → reframe as "a substrate-agnostic KL-routing framework for PEFT-based CL, instantiated on two adapter families" — which is *broader*, sounds more novel, and neutralizes the objection entirely because the objection assumed the substrate was the claim. This is the counterintuitive point: the experiment reviewers demanded as a novelty *attack* is also your best novelty *defense*, because it relocates the contribution from the substrate (contested) to the framework (uncontested — nobody disputed the routing).

**Applicability** specifically is answered by the pieces now in flight: the measured task-agnostic gap (P0-5, infrastructure done), the overhead accounting (warm-up + K probes per task, ido5-Q4), and the localization split (done). Those three convert "is this usable?" from a rhetorical weakness into three tables.

One framing rule for the rewrite: every place the current draft says "first to X," replace with the specific mechanism plus its ablation reference. Reviewers rated Originality 2 partly because "first joint routing/adaptation/merging in representation space" is the kind of claim that invites the identical-to-LoReFT rebuttal; "task-level probe routing without trainable gate parameters, versus token-level learned gating in the MoE-LoRA family (Table X)" is the kind that doesn't.



No. Check what's running against the metareview's four consensus items — that's the acceptance test, and two of the four are still uncovered.

| Metareview consensus item | Covered by what's running? |
|---|---|
| 1. Novelty / LoReFT-vs-LoRA justification | ✓ CRAFT-LoRA (plus the design-decisions + LwF rewrite, which is writing, not compute) |
| 2. Inference protocol / task-ID dependency | ✓ task-agnostic router ablation |
| 3. **KL-measures-and-controls-forgetting justification** | ✗ **nothing running addresses this** |
| 4. Experimental completeness | ~ partial — merge-rule and intra-group help, but the named gaps are elsewhere |

**Gap 1 — β dose-response (consensus item 3).** The merge ablation does not touch the KL claim. The claim "KL predicts and controls forgetting" is the paper's scientific identity, ido5-W1 attacked it directly, and the metareview elevated it to a consensus concern. The missing experiment: β ∈ {0.05, 0.1, 0.3, 0.5, 1.0} on the 15-task stream, plot per-task terminal KL vs in-group forgetting colored by β, negative-forgetting points *included* this time. Monotone co-movement = claim established; without it you're resubmitting the same n=11 correlation that drew the objection. This is 5 jobs on the arm you already have infrastructure for (your β-sweep configs exist — Table 19 came from somewhere), and my intra-group analyzer already computes per-task in-group forgetting F_t from cl_results.json, so the analysis side is done.

**Gap 2 — same-harness TreeLoRA and O-LoRA (consensus item 4, and the credibility floor under every table).** This is the one I'd rank above everything currently running except CRAFT-LoRA. Nothing in your four studies fixes the fact that every headline comparison is CRAFT-in-your-harness vs baselines-borrowed-from-Qian-et-al., with the 77-vs-59 order-table anomaly sitting in plain view. It also gates LA reporting: CRAFT's LA is arithmetic from your own matrices, but the plasticity *comparison* (the answer to jvrh's "does it just barely learn?") needs baseline diagonals, which only same-harness reruns produce. Minimum: TreeLoRA + O-LoRA, TRACE-8, three backbones, 3 seeds — and whatever explains the order-table gap, found and stated.

**Cheap items that cost hours, not GPU-days, and are currently unclaimed:**

1. **LA columns for CRAFT** — pure arithmetic on existing matrices; add now, extend with baseline diagonals when reruns land.
2. **Overhead table (ido5-Q4)** — extractable from existing SLURM logs: warm-up wall-clock is timestamped, probe forwards are countable. No new runs.
3. **Promised-but-unreported sweeps (jvrh)** — §B.2 claims t_pos, rank, and KL-window sweeps were run. If those runs exist on Nova/Delta, this is log archaeology; if they don't, either run them or delete the claim — a second reviewer catching a claimed-but-absent sweep after the checklist incident would be fatal.
4. **Significance handling for 7B/Gemma** — either paired-seed tests or rewrite the claim scope (OP improvement claimed for 1B and 15-task only; forgetting reduction claimed everywhere). Writing, not compute.
5. **The writing bundle** — LwF citation + repositioning, MoSEs/MoE-LoRA differentiation, checklist corrections, compute paragraph, HiDePrompt/OGD cleanup, f/l-stream clarification jvrh asked for.

One more note on what *is* running: merge-rule and intra-group were P1 items — good supporting material, and intra-group is effectively free since the analyzer already produced the 100%/0% split. But they answer one reviewer's secondary questions (ido5 W3/Q2, jvrh Q4) while consensus item 3 answers all of them. If Delta GPU allocation is the constraint, priority for the next submission wave is: TreeLoRA/O-LoRA reruns > β dose-response > everything else queued.

So the full picture: your four studies + reconciliation close roughly half the acceptance test. Add the β dose-response and the baseline reruns and the P0 list is genuinely complete; the five cheap items are then writing-week work, not experiment-week work.