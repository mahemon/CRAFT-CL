**subspace_energy** — Each group's LoReFT intervention contains a rotation matrix R whose rank-8 rows define the subspace that group learned to read and edit. The router runs the eval probe once through the frozen base model only, grabs hidden states at the intervention layers/positions, and measures how much of their energy falls inside each group's subspace (‖R_g h‖² / ‖h‖²). The group whose lens "sees" the most goes. Cost: **1 base forward per decision** (plus a one-time cached R extraction per group). Cheapest, and the most CRAFT-native — the router comes free from LoReFT's own geometry.

**perplexity** — Forward the probe through each group's *intervened* model and compute how well it predicts the probe's own next tokens (prompt NLL, no labels needed). The group trained on similar text predicts it best; lowest loss wins. Cost: **K forwards per decision** (K = number of groups, so 5 forwards for your 5-group topology). Most expensive, but the most direct test of "which group behaves like this data" — same output-space logic as your training-time routing, minus the warmup.

**prototype** — During training, right after each task commits, the frozen base embeds 16 of that task's train examples once and the mean hidden vector is stored as (or averaged into) the group's prototype. At eval, embed the probe the same way and pick the nearest prototype by cosine. Cost: **1 base forward per decision** + one per task at training time + one small stored vector per group. Simple and robust, but the mechanism is borrowed (L2P-style query-key matching), so it serves as the strong baseline rather than the novelty.

Quick comparison:

| | signal space | eval cost | needs training hook | novelty |
|---|---|---|---|---|
| subspace_energy | R subspace geometry | 1 forward | no | high (LoReFT-specific) |
| perplexity | output distribution | K forwards | no | medium (MoSEs-comparable) |
| prototype | base hidden state | 1 forward | yes (prototypes) | low (L2P-style) |

All three decide per eval batch (16 examples), not per example, and all fall back to oracle on any internal error. The story you're hoping the results tell: subspace_energy matches oracle at one forward — task-agnostic inference as a structural consequence of LoReFT, with perplexity and prototype as the comparison points showing you didn't need K forwards or stored prototypes to get there.