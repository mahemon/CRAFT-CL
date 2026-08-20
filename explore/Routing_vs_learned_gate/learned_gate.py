"""
learned_gate.py — support for the Routing-vs-Learned-Gate ablation (C2 as
a result: does gate-free KL probe routing beat a learned gate, at matched
probe data?).

Design: a minimal linear classifier over mean-pooled frozen-backbone
hidden states of the SAME probe batch kl_routing observes, refit from
scratch each time a new task arrives (task counts are small -- 8 to 15 --
so refitting a tiny logistic regression on a growing dataset of
one-vector-per-task is a few milliseconds, not a training cost worth
amortizing). This is a genuine trainable-parameter router: the linear
layer's weights are optimized, unlike kl_routing's KL distance, which
has none. It is intentionally NOT hooked into the group's actual output
distribution (unlike kl_routing) -- it only sees the frozen backbone's
own representation of the prompt, which is what a MoE-style gate
typically conditions on.

This module is self-contained (only needs torch); it is not imported by
anything in the CRAFT-CL pipeline other than a small dispatch branch
added to run_learned_gate_routing_phase's caller in train.py.
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@torch.no_grad()
def pooled_base_features(base_model, probe_batch: Dict[str, torch.Tensor]
                         ) -> torch.Tensor:
    """Mean-pooled final-hidden-layer representation of the frozen
    backbone (no intervention applied) over the probe batch, averaged
    across examples to a single task-level feature vector of shape
    [hidden_size]. Strips reft-specific collator keys before the plain
    HF forward call."""
    b = {k: v for k, v in probe_batch.items()
        if k in ("input_ids", "attention_mask")}
    b["output_hidden_states"] = True
    b["use_cache"] = False
    was_training = base_model.training
    base_model.eval()
    out = base_model(**b)
    if was_training:
        base_model.train()
    hidden = out.hidden_states[-1]                      # [B, T, H]
    attn = probe_batch.get("attention_mask")
    if attn is None:
        attn = torch.ones(hidden.shape[:2], device=hidden.device)
    mask = attn.unsqueeze(-1).to(hidden.dtype)           # [B, T, 1]
    pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
    return pooled.mean(dim=0).float().detach()           # [H]


class LearnedGateRouter:
    """Grows a class per discovered group. Retrains a small linear head
    from scratch on the accumulated (feature, group_id) history whenever
    the number of classes changes, then routes the new task's feature
    vector by argmax confidence against a threshold."""

    def __init__(self, hidden_size: int, device: str,
                confidence_threshold: float = 0.5,
                lr: float = 1e-2, epochs: int = 300,
                weight_decay: float = 1e-3):
        self.hidden_size = hidden_size
        self.device = device
        self.confidence_threshold = float(confidence_threshold)
        self.lr = float(lr)
        self.epochs = int(epochs)
        self.weight_decay = float(weight_decay)
        self.history_feats: List[torch.Tensor] = []
        self.history_labels: List[int] = []
        self.classifier: Optional[nn.Linear] = None

    def _refit(self) -> float:
        n_classes = len(set(self.history_labels))
        X = torch.stack(self.history_feats).to(self.device)
        y = torch.tensor(self.history_labels, device=self.device,
                         dtype=torch.long)
        clf = nn.Linear(self.hidden_size, n_classes).to(self.device)
        opt = torch.optim.Adam(clf.parameters(), lr=self.lr,
                               weight_decay=self.weight_decay)
        last_loss = float("nan")
        for _ in range(self.epochs):
            opt.zero_grad()
            logits = clf(X)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            opt.step()
            last_loss = float(loss.item())
        self.classifier = clf
        return last_loss

    def route(self, feat: torch.Tensor
             ) -> Tuple[Optional[int], bool, float, Optional[float]]:
        """Returns (predicted_gid_or_None, is_new_group, confidence,
        refit_train_loss_or_None). First task always opens a new group
        (no classes to predict from yet)."""
        if not self.history_feats:
            return None, True, 0.0, None
        n_classes_now = len(set(self.history_labels))
        refit_loss = None
        if self.classifier is None or self.classifier.out_features != n_classes_now:
            refit_loss = self._refit()
        with torch.no_grad():
            logits = self.classifier(feat.to(self.device).unsqueeze(0))
            probs = F.softmax(logits, dim=-1).squeeze(0)
            conf, pred = probs.max(dim=0)
        conf_f = float(conf.item())
        if conf_f >= self.confidence_threshold:
            return int(pred.item()), False, conf_f, refit_loss
        return None, True, conf_f, refit_loss

    def register(self, feat: torch.Tensor, gid: int) -> None:
        self.history_feats.append(feat.detach().cpu())
        self.history_labels.append(int(gid))

    def param_count(self) -> int:
        if self.classifier is None:
            return 0
        return sum(p.numel() for p in self.classifier.parameters())
