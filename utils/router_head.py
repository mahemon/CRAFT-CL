"""
router_head.py — a trainable task-identity router for CRAFT.

What it is
----------
A linear classifier from frozen base-model features to task identity. It is
trained DURING the continual stream: at every task commit, features of the new
task are added to a feature bank and the head is refit on the banks of all
tasks seen so far. Because the backbone is frozen, the features never change,
so the bank is an exact replay and the head cannot forget. Nothing in this
file touches the LoReFT interventions or how they train.

Features
--------
For one example: mean of the frozen-base hidden state over the last `n_pos`
valid prompt tokens (the l-stream positions), taken at every intervened layer
and concatenated. Same positions the interventions edit; same pooling the
prototype routers use, but per example rather than per batch.

Training
--------
Multinomial logistic regression on standardized features, L2-regularized,
solved with LBFGS. Deterministic. On Llama-3.2-1B the head is
(16 layers x 2048) x T weights; at T=8 that is ~262K parameters.

Target
------
`target="task"` (default) predicts the task; `target="group"` predicts the
group id directly, pooling every member task's bank under its group at fit
time. Use group mode when there is no private T3-l stream (--all_groupwise):
the group is then all inference needs, and confusions between tasks in the
same group stop counting as errors.

Inference
---------
Predict a task from the query batch (mean of per-example probabilities),
look the task up in task_to_group, and load the group state plus the task's
private T3-l state. The predicted probability doubles as a confidence for an
optional group-only fallback.

Persistence
-----------
save()/load() write one file, router_head.pt, next to the group checkpoint.
It holds the banks (fp16), the fitted head, the layer ids and n_pos used for
the features, so inference reproduces the same feature space.
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F

_KEEP = ("input_ids", "attention_mask", "labels", "intervention_locations", "subspaces")


# ──────────────────────────────────────────────────────────────────────────
# features
# ──────────────────────────────────────────────────────────────────────────
def _unique_layers(task_layers) -> List[int]:
    """train.py's task_layers is base_layers x 2 (f and l share a layer id).
    Accepts ints or intervention keys ('layer.12.comp...')."""
    import re
    out = set()
    for x in task_layers:
        if isinstance(x, int):
            out.add(x)
        else:
            m = re.search(r"layer[._](\d+)", str(x))
            if m:
                out.add(int(m.group(1)))
    return sorted(out)


@torch.no_grad()
def features_from_batch(base_model, batch: Dict, layer_ids: List[int],
                        n_pos: int, prompt_only: bool = False) -> torch.Tensor:
    """(B, len(layer_ids) * hidden) float32 on the model's device.
    One row per example: mean over its last n_pos valid tokens, per layer.

    prompt_only=True pools the last n_pos PROMPT tokens (labels == -100) when
    the batch carries labels of the same shape. Training examples hold prompt
    + answer, test prompts hold the prompt only, so this makes the bank match
    what the head sees at test time. Falls back to all valid tokens when no
    prompt token is marked. False = original behaviour."""
    labels = batch.get("labels", None)
    use_labels = (prompt_only and torch.is_tensor(labels)
                  and tuple(labels.shape) == tuple(batch["attention_mask"].shape))
    out = base_model(input_ids=batch["input_ids"],
                     attention_mask=batch["attention_mask"],
                     output_hidden_states=True, use_cache=False)
    hs = out.hidden_states                      # [emb, layer0, layer1, ...]
    att = batch["attention_mask"]
    rows = []
    for b in range(att.shape[0]):
        valid = att[b].bool()
        if use_labels:
            pv = valid & (labels[b].to(valid.device) == -100)
            if bool(pv.any()):
                valid = pv
        idx = torch.nonzero(valid, as_tuple=False).flatten()
        idx = idx[-n_pos:] if idx.numel() > n_pos else idx
        per_layer = [hs[li + 1][b, idx, :].float().mean(dim=0) for li in layer_ids]
        rows.append(torch.cat(per_layer, dim=0))
    return torch.stack(rows, dim=0)


def _collate(examples, tokenizer):
    from transformers import DataCollatorForSeq2Seq
    from pyreft import ReftDataCollator
    collator = ReftDataCollator(data_collator=DataCollatorForSeq2Seq(
        tokenizer=tokenizer, model=None, label_pad_token_id=-100,
        padding="longest"))
    return collator(examples)


@torch.no_grad()
def features_from_dataset(base_model, dataset, tokenizer, layer_ids: List[int],
                          n_pos: int, n_examples: int, device,
                          seed: int = 0, batch_size: int = 8,
                          prompt_only: bool = False) -> torch.Tensor:
    """Features for n_examples drawn without replacement (seeded) from the
    dataset. Returns (n, dim) float16 on CPU."""
    n = min(int(n_examples), len(dataset))
    g = torch.Generator().manual_seed(int(seed))
    order = torch.randperm(len(dataset), generator=g)[:n].tolist()
    was_training = base_model.training
    base_model.eval()
    feats = []
    try:
        for s in range(0, n, batch_size):
            ex = [{k: v for k, v in dataset[i].items() if k in _KEEP}
                  for i in order[s:s + batch_size]]
            batch = _collate(ex, tokenizer)
            batch = {k: (v.to(device) if torch.is_tensor(v) else v)
                     for k, v in batch.items()}
            feats.append(features_from_batch(base_model, batch, layer_ids,
                                             n_pos, prompt_only=prompt_only
                                             ).to(torch.float16).cpu())
    finally:
        if was_training:
            base_model.train()
    return torch.cat(feats, dim=0)


# ──────────────────────────────────────────────────────────────────────────
# the head
# ──────────────────────────────────────────────────────────────────────────
class RouterHead:
    def __init__(self, layer_ids: Optional[List[int]] = None,
                 n_pos: int = 15, l2: float = 1e-2, iters: int = 200,
                 target: str = "task", prompt_only: bool = False):
        assert target in ("task", "group"), f"target must be task|group, got {target!r}"
        self.layer_ids: List[int] = list(layer_ids or [])
        self.n_pos = int(n_pos)
        self.l2 = float(l2)
        self.iters = int(iters)
        # "task": classes are task names (original behaviour).
        # "group": classes are group ids as strings ("0", "1", ...); the bank
        # is still keyed by task, and tasks are pooled by group at fit time
        # using the task_to_group mapping passed to fit(). Mapping at fit
        # time means a later eviction/merge is picked up by the next refit.
        self.target = target
        # Pool prompt tokens only when building the bank; saved with the head.
        self.prompt_only = bool(prompt_only)
        self.task_to_group: Dict[str, int] = {}
        self.banks: Dict[str, torch.Tensor] = {}   # task -> (n, dim) fp16 cpu
        self.classes: List[str] = []
        self.W: Optional[torch.Tensor] = None      # (T, dim) fp32 cpu
        self.b: Optional[torch.Tensor] = None      # (T,)
        self.mu: Optional[torch.Tensor] = None     # (dim,)
        self.sd: Optional[torch.Tensor] = None     # (dim,)
        self.fit_log: List[dict] = []

    # -- stream-time API ---------------------------------------------------
    def add_task(self, task_name: str, feats: torch.Tensor) -> None:
        self.banks[str(task_name)] = feats.to(torch.float16).cpu()

    def _label_banks(self) -> Dict[str, torch.Tensor]:
        """Banks keyed by classifier label. In task mode this is the task
        bank itself; in group mode every task's bank is concatenated under
        its group id."""
        if self.target == "task":
            return dict(self.banks)
        pooled: Dict[str, List[torch.Tensor]] = {}
        for t, bank in self.banks.items():
            if t not in self.task_to_group:
                raise KeyError(f"router_head(group): no group for task {t!r}; "
                               f"pass task_to_group to fit()")
            pooled.setdefault(str(int(self.task_to_group[t])), []).append(bank)
        return {g: torch.cat(v, dim=0) for g, v in pooled.items()}

    def fit(self, device=None, holdout: int = 4,
            task_to_group: Optional[Dict[str, int]] = None) -> dict:
        """Refit on every stored bank. `holdout` examples per class are kept
        out of the fit and reported as a sanity accuracy (the number the
        stream log should show climbing to 1.0). They are put back for the
        final fit so nothing is wasted. In group mode `task_to_group` must be
        given (or have been given on a previous fit)."""
        if task_to_group is not None:
            self.task_to_group = {str(k): int(v) for k, v in task_to_group.items()}
        banks = self._label_banks()
        self.classes = sorted(banks, key=(int if self.target == "group" else str))
        dev = device if device is not None else (
            "cuda" if torch.cuda.is_available() else "cpu")

        def _stack(use_holdout: bool):
            xs, ys = [], []
            for ci, t in enumerate(self.classes):
                bank = banks[t]
                if use_holdout and bank.shape[0] > holdout:
                    tr, ho = bank[holdout:], bank[:holdout]
                else:
                    tr, ho = bank, bank[:0]
                xs.append((tr, ho)); ys.append(ci)
            return xs, ys

        def _solve(X, y):
            mu = X.mean(0); sd = X.std(0) + 1e-3
            Xn = (X - mu) / sd
            T_, D = len(self.classes), X.shape[1]
            W = torch.zeros(T_, D, device=dev, requires_grad=True)
            b = torch.zeros(T_, device=dev, requires_grad=True)
            opt = torch.optim.LBFGS([W, b], lr=1.0, max_iter=self.iters,
                                    history_size=20, line_search_fn="strong_wolfe")
            def closure():
                opt.zero_grad()
                logits = Xn @ W.t() + b
                loss = F.cross_entropy(logits, y) + self.l2 * (W ** 2).sum() / T_
                loss.backward()
                return loss
            opt.step(closure)
            return W.detach(), b.detach(), mu, sd

        # 1. holdout fit for the sanity number
        xs, ys = _stack(True)
        ho_acc = None
        if all(h.shape[0] > 0 for _, h in xs):
            X = torch.cat([tr for tr, _ in xs]).float().to(dev)
            y = torch.cat([torch.full((tr.shape[0],), c) for (tr, _), c in zip(xs, ys)]).to(dev)
            W, b, mu, sd = _solve(X, y)
            Xh = torch.cat([h for _, h in xs]).float().to(dev)
            yh = torch.cat([torch.full((h.shape[0],), c) for (_, h), c in zip(xs, ys)]).to(dev)
            pred = (((Xh - mu) / sd) @ W.t() + b).argmax(1)
            ho_acc = float((pred == yh).float().mean())

        # 2. final fit on everything
        X = torch.cat([banks[t] for t in self.classes]).float().to(dev)
        y = torch.cat([torch.full((banks[t].shape[0],), ci)
                       for ci, t in enumerate(self.classes)]).to(dev)
        W, b, mu, sd = _solve(X, y)
        train_acc = float(((((X - mu) / sd) @ W.t() + b).argmax(1) == y).float().mean())
        self.W, self.b, self.mu, self.sd = W.cpu(), b.cpu(), mu.cpu(), sd.cpu()
        rec = {"target": self.target,
               "n_classes": len(self.classes),
               "n_tasks": len(self.banks),
               "n_examples": int(X.shape[0]),
               "train_acc": round(train_acc, 4),
               "holdout_acc": (round(ho_acc, 4) if ho_acc is not None else None)}
        self.fit_log.append(rec)
        return rec

    # -- inference API -----------------------------------------------------
    @torch.no_grad()
    def predict_proba(self, feats: torch.Tensor) -> torch.Tensor:
        """(n, T) softmax probabilities on CPU."""
        assert self.W is not None, "RouterHead is not fitted"
        X = (feats.float().cpu() - self.mu) / self.sd
        return torch.softmax(X @ self.W.t() + self.b, dim=-1)

    @torch.no_grad()
    def predict(self, feats: torch.Tensor):
        """Batch decision: mean probability over the examples.
        Returns (label, confidence, {label: prob}). The label is a task name
        in task mode and a group-id string in group mode."""
        p = self.predict_proba(feats).mean(0)
        i = int(p.argmax())
        return self.classes[i], float(p[i]), {t: round(float(p[j]), 4)
                                              for j, t in enumerate(self.classes)}

    @torch.no_grad()
    def predict_each(self, feats: torch.Tensor):
        """Per-example decisions, no pooling across the batch. Returns a list
        of (label, confidence), one per row of `feats`. This is the routing
        rule for a mixed-task query stream."""
        P = self.predict_proba(feats)
        out = []
        for row in P:
            i = int(row.argmax())
            out.append((self.classes[i], float(row[i])))
        return out

    def label_to_gid(self, label: str) -> Optional[int]:
        """Map a predicted label to a group id in either mode."""
        if self.target == "group":
            return int(label)
        g = self.task_to_group.get(str(label))
        return int(g) if g is not None else None

    # -- persistence -------------------------------------------------------
    def save(self, path: str) -> None:
        torch.save({
            "layer_ids": self.layer_ids, "n_pos": self.n_pos, "l2": self.l2,
            "iters": self.iters, "classes": self.classes,
            "target": self.target, "task_to_group": self.task_to_group,
            "prompt_only": self.prompt_only, "banks": self.banks, "W": self.W, "b": self.b,
            "mu": self.mu, "sd": self.sd, "fit_log": self.fit_log,
        }, path)
        with open(os.path.splitext(path)[0] + ".json", "w") as f:
            json.dump({"classes": self.classes, "layer_ids": self.layer_ids,
                       "n_pos": self.n_pos, "target": self.target,
                       "prompt_only": self.prompt_only,
                       "task_to_group": self.task_to_group,
                       "fit_log": self.fit_log,
                       "bank_sizes": {t: int(v.shape[0]) for t, v in self.banks.items()}},
                      f, indent=2)

    @classmethod
    def load(cls, path: str) -> "RouterHead":
        d = torch.load(path, map_location="cpu", weights_only=False)
        h = cls(d["layer_ids"], d["n_pos"], d["l2"], d["iters"],
                target=d.get("target", "task"),
                prompt_only=bool(d.get("prompt_only", False)))
        h.task_to_group = {str(k): int(v)
                           for k, v in d.get("task_to_group", {}).items()}
        h.classes = d["classes"]; h.banks = d["banks"]
        h.W, h.b, h.mu, h.sd = d["W"], d["b"], d["mu"], d["sd"]
        h.fit_log = d.get("fit_log", [])
        return h