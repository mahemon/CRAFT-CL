"""
merge_rules.py — group-state merge rules for the CRAFT-CL merge ablation.

Four rules, selected by --merge_rule:

  overwrite : g["iv_state"] <- trained state          (paper default, unchanged)
  ema       : g["iv_state"] <- (1-a)*old + a*trained  (a = --merge_ema_alpha)
  fisher    : g["iv_state"] <- (F_old*old + F_new*new) / (F_old + F_new)
              per-parameter diagonal Fisher; F accumulated across merges
  prototype : group keeps one prototype per member task;
              g["iv_state"] <- uniform mean over all prototypes
              (task-balanced merge; removes last-writer-wins recency bias)

All rules operate on the raw (unconstrained) state-dict tensors, matching the
existing overwrite transfer path: the Stiefel parametrization on R is
reapplied on the next forward pass, so interpolated states remain valid
inputs to the parametrization exactly as overwritten states do.
"""

import copy
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F


def _clone(d: dict) -> dict:
    return {k: (v.detach().clone() if torch.is_tensor(v) else copy.deepcopy(v))
            for k, v in d.items()}


def merge_states_ema(old: Dict[str, dict], new: Dict[str, dict],
                     alpha: float) -> Dict[str, dict]:
    """(1-alpha)*old + alpha*new, per tensor. Keys absent on one side pass
    through from the side that has them (new wins on both-present non-tensors)."""
    out: Dict[str, dict] = {}
    for iv_key in set(old) | set(new):
        if iv_key not in old:
            out[iv_key] = _clone(new[iv_key]); continue
        if iv_key not in new:
            out[iv_key] = _clone(old[iv_key]); continue
        o, n = old[iv_key], new[iv_key]
        merged = {}
        for p in set(o) | set(n):
            if p in o and p in n and torch.is_tensor(o[p]) and torch.is_tensor(n[p]) \
                    and o[p].shape == n[p].shape:
                merged[p] = ((1.0 - alpha) * o[p].float()
                             + alpha * n[p].float()).to(n[p].dtype)
            else:
                src = n if p in n else o
                merged[p] = (src[p].detach().clone()
                             if torch.is_tensor(src[p]) else copy.deepcopy(src[p]))
        out[iv_key] = merged
    return out


def merge_states_mean(states: List[Dict[str, dict]]) -> Dict[str, dict]:
    """Uniform average over a list of iv_states (prototype rule)."""
    assert states, "merge_states_mean called with empty prototype list"
    if len(states) == 1:
        return {k: _clone(v) for k, v in states[0].items()}
    out: Dict[str, dict] = {}
    iv_keys = set().union(*[set(s.keys()) for s in states])
    for iv_key in iv_keys:
        present = [s[iv_key] for s in states if iv_key in s]
        merged = {}
        for p in present[0]:
            vals = [s[p] for s in present
                    if p in s and torch.is_tensor(s[p])
                    and s[p].shape == present[0][p].shape]
            if torch.is_tensor(present[0][p]) and len(vals) == len(present):
                merged[p] = torch.stack([v.float() for v in vals]) \
                                 .mean(dim=0).to(present[0][p].dtype)
            else:
                merged[p] = (present[-1][p].detach().clone()
                             if torch.is_tensor(present[-1][p])
                             else copy.deepcopy(present[-1][p]))
        out[iv_key] = merged
    return out


def merge_states_fisher(old: Dict[str, dict],
                        fisher_old: Optional[Dict[str, dict]],
                        new: Dict[str, dict],
                        fisher_new: Optional[Dict[str, dict]],
                        floor: float = 1e-12):
    """Fisher-weighted per-parameter merge. Parameters lacking Fisher on
    either side fall back to the new value (overwrite). Returns
    (merged_state, accumulated_fisher)."""
    if fisher_old is None or fisher_new is None:
        # Cannot weight without both sides; degrade gracefully to overwrite
        # but still propagate whatever Fisher exists for the next merge.
        acc = fisher_new if fisher_new is not None else fisher_old
        return {k: _clone(v) for k, v in new.items()}, acc

    out: Dict[str, dict] = {}
    acc: Dict[str, dict] = {}
    for iv_key in set(old) | set(new):
        if iv_key not in old:
            out[iv_key] = _clone(new[iv_key])
            acc[iv_key] = _clone(fisher_new.get(iv_key, {}))
            continue
        if iv_key not in new:
            out[iv_key] = _clone(old[iv_key])
            acc[iv_key] = _clone(fisher_old.get(iv_key, {}))
            continue
        o, n = old[iv_key], new[iv_key]
        fo = fisher_old.get(iv_key, {})
        fn = fisher_new.get(iv_key, {})
        merged, facc = {}, {}
        for p in set(o) | set(n):
            has_all = (p in o and p in n and p in fo and p in fn
                       and torch.is_tensor(o[p]) and torch.is_tensor(n[p])
                       and o[p].shape == n[p].shape
                       and fo[p].shape == o[p].shape
                       and fn[p].shape == n[p].shape)
            if has_all:
                w_o = fo[p].float() + floor
                w_n = fn[p].float() + floor
                merged[p] = ((w_o * o[p].float() + w_n * n[p].float())
                             / (w_o + w_n)).to(n[p].dtype)
                facc[p] = (fo[p].float() + fn[p].float())
            else:
                src = n if p in n else o
                merged[p] = (src[p].detach().clone()
                             if torch.is_tensor(src[p]) else copy.deepcopy(src[p]))
                if p in fn:
                    facc[p] = fn[p].detach().clone()
                elif p in fo:
                    facc[p] = fo[p].detach().clone()
        out[iv_key] = merged
        acc[iv_key] = facc
    return out, acc


@torch.enable_grad()
def estimate_fisher_diag(reft_model, train_dataset, tokenizer,
                         n_batches: int = 8, batch_size: int = 4,
                         device: str = "cuda") -> Dict[str, dict]:
    """Diagonal empirical Fisher of the task CE loss w.r.t. intervention
    parameters: mean over batches of grad^2. Keyed to match iv_state:
    {iv_key: {param_name: tensor}}, where param_name follows
    named_parameters() and therefore aligns with state_dict() keys under
    pyreft's parametrization."""
    import gc
    from transformers import DataCollatorForSeq2Seq
    from pyreft import ReftDataCollator

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    keep = ("input_ids", "attention_mask", "labels",
            "intervention_locations", "subspaces")
    n_total = min(len(train_dataset), n_batches * batch_size)
    batch_size = 1   # micro-batch: backward peak scales with batch x seq
    examples = [{k: v for k, v in train_dataset[i].items() if k in keep}
                for i in range(n_total)]
    collator = ReftDataCollator(data_collator=DataCollatorForSeq2Seq(
        tokenizer=tokenizer, model=None,
        label_pad_token_id=-100, padding="longest"))

    # Collect intervention parameters, grouped by iv_key
    iv_params: Dict[str, Dict[str, torch.nn.Parameter]] = {}
    for iv_key, iv_list in reft_model.interventions.items():
        mod = iv_list[0] if isinstance(iv_list, (list, tuple)) else iv_list
        iv_params[iv_key] = dict(mod.named_parameters())

    fisher: Dict[str, dict] = {
        iv_key: {p: torch.zeros_like(t, dtype=torch.float32)
                 for p, t in params.items()}
        for iv_key, params in iv_params.items()
    }

    was_training = reft_model.training
    reft_model.eval()  # eval mode but grads enabled on iv params
    n_used = 0
    for start in range(0, n_total, batch_size):
        batch = collator(examples[start:start + batch_size])
        batch = {k: (v.to(device) if torch.is_tensor(v) else v)
                 for k, v in batch.items()}
        b = dict(batch)
        iv_locs = b.pop("intervention_locations", None)
        b.pop("subspaces", None)
        labels = b.get("labels", None)
        b["use_cache"] = False
        unit_locations = None
        if iv_locs is not None:
            unit_locations = {"sources->base": (
                None, iv_locs.permute(1, 0, 2).tolist()
                if hasattr(iv_locs, "permute") else iv_locs)}

        for params in iv_params.values():
            for t in params.values():
                if t.grad is not None:
                    t.grad = None

        _, out = reft_model(base=b, unit_locations=unit_locations)
        logits = out.logits if hasattr(out, "logits") else out[0]
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous().to(shift_logits.device)
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1), ignore_index=-100)
        loss.backward()

        for iv_key, params in iv_params.items():
            for p, t in params.items():
                if t.grad is not None:
                    fisher[iv_key][p] += t.grad.detach().float() ** 2
        n_used += 1
        del out, logits, shift_logits, shift_labels, loss
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if n_used > 0:
        for params in fisher.values():
            for p in params:
                params[p] /= float(n_used)
    for params in iv_params.values():
        for t in params.values():
            t.grad = None
    if was_training:
        reft_model.train()
    print(f"  [fisher] Diagonal Fisher estimated on {n_used} batches "
          f"({n_total} examples).")
    return fisher