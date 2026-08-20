"""
agnostic_router.py — task-agnostic inference-time group routing for CRAFT-CL.

Selects which group's intervention state to load for an eval batch WITHOUT
using the task name. Three methods, plus 'oracle' (the existing
task_to_group lookup, handled by the caller):

  subspace_energy : one frozen-base forward; route to the group whose
                    Stiefel rotation R captures the most hidden-state
                    energy at the intervention positions.  (Option E)
  perplexity      : forward the probe through each group's intervened
                    model; route to the lowest prompt next-token loss.
                    (Option C; K forwards)
  prototype       : cosine similarity between the probe's mean frozen
                    hidden state and per-group prototypes stored during
                    training.  Requires attach_group_prototype() to have
                    been called after each task commit.  (Option B)

Every decision is logged; dump_report() writes agnostic_routing_log.json
with per-decision records and the running hit rate vs the oracle table.
All routers are wrapped: on ANY internal error the caller falls back to
oracle loading, so routing experiments can never crash a run.
"""

import json
import os
import re
import traceback
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from transformers import DataCollatorForSeq2Seq
from pyreft import ReftDataCollator

VALID_METHODS = ("oracle", "subspace_energy", "perplexity", "prototype")

_PROBE_KEEP = ("input_ids", "attention_mask", "labels",
               "intervention_locations", "subspaces")

_LOG: List[dict] = []


# ──────────────────────────────────────────────────────────────────────────
# probe construction (mirrors train.py's whitelisted probe builders)
# ──────────────────────────────────────────────────────────────────────────
def _build_probe(dataset, tokenizer, probe_size: int, device) -> Dict:
    n = min(int(probe_size), len(dataset))
    examples = [{k: v for k, v in dataset[i].items() if k in _PROBE_KEEP}
                for i in range(n)]
    collator = ReftDataCollator(data_collator=DataCollatorForSeq2Seq(
        tokenizer=tokenizer, model=None,
        label_pad_token_id=-100, padding="longest"))
    batch = collator(examples)
    return {k: (v.to(device) if torch.is_tensor(v) else v)
            for k, v in batch.items()}


def _fl_positions(attention_mask: torch.Tensor, n_pos: int) -> List[torch.Tensor]:
    """Per-row indices of the first n_pos + last n_pos valid tokens
    (padding-side agnostic)."""
    out = []
    for row in attention_mask:
        idx = torch.nonzero(row, as_tuple=False).flatten()
        if idx.numel() <= 2 * n_pos:
            out.append(idx)
        else:
            out.append(torch.cat([idx[:n_pos], idx[-n_pos:]]))
    return out


def _layer_of_iv_key(key: str) -> Optional[int]:
    m = re.search(r"layer[._](\d+)", key)
    return int(m.group(1)) if m else None


def _iv_module(iv):
    return iv[0] if isinstance(iv, (list, tuple)) else iv


# ──────────────────────────────────────────────────────────────────────────
# shared: base hidden states at intervention layers/positions
# ──────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def _base_hidden_at_positions(base_model, batch, layer_ids: List[int],
                              n_pos: int) -> Dict[int, torch.Tensor]:
    """Returns {layer_idx: (N_tok, hidden) float32} of frozen-base hidden
    states gathered at the f{n}+l{n} positions across the probe batch."""
    out = base_model(input_ids=batch["input_ids"],
                     attention_mask=batch["attention_mask"],
                     output_hidden_states=True, use_cache=False)
    hs = out.hidden_states  # tuple: [emb, layer0_out, layer1_out, ...]
    pos = _fl_positions(batch["attention_mask"], n_pos)
    gathered: Dict[int, torch.Tensor] = {}
    for li in layer_ids:
        h = hs[li + 1]                      # block_output of layer li
        rows = [h[b, p, :] for b, p in enumerate(pos)]
        gathered[li] = torch.cat(rows, dim=0).float()
    return gathered


# ──────────────────────────────────────────────────────────────────────────
# per-group rotation matrices (cached on the group dict)
# ──────────────────────────────────────────────────────────────────────────
def _group_rotations(memory, gid: int, build_reft_fn, teardown_fn
                     ) -> List[Tuple[int, torch.Tensor]]:
    """[(layer_idx, R float32 cpu (hidden, rank))].  Built once per group by
    loading the stored state into a fresh shell (so pyreft's orthogonal
    parametrization re-materializes the true R), then cached."""
    g = memory.groups[gid]
    if "_agn_R" in g:
        return g["_agn_R"]
    shell = build_reft_fn()
    try:
        for key, iv in shell.interventions.items():
            mod = _iv_module(iv)
            if key in g["iv_state"]:
                mod.load_state_dict(g["iv_state"][key])
        rots = []
        for key, iv in shell.interventions.items():
            li = _layer_of_iv_key(key)
            if li is None:
                continue
            rl = getattr(_iv_module(iv), "rotate_layer", None)
            W = getattr(rl, "weight", None) if rl is not None else None
            if W is None:
                continue
            rots.append((li, W.detach().float().cpu()))
    finally:
        teardown_fn(shell)
        del shell
    g["_agn_R"] = rots
    return rots


# ──────────────────────────────────────────────────────────────────────────
# method 1: subspace energy  (one base forward total)
# ──────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def _route_subspace_energy(memory, batch, base_model, args,
                           build_reft_fn, teardown_fn) -> Tuple[int, Dict]:
    gids = sorted(memory.groups.keys())
    rot = {gid: _group_rotations(memory, gid, build_reft_fn, teardown_fn)
           for gid in gids}
    layer_ids = sorted({li for rs in rot.values() for li, _ in rs})
    n_pos = int(getattr(args, "n_positions", 15))
    hidden = _base_hidden_at_positions(base_model, batch, layer_ids, n_pos)
    scores = {}
    for gid in gids:
        num, den = 0.0, 0.0
        for li, R in rot[gid]:
            h = hidden[li]                       # (N, hidden)
            proj = h @ R.to(h.device)            # (N, rank)
            num += proj.pow(2).sum().item()
            den += h.pow(2).sum().item()
        scores[gid] = (num / den) if den > 0 else 0.0
    pred = max(scores, key=scores.get)
    return pred, {"scores": {f"G{k}": round(v, 6) for k, v in scores.items()}}


# ──────────────────────────────────────────────────────────────────────────
# method 2: prompt perplexity  (K intervened forwards)
# ──────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def _route_perplexity(memory, batch, args,
                      build_reft_fn, teardown_fn) -> Tuple[int, Dict]:
    iv_locs = batch.get("intervention_locations", None)
    unit_locations = None
    if iv_locs is not None:
        unit_locations = {"sources->base": (
            None, iv_locs.permute(1, 0, 2).tolist()
            if hasattr(iv_locs, "permute") else iv_locs)}
    base = {"input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
            "use_cache": False}
    ids, att = batch["input_ids"], batch["attention_mask"]
    tgt   = ids[:, 1:]
    tmask = (att[:, 1:] * att[:, :-1]).bool()
    scores = {}
    for gid in sorted(memory.groups.keys()):
        shell = build_reft_fn()
        try:
            loaded = memory.load(shell, gid=gid)
            if not loaded:
                scores[gid] = float("inf")
                continue
            shell.eval()
            _, out = shell(base=dict(base), unit_locations=unit_locations)
            logits = (out.logits if hasattr(out, "logits") else out[0])
            lp = F.log_softmax(logits[:, :-1, :].float(), dim=-1)
            nll = -lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
            scores[gid] = (nll[tmask].mean()).item()
        finally:
            teardown_fn(shell)
            del shell
    pred = min(scores, key=scores.get)
    return pred, {"scores": {f"G{k}": round(v, 6) for k, v in scores.items()}}


# ──────────────────────────────────────────────────────────────────────────
# method 3: prototype matching  (one base forward; needs training-time hook)
# ──────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def attach_group_prototype(memory, gid: int, base_model, train_dataset,
                           tokenizer, args, probe_size: int = 16) -> None:
    """Call once after a task is committed to group `gid`.  Uses ONLY the
    current task's train data (no rehearsal).  Running mean per group."""
    try:
        device = next(base_model.parameters()).device
        batch = _build_probe(train_dataset, tokenizer, probe_size, device)
        layer_ids = sorted({li for li in
                            (_layer_of_iv_key(k) for k in
                             memory.groups[gid]["iv_state"].keys())
                            if li is not None})
        n_pos = int(getattr(args, "n_positions", 15))
        hidden = _base_hidden_at_positions(base_model, batch, layer_ids, n_pos)
        vec = torch.cat([hidden[li].mean(dim=0) for li in layer_ids]).cpu()
        g = memory.groups[gid]
        n = int(g.get("_agn_proto_n", 0))
        if n == 0 or "_agn_proto" not in g:
            g["_agn_proto"] = vec
        else:
            g["_agn_proto"] = (g["_agn_proto"] * n + vec) / (n + 1)
        g["_agn_proto_n"] = n + 1
        print(f"  [agnostic] prototype attached → G{gid} "
              f"(n_tasks_in_proto={n + 1}, dim={vec.numel()})")
    except Exception:
        print("  [agnostic] prototype attach FAILED (non-fatal):")
        traceback.print_exc()


@torch.no_grad()
def _route_prototype(memory, batch, base_model, args) -> Tuple[int, Dict]:
    gids = [g for g in sorted(memory.groups.keys())
            if "_agn_proto" in memory.groups[g]]
    if not gids:
        raise RuntimeError("no prototypes stored; was attach_group_prototype "
                           "called during training?")
    layer_ids = sorted({li for gid in gids for li in
                        (_layer_of_iv_key(k) for k in
                         memory.groups[gid]["iv_state"].keys())
                        if li is not None})
    n_pos = int(getattr(args, "n_positions", 15))
    hidden = _base_hidden_at_positions(base_model, batch, layer_ids, n_pos)
    vec = torch.cat([hidden[li].mean(dim=0) for li in layer_ids]).cpu()
    scores = {gid: F.cosine_similarity(
        vec, memory.groups[gid]["_agn_proto"], dim=0).item() for gid in gids}
    pred = max(scores, key=scores.get)
    return pred, {"scores": {f"G{k}": round(v, 6) for k, v in scores.items()}}


# ──────────────────────────────────────────────────────────────────────────
# entry point used by train.py's eval loop
# ──────────────────────────────────────────────────────────────────────────
def route_and_load(eval_reft, eval_task: str, memory, base_model, tokenizer,
                   args, ev_dataset, build_reft_fn, teardown_fn) -> bool:
    """Task-agnostic replacement for memory.load(eval_reft, task_name=...).
    Decides the group from the eval data, loads it into eval_reft, logs the
    decision vs the oracle table.  Falls back to oracle on any error."""
    method = getattr(args, "inference_router", "oracle")
    oracle_gid = memory.task_to_group.get(eval_task, None)

    if method == "oracle" or not memory.groups:
        return memory.load(eval_reft, task_name=eval_task)

    probe_size = int(getattr(args, "router_probe_size", 16))
    entry = {"task": eval_task, "method": method,
             "oracle_gid": oracle_gid, "probe_size": probe_size}
    try:
        device = next(base_model.parameters()).device
        batch = _build_probe(ev_dataset, tokenizer, probe_size, device)
        if method == "subspace_energy":
            pred, dbg = _route_subspace_energy(
                memory, batch, base_model, args, build_reft_fn, teardown_fn)
        elif method == "perplexity":
            pred, dbg = _route_perplexity(
                memory, batch, args, build_reft_fn, teardown_fn)
        elif method == "prototype":
            pred, dbg = _route_prototype(memory, batch, base_model, args)
        else:
            raise ValueError(f"unknown inference_router={method!r}")
        entry.update(dbg)
        entry["pred_gid"] = int(pred)
        entry["hit"] = (oracle_gid is not None and int(pred) == int(oracle_gid))
        tag = "HIT " if entry["hit"] else "MISS"
        print(f"  [agnostic:{method}] {eval_task:<15} pred=G{pred} "
              f"oracle=G{oracle_gid}  {tag} {entry.get('scores', '')}")
        _LOG.append(entry)
        return memory.load(eval_reft, gid=int(pred))
    except Exception as e:
        entry["error"] = f"{type(e).__name__}: {e}"
        entry["fallback"] = "oracle"
        _LOG.append(entry)
        print(f"  [agnostic:{method}] FAILED for '{eval_task}' → oracle "
              f"fallback  ({entry['error']})")
        traceback.print_exc()
        return memory.load(eval_reft, task_name=eval_task)


def dump_report(output_dir: str) -> None:
    """Write agnostic_routing_log.json (idempotent; call after each task)."""
    if not _LOG:
        return
    decided = [e for e in _LOG if "pred_gid" in e]
    hits = sum(1 for e in decided if e.get("hit"))
    report = {
        "n_decisions": len(decided),
        "n_errors": len(_LOG) - len(decided),
        "hit_rate": (hits / len(decided)) if decided else None,
        "decisions": _LOG,
    }
    path = os.path.join(output_dir, "agnostic_routing_log.json")
    try:
        with open(path, "w") as f:
            json.dump(report, f, indent=2)
        if decided:
            print(f"  [agnostic] routing hit-rate so far: "
                  f"{hits}/{len(decided)}  → {path}")
    except OSError:
        traceback.print_exc()
