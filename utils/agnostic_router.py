"""
agnostic_router.py — inference-time group routing for CRAFT-CL.

Loads the intervention state for an eval batch. Methods supported:

  oracle        : ground-truth task_to_group lookup (task label known)
  learned_head  : trainable task-identity head (router_head.py) on frozen-base
                  features; loads the predicted task's group + private T3-l

The third router, litar (task identification through the learned group
interventions), lives in utils/litar_router.py.
"""

import json
import os
import traceback
from typing import Dict, List

import torch
from transformers import DataCollatorForSeq2Seq
from pyreft import ReftDataCollator
from utils.router_head import features_from_batch as _head_features

VALID_METHODS = (
    "oracle",
    "learned_head",   # trainable task-identity head (router_head.py); loads group + private T3-l
)

_PROBE_KEEP = ("input_ids", "attention_mask", "labels",
               "intervention_locations", "subspaces")

_LOG: List[dict] = []

# ──────────────────────────────────────────────────────────────────────────
# probe construction
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


def _strict() -> bool:
    """ROUTER_STRICT=1: a router that cannot decide raises instead of loading
    the oracle state, so task-agnostic scores never include oracle rescues."""
    return os.environ.get("ROUTER_STRICT", "") in ("1", "true", "True")


# ──────────────────────────────────────────────────────────────────────────
# entry point used by train.py
# ──────────────────────────────────────────────────────────────────────────
def route_and_load(eval_reft, eval_task: str, memory, base_model, tokenizer,
                   args, ev_dataset, build_reft_fn, teardown_fn) -> bool:
    """oracle: memory.load(eval_reft, task_name=...). learned_head: decides the
    task from the eval data, loads its group (and private T3-l) into eval_reft,
    and logs the decision against the oracle table. Without ROUTER_STRICT=1 an
    error falls back to oracle."""
    method = getattr(args, "inference_router", "oracle")
    oracle_gid = memory.task_to_group.get(eval_task, None)

    if method == "oracle" or not memory.groups:
        return memory.load(eval_reft, task_name=eval_task)

    probe_size = int(getattr(args, "router_probe_size", 16))
    entry = {"task": eval_task, "method": method,
             "oracle_gid": oracle_gid, "probe_size": probe_size}

    # Trainable task-identity head (router_head.py). Predicts the TASK from
    # frozen-base features, so it selects the group AND the private T3-l.
    if method == "learned_head":
        head = getattr(memory, "router_head", None)
        if head is None or getattr(head, "W", None) is None:
            entry["error"] = "router_head missing or unfitted (train with --router_head)"
            if _strict():
                raise RuntimeError(f"[agnostic:learned_head] {eval_task}: {entry['error']}")
            entry["fallback"] = "oracle"
            _LOG.append(entry)
            print(f"  [agnostic:learned_head] {eval_task:<15} {entry['error']} -> oracle")
            return memory.load(eval_reft, task_name=eval_task)
        try:
            device = next(base_model.parameters()).device
            batch = _build_probe(ev_dataset, tokenizer, probe_size, device)
            feats = _head_features(base_model, batch, head.layer_ids, head.n_pos,
                                   prompt_only=getattr(head, "prompt_only", False))
            pred_task, conf, probs = head.predict(feats)
            min_conf = float(os.environ.get("ROUTER_HEAD_MIN_CONF", "0.0"))
            pred_gid = memory.task_to_group.get(pred_task, None)
            entry.update({"scores": probs, "pred_task": pred_task,
                          "confidence": round(conf, 4),
                          "task_hit": (pred_task == eval_task),
                          "pred_gid": (int(pred_gid) if pred_gid is not None else None),
                          "hit": (pred_gid is not None and oracle_gid is not None
                                  and int(pred_gid) == int(oracle_gid))})
            if pred_gid is None:
                raise KeyError(f"predicted task {pred_task!r} has no group")
            if conf >= min_conf and pred_task in getattr(memory, "t3l_tasks", {}):
                success = memory.load(eval_reft, task_name=pred_task)
                entry["t3l_loaded"] = bool(success); how = "group + private T3-l"
            else:
                success = memory.load(eval_reft, gid=int(pred_gid))
                entry["t3l_loaded"] = False
                how = ("group only (conf below ROUTER_HEAD_MIN_CONF)"
                       if conf < min_conf else "group only (no private state)")
            _LOG.append(entry)
            tag = ("HIT " if entry["hit"] else "MISS")
            print(f"  [agnostic:learned_head] {eval_task:<15} pred={pred_task} "
                  f"(G{pred_gid}, conf={conf:.3f}) oracle=G{oracle_gid}  {tag}  {how}")
            return success
        except Exception as e:
            if _strict():
                raise
            entry["error"] = f"{type(e).__name__}: {e}"; entry["fallback"] = "oracle"
            _LOG.append(entry)
            print(f"  [agnostic:learned_head] FAILED for '{eval_task}' -> oracle ({entry['error']})")
            traceback.print_exc()
            return memory.load(eval_reft, task_name=eval_task)

    raise ValueError(f"unknown inference_router={method!r} "
                     f"(choose from {VALID_METHODS} or litar)")


def dump_report(output_dir: str) -> None:
    """Write agnostic_routing_log.json (idempotent; call after each task)."""
    if not _LOG:
        return
    decided = [e for e in _LOG if "pred_gid" in e]
    hits = sum(1 for e in decided if e.get("hit"))
    confusion: Dict[str, Dict[str, int]] = {}
    per_task: Dict[str, dict] = {}
    for e in decided:
        o, p = f"G{e.get('oracle_gid')}", f"G{e.get('pred_gid')}"
        confusion.setdefault(o, {})
        confusion[o][p] = confusion[o].get(p, 0) + 1
        per_task[e["task"]] = {"oracle": o, "pred": p, "hit": bool(e.get("hit"))}
    task_decided = [e for e in decided if "task_hit" in e]
    task_hits = sum(1 for e in task_decided if e.get("task_hit"))
    report = {
        "method": decided[0].get("method") if decided else None,
        "n_decisions": len(decided),
        "n_errors": len(_LOG) - len(decided),
        "hit_rate": (hits / len(decided)) if decided else None,
        "task_hit_rate": ((task_hits / len(task_decided))
                          if task_decided else None),
        "t3l_loaded": sum(1 for e in decided if e.get("t3l_loaded")),
        "confusion_oracle_to_pred": confusion,
        "per_task": per_task,
        "missed_tasks": [t for t, v in per_task.items() if not v["hit"]],
        "decisions": _LOG,
    }
    path = os.path.join(output_dir, "agnostic_routing_log.json")
    try:
        with open(path, "w") as f:
            json.dump(report, f, indent=2)
        if decided:
            miss = report["missed_tasks"]
            print(f"  [agnostic] routing hit-rate so far: "
                  f"{hits}/{len(decided)}"
                  + (f"   missed: {', '.join(miss)}" if miss else "")
                  + f"   → {path}")
    except OSError:
        traceback.print_exc()
