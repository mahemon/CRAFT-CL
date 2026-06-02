import json
import os
from typing import Any, Dict, List, Optional, Set

import numpy as np
import torch
from transformers import TrainerCallback


_param_drift_log:   List[Dict] = []
_grad_traj_log:     List[Dict] = []
_causal_router_log: List[Dict] = []
_fisher_log:        List[Dict] = []
_memory_merge_log:  List[Dict] = []   
_tier_event_log:    List[Dict] = []   
_snapshot_buffer:   Dict[str, Any] = {}


def reset_logs():
    """Clear every in-memory log. Call once at the start of a new run."""
    global _param_drift_log, _grad_traj_log, _causal_router_log
    global _fisher_log, _memory_merge_log, _tier_event_log, _snapshot_buffer
    _param_drift_log   = []
    _grad_traj_log     = []
    _causal_router_log = []
    _fisher_log        = []
    _memory_merge_log  = []
    _tier_event_log    = []
    _snapshot_buffer   = {}

def _to_tensor(p) -> Optional[torch.Tensor]:

    if p is None:
        return None
    if isinstance(p, torch.nn.Linear):
        return p.weight
    if isinstance(p, torch.Tensor):
        return p
    return None


def _get_r(iv) -> Optional[torch.Tensor]:
    try:
        return iv.rotate_layer.parametrizations.weight[0].base
    except AttributeError:
        rl = getattr(iv, 'rotate_layer', None)
        return _to_tensor(getattr(rl, 'weight', None)) if rl is not None else None


def _get_w(iv) -> Optional[torch.Tensor]:
    """W (learned_source / weight) — prefers learned_source."""
    w = _to_tensor(getattr(iv, 'learned_source', None))
    return w if w is not None else _to_tensor(getattr(iv, 'weight', None))


def _get_b(iv) -> Optional[torch.Tensor]:
    """Bias tensor."""
    return _to_tensor(getattr(iv, 'bias', None))


def _get_iv(iv_list):
    """Unwrap (list, tuple) → single intervention module."""
    return iv_list[0] if isinstance(iv_list, (list, tuple)) else iv_list

def _tier_label(
    iv_index:    int,
    task_layers: List[int],
    tier1_set:   Set[int],
    tier2_set:   Set[int],
    tier3_set:   Set[int],
    task_type:   int,
) -> str:

    if iv_index >= len(task_layers):
        return "unknown"
    n_iv    = len(task_layers)
    l_start = (n_iv * 2 // 3) if task_type == 2 else (n_iv // 2)
    layer   = task_layers[iv_index]
    is_l    = iv_index >= l_start

    if layer in tier1_set:
        return "T1"
    if layer in tier2_set:
        return "T2"
    return "T3l" if is_l else "T3f"


def snapshot_params(
    reft_model,
    task_name: str,
    task_idx:  int,
    task_type: int,
    moment:    str,
):

    snap: Dict[str, Dict] = {}
    for key, iv_list in reft_model.interventions.items():
        iv = _get_iv(iv_list)
        r  = _get_r(iv)
        w  = _get_w(iv)
        snap[key] = {
            "R": r.detach().float().cpu().tolist() if r is not None else None,
            "W": w.detach().float().cpu().tolist() if w is not None else None,
        }
    _snapshot_buffer[f"{task_idx}_{task_name}_{moment}"] = snap


def capture_before_flat(reft_model) -> Dict[str, torch.Tensor]:
    """Return {param_key: cpu_float32_tensor} for every intervention parameter."""
    flat: Dict[str, torch.Tensor] = {}
    for key, iv_list in reft_model.interventions.items():
        iv = _get_iv(iv_list)
        for suffix, getter in [("R", _get_r), ("W", _get_w), ("b", _get_b)]:
            t = getter(iv)
            if t is not None:
                flat[f"{key}.{suffix}"] = t.detach().float().cpu().clone()
    return flat

def compute_param_delta(
    before_flat: Dict[str, torch.Tensor],
    reft_model,
    task_name:   str,
    task_idx:    int,
    task_type:   int,
    task_layers: Optional[List[int]] = None,
    tier1_set:   Optional[Set[int]]  = None,
    tier2_set:   Optional[Set[int]]  = None,
    tier3_set:   Optional[Set[int]]  = None,
):

    after_flat = capture_before_flat(reft_model)

    R_d, W_d, b_d = [], [], []
    R_c, W_c      = [], []
    have_tiers = (task_layers is not None and tier1_set is not None
                  and tier2_set is not None and tier3_set is not None)
    tier_acc: Dict[str, Dict[str, list]] = {
        t: {"R_d": [], "W_d": [], "b_d": [], "R_c": [], "W_c": []}
        for t in ("T1", "T2", "T3f", "T3l")
    }
    iv_order = {k: i for i, k in enumerate(reft_model.interventions.keys())}

    for param_key, before in before_flat.items():
        after = after_flat.get(param_key)
        if after is None or before.shape != after.shape:
            continue

        delta_n = float((after.float() - before.float()).norm().item())
        cos     = float(torch.nn.functional.cosine_similarity(
            before.float().reshape(1, -1),
            after.float().reshape(1, -1)).item())

        iv_key = param_key.rsplit(".", 1)[0]
        suffix = param_key.rsplit(".", 1)[1]

        if suffix == "R":
            R_d.append(delta_n); R_c.append(cos)
        elif suffix == "W":
            W_d.append(delta_n); W_c.append(cos)
        elif suffix == "b":
            b_d.append(delta_n)

        if have_tiers and iv_key in iv_order:
            label = _tier_label(iv_order[iv_key], task_layers,
                                 tier1_set, tier2_set, tier3_set, task_type)
            td = tier_acc.get(label, tier_acc["T2"])
            if suffix == "R":
                td["R_d"].append(delta_n); td["R_c"].append(cos)
            elif suffix == "W":
                td["W_d"].append(delta_n); td["W_c"].append(cos)
            elif suffix == "b":
                td["b_d"].append(delta_n)

    def _mean(lst):
        return float(np.mean(lst)) if lst else 0.0
    def _max(lst):
        return float(np.max(lst)) if lst else 0.0

    entry: Dict[str, Any] = {
        "task_name":     task_name,
        "task_idx":      task_idx,
        "task_type":     task_type,
        "R_delta_mean":  _mean(R_d),
        "R_delta_max":   _max(R_d),
        "W_delta_mean":  _mean(W_d),
        "W_delta_max":   _max(W_d),
        "b_delta_mean":  _mean(b_d),
        "R_cosine_mean": _mean(R_c),
        "W_cosine_mean": _mean(W_c),
        "n_R_layers":    len(R_d),
        "n_W_layers":    len(W_d),
    }
    if have_tiers:
        entry["tier_delta"] = {
            tier: {
                "R_delta_mean":  _mean(td["R_d"]),
                "R_delta_max":   _max(td["R_d"]),
                "W_delta_mean":  _mean(td["W_d"]),
                "b_delta_mean":  _mean(td["b_d"]),
                "R_cosine_mean": _mean(td["R_c"]),
                "n_layers":      len(td["R_d"]),
            }
            for tier, td in tier_acc.items()
        }

    _param_drift_log.append(entry)

def compute_fisher(
    reft_model,
    task_name:   str,
    task_idx:    int,
    task_type:   int,
    task_layers: Optional[List[int]] = None,
    tier1_set:   Optional[Set[int]]  = None,
    tier2_set:   Optional[Set[int]]  = None,
    tier3_set:   Optional[Set[int]]  = None,
):

    R_sq, W_sq, b_sq = [], [], []
    hooks_fired = 0

    have_tiers = (task_layers is not None and tier1_set is not None
                  and tier2_set is not None and tier3_set is not None)
    tier_sq: Dict[str, Dict[str, list]] = {
        t: {"R": [], "W": [], "b": []} for t in ("T1", "T2", "T3f", "T3l")
    }
    iv_order = {k: i for i, k in enumerate(reft_model.interventions.keys())}

    for key, iv_list in reft_model.interventions.items():
        iv    = _get_iv(iv_list)
        label = (
            _tier_label(iv_order[key], task_layers,
                        tier1_set, tier2_set, tier3_set, task_type)
            if have_tiers and key in iv_order else "T2"
        )
        for getter, bucket, tier_b in [
            (_get_r, R_sq,  tier_sq[label]["R"]),
            (_get_w, W_sq,  tier_sq[label]["W"]),
            (_get_b, b_sq,  tier_sq[label]["b"]),
        ]:
            t = getter(iv)
            if t is not None:
                if t.grad is not None:
                    val = float(t.grad.detach().float().pow(2).mean())
                    bucket.append(val)
                    tier_b.append(val)
                    hooks_fired += 1
                else:
                    bucket.append(0.0)
                    tier_b.append(0.0)

    def _mean(lst):
        return float(np.mean(lst)) if lst else 0.0

    no_grad = "" if hooks_fired > 0 else "  ← no gradients present"
    F_R, F_W, F_b = _mean(R_sq), _mean(W_sq), _mean(b_sq)

    print(f"  [fisher] {task_name}: "
          f"F(R)={F_R:.3e}  F(W)={F_W:.3e}  F(b)={F_b:.3e}{no_grad}")

    if have_tiers:
        for t in ("T1", "T2", "T3f", "T3l"):
            td = tier_sq[t]
            if any(td[k] for k in ("R", "W", "b")):
                print(f"    [{t}]  F(R)={_mean(td['R']):.3e}  "
                      f"F(W)={_mean(td['W']):.3e}  "
                      f"F(b)={_mean(td['b']):.3e}")

    entry: Dict[str, Any] = {
        "task_name":   task_name,
        "task_idx":    task_idx,
        "task_type":   task_type,
        "fisher_R":    F_R,
        "fisher_W":    F_W,
        "fisher_b":    F_b,
        "hooks_fired": hooks_fired,
        "note":        no_grad.strip() or None,
    }
    if have_tiers:
        entry["tier_fisher"] = {
            tier: {
                "R":        _mean(td["R"]),
                "W":        _mean(td["W"]),
                "b":        _mean(td["b"]),
                "n_layers": len(td["R"]),
            }
            for tier, td in tier_sq.items()
        }

    _fisher_log.append(entry)


class GradientTrajectoryCallback(TrainerCallback):

    def __init__(
        self,
        reft_model,
        task_name:   str,
        task_idx:    int,
        log_every:   int = 20,
        task_layers: Optional[List[int]] = None,
        tier1_set:   Optional[Set[int]]  = None,
        tier2_set:   Optional[Set[int]]  = None,
        tier3_set:   Optional[Set[int]]  = None,
        task_type:   int = 1,
    ):
        self.reft_model   = reft_model
        self.task_name    = task_name
        self.task_idx     = task_idx
        self.log_every    = log_every
        self.task_layers  = task_layers
        self.tier1_set    = tier1_set
        self.tier2_set    = tier2_set
        self.tier3_set    = tier3_set
        self.task_type    = task_type
        self._buffer: List[Dict] = []
        self._n_flushed   = 0
        self._have_tiers  = (task_layers is not None and tier1_set is not None
                             and tier2_set is not None and tier3_set is not None)
        self._iv_order    = {
            k: i for i, k in enumerate(reft_model.interventions.keys())
        }

    @classmethod
    def make(
        cls,
        reft_model,
        task_name:   str,
        task_idx:    int,
        log_every:   int = 20,
        task_layers: Optional[List[int]] = None,
        tier1_set:   Optional[Set[int]]  = None,
        tier2_set:   Optional[Set[int]]  = None,
        tier3_set:   Optional[Set[int]]  = None,
        task_type:   int = 1,
    ) -> "GradientTrajectoryCallback":
        return cls(
            reft_model, task_name, task_idx, log_every,
            task_layers, tier1_set, tier2_set, tier3_set, task_type,
        )

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.log_every != 0:
            return

        R_g, W_g, b_g = [], [], []
        tier_g: Dict[str, Dict[str, list]] = {
            t: {"R": [], "W": [], "b": []} for t in ("T1", "T2", "T3f", "T3l")
        }

        for key, iv_list in self.reft_model.interventions.items():
            iv = _get_iv(iv_list)
            label = (
                _tier_label(self._iv_order[key], self.task_layers,
                            self.tier1_set, self.tier2_set, self.tier3_set,
                            self.task_type)
                if self._have_tiers and key in self._iv_order else "T2"
            )
            for getter, bucket, tier_b in [
                (_get_r, R_g, tier_g[label]["R"]),
                (_get_w, W_g, tier_g[label]["W"]),
                (_get_b, b_g, tier_g[label]["b"]),
            ]:
                t = getter(iv)
                if t is not None and t.grad is not None:
                    gn = float(t.grad.norm().item())
                    bucket.append(gn)
                    tier_b.append(gn)

        def _mean(lst):
            return float(np.mean(lst)) if lst else 0.0

        entry: Dict[str, Any] = {
            "task_name":    self.task_name,
            "task_idx":     self.task_idx,
            "step":         state.global_step,
            "R_gnorm_mean": _mean(R_g),
            "W_gnorm_mean": _mean(W_g),
            "b_gnorm_mean": _mean(b_g),
        }
        if self._have_tiers:
            entry["tier_gnorm"] = {
                tier: {"R": _mean(td["R"]), "W": _mean(td["W"]), "b": _mean(td["b"])}
                for tier, td in tier_g.items()
            }
        self._buffer.append(entry)

    def on_train_end(self, args, state, control, **kwargs):
        n = len(self._buffer)
        _grad_traj_log.extend(self._buffer)
        self._n_flushed += n
        self._buffer = []
        n_r = sum(
            1 for iv_list in self.reft_model.interventions.values()
            if (lambda t: t is not None and t.requires_grad)(_get_r(_get_iv(iv_list)))
        )
        n_w = sum(
            1 for iv_list in self.reft_model.interventions.values()
            if _get_w(_get_iv(iv_list)) is not None
        )
        print(f"  [grad_cb] {self.task_name}: flushed {n} entries  "
              f"({self._n_flushed} total steps, every {self.log_every})  "
              f"R:×{n_r}  W×{n_w}")

def log_delta_assignment(
    task_name:         str,
    task_idx:          int,
    proto_id:          int,
    is_new_proto:      bool,
    delta_dim:         int,
    delta_norm:        float,
    distances:         Dict[str, float],
    warmup_steps_used: int,
    skip_steps:        int,
):
    status    = "NEW" if is_new_proto else "MERGED"
    dists_str = ("  ".join(f"P{k}:{v:.4f}" for k, v in distances.items())
                 if distances else "{}")
    entry = {
        "event":        "delta_assignment",
        "task_name":    task_name,
        "task_idx":     task_idx,
        "proto_id":     proto_id,
        "is_new_proto": is_new_proto,
        "delta_dim":    delta_dim,
        "delta_norm":   round(float(delta_norm), 6),
        "distances":    {str(k): round(float(v), 6) for k, v in distances.items()},
        "warmup_steps": warmup_steps_used,
        "skip_steps":   skip_steps,
    }
    _causal_router_log.append(entry)
    print(f"  [delta_assign] {task_name}: P{proto_id} ({status})  "
          f"delta_dim={delta_dim}  delta_norm={delta_norm:.4f}  "
          f"distances={{{dists_str}}}")


def log_prototype_state(
    proto_id:    int,
    n_merged:    int,
    task_names:  List[str],
    delta_norm:  float,
    delta_dim:   int,
):
    _causal_router_log.append({
        "event":       "prototype_state",
        "proto_id":    proto_id,
        "n_merged":    n_merged,
        "task_names":  list(task_names),
        "delta_norm":  round(float(delta_norm), 6),
        "delta_dim":   delta_dim,
    })
    print(f"  [proto_state] P{proto_id}: n_merged={n_merged}  "
          f"tasks={task_names}  delta_dim={delta_dim}  norm={delta_norm:.4f}")


def log_prototype_merge(
    task_name:      str,
    task_idx:       int,
    proto_id:       int,
    cosine_dist:    float,
    merge_alpha:    float,
    n_merged_after: int,
):
    _memory_merge_log.append({
        "event":          "prototype_merge",
        "task_name":      task_name,
        "task_idx":       task_idx,
        "proto_id":       proto_id,
        "cosine_dist":    round(float(cosine_dist), 6),
        "merge_alpha":    merge_alpha,
        "n_merged_after": n_merged_after,
    })
    print(f"  [proto_merge] {task_name}: merged into P{proto_id}  "
          f"dist={cosine_dist:.4f}  alpha={merge_alpha}  "
          f"n_merged_after={n_merged_after}")


def log_memory_merge(
    task_name:  str,
    task_idx:   int,
    tier:       str,
    proto_id:   Optional[int],
    alpha:      float,
    n_iv_keys:  int,
    is_first:   bool,
):
    _memory_merge_log.append({
        "event":     "memory_merge",
        "task_name": task_name,
        "task_idx":  task_idx,
        "tier":      tier,
        "proto_id":  proto_id,
        "alpha":     alpha,
        "n_iv_keys": n_iv_keys,
        "is_first":  is_first,
    })
    mode = "init" if is_first else f"EMA(α={alpha})"
    pid  = f" P{proto_id}" if proto_id is not None else ""
    print(f"  [MemMerge] {task_name}: {tier}{pid}  {mode}  n_keys={n_iv_keys}")


def log_task_private_save(
    task_name: str,
    task_idx:  int,
    n_iv_keys: int,
):
    _memory_merge_log.append({
        "event":     "task_private_save",
        "task_name": task_name,
        "task_idx":  task_idx,
        "tier":      "T3l_private",
        "n_iv_keys": n_iv_keys,
    })
    print(f"  [MemSave ] {task_name}: T3l_private  exact copy  n_keys={n_iv_keys}")

def log_tier_freeze(
    task_name:        str,
    task_idx:         int,
    n_frozen_tensors: int,
    tier1_layers:     List[int],
):
    _tier_event_log.append({
        "event":            "tier_freeze",
        "task_name":        task_name,
        "task_idx":         task_idx,
        "tier":             "T1_shared",
        "n_frozen_tensors": n_frozen_tensors,
        "tier1_layers":     list(tier1_layers),
    })
    print(f"  [TierFreeze] {task_name}: T1 frozen  "
          f"{n_frozen_tensors} tensors  layers={tier1_layers}")


def log_tier_unfreeze(
    task_name: str,
    task_idx:  int,
    tier:      str,
    n_tensors: int,
):
    _tier_event_log.append({
        "event":     "tier_unfreeze",
        "task_name": task_name,
        "task_idx":  task_idx,
        "tier":      tier,
        "n_tensors": n_tensors,
    })
    print(f"  [TierUnfreeze] {task_name}: {tier} unfrozen  {n_tensors} tensors")

def flush_log(output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    def _write(filename: str, data):
        path = os.path.join(output_dir, filename)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        n = len(data) if isinstance(data, (list, dict)) else "?"
        print(f"  [logger] {filename} written  ({n} entries)")

    _write("param_drift_log.json",     _param_drift_log)
    _write("grad_trajectory_log.json", _grad_traj_log)
    _write("causal_router_log.json",   _causal_router_log)
    _write("fisher_log.json",          _fisher_log)
    _write("memory_merge_log.json",    _memory_merge_log)
    _write("tier_event_log.json",      _tier_event_log)
    if _snapshot_buffer:
        _write("snapshot_buffer.json", _snapshot_buffer)
