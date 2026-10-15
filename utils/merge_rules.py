"""
merge_rules.py — group-state merge rules for CRAFT-CL.

Must sit alongside train.py; train.py imports:

    from merge_rules import (MERGE_RULES, merge_group_state,
                             compute_iv_fisher, accumulate_fisher)

A "state" here is the CRAFT-CL intervention-state mapping produced by
train.py's _extract_iv_state / _split_iv_state_by_5tier:

    Dict[iv_key, Dict[param_name, Tensor]]

A "fisher" has exactly the same shape, with each tensor holding a diagonal
empirical-Fisher estimate for the matching parameter.

Rules
-----
overwrite   merged = new                                  (legacy behaviour)
ema         merged = a*new + (1-a)*old,  a = ema_alpha
prototype   merged = a*new + (1-a)*old,  a = 1/(n_old+1)  (running mean over
            the tasks in the group)
kl          merged = a*new + (1-a)*old,  a = 1/(1 + kl_scale*kl_final).
            Low measured drift leans toward the newly trained state; high
            drift preserves the existing group state. UNTESTED — no ablation
            data backs this rule yet.
fisher      merged = w*new + (1-w)*old, elementwise, with
                w = F_new / (F_new + F_old + eps)
            F_new is the diagonal empirical Fisher of THIS task's own CE loss
            (mean grad^2 over `merge_fisher_batches` minibatches); F_old is
            the running sum of the Fishers of the tasks already merged into
            the group. This is a parameter-importance signal computed from
            the task loss only — it is NOT a KL quantity and is independent
            of the KL anchor.

Every rule falls back to `overwrite` (loudly, never silently) if the inputs
it needs are missing — e.g. `kl` with kl_final=None, or `fisher` with no
Fisher estimate.
"""

from typing import Dict, Optional, Tuple, Any
import torch

MERGE_RULES = ("overwrite", "ema", "fisher", "prototype", "kl")

_FISHER_EPS = 1e-12

State  = Dict[str, Dict[str, torch.Tensor]]
Fisher = Dict[str, Dict[str, torch.Tensor]]


# ---------------------------------------------------------------------------
# state helpers
# ---------------------------------------------------------------------------

def _blendable(t_old: Any, t_new: Any) -> bool:
    """Only float tensors of identical shape are blended; anything else
    (ints, bools, shape mismatches, non-tensors) takes the new value."""
    return (torch.is_tensor(t_old) and torch.is_tensor(t_new)
            and t_old.is_floating_point() and t_new.is_floating_point()
            and tuple(t_old.shape) == tuple(t_new.shape))


def _clone(t: Any) -> Any:
    return t.detach().clone() if torch.is_tensor(t) else t


def _blend_scalar(old_state: State, new_state: State, alpha: float) -> Tuple[State, Dict]:
    """merged = alpha*new + (1-alpha)*old, elementwise over matching tensors."""
    alpha = float(alpha)
    merged: State = {}
    n_blend = n_copy = 0
    for iv_key, new_sub in new_state.items():
        old_sub = old_state.get(iv_key, {})
        out: Dict[str, torch.Tensor] = {}
        for pname, t_new in new_sub.items():
            t_old = old_sub.get(pname, None)
            if _blendable(t_old, t_new):
                o = t_old.detach().to(device=t_new.device, dtype=torch.float32)
                n = t_new.detach().to(torch.float32)
                out[pname] = (alpha * n + (1.0 - alpha) * o).to(t_new.dtype)
                n_blend += 1
            else:
                out[pname] = _clone(t_new)
                n_copy += 1
        merged[iv_key] = out
    # keys the new state does not touch survive unchanged
    for iv_key, old_sub in old_state.items():
        if iv_key not in merged:
            merged[iv_key] = {k: _clone(v) for k, v in old_sub.items()}
    return merged, {"alpha": alpha, "n_blended": n_blend, "n_copied": n_copy}


def _blend_fisher(old_state: State, new_state: State,
                  fisher_new: Fisher, fisher_old: Optional[Fisher]) -> Tuple[State, Dict]:
    """merged = w*new + (1-w)*old with w = F_new/(F_new+F_old), elementwise."""
    merged: State = {}
    n_blend = n_copy = 0
    w_sum = 0.0
    w_n = 0
    for iv_key, new_sub in new_state.items():
        old_sub = old_state.get(iv_key, {})
        f_new_sub = (fisher_new or {}).get(iv_key, {})
        f_old_sub = (fisher_old or {}).get(iv_key, {})
        out: Dict[str, torch.Tensor] = {}
        for pname, t_new in new_sub.items():
            t_old = old_sub.get(pname, None)
            f_new = f_new_sub.get(pname, None)
            if not _blendable(t_old, t_new) or not torch.is_tensor(f_new):
                out[pname] = _clone(t_new)
                n_copy += 1
                continue
            n = t_new.detach().to(torch.float32)
            o = t_old.detach().to(device=t_new.device, dtype=torch.float32)
            fn = f_new.detach().to(device=t_new.device, dtype=torch.float32)
            fo = f_old_sub.get(pname, None)
            fo = (fo.detach().to(device=t_new.device, dtype=torch.float32)
                  if torch.is_tensor(fo) else torch.zeros_like(fn))
            w = fn / (fn + fo + _FISHER_EPS)
            out[pname] = (w * n + (1.0 - w) * o).to(t_new.dtype)
            w_sum += float(w.mean().item())
            w_n += 1
            n_blend += 1
        merged[iv_key] = out
    for iv_key, old_sub in old_state.items():
        if iv_key not in merged:
            merged[iv_key] = {k: _clone(v) for k, v in old_sub.items()}
    return merged, {"alpha": (w_sum / w_n) if w_n else float("nan"),
                    "n_blended": n_blend, "n_copied": n_copy}


def merge_group_state(
    old_state: State,
    new_state: State,
    *,
    rule: str = "overwrite",
    ema_alpha: float = 0.5,
    n_old: int = 1,
    kl_final: Optional[float] = None,
    kl_scale: float = 1.0,
    fisher_new: Optional[Fisher] = None,
    fisher_old: Optional[Fisher] = None,
) -> Tuple[State, Dict]:
    """Combine an existing group state with a freshly trained task state.

    Returns (merged_state, info) where info carries the effective blend
    weight and the rule actually applied (which may differ from `rule` if a
    required input was missing).
    """
    rule = str(rule).lower()
    if rule not in MERGE_RULES:
        raise ValueError(f"unknown merge_rule={rule!r}; expected one of {MERGE_RULES}")

    if not old_state:
        return ({k: {p: _clone(t) for p, t in sub.items()}
                 for k, sub in new_state.items()},
                {"rule": "overwrite", "reason": "no existing group state",
                 "alpha": 1.0, "n_blended": 0, "n_copied": 0})

    if rule == "overwrite":
        merged, info = _blend_scalar(old_state, new_state, 1.0)
        info["rule"] = "overwrite"
        return merged, info

    if rule == "ema":
        merged, info = _blend_scalar(old_state, new_state, ema_alpha)
        info["rule"] = "ema"
        return merged, info

    if rule == "prototype":
        alpha = 1.0 / max(1, int(n_old) + 1)
        merged, info = _blend_scalar(old_state, new_state, alpha)
        info["rule"] = "prototype"
        info["n_old"] = int(n_old)
        return merged, info

    if rule == "kl":
        if kl_final is None or kl_final != kl_final:      # None or NaN
            merged, info = _blend_scalar(old_state, new_state, 1.0)
            info.update(rule="overwrite", requested="kl",
                        reason="kl_final unavailable (KL anchor off or "
                               "save-time KL eval failed)")
            return merged, info
        # scale is set by the caller from --merge_kl_halflife when that is
        # given: halflife H is the kl_final at which old and new are weighted
        # equally, so scale = 1/H. See the note in kl_scale_from_halflife on
        # why the default scale=1.0 is inert at the observed KL magnitudes.
        alpha = 1.0 / (1.0 + float(kl_scale) * float(kl_final))
        merged, info = _blend_scalar(old_state, new_state, alpha)
        info.update(rule="kl", kl_final=float(kl_final), kl_scale=float(kl_scale),
                    halflife=(1.0 / float(kl_scale)) if kl_scale else float("inf"))
        if alpha > 0.95:
            info["note"] = (f"alpha={alpha:.4f} — within 5% of overwrite. "
                            f"kl_final={kl_final:.4g} is small relative to "
                            f"1/scale={1.0/float(kl_scale) if kl_scale else float('inf'):.4g}; "
                            f"raise --merge_kl_scale (or lower --merge_kl_halflife) "
                            f"for this rule to do anything.")
        return merged, info

    # fisher
    if not fisher_new:
        merged, info = _blend_scalar(old_state, new_state, 1.0)
        info.update(rule="overwrite", requested="fisher",
                    reason="no Fisher estimate for the new task")
        return merged, info
    merged, info = _blend_fisher(old_state, new_state, fisher_new, fisher_old)
    info["rule"] = "fisher"
    info["had_group_fisher"] = bool(fisher_old)
    if info.get("n_blended", 0) == 0:
        # Every tensor took the copy path, so this merge WAS an overwrite.
        # Do not let that pass as a successful fisher merge -- it is how a
        # whole ablation ends up comparing a config against itself.
        info["note"] = (
            f"blended 0 of {info.get('n_copied', 0)} tensors — the Fisher keys "
            f"do not line up with the state-dict keys, so this merge was an "
            f"overwrite. Expected state keys are the intervention's own "
            f"(e.g. 'weight', 'bias', 'rotate_layer'), not named_parameters() "
            f"paths. Check the [fisher] keys= line above.")
    return merged, info


def kl_scale_from_halflife(halflife: Optional[float]) -> Optional[float]:
    """Convert an interpretable half-life into the raw scale.

    alpha = 1/(1 + scale*kl) equals 0.5 exactly when kl == 1/scale, so the
    half-life H (the kl_final at which the group state and the new state get
    equal weight) is H = 1/scale.

    This matters because the observed post-training kl_final in this codebase
    is ~0.03, four to forty times below every eviction threshold tried. At the
    default scale=1.0 the half-life is 1.0, so alpha = 1/(1+0.03) = 0.97 and
    merge_rule=kl reproduces overwrite to three decimals. Useful half-lives
    are the same order as the measured KL: H in [0.01, 0.10], i.e. scale in
    [10, 100].
    """
    if halflife is None:
        return None
    h = float(halflife)
    if h <= 0:
        raise ValueError(f"merge_kl_halflife must be > 0 (got {h})")
    return 1.0 / h


def accumulate_fisher(fisher_old: Optional[Fisher], fisher_new: Optional[Fisher]) -> Fisher:
    """Running sum of per-parameter Fisher, used as the group's F_old."""
    if not fisher_new:
        return {k: {p: t.clone() for p, t in sub.items()}
                for k, sub in (fisher_old or {}).items()}
    out: Fisher = {}
    for iv_key, sub_new in fisher_new.items():
        sub_old = (fisher_old or {}).get(iv_key, {})
        acc: Dict[str, torch.Tensor] = {}
        for pname, t_new in sub_new.items():
            t_old = sub_old.get(pname, None)
            acc[pname] = (t_new.clone() if not torch.is_tensor(t_old)
                          else (t_old.to(t_new.dtype) + t_new))
        out[iv_key] = acc
    for iv_key, sub_old in (fisher_old or {}).items():
        if iv_key not in out:
            out[iv_key] = {p: t.clone() for p, t in sub_old.items()}
    return out


# ---------------------------------------------------------------------------
# diagonal empirical Fisher over the intervention parameters
# ---------------------------------------------------------------------------

def _default_call(reft_model, batch):
    """Mirror of train.py's KLRazorController._call_intervenable."""
    keep = ("input_ids", "attention_mask", "labels",
            "intervention_locations", "subspaces")
    b = {k: v for k, v in batch.items() if k in keep}
    iv_locs = b.pop("intervention_locations", None)
    b.pop("subspaces", None)
    b["use_cache"] = False
    unit_locations = None
    if iv_locs is not None:
        unit_locations = {"sources->base": (None, iv_locs.permute(1, 0, 2).tolist()
                                            if hasattr(iv_locs, "permute") else iv_locs)}
    _, outputs = reft_model(base=b, unit_locations=unit_locations)
    return outputs


def _iv_modules(reft_model) -> Dict[str, torch.nn.Module]:
    return {k: (v[0] if isinstance(v, (list, tuple)) else v)
            for k, v in reft_model.interventions.items()}


def _remap_to_state_keys(mod, fisher_sub: Dict[str, torch.Tensor], verbose=True):
    """Re-key a Fisher dict from named_parameters() names onto state_dict() names.

    These are NOT the same namespace for pyreft interventions. state_dict()
    is overridden to emit flattened custom names -- 'weight', 'bias',
    'rotate_layer' -- while named_parameters() yields the module paths
    ('learned_source.weight', 'rotate_layer.parametrizations.weight.original',
    ...). merge_group_state() walks the STATE dicts, so a Fisher keyed the
    other way misses on every lookup and _blend_fisher copies the new tensor
    wholesale for every parameter: 'blended=0 copied=96', i.e. a silent
    overwrite wearing a fisher label.

    Matching is tried in order: exact name, suffix/containment, then unique
    shape within this intervention (the three tensors have distinct shapes:
    (r, d) source weight, (r,) bias, (d, r) rotation).
    """
    if not fisher_sub:
        return fisher_sub, []
    try:
        sd = mod.state_dict()
    except Exception:
        return fisher_sub, []
    if set(fisher_sub) <= set(sd):
        return fisher_sub, []          # already aligned, nothing to do

    by_shape: Dict[tuple, list] = {}
    for k, v in sd.items():
        if hasattr(v, "shape"):
            by_shape.setdefault(tuple(v.shape), []).append(k)

    out: Dict[str, torch.Tensor] = {}
    unmapped = []
    for pname, t in fisher_sub.items():
        if pname in sd:
            out[pname] = t
            continue
        cands = [k for k in sd if pname.endswith(k) or k in pname]
        if len(cands) == 1:
            out[cands[0]] = t
            continue
        cands = by_shape.get(tuple(t.shape), [])
        if len(cands) == 1:
            out[cands[0]] = t
            continue
        unmapped.append(pname)
    return out, unmapped


def compute_iv_fisher(
    reft_model,
    dataset,
    data_collator,
    *,
    n_batches: int = 8,
    batch_size: int = 4,
    device: str = "cuda",
    seed: int = 42,
    call_fn=None,
    verbose: bool = True,
) -> Fisher:
    """Diagonal empirical Fisher of the task's own CE loss w.r.t. the
    intervention parameters: mean of grad^2 over `n_batches` minibatches.

    Returned tensors live on CPU in float32, keyed exactly like the state
    dicts train.py stores, so they line up with merge_group_state().
    """
    import torch.nn.functional as F
    from torch.utils.data import DataLoader

    call_fn = call_fn or _default_call
    mods = _iv_modules(reft_model)

    prev_req = {}
    for iv_key, mod in mods.items():
        for pname, p in mod.named_parameters():
            prev_req[(iv_key, pname)] = p.requires_grad
            p.requires_grad_(True)
            if p.grad is not None:
                p.grad = None

    fisher: Fisher = {iv_key: {} for iv_key in mods}
    g = torch.Generator()
    g.manual_seed(int(seed))
    loader = DataLoader(dataset, batch_size=int(batch_size), shuffle=True,
                        collate_fn=data_collator, generator=g)

    n_used = 0
    for i, batch in enumerate(loader):
        if n_used >= int(n_batches):
            break
        batch = {k: (v.to(device) if torch.is_tensor(v) else v)
                 for k, v in batch.items()}
        try:
            with torch.enable_grad():
                out = call_fn(reft_model, batch)
                logits = out.logits if hasattr(out, "logits") else out[0]
                labels = batch.get("labels", None)
                if labels is not None and logits is not None:
                    shift_logits = logits[..., :-1, :].contiguous().float()
                    shift_labels = labels[..., 1:].contiguous().to(shift_logits.device)
                    loss = F.cross_entropy(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1), ignore_index=-100)
                else:
                    loss = out.loss
                if loss is None or not torch.isfinite(loss):
                    continue
                loss.backward()
        except Exception as e:
            if verbose:
                print(f"  [fisher] batch {i} failed "
                      f"({type(e).__name__}: {e}); skipped.")
            for mod in mods.values():
                for p in mod.parameters():
                    p.grad = None
            continue

        for iv_key, mod in mods.items():
            for pname, p in mod.named_parameters():
                if p.grad is None:
                    continue
                g2 = p.grad.detach().to(torch.float32).pow(2).cpu()
                prev = fisher[iv_key].get(pname, None)
                fisher[iv_key][pname] = g2 if prev is None else prev + g2
                p.grad = None
        n_used += 1

    if n_used == 0:
        if verbose:
            print("  [fisher] no usable batches — returning empty Fisher "
                  "(merge will fall back to overwrite).")
        fisher = {}
    else:
        for iv_key in list(fisher.keys()):
            for pname in list(fisher[iv_key].keys()):
                fisher[iv_key][pname] /= float(n_used)
        fisher = {k: v for k, v in fisher.items() if v}

        # Re-key onto state_dict() names, which is what merge_group_state walks.
        all_unmapped = []
        for iv_key in list(fisher.keys()):
            remapped, unmapped = _remap_to_state_keys(mods[iv_key], fisher[iv_key])
            fisher[iv_key] = remapped
            all_unmapped.extend(unmapped)
        fisher = {k: v for k, v in fisher.items() if v}

        if verbose:
            n_t = sum(len(v) for v in fisher.values())
            mean_f = (sum(float(t.mean()) for v in fisher.values() for t in v.values())
                      / max(1, n_t))
            example = ""
            if fisher:
                k0 = next(iter(fisher))
                example = f"  keys={sorted(fisher[k0])}"
            print(f"  [fisher] diagonal empirical Fisher over {n_used} batch(es): "
                  f"{len(fisher)} interventions, {n_t} tensors, "
                  f"mean={mean_f:.3e}{example}")
            if all_unmapped:
                print(f"  [fisher] WARNING: {len(all_unmapped)} parameter(s) could "
                      f"not be matched to state_dict names "
                      f"(e.g. {sorted(set(all_unmapped))[:3]}); those tensors will "
                      f"fall back to overwrite in the merge.")

    for iv_key, mod in mods.items():
        for pname, p in mod.named_parameters():
            p.grad = None
            p.requires_grad_(prev_req.get((iv_key, pname), True))

    return fisher
