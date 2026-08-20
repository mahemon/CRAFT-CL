"""
craft_lora.py — CRAFT-LoRA backend.

Runs the full CRAFT-CL pipeline (KL routing, group memory, KL anchor,
eviction, per-task backward eval) over LoRA adapters instead of LoReFT
interventions, with NO changes to the pipeline itself.

How: CraftLoRAModel duck-types the slice of the pyreft ReftModel interface
that train.py actually uses —

    .interventions                    dict  key -> adapter module
    .model                            the wrapped base model
    __call__(base=..., unit_locations=..., labels=..., subspaces=...)
                                      -> (None, model outputs)
    .generate(base=..., unit_locations=..., intervene_on_prompt=..., **gen)
                                      -> (None, sequences)
    .print_trainable_parameters()
    .remove_all_hooks()               restores the base model (teardown)
    .eval()/.train()/.parameters()    (nn.Module)

so _extract_iv_state, _load_iv_state, GroupMemory, the KL-Razor controller,
the routing probes, build_stratified_optimizer, and compute_metrics all work
as-is.  unit_locations / intervene_on_prompt / subspaces are accepted and
ignored (LoRA applies to every token).

One adapter per unique layer, attached to args.lora_modules (default
'o_proj') of that decoder layer.  Adapter = standard LoRA:
y += (alpha/r)·B(A(x)), A ~ N(0, 0.02), B = 0 (identity at init).
Fixed-seed init mirrors the LoReFT fixed_intervention_seed semantics.
Adapters are injected by patching the target Linear's forward; injection is
idempotent (rebuilding replaces the adapter, never stacks) and
remove_all_hooks() restores the original forwards.
"""

from typing import Dict, List

import torch
import torch.nn as nn

_MODEL_KW = ("input_ids", "attention_mask", "labels", "use_cache",
             "position_ids", "past_key_values", "inputs_embeds")

_GEN_DROP = ("base", "unit_locations", "intervene_on_prompt", "subspaces")

_PATCH_ATTR = "_craft_lora_adapter"
_ORIG_ATTR = "_craft_lora_orig_forward"


class LoRAAdapter(nn.Module):
    """Holds ONLY the trainable A/B (state_dict stays small for GroupMemory)."""

    def __init__(self, in_features: int, out_features: int,
                 rank: int, alpha: int, device):
        super().__init__()
        self.rank = rank
        self.scaling = float(alpha) / float(rank)
        self.lora_A = nn.Parameter(
            torch.randn(rank, in_features, dtype=torch.float32) * 0.02)
        self.lora_B = nn.Parameter(
            torch.zeros(out_features, rank, dtype=torch.float32))
        self.to(device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.nn.functional.linear(x.to(self.lora_A.dtype), self.lora_A)
        h = torch.nn.functional.linear(h, self.lora_B)
        return (h * self.scaling).to(x.dtype)


def _resolve_target_linear(base_model, layer_idx: int, module_name: str):
    """decoder layer -> named sub-Linear (searched by suffix, e.g. 'o_proj')."""
    dec_layers = base_model.model.layers if hasattr(base_model, "model") \
        else base_model.layers
    layer = dec_layers[layer_idx]
    for name, mod in layer.named_modules():
        if name.endswith(module_name) and isinstance(mod, nn.Linear):
            return mod
    raise ValueError(f"module '{module_name}' not found in decoder layer "
                     f"{layer_idx}")


def _inject(linear: nn.Linear, adapter: LoRAAdapter) -> None:
    """Idempotent: replaces any existing adapter instead of stacking."""
    if not hasattr(linear, _ORIG_ATTR):
        setattr(linear, _ORIG_ATTR, linear.forward)
    orig = getattr(linear, _ORIG_ATTR)
    setattr(linear, _PATCH_ATTR, adapter)

    def patched(x, _orig=orig, _lin=linear):
        return _orig(x) + getattr(_lin, _PATCH_ATTR)(x)

    linear.forward = patched


def _restore(linear: nn.Linear) -> None:
    if hasattr(linear, _ORIG_ATTR):
        linear.forward = getattr(linear, _ORIG_ATTR)
        delattr(linear, _ORIG_ATTR)
    if hasattr(linear, _PATCH_ATTR):
        delattr(linear, _PATCH_ATTR)


class CraftLoRAModel(nn.Module):

    def __init__(self, base_model, layers: List[int], rank: int, alpha: int,
                 module_name: str, device):
        super().__init__()
        object.__setattr__(self, "model", base_model)  # not a submodule:
        # keeps base params out of .parameters()/.state_dict() so the Trainer
        # and GroupMemory only ever see adapter weights.
        self.interventions: Dict[str, LoRAAdapter] = {}
        self._targets: Dict[str, nn.Linear] = {}
        uniq = sorted(dict.fromkeys(int(l) for l in layers))
        for l in uniq:
            lin = _resolve_target_linear(base_model, l, module_name)
            adapter = LoRAAdapter(lin.in_features, lin.out_features,
                                  rank, alpha, device)
            _inject(lin, adapter)
            key = f"layer.{l}.comp.lora_{module_name}.unit.pos.nunit.1#0"
            self.interventions[key] = adapter
            self._targets[key] = lin
            self.add_module(f"lora_{l}", adapter)

    def _activate(self):
        """Bind THIS instance's adapters as the live ones on the shared
        base model.  Several CraftLoRAModel instances can exist at once
        (live trainer model, KL-anchor shadow, eval shells) over one base;
        whichever is forwarding must own the patch."""
        for key, lin in self._targets.items():
            if not hasattr(lin, _ORIG_ATTR):
                _inject(lin, self.interventions[key])
            else:
                setattr(lin, _PATCH_ATTR, self.interventions[key])

    # ── pyreft-compatible call surface ────────────────────────────────────
    def forward(self, base=None, unit_locations=None, labels=None,
                subspaces=None, **kwargs):
        self._activate()
        b = dict(base or {})
        if labels is not None:
            b["labels"] = labels
        b.setdefault("use_cache", False)
        b = {k: v for k, v in b.items() if k in _MODEL_KW}
        out = self.model(**b)
        return None, out

    @torch.no_grad()
    def generate(self, **generation_args):
        self._activate()
        base = dict(generation_args.get("base") or {})
        gen = {k: v for k, v in generation_args.items() if k not in _GEN_DROP}
        seqs = self.model.generate(
            input_ids=base.get("input_ids"),
            attention_mask=base.get("attention_mask"), **gen)
        return None, seqs

    # ── housekeeping used by train.py ─────────────────────────────────────
    def print_trainable_parameters(self):
        iv = sum(p.numel() for a in self.interventions.values()
                 for p in a.parameters() if p.requires_grad)
        mp = sum(p.numel() for p in self.model.parameters())
        tm = sum(p.numel() for p in self.model.parameters()
                 if p.requires_grad)
        print(f"trainable intervention params: {iv:,} || "
              f"trainable model params: {tm:,}")
        print(f"model params: {mp:,} || "
              f"trainable%: {100.0 * iv / (mp + iv)}")

    def save(self, save_directory, **kwargs):
        """pyreft-compatible checkpoint save: adapters only."""
        import os
        os.makedirs(save_directory, exist_ok=True)
        state = {k: {n: p.detach().cpu() for n, p in a.state_dict().items()}
                 for k, a in self.interventions.items()}
        torch.save(state, os.path.join(save_directory,
                                       "craft_lora_adapters.pt"))
        
    def remove_all_hooks(self):
        for lin in self._targets.values():
            _restore(lin)

    def disable_forward_hooks(self):     # teardown alias
        self.remove_all_hooks()

    def get_device(self):
        return next(self.model.parameters()).device

    def set_device(self, device):
        for a in self.interventions.values():
            a.to(device)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(object.__getattribute__(self, "model"), name)


def build_craft_lora_model(base_model, layers, args):
    """Drop-in replacement for _build_reft_model when args.use_lora is set.
    Same fixed-seed semantics: every build starts from the identical init."""
    seed = getattr(args, "fixed_intervention_seed", None)
    if seed is not None:
        torch.manual_seed(int(seed))
        torch.cuda.manual_seed_all(int(seed))
    device = next(base_model.parameters()).device
    m = CraftLoRAModel(
        base_model, list(layers),
        rank=int(getattr(args, "lora_rank", 8)),
        alpha=int(getattr(args, "lora_alpha", 32)),
        module_name=str(getattr(args, "lora_modules", "o_proj")),
        device=device)
    print(f"  [build_lora] {len(m.interventions)} LoRA adapters "
          f"(rank={getattr(args, 'lora_rank', 8)}, "
          f"alpha={getattr(args, 'lora_alpha', 32)}, "
          f"target={getattr(args, 'lora_modules', 'o_proj')}).")
    return m