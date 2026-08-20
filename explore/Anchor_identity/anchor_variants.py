"""
anchor_variants.py — support for the Anchor Identity ablation.

Provides one utility: converting a normally-built shadow ReFT model into
a pi0 anchor, i.e. a model whose interventions are forced to the identity
function so its forward pass reproduces the frozen backbone with no
adaptation applied at all, while keeping the exact same calling
interface (`.interventions`, `.eval()`, callable via
`reft_model(base=..., unit_locations=...)`) that KLRazorController and
AGemShadowProjector already expect. This means no other code needs to
change to support the pi0 arm: it is purely a matter of which iv_state
(or non-state) the shadow model is built with.

Zeroing the loaded LoReFT parameters (W=0, b=0) does NOT give a true
no-op: the intervention is
    Phi(h) = h + R^T (W h + b - R h)
so W=0, b=0 collapses to h - R^T R h, which is not h unless R^T R = 0 --
false for an orthonormal R. A genuine pi0 anchor requires bypassing the
intervention's forward entirely, which is what this module does via a
per-instance method override (a standard, non-invasive technique; it
does not modify the pyreft/pyvene library, only the individual module
instances of this one throwaway shadow copy).
"""

import types
from typing import Dict


def _identity_forward(self, base, source=None, subspaces=None):
    """Replacement forward: returns the hidden state unchanged, i.e. the
    intervention contributes nothing. Signature matches pyreft's
    intervention modules (base, source, subspaces)."""
    return base


def make_identity_shadow(shadow_reft) -> None:
    """Mutates shadow_reft in place: every intervention module's forward
    method is overridden to the identity function. After this call,
    forwarding shadow_reft with any unit_locations reproduces the frozen
    backbone's own output distribution (pi0), regardless of whatever
    (untrained, randomly-initialized) parameters the modules hold."""
    n = 0
    for iv_key, iv_list in shadow_reft.interventions.items():
        mods = iv_list if isinstance(iv_list, (list, tuple)) else [iv_list]
        for mod in mods:
            mod.forward = types.MethodType(_identity_forward, mod)
            n += 1
    print(f"  [anchor_identity=pi0] {n} intervention module(s) forced to "
          f"identity forward; shadow now reproduces the no-adaptation "
          f"backbone.")


def snapshot_group_state(memory, gid: int) -> Dict[str, dict]:
    """Deep-copies a group's current iv_state, for use as the prev_task
    anchor snapshot taken immediately after a task's save() call. Under
    the default overwrite merge rule this equals exactly that task's own
    trained state (overwrite replaces the group state with the trained
    state), which is the intended meaning of "the immediately preceding
    task's state" for this ablation. Callers should note in the run
    header if merge_rule != overwrite, since then this would reflect the
    blended group state rather than the pure prior task's state."""
    from copy import deepcopy
    if gid not in memory.groups:
        return {}
    return {k: deepcopy(v) for k, v in memory.groups[gid]["iv_state"].items()}
