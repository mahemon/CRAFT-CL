"""
litar_router.py - LoReFT-intervention task-agnostic routing (LITAR) for CRAFT.

LITAR is the `cond_head` signal of group_signal_test/group_signals.py, moved
into the training run. It uses the same classifier, pooling and decision rule
as the learned head (router_head.py), but every fingerprint is read THROUGH a
stored group intervention state instead of the plain backbone.

  bank      n_samples train prompts per task (cut to their prompt tokens),
            stored when the task is committed. Same seeded draw as the head.
  z_k(x)    group k's state loaded, task-private T3-l sites set to identity
            (W = R^T, b = 0), one forward pass; mean of the last n_pos tokens
            at every layer, concatenated.
  p_k       one multinomial logistic regression per group on bank B_k
            (RouterHead: standardised features, CE + l2 * ||A||^2 / T, LBFGS),
            refit from zero.
  decision  p(tau | x) = mean_k p_k(tau | x), averaged over the first m test
            prompts of the task (--router_probe_size, default 16); argmax.
  load      group state of M[tau_hat] plus the private psi of tau_hat,
            exactly as the head does.

Keeping it exact. A group's bank is rebuilt from zero whenever its stored
state changes (checked by a checksum of the state at every routing call), and
every group gets the rows of each new task. Banks and classifiers are built
lazily on the first routing call that needs them, so a run that never routes
with LITAR pays only for storing the prompts. Banks are not pickled into
resume_state.pt or the eval snapshots; they are recomputed from the stored
prompts on first use.

Selected with --inference_router litar or --extra_eval_routers litar.
"""
from __future__ import annotations

import gc
import json
import os
from typing import Dict, List, Optional

import torch

from utils.router_head import RouterHead, _collate

LITAR_LOG_FILE = "litar_routing_log.jsonl"    # one JSON line per decision


# ──────────────────────────────────────────────────────────────────────────
# helpers (same rules as group_signals.py)
# ──────────────────────────────────────────────────────────────────────────
def _prompt_example(item) -> dict:
    """Keep what the collator needs and cut a train example to its prompt
    tokens (the labels == -100 prefix), so bank prompts look like test prompts."""
    ids = torch.as_tensor(item["input_ids"]).clone()
    locs = torch.as_tensor(item["intervention_locations"]).clone()
    L = int(ids.numel())
    lab = item.get("labels", None)
    if lab is not None:
        nz = torch.nonzero(torch.as_tensor(lab) != -100).flatten()
        if nz.numel():
            L = max(1, int(nz[0]))
    ex = {"input_ids": ids[:L].clone()}
    if locs.numel() and int(locs.max()) >= L:
        locs = locs.clamp(max=L - 1)
    ex["intervention_locations"] = locs.tolist()
    if "subspaces" in item:
        ex["subspaces"] = item["subspaces"]
    return ex


def _collate_right(examples, tokenizer):
    """Right padding, as in group_signals.py. compute_metrics leaves the
    tokenizer on left padding after an evaluation, and left padding would move
    the prompts away from their intervention locations."""
    old = tokenizer.padding_side
    tokenizer.padding_side = "right"
    try:
        return _collate(examples, tokenizer)
    finally:
        tokenizer.padding_side = old


def _iv_module(iv):
    return iv[0] if isinstance(iv, (list, tuple)) else iv


@torch.no_grad()
def _identity_private_sites(shell, keys) -> int:
    """Private T3-l sites become a no-op edit: W = R^T, b = 0, so
    h + R^T(W h + b - R h) = h. Otherwise they keep the fixed-seed random
    init and add the same random edit under every group."""
    n = 0
    for k in keys:
        if k not in shell.interventions:
            continue
        mod = _iv_module(shell.interventions[k])
        R = mod.rotate_layer.weight                       # (d, r), orthogonal
        ls = mod.learned_source                           # Linear d -> r
        ls.weight.copy_(R.detach().t().to(ls.weight.dtype))
        if ls.bias is not None:
            ls.bias.zero_()
        n += 1
    return n


def _private_keys(memory) -> set:
    keys = set()
    for st in getattr(memory, "t3l_tasks", {}).values():
        keys |= set(st.keys())
    return keys


@torch.no_grad()
def _pool(hs, att, layer_ids, n_pos) -> torch.Tensor:
    """Mean of the last n_pos valid tokens, per layer, concatenated (fp16, CPU)."""
    rows = []
    for b in range(att.shape[0]):
        idx = torch.nonzero(att[b].bool(), as_tuple=False).flatten()
        idx = idx[-n_pos:] if idx.numel() > n_pos else idx
        rows.append(torch.cat([hs[li + 1][b, idx, :].float().mean(0) for li in layer_ids]))
    return torch.stack(rows).to(torch.float16).cpu()


def _state_sig(iv_state) -> tuple:
    """Checksum of a stored group state; any change forces a bank rebuild."""
    out = []
    for k in sorted(iv_state):
        v = iv_state[k]
        items = sorted(v.items()) if isinstance(v, dict) else [("", v)]
        for n, t in items:
            if torch.is_tensor(t):
                t64 = t.detach().double()
                out.append((k, n, tuple(t.shape), float(t64.sum()), float(t64.abs().sum())))
    return tuple(out)


def _release():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ──────────────────────────────────────────────────────────────────────────
# the router
# ──────────────────────────────────────────────────────────────────────────
class LitarRouter:
    def __init__(self, layer_ids: List[int], n_pos: int, n_samples: int = 64,
                 l2: float = 1e-2, iters: int = 200):
        self.layer_ids = list(layer_ids)
        self.n_pos = int(n_pos)
        self.n_samples = int(n_samples)
        self.l2 = float(l2)
        self.iters = int(iters)
        self.prompts: Dict[str, List[dict]] = {}     # task -> bank prompts
        self.log: List[dict] = []
        self._reset_cache()

    def _reset_cache(self):
        self._sig: Dict[int, tuple] = {}
        self._banks: Dict[int, Dict[str, torch.Tensor]] = {}
        self._heads: Dict[int, RouterHead] = {}

    def __getstate__(self):
        d = dict(self.__dict__)
        d["_sig"], d["_banks"], d["_heads"] = {}, {}, {}
        return d

    def __setstate__(self, d):
        self.__dict__.update(d)
        if not hasattr(self, "_banks"):
            self._reset_cache()

    # -- stream time -------------------------------------------------------
    def add_task(self, task_name: str, dataset, seed: int) -> None:
        """Store the bank prompts of a committed task. Seeded draw without
        replacement, the same one features_from_dataset makes for the head.
        Uses its own generator, so the global RNG is untouched."""
        n = min(self.n_samples, len(dataset))
        g = torch.Generator().manual_seed(int(seed))
        order = torch.randperm(len(dataset), generator=g)[:n].tolist()
        self.prompts[str(task_name)] = [_prompt_example(dataset[i]) for i in order]
        print(f"  [litar] stored {n} bank prompts for {task_name} "
              f"(tasks stored: {len(self.prompts)})")

    # -- fingerprints ------------------------------------------------------
    @torch.no_grad()
    def _fingerprints(self, memory, gid, ex_by_task, tokenizer, build_reft_fn,
                      teardown_fn, bs, dev) -> Dict[str, torch.Tensor]:
        """One fresh shell with group gid loaded and private sites at identity;
        fingerprints for every list of examples in ex_by_task. Batches never
        mix tasks (per-task positions can differ in shape)."""
        shell = build_reft_fn()
        out = {}
        try:
            if not memory.load(shell, gid=gid):
                raise RuntimeError(f"memory.load failed for G{gid}")
            _identity_private_sites(shell, _private_keys(memory))
            shell.eval()
            for iv in shell.interventions.values():
                _iv_module(iv).eval()
            for key, exs in ex_by_task.items():
                feats = []
                for s in range(0, len(exs), bs):
                    batch = _collate_right(exs[s:s + bs], tokenizer)
                    att = batch["attention_mask"].to(dev)
                    base = {"input_ids": batch["input_ids"].to(dev),
                            "attention_mask": att, "use_cache": False,
                            "output_hidden_states": True}
                    locs = batch["intervention_locations"]
                    locs = (locs.permute(1, 0, 2).tolist() if hasattr(locs, "permute")
                            else locs)
                    _, o = shell(base=base, unit_locations={"sources->base": (None, locs)})
                    feats.append(_pool(o.hidden_states, att, self.layer_ids, self.n_pos))
                    del o
                out[key] = torch.cat(feats, dim=0)
        finally:
            if teardown_fn is not None:
                teardown_fn(shell)
            del shell
            _release()
        return out

    def _ensure(self, memory, tokenizer, build_reft_fn, teardown_fn, bs, dev, strict):
        """Bring banks and per-group classifiers in line with the current memory."""
        t2g = {t: int(g) for t, g in memory.task_to_group.items() if int(g) in memory.groups}
        missing = [t for t in t2g if t not in self.prompts]
        if missing:
            msg = (f"no stored LITAR prompts for {missing}; this state was trained "
                   f"by a train.py without LITAR")
            if strict:
                raise RuntimeError(msg)
            print(f"  [litar] WARNING: {msg}; those tasks cannot be predicted.")
        tasks = [t for t in t2g if t in self.prompts]
        gids = sorted(set(t2g.values()))
        for g in list(self._banks):
            if g not in gids:
                self._banks.pop(g, None); self._sig.pop(g, None); self._heads.pop(g, None)
        for g in gids:
            sig = _state_sig(memory.groups[g]["iv_state"])
            if self._sig.get(g) != sig:
                self._banks[g], self._sig[g] = {}, sig
                self._heads.pop(g, None)
            need = {t: self.prompts[t] for t in tasks if t not in self._banks[g]}
            if need:
                self._banks[g].update(self._fingerprints(
                    memory, g, need, tokenizer, build_reft_fn, teardown_fn, bs, dev))
                self._heads.pop(g, None)
            h = self._heads.get(g)
            if h is None or sorted(h.classes) != sorted(tasks):
                h = RouterHead(layer_ids=[], n_pos=0, l2=self.l2, iters=self.iters)
                for t in tasks:
                    h.add_task(t, self._banks[g][t])
                rec = h.fit(device=dev)
                self._heads[g] = h
                print(f"  [litar] G{g}: {len(need)} task(s) fingerprinted, refit from zero: "
                      f"classes={rec['n_classes']} examples={rec['n_examples']} "
                      f"train_acc={rec['train_acc']} holdout_acc={rec['holdout_acc']}")
        return tasks, gids

    # -- inference ---------------------------------------------------------
    @torch.no_grad()
    def route(self, eval_reft, eval_task, memory, base_model, tokenizer, args,
              dataset, build_reft_fn=None, teardown_fn=None) -> bool:
        strict = os.environ.get("ROUTER_STRICT", "0") == "1"
        if getattr(args, "act_fn", None) not in (None, "linear"):
            print(f"  [litar] WARNING: act_fn={args.act_fn}; W = R^T, b = 0 is not an "
                  f"identity edit for a non-linear act_fn, so private sites still act.")
        dev = next(base_model.parameters()).device
        bs = max(1, int(getattr(args, "eval_batch_size", 4) or 4))
        m = min(int(getattr(args, "router_probe_size", 16) or 16), len(dataset))
        try:
            if build_reft_fn is None:
                raise RuntimeError("route_and_load was called without build_reft_fn")
            if m <= 0:
                raise RuntimeError("no test prompts")
            tasks, gids = self._ensure(memory, tokenizer, build_reft_fn,
                                       teardown_fn, bs, dev, strict)
            if not tasks or not gids:
                raise RuntimeError("no stored tasks or groups to route to")
            test = [_prompt_example(dataset[i]) for i in range(m)]
            P = torch.zeros(m, len(tasks))
            for g in gids:
                z = self._fingerprints(memory, g, {"test": test}, tokenizer,
                                       build_reft_fn, teardown_fn, bs, dev)["test"]
                Pg = self._heads[g].predict_proba(z)              # columns = head.classes
                col = {c: j for j, c in enumerate(self._heads[g].classes)}
                P += Pg[:, [col[t] for t in tasks]]
            P /= len(gids)
        except Exception as e:
            if strict:
                raise
            print(f"  [litar] cannot route {eval_task} ({type(e).__name__}: {e}); "
                  f"oracle fallback")
            self._write(args, {"router": "litar", "task": eval_task,
                               "fallback": f"{type(e).__name__}: {e}"})
            return memory.load(eval_reft, task_name=eval_task)

        p = P.mean(0)
        j = int(p.argmax())
        pred = tasks[j]
        pred_gid = int(memory.task_to_group[pred])
        oracle_gid = memory.task_to_group.get(eval_task)
        oracle_gid = int(oracle_gid) if oracle_gid is not None else None
        per_prompt = [int(memory.task_to_group[tasks[int(r.argmax())]]) for r in P]
        loaded = memory.load(eval_reft, task_name=pred)
        entry = {
            "router": "litar", "task": eval_task, "pred_task": pred,
            "pred_gid": pred_gid, "oracle_gid": oracle_gid,
            "hit": (oracle_gid is not None and pred_gid == oracle_gid),
            "task_hit": pred == eval_task, "conf": round(float(p[j]), 4),
            "m": m, "n_groups": len(gids),
            "per_prompt_group_acc": (round(sum(g == oracle_gid for g in per_prompt) / m, 4)
                                     if oracle_gid is not None else None),
            "probs": {t: round(float(p[i]), 4) for i, t in enumerate(tasks)},
            "t3l_loaded": pred in getattr(memory, "t3l_tasks", {}),
            "loaded": bool(loaded),
        }
        self.log.append(entry)
        hits = sum(1 for e in self.log if e["hit"])
        print(f"  [litar] {eval_task} -> {pred} (G{pred_gid}, oracle G{oracle_gid})  "
              f"conf={entry['conf']:.3f}  per-prompt group acc="
              f"{entry['per_prompt_group_acc']}  m={m}  K={len(gids)}  "
              f"group hits so far {hits}/{len(self.log)}")
        self._write(args, entry)
        return loaded

    @staticmethod
    def _write(args, entry) -> None:
        """Append one decision to <output_dir>/litar_routing_log.jsonl. Appending
        keeps every decision in --eval_only, where each task restores its own memory."""
        out_dir = getattr(args, "output_dir", None)
        if not out_dir:
            return
        try:
            with open(os.path.join(out_dir, LITAR_LOG_FILE), "a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            print(f"  [litar] WARNING: could not write {LITAR_LOG_FILE} "
                  f"({type(e).__name__}: {e})")


def litar_route_and_load(eval_reft, eval_task, memory, base_model, tokenizer, args,
                         dataset, build_reft_fn=None, teardown_fn=None) -> bool:
    """Same signature and return value as agnostic_router.route_and_load."""
    litar: Optional[LitarRouter] = getattr(memory, "litar", None)
    if litar is None:
        msg = "this memory has no LITAR state (trained by a train.py without LITAR)"
        if os.environ.get("ROUTER_STRICT", "0") == "1":
            raise RuntimeError(msg)
        print(f"  [litar] {msg}; oracle fallback for {eval_task}")
        return memory.load(eval_reft, task_name=eval_task)
    # Leave the global RNG exactly as LITAR found it. Every build_reft_fn() call
    # reseeds torch, and evaluation samples (temperature 0.6 without
    # --greedy_decoding), so without this the generated answers, and the scores,
    # differ from the oracle row even when LITAR loads the same states.
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        return litar.route(eval_reft, eval_task, memory, base_model, tokenizer, args,
                           dataset, build_reft_fn=build_reft_fn, teardown_fn=teardown_fn)
    finally:
        torch.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)
