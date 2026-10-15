# Use Tuned_router/utils, not CRAFT-CL/utils. Neither folder has an
# __init__.py, so "utils" is a namespace package spanning both; putting this
# folder first makes every utils.* module resolve here, and the check below
# stops the run if any of them was picked up from somewhere else.
import sys
import os

_STUDY_DIR = os.path.dirname(os.path.abspath(__file__))
if sys.path[0] != _STUDY_DIR:
    sys.path.insert(0, _STUDY_DIR)

from utils.task_config import task_config
from utils.dataset import LoReftGLUEDataset, LoReftTRACEDataset
from utils.compute_metrics import compute_metrics

from html import parser

import copy
import gc
import json
import datetime
import argparse

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Set, Tuple
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    DataCollatorForSeq2Seq, set_seed, TrainingArguments,
    TrainerCallback,
)
from huggingface_hub import login

# from utils.task_config import task_config
# from utils.dataset import LoReftGLUEDataset, LoReftTRACEDataset
# from utils.compute_metrics import compute_metrics

# from utils.CL_topology import (
#     capture_before_flat, compute_param_delta,
#     compute_fisher, GradientTrajectoryCallback, flush_log,
#     log_delta_assignment, log_prototype_state, log_prototype_merge,
#     log_memory_merge, log_task_private_save, reset_logs,
# )

from utils.CL_topology import (
    capture_before_flat, compute_param_delta,
    compute_fisher, GradientTrajectoryCallback, flush_log,
    log_delta_assignment, log_prototype_state, log_prototype_merge,
    log_memory_merge, log_task_private_save, reset_logs,
)

_LOCAL_UTILS = os.path.join(_STUDY_DIR, "utils") + os.sep
for _m in ("utils.task_config", "utils.dataset",
           "utils.compute_metrics", "utils.CL_topology"):
    _f = os.path.abspath(sys.modules[_m].__file__)
    if not _f.startswith(_LOCAL_UTILS):
        raise SystemExit(f"FATAL: {_m} loaded from {_f}, expected {_LOCAL_UTILS}")
print(f"  [utils] task_config, dataset, compute_metrics, CL_topology <- {_LOCAL_UTILS}")

from pyreft import (
    get_reft_model, ReftConfig, ReftTrainerForCausalLM,
    NoreftIntervention, LoreftIntervention, ConsreftIntervention,
    LobireftIntervention, DireftIntervention, NodireftIntervention,
    ReftDataCollator,
)

from utils.router_head import RouterHead, features_from_dataset, _unique_layers
from utils.agnostic_router import (
    route_and_load as _agn_route_and_load, dump_report as agn_dump_report,
)
from utils.litar_router import LitarRouter, litar_route_and_load

for _m in ("utils.router_head", "utils.agnostic_router", "utils.litar_router"):
    _f = os.path.abspath(sys.modules[_m].__file__)
    if not _f.startswith(_LOCAL_UTILS):
        raise SystemExit(f"FATAL: {_m} loaded from {_f}, expected {_LOCAL_UTILS}")


def route_and_load(eval_reft, eval_task, memory, base_model, tokenizer, args,
                   dataset, build_reft_fn=None, teardown_fn=None):
    """--inference_router litar goes to utils/litar_router.py; oracle and
    learned_head go to utils/agnostic_router.py."""
    _route = (litar_route_and_load if args.inference_router == "litar"
              else _agn_route_and_load)
    return _route(eval_reft, eval_task, memory, base_model, tokenizer, args,
                  dataset, build_reft_fn=build_reft_fn, teardown_fn=teardown_fn)


# Group-state merge rules. Optional: only --merge_rule != "overwrite" needs it,
# and main() fails fast with a clear message if it is missing in that case.
try:
    from utils.merge_rules import (
        MERGE_RULES, merge_group_state, compute_iv_fisher, accumulate_fisher,
        kl_scale_from_halflife,
    )
    _HAS_MERGE_RULES = True
except ImportError as _merge_import_err:
    MERGE_RULES = ("overwrite", "ema", "fisher", "prototype", "kl")
    merge_group_state = compute_iv_fisher = accumulate_fisher = None
    kl_scale_from_halflife = None
    _HAS_MERGE_RULES = False

try:
    import peft
    is_peft_available = True
except ModuleNotFoundError:
    is_peft_available = False

device = "cuda" if torch.cuda.is_available() else "cpu"

TRACE_CL_ORDER = [
    "C-STANCE", "FOMC", "MeetingBank", "Py150",
    "ScienceQA", "NumGLUE-cm", "NumGLUE-ds", "20Minuten",
]
TRACE_TASK_TYPES = {
    "C-STANCE": 1, "FOMC": 1, "ScienceQA": 1, "dbpedia": 1,
    "amazon": 1,   "yahoo": 1, "agnews": 1,   "yelp": 1,
    "BoolQA": 1,   "QQP": 1,
    "MeetingBank": 2, "20Minuten": 2,
    "Py150": 3,
    "NumGLUE-cm": 4, "NumGLUE-ds": 4,
}
TASK_TYPE_NAMES = {1: "cls", 2: "summ", 3: "code", 4: "reasoning"}
TASK_EPOCH_SCHEDULE = {
    "C-STANCE": 3, "FOMC": 2,   "MeetingBank": 5, "Py150": 5,
    "ScienceQA": 2, "NumGLUE-cm": 5, "NumGLUE-ds": 5, "20Minuten": 5,
    "dbpedia": 3, "amazon": 2,  "yahoo": 3, "agnews": 2,
    "yelp": 3, "BoolQA": 3, "QQP": 3,
}

MANUAL_TASK_GROUPS: Dict[str, str] = {
    "C-STANCE":   "classification",
    "FOMC":       "classification",
    "ScienceQA":  "classification",
    "dbpedia":    "classification",
    "amazon":     "classification",
    "yahoo":      "classification",
    "agnews":     "classification",
    "yelp":       "classification",
    "BoolQA":     "classification",
    "QQP":        "classification",
    "MeetingBank":"summarisation",
    "20Minuten":  "summarisation",
    "Py150":      "code",
    "NumGLUE-cm": "reasoning",
    "NumGLUE-ds": "reasoning",
}

def _parse_manual_groups_arg(spec: Optional[str]) -> Dict[str, str]:
    """
    Parse a --manual_groups override of the form
      'C-STANCE:classification,FOMC:classification,MeetingBank:summarisation'
    Returns a dict overlaid on MANUAL_TASK_GROUPS (later entries win).
    """
    if not spec:
        return dict(MANUAL_TASK_GROUPS)
    out = dict(MANUAL_TASK_GROUPS)
    for pair in spec.split(","):
        pair = pair.strip()
        if not pair:
            continue
        if ":" not in pair:
            raise ValueError(f"Bad manual_groups entry {pair!r}: need 'task:group'.")
        t, g = pair.split(":", 1)
        out[t.strip()] = g.strip()
    return out
LAYER_MULT = 2

TRACE_ALPACA_HEADER = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n"
)

# Mistral-7B-Instruct-v0.3's own template puts the system prompt only into
# the LAST user turn, so utils/dataset.py's prompt-only and prompt+answer
# sequences diverge after "[INST]" and the prompt is left in the loss. This
# template puts it into the FIRST user turn instead, so the prompt is an exact
# prefix of the full sequence. Output is otherwise identical to Mistral's.
MISTRAL_CHAT_TEMPLATE = (
    "{%- if messages[0]['role'] == 'system' %}"
    "{%- set system_message = messages[0]['content'] %}"
    "{%- set loop_messages = messages[1:] %}"
    "{%- else %}{%- set loop_messages = messages %}{%- endif %}"
    "{{- bos_token }}"
    "{%- for message in loop_messages %}"
    "{%- if message['role'] == 'user' %}"
    "{%- if loop.first and system_message is defined %}"
    "{{- '[INST] ' + system_message + '\\n\\n' + message['content'] + '[/INST]' }}"
    "{%- else %}{{- '[INST] ' + message['content'] + '[/INST]' }}{%- endif %}"
    "{%- elif message['role'] == 'assistant' %}"
    "{{- ' ' + message['content'] + eos_token }}"
    "{%- endif %}"
    "{%- endfor %}"
)

def _find_question_start(tokenizer) -> int:
    header_ids = tokenizer(
        TRACE_ALPACA_HEADER,
        add_special_tokens=False,
        return_tensors="pt",
    )["input_ids"][0]
    # Special tokens the tokenizer prepends (BOS). 1 for Llama / Gemma /
    # Mistral / DeepSeek-R1-Distill-Llama (unchanged from the old "+ 1");
    # 0 for Qwen2.5, which adds no BOS.
    _n_special = (len(tokenizer("x")["input_ids"])
                  - len(tokenizer("x", add_special_tokens=False)["input_ids"]))
    question_start = len(header_ids) + _n_special
    print(f"  [question_start] header={len(header_ids)} tokens  "
          f"special_prefix={_n_special}  "
          f"→ f-stream starts at position {question_start}")
    return question_start

def _stratify_layers(
    base_layers: List[int],
    shallow_frac: float = 0.33,
    deep_frac:    float = 0.38,
) -> Tuple[List[int], List[int], List[int]]:

    n  = len(base_layers)
    n1 = max(1, round(n * shallow_frac))
    n3 = max(1, round(n * deep_frac))
    n2 = max(1, n - n1 - n3)
    t1 = base_layers[:n1]
    t2 = base_layers[n1: n1 + n2]
    t3 = base_layers[n1 + n2:]
    print(f"  [stratify] n={n}  "
          f"T1={len(t1)} (layers {t1[0]}-{t1[-1]})  "
          f"T2={len(t2)} (layers {t2[0]}-{t2[-1]})  "
          f"T3={len(t3)} (layers {t3[0]}-{t3[-1]})")
    return t1, t2, t3


def _l_start_idx(n_interventions: int) -> int:
    """l-stream starts at the midpoint. layer_mult is always 2."""
    return n_interventions // 2


def _get_task_layers(base_layers: List[int]) -> List[int]:
    """Task layers = base × 2 (fN + lN, same N)."""
    return list(base_layers) * LAYER_MULT

def _extract_iv_state(reft_model) -> Dict[str, dict]:
    return {
        key: copy.deepcopy(
            (iv[0] if isinstance(iv, (list, tuple)) else iv).state_dict()
        )
        for key, iv in reft_model.interventions.items()
    }

def _load_iv_state(reft_model, iv_state: Dict[str, dict]):
    for key, iv_list in reft_model.interventions.items():
        mod = iv_list[0] if isinstance(iv_list, (list, tuple)) else iv_list
        if key in iv_state:
            mod.load_state_dict(iv_state[key])

def _teardown_reft_model(reft_model) -> None:
    try:
        ivs = getattr(reft_model, "interventions", {})
        for _key, iv_list in ivs.items():
            mod = iv_list[0] if isinstance(iv_list, (list, tuple)) else iv_list
            try:
                mod.to("cpu")
            except Exception:
                pass
    except Exception:
        pass

    for meth in ("disable_forward_hooks",
                 "disable_model_hooks",
                 "remove_all_hooks",
                 "_cleanup_hooks"):
        fn = getattr(reft_model, meth, None)
        if callable(fn):
            try:
                fn()
                break
            except Exception:
                continue

def _gpu_mem(tag: str) -> None:
    """Print allocated GPU memory and the peak since the current task started."""
    if torch.cuda.is_available():
        print(f"  [gpu_mem] {tag}: "
              f"{torch.cuda.memory_allocated() / 2**30:.2f} GiB allocated, "
              f"peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB since task start")


def _print_cuda_census(top: int = 15) -> None:
    """On OOM: list the largest live CUDA tensors, so the log shows what holds memory."""
    if not torch.cuda.is_available():
        return
    try:
        print(f"\n  [oom-census] allocated={torch.cuda.memory_allocated() / 2**30:.2f} GiB  "
              f"reserved={torch.cuda.memory_reserved() / 2**30:.2f} GiB")
        agg = {}
        for obj in gc.get_objects():
            try:
                if not (torch.is_tensor(obj) and obj.is_cuda):
                    continue
                key = (tuple(obj.shape), str(obj.dtype))
                n, b = agg.get(key, (0, 0))
                agg[key] = (n + 1, b + obj.element_size() * obj.nelement())
            except Exception:
                continue
        for (shape, dt), (n, b) in sorted(agg.items(), key=lambda kv: -kv[1][1])[:top]:
            print(f"  [oom-census] {b / 2**30:8.2f} GiB  x{n:<4d} {dt:<15} {list(shape)}")
    except Exception as e:
        print(f"  [oom-census] failed: {type(e).__name__}: {e}")


RESUME_STATE_FILE = "resume_state.pt"


def _save_resume_state(output_dir: str, task_idx: int, cl_order, perf_matrix,
                       kl_per_task_log, task_to_group_log, routing_log,
                       memory, router_head) -> None:
    """Everything the task loop carries from one task to the next, written after
    each task so a crashed run can continue with --resume_from. Never fatal."""
    import random
    path = os.path.join(output_dir, RESUME_STATE_FILE)
    tmp = path + ".tmp"
    try:
        state = {
            "version": 1,
            "completed_task_idx": int(task_idx),
            "cl_order": list(cl_order),
            "perf_matrix": perf_matrix,
            "kl_per_task_log": kl_per_task_log,
            "task_to_group_log": task_to_group_log,
            "routing_log": routing_log,
            "memory": memory,
            "router_head": router_head,
            "rng": {
                "torch": torch.get_rng_state(),
                "cuda": (torch.cuda.get_rng_state_all()
                         if torch.cuda.is_available() else None),
                "numpy": np.random.get_state(),
                "python": random.getstate(),
            },
        }
        torch.save(state, tmp)
        os.replace(tmp, path)
        print(f"  [resume] state after task {task_idx + 1} saved → {path}")
    except Exception as e:
        print(f"  [resume] WARNING: could not save resume state "
              f"({type(e).__name__}: {e}); training continues.")
        try:
            os.remove(tmp)
        except OSError:
            pass


def _load_resume_state(path: str, cl_order) -> dict:
    if os.path.isdir(path):
        path = os.path.join(path, RESUME_STATE_FILE)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"--resume_from: no {RESUME_STATE_FILE} at {path}")
    try:
        state = torch.load(path, map_location=None, weights_only=False)
    except TypeError:            # torch without the weights_only argument
        state = torch.load(path, map_location=None)
    if list(state["cl_order"]) != list(cl_order):
        raise ValueError(f"--resume_from: task order differs "
                         f"(saved {state['cl_order']}, now {list(cl_order)})")
    return state


EVAL_SNAPSHOT_DIR = "eval_snapshots"
TRAIN_ARGS_FILE = "train_args.json"
TRAIN_SUMMARY_FILE = "train_summary.json"


def _eval_snapshot_path(output_dir: str, task_idx: int, task_name: str) -> str:
    return os.path.join(output_dir, EVAL_SNAPSHOT_DIR, f"after_{task_idx:02d}_{task_name}.pt")


def _save_eval_snapshot(output_dir: str, task_idx: int, cl_order, kl_per_task_log,
                        task_to_group_log, routing_log, memory, router_head) -> None:
    """State after task task_idx is committed: everything an evaluation of row
    task_idx needs (--skip_eval / --save_eval_snapshots). One file per task, kept."""
    path = _eval_snapshot_path(output_dir, task_idx, cl_order[task_idx])
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    try:
        torch.save({
            "version": 1,
            "task_idx": int(task_idx),
            "task_name": cl_order[task_idx],
            "cl_order": list(cl_order),
            "kl_per_task_log": kl_per_task_log,
            "task_to_group_log": task_to_group_log,
            "routing_log": routing_log,
            "memory": memory,
            "router_head": router_head,
        }, tmp)
        os.replace(tmp, path)
        print(f"  [snapshot] state after task {task_idx + 1} saved → {path}")
    except Exception as e:
        print(f"  [snapshot] WARNING: could not save the evaluation snapshot "
              f"({type(e).__name__}: {e}); this task cannot be evaluated later.")
        try:
            os.remove(tmp)
        except OSError:
            pass


def _load_eval_snapshot(train_dir: str, task_idx: int, cl_order) -> dict:
    path = _eval_snapshot_path(train_dir, task_idx, cl_order[task_idx])
    if not os.path.isfile(path):
        raise FileNotFoundError(f"--eval_only: snapshot not found: {path}")
    try:
        snap = torch.load(path, map_location=None, weights_only=False)
    except TypeError:
        snap = torch.load(path, map_location=None)
    if list(snap["cl_order"]) != list(cl_order):
        raise ValueError(f"--eval_only: task order differs (snapshot {snap['cl_order']}, "
                         f"now {list(cl_order)})")
    return snap


# Arguments an evaluation-only run may set itself; every other argument is
# restored from the training run's train_args.json.
EVAL_ONLY_KEYS = {"eval_only", "eval_from", "output_dir", "cl_eval_file", "per_task_bwd_eval",
                  "inference_router", "extra_eval_routers", "eval_batch_size",
                  "max_n_eval_example", "greedy_decoding", "temperature", "top_p", "top_k",
                  "router_probe_size"}


def _restore_train_args(args):
    """--eval_only: take the training configuration from <eval_from>/train_args.json.
    Evaluation settings (EVAL_ONLY_KEYS) keep the value given on this command line;
    if one is not given, the training run's value is used."""
    import sys
    if not args.eval_from:
        raise SystemExit("--eval_only needs --eval_from <training output dir>")
    path = os.path.join(args.eval_from, TRAIN_ARGS_FILE)
    if not os.path.isfile(path):
        raise SystemExit(f"--eval_only: {path} not found (was the run trained with this train.py?)")
    with open(path) as f:
        saved = json.load(f)
    given = {a[2:].split("=")[0].replace("-", "_") for a in sys.argv[1:] if a.startswith("--")}
    given |= {g[3:] for g in given if g.startswith("no_")}        # --no_per_task_bwd_eval
    for k, v in saved.items():
        if k in EVAL_ONLY_KEYS and k in given:
            continue
        if k in ("skip_eval", "save_eval_snapshots", "resume_from", "eval_only", "eval_from"):
            continue
        setattr(args, k, v)
    args.skip_eval = False
    args.save_eval_snapshots = False
    args.resume_from = None
    if not args.output_dir or "output_dir" not in given:
        args.output_dir = os.path.join(args.eval_from, "eval_" + (
            "val" if str(args.cl_eval_file) == "eval.json" else "test")
            + ("_bwd" if args.per_task_bwd_eval else ""))
    print(f"  [eval-only] training configuration restored from {path}")
    return args


def _release_gpu() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass


# --kl_pred_mask: logits[t] predict token t+1, so the KL keeps position t when
# labels[t+1] is a target, the same positions the shifted CE loss uses. The
# original mask (labels != -100) is one position late and never scores the
# answer distribution of a one-token answer. Off = original behaviour.
_KL_PRED_MASK = False


def _kl_mask(labels, device):
    if not _KL_PRED_MASK:
        return (labels != -100).to(device)
    m = torch.zeros_like(labels, dtype=torch.bool)
    m[..., :-1] = labels[..., 1:] != -100
    return m.to(device)


def _split_iv_state_by_5tier(
    reft_model, iv_state, task_layers,
    tier1_set: Set[int], tier2_set: Set[int], tier3_set: Set[int],
    all_groupwise: bool = False,
) -> Dict[str, Dict[str, dict]]:
    iv_keys = list(reft_model.interventions.keys())
    l_start = _l_start_idx(len(iv_keys))
    buckets = {"t1_f": {}, "t1_l": {}, "t2_f": {}, "t2_l": {},
               "t3_f": {}, "t3_l": {}}
    for i, iv_key in enumerate(iv_keys):
        if i >= len(task_layers): continue
        layer = task_layers[i]
        is_l  = (i >= l_start)
        state = iv_state.get(iv_key, {})
        if layer in tier1_set:
            bucket = "t1_l" if is_l else "t1_f"
        elif layer in tier2_set:
            bucket = "t2_l" if is_l else "t2_f"
        else:   
            bucket = "t3_l" if is_l else "t3_f"
        if all_groupwise and bucket == "t3_l":
            bucket = "t3_f"
        buckets[bucket][iv_key] = state
    return buckets


def _flatten_routing_buckets(
    buckets: Dict[str, Dict[str, dict]],
) -> Tuple[Dict[str, dict], Dict[str, str]]:

    flat: Dict[str, dict] = {}
    iv_to_tier: Dict[str, str] = {}
    for tier_key in ("t1_f", "t1_l", "t2_f", "t2_l", "t3_f"):
        sub = buckets.get(tier_key, {})
        for iv_key, state in sub.items():
            flat[iv_key] = state
            iv_to_tier[iv_key] = tier_key
    return flat, iv_to_tier

def _get_w(iv):
    w = getattr(iv, 'learned_source', None)
    if isinstance(w, torch.nn.Linear): w = w.weight
    if w is None: w = getattr(iv, 'weight', None)
    return w

def _get_r(iv):
    try:
        return iv.rotate_layer.parametrizations.weight[0].base
    except AttributeError:
        rl = getattr(iv, 'rotate_layer', None)
        return getattr(rl, 'weight', None) if rl else None

def _get_b(iv):
    b = getattr(iv, 'bias', None)
    if isinstance(b, torch.nn.Linear): b = b.weight
    return b

_STATE_KEY_DEBUG_PRINTED = False

def _debug_print_state_keys_once(iv_key: str, state: dict) -> None:
    """Print actual state_dict keys the first time they're inspected.
    Helps diagnose pyreft version differences in key naming."""
    global _STATE_KEY_DEBUG_PRINTED
    if _STATE_KEY_DEBUG_PRINTED:
        return
    _STATE_KEY_DEBUG_PRINTED = True
    print(f"  [state_dict_debug] first iv '{iv_key}' has state_dict keys:")
    for k, v in state.items():
        if isinstance(v, torch.Tensor):
            print(f"    {k!r}: Tensor shape={tuple(v.shape)} dtype={v.dtype}")
        else:
            print(f"    {k!r}: {type(v).__name__}")


def _get_w_from_state(state: dict) -> Optional[torch.Tensor]:
    """
    Robust extraction of the W matrix from an intervention state dict,
    tolerant to pyreft version differences.
    """
    if not state:
        return None
    candidates = (
        "learned_source.weight",
        "learned_source.0.weight",
        "learned_source.linear.weight",
        "W",
        "weight",
    )
    for key in candidates:
        v = state.get(key, None)
        if isinstance(v, torch.Tensor) and v.dim() == 2:
            return v
    for key, val in state.items():
        if not isinstance(val, torch.Tensor) or val.dim() != 2:
            continue
        k_low = key.lower()
        if "rotate" in k_low or "original" in k_low or "parametriz" in k_low:
            continue
        if "source" in k_low or "learned" in k_low or "weight" in k_low:
            return val
    return None

def _get_r_from_state(state: dict) -> Optional[torch.Tensor]:
    """Extract R (rotate_layer base) from state dict, tolerant to key variants."""
    if not state:
        return None
    candidates = (
        "rotate_layer.parametrizations.weight.original",
        "rotate_layer.parametrizations.weight.original0",
        "rotate_layer.weight",
        "R",
    )
    for key in candidates:
        v = state.get(key, None)
        if isinstance(v, torch.Tensor):
            return v
    for key, val in state.items():
        if isinstance(val, torch.Tensor) and val.dim() == 2 and "rotate" in key.lower():
            return val
    return None

class WeightDistanceMemory:

    DEFAULT_SEED = 42

    def __init__(
        self,
        tier1_set: Set[int], tier2_set: Set[int], tier3_set: Set[int],
        group_alpha:           float = 0.50,
        assign_threshold:      float = 0.30,
        normalize:             str   = "zscore",   
        aggregate:             str   = "mean_all", 
        tiered_weights:        Tuple[float, float, float] = (0.1, 0.2, 0.7),
        fixed_intervention_seed: int = DEFAULT_SEED,
        flat_routing:          bool = True,
        flat_aggregator:       str  = "vote",  
        flat_vote_quantile:    float = 0.5,
        vote_max_dist:         float = 0.20,
        group_capacity:        int   = 0,
        merge_rule:            str   = "overwrite",
        merge_ema_alpha:       float = 0.5,
        merge_kl_scale:        float = 1.0,
    ):
        assert normalize in ("none", "minmax", "zscore"), f"bad normalize={normalize}"
        assert aggregate in ("mean_all", "tiered"), f"bad aggregate={aggregate}"
        assert flat_aggregator in ("vote", "mean", "robust_mean"), \
            f"bad flat_aggregator={flat_aggregator}"
        self.tier1_set = set(tier1_set)
        self.tier2_set = set(tier2_set)
        self.tier3_set = set(tier3_set)

        self.all_groupwise = False 

        self.group_alpha      = float(group_alpha)
        self.assign_threshold = float(assign_threshold)
        self.normalize        = normalize
        self.aggregate        = aggregate
        self.tiered_weights   = tuple(tiered_weights)
        self.fixed_seed       = int(fixed_intervention_seed)

        self.flat_routing       = bool(flat_routing)
        self.flat_aggregator    = flat_aggregator
        self.flat_vote_quantile = float(flat_vote_quantile)

        self.vote_max_dist      = float(vote_max_dist) if vote_max_dist is not None else 0.0

        # Max tasks per group. <=0 means uncapped (original behaviour). A full
        # group is excluded from routing candidates, so a task that would have
        # joined it opens a fresh group instead.
        self.group_capacity = int(group_capacity or 0)

        assert str(merge_rule).lower() in MERGE_RULES, \
            f"bad merge_rule={merge_rule}; expected one of {MERGE_RULES}"
        self.merge_rule      = str(merge_rule).lower()
        self.merge_ema_alpha = float(merge_ema_alpha)
        self.merge_kl_scale  = float(merge_kl_scale)
        # Running per-parameter Fisher per group, used as F_old by merge_rule=fisher.
        self.group_fisher: Dict[int, Dict[str, Dict[str, torch.Tensor]]] = {}

        # Frozen probe batches, one per member task, kept on CPU. These are
        # what makes incumbent-side routing possible: the router can only ask
        # "what does this arrival do to the group's own data" if the group's
        # own data is still around. 16 examples per task.
        self.group_probes: Dict[int, List[dict]] = {}
        self.merge_log: List[dict] = []

        self.groups: Dict[int, Dict] = {}
        self.t3l_tasks: Dict[str, Dict[str, dict]] = {} 
        self.task_to_group: Dict[str, int] = {}        
        self.group_name_to_gid: Dict[str, int] = {}      
        self.gid_to_group_name: Dict[int, str] = {}     
        self.task_warmup_snapshots: Dict[str, Dict[str, dict]] = {}

        self.init_state: Optional[Dict[str, dict]] = None

        self._init_buckets: Optional[Dict[str, Dict[str, dict]]] = None

        self._next_gid = 0
        self._routing_log: List[dict] = []

        self.router = None

    def set_init_state(self, iv_state: Dict[str, dict]) -> None:
        """Cache the fixed-seed init iv_state (CPU) for delta-based routing."""
        self.init_state = {
            k: _clone_state(v) for k, v in iv_state.items()
        }
        self._init_buckets = None
        n = len(self.init_state)

    def route_manual(
        self,
        task_name: str,
        group_name: str,
    ) -> Tuple[int, bool, Dict]:

        if group_name in self.group_name_to_gid:
            gid = self.group_name_to_gid[group_name]
            is_new = False
        else:
            gid = self._next_gid
            self.group_name_to_gid[group_name] = gid
            self.gid_to_group_name[gid] = group_name
            is_new = True

        debug = {
            "task":              task_name,
            "group_name":        group_name,
            "assigned_gid":      int(gid),
            "n_existing_groups": len(self.groups),
            "decision":          "new_group" if is_new else f"join_G{gid}",
            "routing_mode":      "manual",
            "threshold":         float("nan"),
            "normalize_mode":    "n/a",
            "aggregate_mode":    "n/a",
            "group_n_tasks":     {str(g): len(gd["task_names"])
                                   for g, gd in self.groups.items()},
            "group_task_names":  {str(g): list(gd["task_names"])
                                   for g, gd in self.groups.items()},
            "group_name_map":    {str(g): self.gid_to_group_name.get(g, "?")
                                   for g in self.groups.keys()},
            "raw_distances":     {},
            "normalized":        {},
            "per_tier":          {},
            "per_tier_raw":      {},
            "per_tier_n":        {},
            "per_tier_wn":       {},
            "closest_gid":       int(gid),
            "closest_raw_dist":  float("nan"),
            "closest_norm_dist": float("nan"),
        }
        self._routing_log.append(debug)
        return int(gid), is_new, debug

    def group_is_full(self, gid: int) -> bool:
        if self.group_capacity <= 0:
            return False
        g = self.groups.get(gid)
        return g is not None and len(g.get("task_names", [])) >= self.group_capacity

    def open_gids(self) -> List[int]:
        """Groups still eligible to accept a task, in gid order."""
        return [gid for gid in sorted(self.groups.keys())
                if not self.group_is_full(gid)]

    def has_task(self, task_name: str) -> bool:
        """True if this task has already been trained and stored."""
        return task_name in self.task_to_group

    def n_groups(self) -> int:
        return len(self.groups)

    @property
    def n_protos(self) -> int:
        return self.n_groups()

    def route(
        self,
        warmup_state: Dict[str, dict],
        reft_model,
        task_layers: List[int],
        task_name: str,
    ) -> Tuple[int, bool, Dict]:

        buckets_K = _split_iv_state_by_5tier(
            reft_model, warmup_state, task_layers,
            self.tier1_set, self.tier2_set, self.tier3_set)

        if not self.groups:
            debug = {
                "task":              task_name,
                "n_existing_groups": 0,
                "decision":          "new_group",
                "assigned_gid":      self._next_gid,
                "raw_distances":     {},
                "normalized":        {},
                "threshold":         self.assign_threshold,
                "normalize_mode":    self.normalize,
                "aggregate_mode":    self.aggregate,
            }
            self._routing_log.append(debug)
            return self._next_gid, True, debug

        candidate_gids = self.open_gids()
        if not candidate_gids:
            print(f"  [capacity] every group is at capacity "
                  f"({self.group_capacity} task(s)); '{task_name}' → NEW GROUP.")
            debug = {
                "task":              task_name,
                "n_existing_groups": len(self.groups),
                "group_capacity":    self.group_capacity,
                "decision":          "new_group_capacity_full",
                "assigned_gid":      self._next_gid,
                "raw_distances":     {},
                "normalized":        {},
                "threshold":         self.assign_threshold,
                "normalize_mode":    self.normalize,
                "aggregate_mode":    self.aggregate,
            }
            self._routing_log.append(debug)
            return self._next_gid, True, debug
        if len(candidate_gids) < len(self.groups):
            full = [g for g in sorted(self.groups) if g not in candidate_gids]
            print(f"  [capacity] cap={self.group_capacity}; skipping full "
                  f"group(s) {full}; candidates={candidate_gids}")

        raw_dists: Dict[int, float] = {}
        per_tier_dists: Dict[int, Dict[str, float]] = {}
        per_tier_n_all: Dict[int, Dict[str, Tuple[int, int]]] = {}
        per_tier_wn_all: Dict[int, Dict[str, Tuple[float, float]]] = {}
        per_tier_raw_all: Dict[int, Dict[str, float]] = {}
        per_iv_per_group_d: Dict[int, Dict[str, float]] = {}
        for gid in candidate_gids:
            g = self.groups[gid]
            dist, per_tier, per_tier_n, per_tier_wn, per_tier_raw = self._group_distance(
                buckets_K, g["iv_state"], reft_model, task_layers)
            raw_dists[gid]        = dist
            per_tier_dists[gid]   = per_tier
            per_tier_n_all[gid]   = per_tier_n
            per_tier_wn_all[gid]  = per_tier_wn
            per_tier_raw_all[gid] = per_tier_raw

            if self.flat_routing:
                buckets_G = _split_iv_state_by_5tier(
                    reft_model, g["iv_state"], task_layers,
                    self.tier1_set, self.tier2_set, self.tier3_set)
                init_buckets_local = self._init_buckets if self.init_state is not None else None
                flat_K, _ = _flatten_routing_buckets(buckets_K)
                flat_G, _ = _flatten_routing_buckets(buckets_G)
                flat_init = (_flatten_routing_buckets(init_buckets_local)[0]
                             if init_buckets_local is not None else None)
                per_iv_d, _ = self._per_iv_cosine_distances(flat_K, flat_G, flat_init)
                per_iv_per_group_d[gid] = per_iv_d

        norm_dists = self._normalize_dists(raw_dists)
        vote_counts: Dict[int, int] = {}
        vote_share: Dict[int, float] = {}
        vote_winner: Optional[int] = None
        vote_margin: float = float("nan")
        if (self.flat_routing
                and self.flat_aggregator == "vote"
                and len(per_iv_per_group_d) >= 2
                and per_iv_per_group_d):
            iv_sets = [set(d.keys()) for d in per_iv_per_group_d.values()]
            common_ivs = set.intersection(*iv_sets) if iv_sets else set()
            for iv in common_ivs:
                best_gid = min(per_iv_per_group_d.keys(),
                               key=lambda g: per_iv_per_group_d[g][iv])
                vote_counts[best_gid] = vote_counts.get(best_gid, 0) + 1
            total_votes = sum(vote_counts.values())
            if total_votes > 0:
                for gid in per_iv_per_group_d.keys():
                    vote_share[gid] = vote_counts.get(gid, 0) / total_votes
                vote_winner = max(vote_share, key=vote_share.get)
                sorted_shares = sorted(vote_share.values(), reverse=True)
                vote_margin = (sorted_shares[0] - sorted_shares[1]
                               if len(sorted_shares) > 1 else sorted_shares[0])

        if (self.flat_routing
                and self.flat_aggregator == "vote"
                and vote_winner is not None
                and not np.isnan(vote_margin)):
            closest_gid = int(vote_winner)
            closest_norm_dist = float(norm_dists.get(closest_gid, float("nan")))
            closest_raw_dist  = float(raw_dists.get(closest_gid, float("nan")))
            winner_d = per_iv_per_group_d.get(closest_gid, {})
            if winner_d:
                winner_med = float(np.median(list(winner_d.values())))
            else:
                winner_med = float("nan")
            margin_ok = vote_margin >= self.assign_threshold
            absdist_ok = (np.isnan(winner_med)
                          or self.vote_max_dist is None
                          or self.vote_max_dist <= 0
                          or winner_med <= self.vote_max_dist)
            if margin_ok and absdist_ok:
                is_new = False
                assigned_gid = closest_gid
            else:
                is_new = True
                assigned_gid = self._next_gid
        else:
            closest_gid = min(norm_dists, key=norm_dists.get)
            closest_norm_dist = norm_dists[closest_gid]
            closest_raw_dist  = raw_dists[closest_gid]
            check_dist = closest_norm_dist if self.normalize != "none" else closest_raw_dist
            if check_dist <= self.assign_threshold:
                is_new = False
                assigned_gid = closest_gid
            else:
                is_new = True
                assigned_gid = self._next_gid

        debug = {
            "task":              task_name,
            "n_existing_groups": len(self.groups),
            "routing_mode":      "delta" if self.init_state is not None else "raw",
            "router_method":     getattr(self.router, "name", "unknown"),
            "flat_routing":      bool(self.flat_routing),
            "flat_aggregator":   self.flat_aggregator,
            "group_n_tasks":     {str(gid): len(g["task_names"])
                                   for gid, g in self.groups.items()},
            "group_task_names":  {str(gid): list(g["task_names"])
                                   for gid, g in self.groups.items()},
            "raw_distances":     {str(k): float(v) for k, v in raw_dists.items()},
            "normalized":        {str(k): float(v) for k, v in norm_dists.items()},
            "per_tier":          {str(k): {t: float(d) for t, d in pt.items()}
                                   for k, pt in per_tier_dists.items()},
            "per_tier_raw":      {str(k): {t: float(d) for t, d in pt.items()}
                                   for k, pt in per_tier_raw_all.items()},
            "per_tier_n":        {str(k): {t: [int(nm), int(nt)] for t, (nm, nt) in ptn.items()}
                                   for k, ptn in per_tier_n_all.items()},
            "per_tier_wn":       {str(k): {t: [float(wk), float(wg)] for t, (wk, wg) in ptw.items()}
                                   for k, ptw in per_tier_wn_all.items()},
            "per_iv_dists":      {str(gid): {iv: float(d) for iv, d in piv.items()}
                                   for gid, piv in per_iv_per_group_d.items()},
            "vote_counts":       {str(k): int(v) for k, v in vote_counts.items()},
            "vote_share":        {str(k): float(v) for k, v in vote_share.items()},
            "vote_winner":       (int(vote_winner) if vote_winner is not None else None),
            "vote_margin":       float(vote_margin),
            "closest_gid":       int(closest_gid),
            "closest_raw_dist":  float(closest_raw_dist),
            "closest_norm_dist": float(closest_norm_dist),
            "threshold":         self.assign_threshold,
            "normalize_mode":    self.normalize,
            "aggregate_mode":    self.aggregate,
            "decision":          "new_group" if is_new else f"join_G{assigned_gid}",
            "assigned_gid":      int(assigned_gid),
        }
        self._routing_log.append(debug)
        return int(assigned_gid), is_new, debug

    def save(
        self,
        reft_model,
        task_layers: List[int],
        task_name: str,
        task_idx: int,
        assigned_gid: int,
        is_new_group: bool,
        override_alpha: Optional[float] = None,
        kl_final: Optional[float] = None,
        fisher_new: Optional[Dict[str, Dict[str, torch.Tensor]]] = None,
        task_probe: Optional[dict] = None,
    ) -> Tuple[int, bool]:

        iv_state = _extract_iv_state(reft_model)
        buckets = _split_iv_state_by_5tier(
            reft_model, iv_state, task_layers,
            self.tier1_set, self.tier2_set, self.tier3_set,
            all_groupwise=getattr(self, "all_groupwise", False))

        groupwise_K = {}
        for tier_key in ("t1_f", "t1_l", "t2_f", "t2_l", "t3_f"):
            groupwise_K.update(buckets[tier_key])

        t3_K = buckets["t3_l"]
        self.t3l_tasks[task_name] = {k: _clone_state(v) for k, v in t3_K.items()}

        merge_info: Dict = {"rule": "new_group"}
        if is_new_group:
            assert assigned_gid not in self.groups
            self.groups[assigned_gid] = {
                "iv_state":   {k: _clone_state(v) for k, v in groupwise_K.items()},
                "task_names": [task_name],
                "n_merged":   1,
                "kl_history": [],
            }
            if assigned_gid >= self._next_gid:
                self._next_gid = assigned_gid + 1
            if fisher_new:
                self.group_fisher[assigned_gid] = accumulate_fisher(None, fisher_new)
        else:
            assert assigned_gid in self.groups
            g = self.groups[assigned_gid]
            if self.group_is_full(assigned_gid):
                print(f"  [capacity] WARNING: G{assigned_gid} already holds "
                      f"{len(g['task_names'])} task(s) with cap="
                      f"{self.group_capacity}; joining anyway "
                      f"(routing bypassed the capacity filter).")
            n_old = len(g["task_names"])
            if self.merge_rule == "overwrite" or merge_group_state is None:
                g["iv_state"] = {k: _clone_state(v) for k, v in groupwise_K.items()}
                merge_info = {"rule": "overwrite", "alpha": 1.0}
            else:
                merged, merge_info = merge_group_state(
                    g["iv_state"], groupwise_K,
                    rule=self.merge_rule,
                    ema_alpha=self.merge_ema_alpha,
                    n_old=n_old,
                    kl_final=kl_final,
                    kl_scale=self.merge_kl_scale,
                    fisher_new=fisher_new,
                    fisher_old=self.group_fisher.get(assigned_gid),
                )
                g["iv_state"] = merged
            if fisher_new:
                self.group_fisher[assigned_gid] = accumulate_fisher(
                    self.group_fisher.get(assigned_gid), fisher_new)
            g["task_names"].append(task_name)
            g["n_merged"] += 1
            if "kl_history" not in g:
                g["kl_history"] = []
            _a = merge_info.get("alpha", float("nan"))
            print(f"  [merge] G{assigned_gid} ← '{task_name}'  "
                  f"rule={merge_info.get('rule')}"
                  f"{' (requested=' + merge_info['requested'] + ')' if merge_info.get('requested') else ''}  "
                  f"mean_weight_on_new={_a:.4f}  "
                  f"blended={merge_info.get('n_blended', 0)} "
                  f"copied={merge_info.get('n_copied', 0)}"
                  + (f"  [{merge_info['reason']}]" if merge_info.get("reason") else ""))
        self.merge_log.append({"task": task_name, "gid": int(assigned_gid),
                               "is_new_group": bool(is_new_group), **merge_info})
        if kl_final is not None and assigned_gid in self.groups:
            self.groups[assigned_gid]["kl_history"].append(float(kl_final))

        self.task_to_group[task_name] = assigned_gid
        self.task_warmup_snapshots[task_name] = {
            k: _clone_state(v) for k, v in groupwise_K.items()
        }

        # Retain this task's probe as part of the group's evidence, on CPU.
        if task_probe is not None:
            self.group_probes.setdefault(assigned_gid, []).append(
                {k: (v.detach().cpu() if torch.is_tensor(v) else v)
                 for k, v in task_probe.items()})

        g_now = self.groups[assigned_gid]
        n_groupwise_keys = len(groupwise_K)
        n_t3_keys        = len(t3_K)
        gname = self.gid_to_group_name.get(assigned_gid, None)
        gtag  = f" [{gname}]" if gname else ""
        kl_str = f"  kl_final={kl_final:.4f}" if kl_final is not None else ""
        if is_new_group:
            print(f"  [memory] Task '{task_name}' → G{assigned_gid}{gtag} [NEW GROUP]  "
                  f"groupwise_ivs={n_groupwise_keys} (T1+T2+T3_f)  "
                  f"t3l_taskwise_ivs={n_t3_keys} (T3_l){kl_str}  "
                  f"(now {g_now['n_merged']} task in G{assigned_gid})")
        else:
            print(f"  [memory] Task '{task_name}' → G{assigned_gid}{gtag} "
                  f"[OVERWRITE — KL-restrained trained W stored as new group state]  "
                  f"groupwise_ivs={n_groupwise_keys} (T1+T2+T3_f)  "
                  f"t3l_taskwise_ivs={n_t3_keys} (T3_l){kl_str}  "
                  f"(now {g_now['n_merged']} tasks in G{assigned_gid}: {g_now['task_names']})")

        return assigned_gid, is_new_group

    def evict_task_and_allocate_new_group(
        self, task_name: str, previous_gid: int
    ) -> int:

        if previous_gid in self.groups:
            g = self.groups[previous_gid]
            if task_name in g.get("task_names", []):
                g["task_names"].remove(task_name)
                g["n_merged"] = max(0, int(g.get("n_merged", 1)) - 1)

        self.t3l_tasks.pop(task_name, None)
        self.task_warmup_snapshots.pop(task_name, None)

        new_gid = int(self._next_gid)
        self._next_gid = new_gid + 1
        self.task_to_group[task_name] = new_gid
        print(f"  [evict] Task '{task_name}' removed from G{previous_gid}; "
              f"allocated fresh group G{new_gid} for retraining.")
        return new_gid

    def load(self, reft_model, task_name: str = None, gid: int = None) -> bool:

        if task_name is not None:
            if task_name not in self.task_to_group:
                return False
            gid = self.task_to_group[task_name]
            if gid not in self.groups:
                return False
            _load_iv_state(reft_model, self.groups[gid]["iv_state"])
            if task_name in self.t3l_tasks:
                _load_iv_state(reft_model, self.t3l_tasks[task_name])
            return True
        elif gid is not None:
            if gid not in self.groups:
                return False
            _load_iv_state(reft_model, self.groups[gid]["iv_state"])
            return True
        else:
            raise ValueError("must pass task_name or gid")

    def save_to_disk(self, ckpt_dir: str):
        """Checkpoint group and t3l state to disk."""
        os.makedirs(ckpt_dir, exist_ok=True)
        for gid, g in self.groups.items():
            gdir = os.path.join(ckpt_dir, f"group_{gid}")
            os.makedirs(gdir, exist_ok=True)
            _save_state_dict(g["iv_state"], os.path.join(gdir, "iv_state.pt"))
            with open(os.path.join(gdir, "meta.json"), "w") as f:
                json.dump({"task_names": g["task_names"],
                           "n_merged":   g["n_merged"]}, f, indent=2)
        tdir = os.path.join(ckpt_dir, "t3l_private")
        os.makedirs(tdir, exist_ok=True)
        for tname, st in self.t3l_tasks.items():
            safe = tname.replace("/", "_")
            _save_state_dict(st, os.path.join(tdir, f"{safe}.pt"))
        with open(os.path.join(ckpt_dir, "task_to_group.json"), "w") as f:
            json.dump(self.task_to_group, f, indent=2)

    def get_summary(self) -> Dict:
        return {
            "n_groups":       self.n_groups(),
            "task_to_group":  dict(self.task_to_group),
            "groups": {
                str(gid): {"task_names": g["task_names"], "n_merged": g["n_merged"]}
                for gid, g in self.groups.items()
            },
            "n_t3l_tasks":    len(self.t3l_tasks),
            "router": {
                "name":   getattr(self.router, "name", "unknown"),
                "config": dict(getattr(self.router, "kwargs", {})),
            },
            "hyperparams": {
                "group_alpha":      self.group_alpha,
                "assign_threshold": self.assign_threshold,
                "normalize":        self.normalize,
                "aggregate":        self.aggregate,
                "tiered_weights":   list(self.tiered_weights),
                "fixed_seed":       self.fixed_seed,
                "all_groupwise":    bool(getattr(self, "all_groupwise", False)),
            },
        }

    def get_routing_log(self) -> List[dict]:
        return list(self._routing_log)

    def _group_distance(
        self,
        buckets_K: Dict[str, Dict[str, dict]],
        group_state: Dict[str, dict],
        reft_model,
        task_layers: List[int],
    ) -> Tuple[float,
               Dict[str, float],
               Dict[str, Tuple[int, int]],
               Dict[str, Tuple[float, float]],
               Dict[str, float]]:

        buckets_G = _split_iv_state_by_5tier(
            reft_model, group_state, task_layers,
            self.tier1_set, self.tier2_set, self.tier3_set)
        use_delta = self.init_state is not None
        init_buckets: Optional[Dict[str, Dict[str, dict]]] = None
        if use_delta:
            if self._init_buckets is None:
                self._init_buckets = _split_iv_state_by_5tier(
                    reft_model, self.init_state, task_layers,
                    self.tier1_set, self.tier2_set, self.tier3_set)
            init_buckets = self._init_buckets

        if self.flat_routing:
            return self._group_distance_flat(
                buckets_K, buckets_G, init_buckets, reft_model, task_layers,
            )

        per_tier:     Dict[str, float]              = {}
        per_tier_n:   Dict[str, Tuple[int, int]]    = {}  
        per_tier_wn:  Dict[str, Tuple[float, float]] = {}  
        per_tier_raw: Dict[str, float]              = {} 

        for tier_key in ("t1_f", "t1_l", "t2_f", "t2_l"):
            kb = buckets_K.get(tier_key, {})
            gb = buckets_G.get(tier_key, {})
            n_total = len(kb)
            ib = init_buckets.get(tier_key, {}) if init_buckets is not None else None

            if kb:
                first = next(iter(kb))
                _debug_print_state_keys_once(first, kb[first])

            dist, dist_raw, n_matched, ntot, wk_norm, wg_norm = (
                self.router.compute_tier_distance(kb, gb, ib))

            n_total_logged = ntot if ntot > 0 else n_total

            per_tier[tier_key]     = float(dist)
            per_tier_raw[tier_key] = float(dist_raw)
            per_tier_n[tier_key]   = (int(n_matched), int(n_total_logged))
            per_tier_wn[tier_key]  = (float(wk_norm), float(wg_norm))

        if self.aggregate == "tiered":
            w1, w2 = self.tiered_weights[0], self.tiered_weights[1]
            t1_vals = [v for k, v in per_tier.items() if k.startswith("t1_") and not np.isnan(v)]
            t2_vals = [v for k, v in per_tier.items() if k.startswith("t2_") and not np.isnan(v)]
            parts, weights = [], []
            if t1_vals: parts.append(float(np.mean(t1_vals))); weights.append(w1)
            if t2_vals: parts.append(float(np.mean(t2_vals))); weights.append(w2)
            if parts:
                w_sum = sum(weights)
                agg = sum(p * w for p, w in zip(parts, weights)) / w_sum
            else:
                agg = float("nan")
        else:  
            valid = [v for v in per_tier.values() if not np.isnan(v)]
            agg = float(np.mean(valid)) if valid else float("nan")

        return float(agg), per_tier, per_tier_n, per_tier_wn, per_tier_raw

    def _group_distance_flat(
        self,
        buckets_K: Dict[str, Dict[str, dict]],
        buckets_G: Dict[str, Dict[str, dict]],
        init_buckets: Optional[Dict[str, Dict[str, dict]]],
        reft_model,
        task_layers: List[int],
    ) -> Tuple[float, Dict[str, float], Dict[str, Tuple[int, int]],
               Dict[str, Tuple[float, float]], Dict[str, float]]:
       
        flat_K, iv_to_tier = _flatten_routing_buckets(buckets_K)
        flat_G, _          = _flatten_routing_buckets(buckets_G)
        flat_init: Optional[Dict[str, dict]] = None
        if init_buckets is not None:
            flat_init, _ = _flatten_routing_buckets(init_buckets)

        if flat_K:
            first = next(iter(flat_K))
            _debug_print_state_keys_once(first, flat_K[first])

        dist, dist_raw, n_matched, n_total, wk_norm, wg_norm = (
            self.router.compute_tier_distance(flat_K, flat_G, flat_init))
        per_iv_d: Dict[str, float] = {}
        per_iv_d_raw: Dict[str, float] = {}
        if not getattr(self.router, "operates_on_full_tier", False):
            per_iv_d, per_iv_d_raw = self._per_iv_cosine_distances(
                flat_K, flat_G, flat_init)

        per_tier:     Dict[str, float]             = {}
        per_tier_n:   Dict[str, Tuple[int, int]]   = {}
        per_tier_wn:  Dict[str, Tuple[float, float]] = {}
        per_tier_raw: Dict[str, float]             = {}
        for tk in ("t1_f", "t1_l", "t2_f", "t2_l", "t3_f"):
            kb = buckets_K.get(tk, {})
            gb = buckets_G.get(tk, {})
            ntot = len(kb)
            tier_ivs = [iv for iv, t in iv_to_tier.items() if t == tk]
            d_vals     = [per_iv_d[iv]     for iv in tier_ivs if iv in per_iv_d]
            d_raw_vals = [per_iv_d_raw[iv] for iv in tier_ivs if iv in per_iv_d_raw]
            if d_vals:
                per_tier[tk] = float(np.mean(d_vals))
                per_tier_n[tk] = (len(d_vals), ntot)
            else:
                per_tier[tk] = float(dist)
                per_tier_n[tk] = (len(kb) if kb and gb else 0, ntot)
            per_tier_raw[tk] = (float(np.mean(d_raw_vals)) if d_raw_vals
                                else float(dist_raw))

            wk_norms_t, wg_norms_t = [], []
            for iv in tier_ivs:
                if iv not in flat_K or iv not in flat_G:
                    continue
                wK = _flatten_w(flat_K[iv]); wG = _flatten_w(flat_G[iv])
                if wK is None or wG is None: continue
                if flat_init is not None and iv in flat_init:
                    w0 = _flatten_w(flat_init[iv])
                    if w0 is not None and w0.numel() == wK.numel():
                        wk_norms_t.append(float((wK - w0).norm().item()))
                        wg_norms_t.append(float((wG - w0).norm().item()))
                        continue
                wk_norms_t.append(float(wK.norm().item()))
                wg_norms_t.append(float(wG.norm().item()))
            if wk_norms_t and wg_norms_t:
                per_tier_wn[tk] = (float(np.mean(wk_norms_t)),
                                   float(np.mean(wg_norms_t)))
            else:
                per_tier_wn[tk] = (float(wk_norm), float(wg_norm))
        if per_iv_d and self.flat_aggregator != "mean":
            d_arr = np.array(list(per_iv_d.values()), dtype=np.float64)
            if self.flat_aggregator == "robust_mean":
                q = np.percentile(d_arr, 75.0)
                kept = d_arr[d_arr <= q]
                final_dist = float(kept.mean()) if kept.size > 0 else float(d_arr.mean())
                final_dist = float(np.median(d_arr))
        else:
            final_dist = float(dist)

        return (final_dist, per_tier, per_tier_n, per_tier_wn, per_tier_raw)

    def _per_iv_cosine_distances(
        self,
        flat_K: Dict[str, dict],
        flat_G: Dict[str, dict],
        flat_init: Optional[Dict[str, dict]],
    ) -> Tuple[Dict[str, float], Dict[str, float]]:

        per_iv_d: Dict[str, float] = {}
        per_iv_d_raw: Dict[str, float] = {}
        for iv in flat_K.keys():
            if iv not in flat_G:
                continue
            wK = _flatten_w(flat_K[iv])
            wG = _flatten_w(flat_G[iv])
            if wK is None or wG is None or wK.numel() != wG.numel():
                continue
            cos_raw = _safe_cosine(wK, wG)
            if cos_raw is not None:
                per_iv_d_raw[iv] = float(1.0 - cos_raw)
            d_used: Optional[float] = None
            if flat_init is not None and iv in flat_init:
                w0 = _flatten_w(flat_init[iv])
                if w0 is not None and w0.numel() == wK.numel():
                    dK, dG = wK - w0, wG - w0
                    cos_d = _safe_cosine(dK, dG)
                    if cos_d is not None:
                        d_used = float(1.0 - cos_d)
            if d_used is None and cos_raw is not None:
                d_used = float(1.0 - cos_raw)
            if d_used is not None:
                per_iv_d[iv] = d_used
        return per_iv_d, per_iv_d_raw

    def _normalize_dists(self, raw: Dict[int, float]) -> Dict[int, float]:
        """Normalize raw distances across candidate groups."""
        if self.normalize == "none":
            return {k: float(v) for k, v in raw.items()}
        vals = np.array(list(raw.values()), dtype=np.float64)
        keys = list(raw.keys())
        if len(vals) == 1:
            return {k: float(v) for k, v in raw.items()}
        if self.normalize == "minmax":
            lo, hi = vals.min(), vals.max()
            if hi - lo < 1e-12:
                norm = np.zeros_like(vals)
            else:
                norm = (vals - lo) / (hi - lo)
        else: 
            mu, sd = vals.mean(), vals.std()
            if sd < 1e-12:
                norm = np.zeros_like(vals)
            else:
                norm = (vals - mu) / sd

                norm = norm - norm.min()
        return {k: float(v) for k, v in zip(keys, norm)}


def _clone_state(d: dict) -> dict:
    """Deep-clone a single intervention state dict (tensors detached on CPU)."""
    out = {}
    for k, v in d.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.detach().clone().cpu()
        else:
            out[k] = v
    return out


def _save_state_dict(iv_state: Dict[str, dict], path: str):
    """Save an iv_state (nested dict of tensors) to disk via torch.save."""
    to_save = {}
    for iv_key, st in iv_state.items():
        to_save[iv_key] = {
            k: (v.detach().cpu() if isinstance(v, torch.Tensor) else v)
            for k, v in st.items()
        }
    torch.save(to_save, path)


def _print_routing_debug(debug: Dict) -> None:

    n_exist = debug.get("n_existing_groups", 0)
    if n_exist == 0:
        gid_new = debug.get("assigned_gid", "?")
        print(f"    [no existing groups] → create G{gid_new}")
        return

    task_names_by_gid = debug.get("group_task_names", {})
    n_tasks_by_gid    = debug.get("group_n_tasks", {})
    per_tier          = debug.get("per_tier", {})
    per_tier_n        = debug.get("per_tier_n", {})
    per_tier_wn       = debug.get("per_tier_wn", {})
    raw_dists         = debug.get("raw_distances", {})
    norm_dists        = debug.get("normalized", {})
    per_tier_raw      = debug.get("per_tier_raw", {})
    routing_mode      = debug.get("routing_mode", "raw")
    closest           = debug.get("closest_gid", None)
    threshold         = debug.get("threshold", float("nan"))
    norm_mode         = debug.get("normalize_mode", "?")
    agg_mode          = debug.get("aggregate_mode", "?")
    decision          = debug.get("decision", "?")

    print(f"    n_existing_groups={n_exist}  mode={routing_mode}  "
          f"router={debug.get('router_method', 'w_cosine')}  "
          f"aggregate={agg_mode}  normalize={norm_mode}  threshold={threshold:.3f}")
    header = (f"    {'gid':>4} {'n_tk':>4}  "
              f"{'t1_f':>7} {'t1_l':>7} {'t2_f':>7} {'t2_l':>7}  "
              f"{'raw':>7} {'norm':>7}   tasks")
    print(header)
    print("    " + "-" * (len(header) - 4))
    def _f(x): return f"{x:>7.3f}" if not (isinstance(x, float) and np.isnan(x)) else "    nan"
    for gid_str in sorted(raw_dists.keys(), key=lambda x: int(x)):
        pt   = per_tier.get(gid_str, {})
        raw  = raw_dists.get(gid_str, float("nan"))
        norm = norm_dists.get(gid_str, float("nan"))
        nt   = n_tasks_by_gid.get(gid_str, "?")
        tnm  = ",".join(task_names_by_gid.get(gid_str, []))[:28]
        print(f"    G{gid_str:<3} {nt:>4}  "
              f"{_f(pt.get('t1_f', float('nan')))} {_f(pt.get('t1_l', float('nan')))} "
              f"{_f(pt.get('t2_f', float('nan')))} {_f(pt.get('t2_l', float('nan')))}  "
              f"{_f(raw)} {_f(norm)}   {tnm}")
        if routing_mode == "delta":
            ptr = per_tier_raw.get(gid_str, {})
            if ptr:
                print(f"      (raw)      "
                      f"{_f(ptr.get('t1_f', float('nan')))} {_f(ptr.get('t1_l', float('nan')))} "
                      f"{_f(ptr.get('t2_f', float('nan')))} {_f(ptr.get('t2_l', float('nan')))}")

    if closest is not None:
        ptn = per_tier_n.get(str(closest), {})
        ptw = per_tier_wn.get(str(closest), {})
        mt  = " ".join(f"{k}={v[0]}/{v[1]}" for k, v in ptn.items())
        wn_label = "|ΔW|" if routing_mode == "delta" else "|W|"
        wn  = " ".join(
            f"{k}=({wn_label}k={v[0]:.2f},{wn_label}g={v[1]:.2f})"
            for k, v in ptw.items() if not np.isnan(v[0])
        )
        print(f"    closest=G{closest}  iv_match: {mt}")
        if wn:
            print(f"    closest=G{closest}  norms:    {wn}")

    flat_routing  = debug.get("flat_routing", False)
    flat_agg      = debug.get("flat_aggregator", "?")
    per_iv_dists  = debug.get("per_iv_dists", {})
    vote_counts   = debug.get("vote_counts", {})
    vote_share    = debug.get("vote_share", {})
    vote_winner   = debug.get("vote_winner", None)
    vote_margin   = debug.get("vote_margin", float("nan"))

    if flat_routing and per_iv_dists:
        n_ivs = max((len(v) for v in per_iv_dists.values()), default=0)
        print(f"    flat_routing: aggregator={flat_agg}  n_routing_ivs={n_ivs}")
        for gid_str in sorted(per_iv_dists.keys(), key=lambda x: int(x)):
            d_arr = np.array(list(per_iv_dists[gid_str].values()), dtype=np.float64)
            if d_arr.size == 0:
                continue
            print(f"      G{gid_str}  per_iv  min={d_arr.min():.3f} "
                  f"q25={np.percentile(d_arr, 25):.3f} "
                  f"med={np.median(d_arr):.3f} "
                  f"q75={np.percentile(d_arr, 75):.3f} "
                  f"max={d_arr.max():.3f}  mean={d_arr.mean():.3f}")
        if vote_counts:
            counts_str = "  ".join(
                f"G{k}={vote_counts.get(k, 0)} ({vote_share.get(k, 0.0)*100:.1f}%)"
                for k in sorted(vote_counts.keys(), key=lambda x: int(x))
            )
            margin_str = (f"margin={vote_margin:.3f}"
                          if not np.isnan(vote_margin) else "margin=nan")
            winner_str = (f"winner=G{vote_winner}"
                          if vote_winner is not None else "winner=None")
            print(f"    vote: {counts_str}  {winner_str}  {margin_str}")

    print(f"    → decision: {decision}  (assigned_gid=G{debug.get('assigned_gid', '?')})")


def _get_b_from_state(state: dict) -> Optional[torch.Tensor]:
    if not state:
        return None
    candidates = (
        "learned_source.bias",
        "learned_source.0.bias",
        "learned_source.linear.bias",
        "bias",
        "b",
    )
    for key in candidates:
        v = state.get(key, None)
        if isinstance(v, torch.Tensor):
            return v
    for key, val in state.items():
        if not isinstance(val, torch.Tensor):
            continue
        k_low = key.lower()
        if "rotate" in k_low or "weight" in k_low:
            continue
        if "bias" in k_low:
            return val
    return None


def _flatten_w(state: dict) -> Optional[torch.Tensor]:
    """Flat float32 CPU vector of W (or None if not extractable)."""
    W = _get_w_from_state(state)
    if W is None:
        return None
    return W.detach().float().flatten().cpu()

def _safe_cosine(a: torch.Tensor, b: torch.Tensor) -> Optional[float]:
    """Cosine of two flat tensors of the same length; None on degeneracy."""
    if a.numel() == 0 or b.numel() == 0 or a.numel() != b.numel():
        return None
    if a.norm().item() < 1e-12 or b.norm().item() < 1e-12:
        return None
    return float(F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item())


@torch.no_grad()
def _probe_forward_logits(reft_model, probe_batch, top_k: int = 256,
                          mini_batch_size: int = 8):
    reft_model.eval()
    full_b = dict(probe_batch)
    total_B = full_b["input_ids"].shape[0]
    mb = max(1, int(mini_batch_size))

    out_top_vals_chunks = []
    out_top_idx_chunks = []
    out_mask_chunks = []

    for start in range(0, total_B, mb):
        end = min(start + mb, total_B)
        b = {}
        for key, val in full_b.items():
            if torch.is_tensor(val) and val.dim() >= 1 and val.shape[0] == total_B:
                b[key] = val[start:end]
            elif isinstance(val, list) and len(val) == total_B:
                b[key] = val[start:end]
            else:
                b[key] = val

        iv_locs = b.pop("intervention_locations", None)
        b.pop("subspaces", None)
        b.pop("labels", None)             
        b["use_cache"] = False
        unit_locations = None
        if iv_locs is not None:
            unit_locations = {
                "sources->base": (None,
                                  iv_locs.permute(1, 0, 2).tolist()
                                  if hasattr(iv_locs, "permute")
                                  else iv_locs)
            }
        _, outputs = reft_model(base=b, unit_locations=unit_locations)
        logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]

        labels_full = full_b.get("labels", None)
        if labels_full is not None and torch.is_tensor(labels_full):
            labels_slice = labels_full[start:end].to(logits.device)
            mask = _kl_mask(labels_slice, logits.device)
        else:
            mask = torch.ones(logits.shape[:2],
                              dtype=torch.bool, device=logits.device)

        k = min(int(top_k), logits.shape[-1])
        top_vals, top_idx = logits.topk(k, dim=-1)
        out_top_vals_chunks.append(top_vals.detach())
        out_top_idx_chunks.append(top_idx.detach())
        out_mask_chunks.append(mask.detach())
        del logits, outputs, top_vals, top_idx
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    top_vals = torch.cat(out_top_vals_chunks, dim=0)
    top_idx  = torch.cat(out_top_idx_chunks, dim=0)
    mask     = torch.cat(out_mask_chunks, dim=0)
    return top_vals, top_idx, mask


@torch.no_grad()
def _probe_forward_logits_baseline(base_model, probe_batch, top_k: int = 256,
                                   mini_batch_size: int = 8):
    """Same chunked top-k extraction as _probe_forward_logits, but for the
    frozen backbone with NO intervention applied (pi0). Needed for Eq. 4's
    denominator max(min(D_K, D_G), eps), which requires each side's KL
    distance from the untouched baseline -- not just D_KG between the two
    intervened snapshots. Restores base_model's train/eval mode afterward
    since, unlike the transient reft/lora wrappers, it is reused for the
    rest of this task's training."""
    was_training = base_model.training
    base_model.eval()
    full_b = dict(probe_batch)
    total_B = full_b["input_ids"].shape[0]
    mb = max(1, int(mini_batch_size))

    out_top_vals_chunks, out_top_idx_chunks, out_mask_chunks = [], [], []
    for start in range(0, total_B, mb):
        end = min(start + mb, total_B)
        b = {}
        for key, val in full_b.items():
            if torch.is_tensor(val) and val.dim() >= 1 and val.shape[0] == total_B:
                b[key] = val[start:end]
            elif isinstance(val, list) and len(val) == total_B:
                b[key] = val[start:end]
            else:
                b[key] = val
        b.pop("intervention_locations", None)
        b.pop("subspaces", None)
        b.pop("labels", None)
        b["use_cache"] = False
        outputs = base_model(**b)
        logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]

        labels_full = full_b.get("labels", None)
        if labels_full is not None and torch.is_tensor(labels_full):
            labels_slice = labels_full[start:end].to(logits.device)
            mask = _kl_mask(labels_slice, logits.device)
        else:
            mask = torch.ones(logits.shape[:2],
                              dtype=torch.bool, device=logits.device)

        k = min(int(top_k), logits.shape[-1])
        top_vals, top_idx = logits.topk(k, dim=-1)
        out_top_vals_chunks.append(top_vals.detach())
        out_top_idx_chunks.append(top_idx.detach())
        out_mask_chunks.append(mask.detach())
        del logits, outputs, top_vals, top_idx
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if was_training:
        base_model.train()

    top_vals = torch.cat(out_top_vals_chunks, dim=0)
    top_idx  = torch.cat(out_top_idx_chunks, dim=0)
    mask     = torch.cat(out_mask_chunks, dim=0)
    return top_vals, top_idx, mask


@torch.no_grad()
def _kl_logits_distance(
    snap_a, snap_b,
    temperature: float = 2.0,
    top_k: int = 64,
) -> float:

    top_vals_a, top_idx_a, mask_a = snap_a
    top_vals_b, top_idx_b, mask_b = snap_b
    label_mask = mask_a & mask_b  
    device = top_vals_a.device
    top_vals_b = top_vals_b.to(device)
    top_idx_b  = top_idx_b.to(device)
    label_mask = label_mask.to(device)

    B, T, k_snap = top_vals_a.shape
    k_use = min(int(top_k) if top_k and top_k > 0 else k_snap, k_snap)
    top_vals_a = top_vals_a.float()
    top_vals_b = top_vals_b.float()

    if k_use < k_snap:
        top_vals_a = top_vals_a[..., :k_use]
        top_idx_a  = top_idx_a[..., :k_use]
        top_vals_b = top_vals_b[..., :k_use]
        top_idx_b  = top_idx_b[..., :k_use]
    union_idx = torch.cat([top_idx_a, top_idx_b], dim=-1)       

    NEG_INF = float("-inf")

    def _gather_values_for_union(top_vals_X, top_idx_X, union_idx):
        # Exact index matching via searchsorted instead of the [B,T,2k,k]
        # broadcast comparison. Identical results (top-k indices are unique
        # within a row, so the match is unique), but memory is O(B*T*2k)
        # rather than O(B*T*2k*k) -- at probe_size=16, T=1024, k=512 that is
        # ~17M elements instead of ~8.6e9, which is what made top_k>256
        # unusable and is why the snapshots were pinned at k=256.
        sort_idx, order = top_idx_X.sort(dim=-1)
        sort_vals = torch.gather(top_vals_X, dim=-1, index=order)
        pos = torch.searchsorted(sort_idx.contiguous(),
                                 union_idx.contiguous())
        pos = pos.clamp_(max=sort_idx.shape[-1] - 1)
        any_match = torch.gather(sort_idx, dim=-1, index=pos) == union_idx
        gathered = torch.gather(sort_vals, dim=-1, index=pos)
        gathered = torch.where(any_match, gathered,
                               torch.full_like(gathered, NEG_INF))
        return gathered

    val_a_on_union = _gather_values_for_union(top_vals_a, top_idx_a, union_idx)
    val_b_on_union = _gather_values_for_union(top_vals_b, top_idx_b, union_idx)

    la = val_a_on_union / float(temperature)
    lb = val_b_on_union / float(temperature)
    sorted_union, sort_perm = union_idx.sort(dim=-1)
    dup_mask = (sorted_union[..., 1:] == sorted_union[..., :-1])
    dup_mask = torch.cat([torch.zeros_like(dup_mask[..., :1]),
                          dup_mask], dim=-1)                         
    unsort_perm = sort_perm.argsort(dim=-1)
    dup_mask = torch.gather(dup_mask, dim=-1, index=unsort_perm)
    la = la.masked_fill(dup_mask, NEG_INF)
    lb = lb.masked_fill(dup_mask, NEG_INF)

    lp_a = F.log_softmax(la, dim=-1)
    lp_b = F.log_softmax(lb, dim=-1)
    p_a  = lp_a.exp()
    p_b  = lp_b.exp()


    diff_ab = lp_a - lp_b
    diff_ba = lp_b - lp_a
    diff_ab = torch.where(torch.isfinite(diff_ab), diff_ab,
                          torch.zeros_like(diff_ab))
    diff_ba = torch.where(torch.isfinite(diff_ba), diff_ba,
                          torch.zeros_like(diff_ba))

    diff_ab = torch.where(p_a > 0, diff_ab, torch.zeros_like(diff_ab))
    diff_ba = torch.where(p_b > 0, diff_ba, torch.zeros_like(diff_ba))

    kl_ab = (p_a * diff_ab).sum(dim=-1)             
    kl_ba = (p_b * diff_ba).sum(dim=-1)              

    sym = 0.5 * (kl_ab + kl_ba)
    mask_f = label_mask.to(sym.dtype)
    denom = mask_f.sum().clamp_min(1.0)
    return float(((sym * mask_f).sum() / denom).item())



def _build_current_task_probe(train_dataset, tokenizer,
                              probe_size: int, device: str,
                              max_seq_len: int = 256):
    
    n = min(int(probe_size), len(train_dataset))
    keep = ("input_ids", "attention_mask", "labels",
            "intervention_locations", "subspaces")
    examples = [{k: v for k, v in train_dataset[i].items() if k in keep}
                for i in range(n)]
    print(f"  [task_probe] Built probe from current task "
          f"({n} examples) — no past-task data.")
    collator = ReftDataCollator(data_collator=DataCollatorForSeq2Seq(
        tokenizer=tokenizer, model=None,
        label_pad_token_id=-100, padding="longest"))
    batch = collator(examples)
    return {k: (v.to(device) if torch.is_tensor(v) else v)
            for k, v in batch.items()}

def run_kl_routing_phase(
    base_model, task_layers, args,
    intervention_cls, iv_dtype,
    train_dataset, tokenizer,
    memory,
    task_name: str, task_idx: int,
    warmup_steps: int,
    probe_size: int,
    kl_threshold: float,
    output_dir: str,
    seed: int,
    batch_size: int,
    lr: float,
    gradient_accumulation_steps: int,
    *,
    top_k: int = 64,
    temperature: float = 2.0,
    normalize: str = "raw",
    min_dkg: float = 0.01,
    direction: str = "forward",
    base_model_for_bwd=None,
    task_probe: Optional[dict] = None,
):
    
    print(f"\n  [kl_routing] Task '{task_name}' against "
          f"{len(memory.groups)} existing group(s)  "
          f"(warmup_steps={warmup_steps}  probe_size={probe_size}  "
          f"threshold={kl_threshold:.4f}  top_k={top_k}  "
          f"T={temperature:.2f}  normalize={normalize}  "
          f"min_dkg={min_dkg:.4f})")

    device_str = "cuda" if torch.cuda.is_available() else "cpu"

    if not memory.groups:
        gid = memory._next_gid
        debug = {
            "task":              task_name,
            "router_method":     "kl_routing",
            "n_existing_groups": 0,
            "decision":          "new_group_no_existing",
            "assigned_gid":      gid,
            "raw_kl_per_group":  {},
            "distance_per_group":{},
            "threshold":         float(kl_threshold),
            "min_dkg":           float(min_dkg),
            "normalize":         str(normalize),
            "raw_distances":     {},
            "normalized":        {},
            "group_n_tasks":     {},
            "group_task_names":  {},
            "per_tier":          {},
            "per_tier_raw":      {},
            "per_tier_n":        {},
            "per_tier_wn":       {},
            "closest_gid":       gid,
            "closest_raw_dist":  float("nan"),
            "closest_norm_dist": float("nan"),
        }
        memory._routing_log.append(debug)
        return gid, True, debug

    # --group_capacity: groups already holding `capacity` tasks are not
    # candidates. Filtering here also skips their probe forwards entirely.
    candidate_gids = memory.open_gids()
    if not candidate_gids:
        gid = memory._next_gid
        print(f"  [kl_routing] all {len(memory.groups)} group(s) at capacity "
              f"({memory.group_capacity}); '{task_name}' → NEW GROUP "
              f"(no KL probes run).")
        debug = {
            "task":              task_name,
            "router_method":     "kl_routing",
            "n_existing_groups": len(memory.groups),
            "group_capacity":    memory.group_capacity,
            "decision":          "new_group_capacity_full",
            "assigned_gid":      gid,
            "raw_kl_per_group":  {}, "distance_per_group": {},
            "threshold":         float(kl_threshold),
            "min_dkg":           float(min_dkg),
            "normalize":         str(normalize),
            "raw_distances":     {}, "normalized": {},
            "group_n_tasks":     {str(g): len(v["task_names"])
                                  for g, v in memory.groups.items()},
            "group_task_names":  {str(g): list(v["task_names"])
                                  for g, v in memory.groups.items()},
            "per_tier": {}, "per_tier_raw": {}, "per_tier_n": {}, "per_tier_wn": {},
            "closest_gid":       gid,
            "closest_raw_dist":  float("nan"),
            "closest_norm_dist": float("nan"),
        }
        memory._routing_log.append(debug)
        return gid, True, debug
    if len(candidate_gids) < len(memory.groups):
        full = [g for g in sorted(memory.groups) if g not in candidate_gids]
        print(f"  [kl_routing] capacity={memory.group_capacity}: skipping full "
              f"group(s) {full}; candidates={candidate_gids}")

    probe_batch = (task_probe if task_probe is not None else
                   _build_current_task_probe(
                       train_dataset, tokenizer, probe_size, device_str))

    # Snapshots must retain at least as many logits as the routing top_k, or
    # --kl_routing_top_k is silently clamped by k_use = min(top_k, k_snap)
    # inside _kl_logits_distance. They used to be pinned at the k=256 default,
    # which made every setting above 256 identical to 256.
    probe_top_k = int(top_k) if top_k and int(top_k) > 0 else 256

    # pi0 taken BEFORE any intervention wrapper for this task exists, so the
    # baseline cannot be contaminated by a live reft model on the same backbone.
    snap_0 = _probe_forward_logits_baseline(
        base_model, probe_batch, top_k=probe_top_k)

    _set_intervention_seed(int(args.fixed_intervention_seed))
    warm_reft = _build_reft_model(
        base_model, task_layers, args.rank,
        intervention_cls, iv_dtype, args)
    print(f"  [kl_routing] Running warmup ({warmup_steps} steps) "
          f"on '{task_name}' ...")
    _run_warmup_steps(
        warm_reft, train_dataset, tokenizer,
        steps=int(warmup_steps),
        batch_size=int(batch_size), lr=float(lr),
        output_dir=output_dir, seed=int(seed),
        gradient_accumulation_steps=int(gradient_accumulation_steps),
    )

    snap_K = _probe_forward_logits(warm_reft, probe_batch, top_k=probe_top_k)

    # This task's KL-from-baseline. Needed for Eq. 4's ratio denominator
    # max(min(D_K, D_G), eps) -- computed once, independent of how many groups
    # exist, so it no longer degenerates to a fixed 1.0 whenever there's only
    # one candidate group to compare against (the old d_max = max(D_KG among
    # candidates) did exactly that).
    D_K = _kl_logits_distance(snap_0, snap_K, temperature=temperature, top_k=top_k)
    print(f"  [kl_routing] D_K (pi0 -> warmup) = {D_K:.4f}  "
          f"(snapshot k={probe_top_k}, routing top_k={top_k})")

    raw_kl: Dict[int, float] = {}
    raw_dg: Dict[int, float] = {}
    raw_bwd: Dict[int, float] = {}
    for gid in candidate_gids:
        g = memory.groups[gid]
        n_tasks = len(g["task_names"])
        print(f"  [kl_routing] G{gid} (n={n_tasks}, tasks={g['task_names']}) "
              f"— forward + KL ...")

        _set_intervention_seed(int(args.fixed_intervention_seed))
        g_reft = _build_reft_model(
            base_model, task_layers, args.rank,
            intervention_cls, iv_dtype, args)
        _load_iv_state(g_reft, g["iv_state"])

        try:
            snap_G = _probe_forward_logits(g_reft, probe_batch, top_k=probe_top_k)
            D_KG = _kl_logits_distance(
                snap_K, snap_G,
                temperature=temperature, top_k=top_k)
            raw_kl[gid] = D_KG
            raw_dg[gid] = _kl_logits_distance(
                snap_0, snap_G,
                temperature=temperature, top_k=top_k)
            # --- incumbent-side (backward) score --------------------------
            # Everything above is measured on the ARRIVING task's probe, so
            # it can only see whether K and G agree where the newcomer lives.
            # The damage that actually matters happens in the group's own
            # distribution. Measured on the study's 56-pair matrix, the same
            # numbers read in this direction rank interference at rho=0.517
            # against rho=0.192 for the forward direction -- identical values,
            # transposed, +0.33 rho for free.
            #
            # max over the group's members: joining must not hurt ANY of them.
            if direction in ("backward", "both"):
                member_probes = memory.group_probes.get(gid, [])
                d_bwds = []
                bwd_nums, bwd_dens = [], []
                n_floored = 0
                bwd_eps = float(getattr(args, "kl_routing_bwd_eps", 0.05))
                for mp in member_probes:
                    mp_dev = {k: (v.to(device_str) if torch.is_tensor(v) else v)
                              for k, v in mp.items()}
                    snap_G_own = _probe_forward_logits(g_reft, mp_dev, top_k=probe_top_k)
                    snap_K_own = _probe_forward_logits(warm_reft, mp_dev, top_k=probe_top_k)
                    d_raw = _kl_logits_distance(snap_G_own, snap_K_own,
                                                temperature=temperature, top_k=top_k)
                    # Eq. 4's ratio, in the incumbent's frame. The denominator
                    # mirrors the forward metric: min(D_K, D_G) measured on the
                    # MEMBER's data, with its own epsilon. Using min_dkg here
                    # (0.01) let the ratio be scaled by up to 100x whenever a
                    # group had barely moved from pi0 on its own probe.
                    if base_model_for_bwd is not None:
                        snap_0_own = _probe_forward_logits_baseline(
                            base_model_for_bwd, mp_dev, top_k=probe_top_k)
                        d_g_own = _kl_logits_distance(
                            snap_0_own, snap_G_own,
                            temperature=temperature, top_k=top_k)
                        d_k_own = _kl_logits_distance(
                            snap_0_own, snap_K_own,
                            temperature=temperature, top_k=top_k)
                        denom = min(d_k_own, d_g_own)
                        if denom <= bwd_eps:
                            n_floored += 1
                        denom = max(denom, bwd_eps)
                        del snap_0_own
                    else:
                        denom = 1.0
                    bwd_nums.append(float(d_raw))
                    bwd_dens.append(float(denom))
                    d_bwds.append(d_raw / denom if denom > 0 else d_raw)
                    del snap_G_own, snap_K_own, mp_dev
                if d_bwds:
                    # Ratio of sums, not mean/max of ratios: one member with a
                    # near-zero denominator can no longer decide the score.
                    raw_bwd[gid] = float(sum(bwd_nums) / max(sum(bwd_dens), bwd_eps))
                    print(f"  [kl_routing] G{gid}  D_bwd={raw_bwd[gid]:.4f} "
                          f"(ratio-of-sums over {len(d_bwds)} member probe(s); "
                          f"per-probe max={max(d_bwds):.4f})")
                    if n_floored:
                        print(f"  [kl_routing] G{gid}  WARNING: {n_floored}/"
                              f"{len(d_bwds)} member probe(s) hit the denominator "
                              f"floor eps={bwd_eps} — D_bwd may be unreliable")
                else:
                    raw_bwd[gid] = float("nan")
                    print(f"  [kl_routing] G{gid}  no stored member probe; "
                          f"D_bwd unavailable.")

            print(f"  [kl_routing] G{gid}  D_KG={D_KG:.4f}")
        except Exception as e:
            print(f"  [kl_routing] G{gid} forward FAILED "
                  f"({type(e).__name__}: {e}); marking D_KG=inf.")
            raw_kl[gid] = float("inf")
            raw_dg[gid] = float("inf")
            raw_bwd[gid] = float("inf")

        try:
            _teardown_reft_model(g_reft)
        except Exception:
            pass
        del g_reft
        # gc.collect as well: a torn-down model that sits in a reference cycle
        # is not freed by empty_cache() alone and would stay on the GPU until
        # the end-of-task collection, one more per group probed.
        _release_gpu()

    try:
        _teardown_reft_model(warm_reft)
    except Exception:
        pass
    del warm_reft, snap_K
    _release_gpu()

    finite = {k: v for k, v in raw_kl.items()
              if v != float("inf") and not np.isnan(v)}

    if not finite:
        is_new = True
        assigned_gid = memory._next_gid
        closest_gid = assigned_gid
        closest_dist = float("nan")
        closest_raw_dkg = float("nan")
        distance: Dict[int, float] = {k: float("inf")
                                      for k in raw_kl.keys()}
        print(f"  [kl_routing] All forwards failed; allocating new group.")
    else:
        if str(normalize).lower() == "ratio":
            # Eq. 4: d(t,k) = D_KG_k / max(min(D_K, D_G_k), eps). Each
            # candidate gets its OWN denominator (how far the new task and
            # THAT group have each moved from the untouched baseline), not
            # a value shared across candidates -- so a single-candidate
            # comparison no longer collapses to a fixed 1.0 by construction.
            distance = {}
            for k, v in raw_kl.items():
                dg = raw_dg.get(k, float("inf"))
                denom = max(min(D_K, dg), float(min_dkg))
                distance[k] = (v / denom) if denom > 0 else v
            for gid, d in distance.items():
                print(f"  [kl_routing] G{gid}  distance(ratio)={d:.4f}  "
                      f"(D_KG={raw_kl[gid]:.4f}  D_K={D_K:.4f}  "
                      f"D_G={raw_dg.get(gid, float('nan')):.4f})")
        else:
            distance = dict(raw_kl)
            for gid, d in distance.items():
                print(f"  [kl_routing] G{gid}  distance(raw)={d:.4f}")

        # --- direction selects which score gates the join -------------------
        # forward  : legacy. distance measured on the arriving task's data.
        # backward : distance measured on the group's own members' data. The
        #            validated choice (rho 0.517 vs 0.192 on the 56-pair
        #            interference matrix).
        # both     : must pass on BOTH sides. Strictly more conservative, so
        #            it raises K; use when forgetting matters more than
        #            parameter savings.
        if direction in ("backward", "both"):
            usable_bwd = {k: v for k, v in raw_bwd.items()
                          if v == v and v != float("inf")}
            if not usable_bwd:
                print(f"  [kl_routing] direction={direction} requested but no "
                      f"member probes were available; falling back to forward.")
            elif direction == "backward":
                distance = dict(usable_bwd)
                for gid, d in distance.items():
                    print(f"  [kl_routing] G{gid}  distance(backward)={d:.4f}")
                print(f"  [kl_routing] decision metric = backward "
                      f"(--kl_routing_normalize={normalize} applies to the "
                      f"forward score only and is NOT used here)")
            else:  # both
                distance = {k: max(distance.get(k, float("inf")),
                                   usable_bwd.get(k, float("inf")))
                            for k in set(distance) | set(usable_bwd)}
                for gid, d in distance.items():
                    print(f"  [kl_routing] G{gid}  distance(both)={d:.4f}  "
                          f"(fwd={raw_kl.get(gid, float('nan')):.4f} "
                          f"bwd={raw_bwd.get(gid, float('nan')):.4f})")

        finite_d = {k: v for k, v in distance.items()
                    if v != float("inf") and not np.isnan(v)}
        if not finite_d:
            finite_d = {k: v for k, v in dict(raw_kl).items()
                        if v != float("inf") and not np.isnan(v)}
        closest_gid = min(finite_d, key=finite_d.get)
        closest_dist = finite_d[closest_gid]
        closest_raw_dkg = float(raw_kl[closest_gid])

        if closest_raw_dkg < float(min_dkg):
            is_new = True
            assigned_gid = memory._next_gid
            print(f"  [kl_routing] closest=G{closest_gid}  "
                  f"distance={closest_dist:.4f}  "
                  f"raw D_KG={closest_raw_dkg:.4f} < min_dkg={min_dkg:.4f}  → "
                  f"NEW GROUP (no-signal floor triggered)")
        elif closest_dist <= kl_threshold:
            is_new = False
            assigned_gid = closest_gid
            print(f"  [kl_routing] closest=G{closest_gid}  "
                  f"distance={closest_dist:.4f}  threshold={kl_threshold:.4f}  → "
                  f"JOIN G{assigned_gid}")
        else:
            is_new = True
            assigned_gid = memory._next_gid
            print(f"  [kl_routing] closest=G{closest_gid}  "
                  f"distance={closest_dist:.4f}  threshold={kl_threshold:.4f}  → "
                  f"NEW GROUP")

    debug = {
        "task":              task_name,
        "router_method":     "kl_routing",
        "n_existing_groups": len(memory.groups),
        "warmup_steps":      int(warmup_steps),
        "probe_size":        int(probe_size),
        "threshold":         float(kl_threshold),
        "min_dkg":           float(min_dkg),
        "top_k":             int(top_k),
        "temperature":       float(temperature),
        "normalize":         str(normalize),
        "raw_kl_per_group":  {str(k): float(v) for k, v in raw_kl.items()},
        "distance_per_group":{str(k): float(v) for k, v in distance.items()},
        "decision":          "new_group" if is_new else f"join_G{assigned_gid}",
        "assigned_gid":      int(assigned_gid),
        "closest_raw_dkg":   float(closest_raw_dkg),
        "raw_distances":     {str(k): float(v) for k, v in distance.items()},
        "normalized":        {str(k): float(v) for k, v in distance.items()},
        "group_n_tasks":     {str(gid): len(g["task_names"])
                              for gid, g in memory.groups.items()},
        "group_task_names":  {str(gid): list(g["task_names"])
                              for gid, g in memory.groups.items()},
        "per_tier":          {str(k): {"agg": float(v)}
                              for k, v in distance.items()},
        "per_tier_raw":      {str(k): {"agg": float(v)}
                              for k, v in distance.items()},
        "per_tier_n":        {str(k): {"agg": [0, 0]}
                              for k in distance.keys()},
        "per_tier_wn":       {str(k): {"agg": [float("nan"), float("nan")]}
                              for k in distance.keys()},
        "closest_gid":       int(closest_gid),
        "closest_raw_dist":  float(closest_dist),
        "closest_norm_dist": float(closest_dist),
    }
    memory._routing_log.append(debug)
    return int(assigned_gid), is_new, debug
def _run_warmup_steps(
    reft_model, train_dataset, tokenizer,
    steps: int, batch_size: int, lr: float,
    output_dir: str, seed: int,
    gradient_accumulation_steps: int,
):
    from transformers import TrainingArguments
    set_seed(int(seed))
    targs = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=1.0,                 
        max_steps=int(steps),
        per_device_train_batch_size=int(batch_size),
        gradient_accumulation_steps=int(gradient_accumulation_steps),
        learning_rate=float(lr),
        warmup_ratio=0.0,
        weight_decay=0.0,
        logging_steps=max(1, int(steps) // 4),
        save_strategy="no",
        report_to="none",
        bf16=(reft_model.dtype == torch.bfloat16) if hasattr(reft_model, "dtype") else False,
        seed=int(seed),
        disable_tqdm=True,
    )
    collator = ReftDataCollator(data_collator=DataCollatorForSeq2Seq(
        tokenizer=tokenizer, model=None,
        label_pad_token_id=-100, padding="longest"))
    trainer = ReftTrainerForCausalLM(
        model=reft_model,
        args=targs,
        train_dataset=train_dataset,
        eval_dataset=None,
        tokenizer=tokenizer,
        data_collator=collator,
    )
    trainer.train()
    reft_model.eval()
def _set_intervention_seed(seed: int):

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def build_stratified_optimizer(
    reft_model, task_layers,
    tier1_set, tier2_set, tier3_set,
    base_lr, weight_decay=0.02,
):

    n_iv    = len(list(reft_model.interventions.keys()))
    l_start = _l_start_idx(n_iv)
    p_t1, p_t2, p_t3f, p_t3l = [], [], [], []
    for i, (_, iv_list) in enumerate(reft_model.interventions.items()):
        iv = iv_list[0] if isinstance(iv_list, (list, tuple)) else iv_list
        ps = [p for p in iv.parameters() if p.requires_grad]
        if not ps or i >= len(task_layers): continue
        layer, is_l = task_layers[i], (i >= l_start)
        if layer in tier1_set:
            p_t1.extend(ps)
        elif layer in tier2_set:
            p_t2.extend(ps)
        elif layer in tier3_set and not is_l:
            p_t3f.extend(ps)
        else:   
            p_t3l.extend(ps)
    groups = []
    if p_t1:  groups.append({"params": p_t1,  "lr": base_lr*1.0, "name": "T1_groupwise"})
    if p_t2:  groups.append({"params": p_t2,  "lr": base_lr*1.0, "name": "T2_groupwise"})
    if p_t3f: groups.append({"params": p_t3f, "lr": base_lr*1.0, "name": "T3f_groupwise"})
    if p_t3l: groups.append({"params": p_t3l, "lr": base_lr*1.0, "name": "T3l_taskwise"})
    if not groups:
        groups = [{"params": [p for p in reft_model.parameters() if p.requires_grad],
                   "lr": base_lr, "name": "fallback"}]
    opt = torch.optim.AdamW(groups, weight_decay=weight_decay)
    print("  [StratOptim]")
    for g in groups:
        print(f"    {g['name']:<18}  lr={g['lr']:.2e}  "
              f"params={sum(p.numel() for p in g['params']):,}")
    return opt


dtype_mapping = {
    "float32": torch.float32, "float16": torch.float16,
    "bfloat16": torch.bfloat16, "float8": "float8",
}
intervention_mapping = {
    "NoreftIntervention":   NoreftIntervention,
    "LoreftIntervention":   LoreftIntervention,
    "ConsreftIntervention": ConsreftIntervention,
    "LobireftIntervention": LobireftIntervention,
    "DireftIntervention":   DireftIntervention,
    "NodireftIntervention": NodireftIntervention,
}

def _build_reft_model(base_model, layers, rank, intervention_cls, iv_dtype, args):

    fixed_seed = getattr(args, "fixed_intervention_seed", None)
    if fixed_seed is not None:
        _set_intervention_seed(int(fixed_seed))
    reps = [{"layer": l, "component": "block_output",
              "low_rank_dimension": rank,
              "intervention": intervention_cls(
                  embed_dim=base_model.config.hidden_size,
                  low_rank_dimension=rank, dropout=args.dropout,
                  dtype=iv_dtype, act_fn=args.act_fn,
                  device=device, add_bias=args.add_bias,
              )} for l in layers]
    reft_model = get_reft_model(
        base_model, ReftConfig(representations=reps),
        set_device=not isinstance(iv_dtype, str))
    n = 0
    for iv_list in reft_model.interventions.values():
        iv = iv_list[0] if isinstance(iv_list, (list, tuple)) else iv_list
        try:
            R = iv.rotate_layer.parametrizations.weight[0].base
        except AttributeError:
            R = iv.rotate_layer.weight
        if not R.requires_grad:
            R.requires_grad_(True); n += 1
    if n: print(f"  [build_reft] R restored on {n} layers.")
    return reft_model

def _build_shadow_reft(base_model, task_layers, rank, intervention_cls,
                       iv_dtype, args, anchor_iv_state: Dict[str, dict]):

    _set_intervention_seed(args.fixed_intervention_seed)
    shadow = _build_reft_model(
        base_model, task_layers, rank, intervention_cls, iv_dtype, args)
    _load_iv_state(shadow, anchor_iv_state)
    return shadow


class AGemShadowProjector:
    def __init__(self, live_reft, shadow_reft, groupwise_iv_keys: Set[str]):
        self.live_reft   = live_reft
        self.shadow_reft = shadow_reft
        self.groupwise_iv_keys = set(groupwise_iv_keys)
        self.n_steps          = 0
        self.n_w_tensors_seen = 0
        self.n_w_projected    = 0
        self.sum_cosines      = 0.0
        self.n_cosines        = 0
        self._shadow_W_refs: Dict[str, torch.nn.Parameter] = {}
        self._freeze_shadow_except_groupwise_W()

    def _freeze_shadow_except_groupwise_W(self):
        kept = 0
        frozen = 0
        for iv_key, iv_list in self.shadow_reft.interventions.items():
            mod = iv_list[0] if isinstance(iv_list, (list, tuple)) else iv_list
            is_groupwise = iv_key in self.groupwise_iv_keys
            W = getattr(mod, "learned_source", None)
            W_weight = W.weight if (W is not None and hasattr(W, "weight")) else None
            for p in mod.parameters():
                if is_groupwise and (p is W_weight):
                    p.requires_grad_(True)
                    kept += 1
                else:
                    p.requires_grad_(False)
                    frozen += 1
            if is_groupwise and W_weight is not None:
                self._shadow_W_refs[iv_key] = W_weight
        print(f"  [A-GEM] Shadow params: grad-enabled={kept} (groupwise W), "
              f"frozen={frozen} (R, b, and taskwise params)")

    @torch.no_grad()
    def _reset_shadow_grads(self):
        for W_param in self._shadow_W_refs.values():
            if W_param.grad is not None:
                W_param.grad = None

    def compute_anchor_W_grads(self, inputs) -> Dict[str, torch.Tensor]:
        self._reset_shadow_grads()
        self.shadow_reft.train()
        shadow_inputs = dict(inputs)
        shadow_inputs.setdefault("use_cache", False)

        with torch.enable_grad():
            outputs = self.shadow_reft(**shadow_inputs)
            loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]
            loss.backward()
        anchor_W_grads: Dict[str, torch.Tensor] = {}
        for iv_key, W_weight in self._shadow_W_refs.items():
            if W_weight.grad is not None:
                anchor_W_grads[iv_key] = W_weight.grad.detach().clone()
        self._reset_shadow_grads()
        del loss
        del outputs
        del shadow_inputs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return anchor_W_grads

    def project_live_W_grads(self, anchor_W_grads: Dict[str, torch.Tensor]):
        self.n_steps += 1
        for iv_key, iv_list in self.live_reft.interventions.items():
            if iv_key not in self.groupwise_iv_keys:
                continue
            if iv_key not in anchor_W_grads:
                continue
            mod = iv_list[0] if isinstance(iv_list, (list, tuple)) else iv_list
            W   = getattr(mod, "learned_source", None)
            if W is None or W.weight.grad is None:
                continue
            g_task = W.weight.grad 
            g_anchor = anchor_W_grads[iv_key]
            if g_anchor.device != g_task.device or g_anchor.dtype != g_task.dtype:
                g_anchor = g_anchor.to(device=g_task.device, dtype=g_task.dtype)

            gt = g_task.reshape(-1)
            ga = g_anchor.reshape(-1)
            gt_norm = gt.float().norm()
            ga_norm = ga.float().norm()
            if gt_norm.item() < 1e-12 or ga_norm.item() < 1e-12:
                continue
            dot = torch.dot(gt.float(), ga.float())
            cos = float(dot.item() / (gt_norm.item() * ga_norm.item()))
            self.n_w_tensors_seen += 1
            self.sum_cosines += cos
            self.n_cosines   += 1
            if cos < 0.0:
                scale = float(dot.item() / (ga_norm.item() ** 2))
                g_task.add_(g_anchor, alpha=-scale)
                self.n_w_projected += 1
            anchor_W_grads[iv_key] = None

    def epoch_stats(self) -> Dict[str, float]:
        if self.n_cosines == 0:
            return {"mean_cos": float("nan"),
                    "proj_rate": 0.0,
                    "n_steps": self.n_steps,
                    "n_projected": 0,
                    "n_tensors": 0}
        return {
            "mean_cos":   self.sum_cosines / self.n_cosines,
            "proj_rate":  self.n_w_projected / max(self.n_w_tensors_seen, 1),
            "n_steps":    self.n_steps,
            "n_projected": self.n_w_projected,
            "n_tensors":   self.n_w_tensors_seen,
        }

    def teardown(self):
        try:
            self._reset_shadow_grads()
        except Exception:
            pass
        try:
            _teardown_reft_model(self.shadow_reft)
        except Exception:
            pass
        self._shadow_W_refs.clear()
        self.shadow_reft = None
        self.live_reft   = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class KLRazorController:
    def __init__(
        self,
        shadow_reft,
        groupwise_iv_keys: Set[str],
        beta: float = 0.1,
        early_stop: bool = True,
        rolling_window: int = 20,
        eviction_enabled: bool = False,
        eviction_severity_threshold: float = 1.3,
        eviction_early_steps: int = 30,
        eviction_absolute_threshold: float = 0.0,
        adaptive_beta: bool = False,
        normalize_kl: bool = False,
        kl_target_contribution: float = 0.5,
        beta_floor: float = 0.01,
        beta_ceiling: float = 1.0,
        running_mean_ema: float = 0.05,
        kl_mode: str = "output",
        rep_layer_indices: Optional[List[int]] = None,
        beta_warmup_steps: int = 0,
        beta_ramp_shape: str = "linear",
    ):
        assert beta_ramp_shape in ("linear", "cosine"), \
            f"bad beta_ramp_shape={beta_ramp_shape}"
        self.beta_ramp_shape = str(beta_ramp_shape)
        self._deferred_boundary_logged = False
        # Plasticity ramp: beta_eff = beta * min(1, step / beta_warmup_steps).
        # 0 = off (full anchor from step 0, the original behaviour). The point
        # is to let a new task descend its own loss before retention starts
        # pulling, which is where LA is lost in the beta dose-response.
        self.beta_warmup_steps = max(0, int(beta_warmup_steps))
        self.shadow_reft = shadow_reft
        self.groupwise_iv_keys = set(groupwise_iv_keys)
        self.beta = float(beta)
        self._beta_initial = float(beta)
        self.early_stop = bool(early_stop)
        self.rolling_window = int(rolling_window)
        self.eviction_enabled = bool(eviction_enabled)
        self.eviction_severity_threshold = float(eviction_severity_threshold)
        self.eviction_early_steps = int(eviction_early_steps)
        self.eviction_absolute_threshold = float(eviction_absolute_threshold)
        self.adaptive_beta = bool(adaptive_beta)
        self.normalize_kl = bool(normalize_kl)
        self.kl_target_contribution = float(kl_target_contribution)
        self.beta_floor = float(beta_floor)
        self.beta_ceiling = float(beta_ceiling)
        self.running_mean_ema = float(running_mean_ema)
        assert kl_mode in ("output", "representation"), \
            f"kl_mode must be 'output' or 'representation', got {kl_mode!r}"
        self.kl_mode = str(kl_mode)
        self.rep_layer_indices = (list(rep_layer_indices)
                                  if rep_layer_indices is not None else None)
        for p in self.shadow_reft.parameters():
            p.requires_grad_(False)
        for iv_list in self.shadow_reft.interventions.values():
            mod = iv_list[0] if isinstance(iv_list, (list, tuple)) else iv_list
            for p in mod.parameters():
                p.requires_grad_(False)

        self.kl_history: List[float] = []
        self._first_epoch_end: Optional[int] = None
        self._threshold: Optional[float] = None
        self.stop_requested: bool = False
        self.stop_action: Optional[str] = None
        self.stop_step: Optional[int] = None
        self.n_steps_seen: int = 0
        self.total_kl_sum: float = 0.0

        self._task_loss_running_mean: Optional[float] = None
        self._kl_running_mean: Optional[float] = None

    def effective_beta(self, step_idx: int) -> float:
        """beta after the plasticity ramp. Identical to self.beta once the
        ramp is done, and identical at every step when the ramp is off.

        linear: frac = t. Reaches full strength with a slope discontinuity in
        the schedule (the loss stays continuous, and the KL term is already
        divided by its running mean, so this is mild -- but it is a corner).
        cosine: frac = (1 - cos(pi*t))/2. Flat at both ends, so the anchor
        arrives gently and asymptotes into its cap with matching slope.
        """
        if self.beta_warmup_steps <= 0:
            return float(self.beta)
        t = min(1.0, max(0, int(step_idx)) / float(self.beta_warmup_steps))
        frac = t if self.beta_ramp_shape == "linear" else 0.5 * (1.0 - np.cos(np.pi * t))
        return float(self.beta) * float(frac)

    def set_first_epoch_boundary(self, step_idx: int) -> bool:
        """Freeze the early-stop / eviction threshold from the KL seen so far.

        Returns False if the call was deferred, in which case the caller must
        try again on a later step.

        Deferred while the beta ramp is still running. The threshold is
        mean+std of the KL history, and it is compared against the rolling
        window of the steps that follow -- so both sides have to be measured
        under the same anchor strength or the comparison is not like-for-like.
        With 500 samples at batch 4 an epoch is 125 steps, so
        --kl_beta_warmup_steps 300 would otherwise set the threshold from a
        window where beta was ramping 0 -> 0.125 (KL rising fast, low mean,
        wide std) and then judge steps trained at a stronger anchor against it.
        The likely failure there is a spurious EARLY STOP, which costs the arm
        training steps and so directly damages the LA it is meant to measure;
        eviction proper also needs severity > threshold*1.3 within
        eviction_early_steps of this boundary.

        Deferring rather than disabling matters: gating eviction on
        step > beta_warmup_steps (300) would skip the entire eviction window
        (steps 125-155) for ramp300 only, silently turning off a mechanism
        that stays live in every other arm.
        """
        if self._first_epoch_end is not None:
            return True
        if self.beta_warmup_steps > 0 and step_idx < self.beta_warmup_steps:
            if not self._deferred_boundary_logged:
                print(f"  [KL] First epoch ended at step {step_idx}, but the "
                      f"beta ramp runs to {self.beta_warmup_steps}; deferring "
                      f"the early-stop/eviction threshold until the anchor is "
                      f"at full strength.")
                self._deferred_boundary_logged = True
            return False
        self._first_epoch_end = int(step_idx)
        if len(self.kl_history) >= 5:
            arr = np.array(self.kl_history, dtype=np.float64)
            self._threshold = float(arr.mean() + arr.std())
            print(f"  [KL] First-epoch done at step {step_idx}: "
                  f"KL mean={arr.mean():.4f} std={arr.std():.4f} "
                  f"→ early-stop threshold = {self._threshold:.4f}")

            if (self.eviction_enabled
                    and self.eviction_absolute_threshold > 0.0
                    and float(arr.mean()) > self.eviction_absolute_threshold):
                self.stop_requested = True
                self.stop_step = int(step_idx)
                self.stop_action = "evict"
                print(f"  [KL] EVICT at first-epoch boundary "
                      f"(step {step_idx}): first-epoch mean KL "
                      f"= {arr.mean():.4f} > absolute threshold "
                      f"{self.eviction_absolute_threshold:.4f}. "
                      f"Task will retrain as own new group.")

            if self.adaptive_beta:
                kl_mean = float(arr.mean())
                task_mean = (float(self._task_loss_running_mean)
                             if self._task_loss_running_mean is not None else None)
                if task_mean is None or task_mean <= 0 or kl_mean <= 1e-8:
                    print(f"  [KL] adaptive_beta skipped "
                          f"(task_mean={task_mean}, kl_mean={kl_mean:.6f}); "
                          f"keeping β={self.beta:.4f}")
                else:
                    if self.normalize_kl:
                        new_beta = self.kl_target_contribution * task_mean
                    else:
                        new_beta = (self.kl_target_contribution
                                    * task_mean / kl_mean)
                    new_beta = max(self.beta_floor,
                                   min(new_beta, self.beta_ceiling))
                    print(f"  [KL] adaptive_beta: task_mean={task_mean:.4f} "
                          f"kl_mean={kl_mean:.4f} "
                          f"target={self.kl_target_contribution:.2f} → "
                          f"β: {self.beta:.4f} → {new_beta:.4f}")
                    self.beta = float(new_beta)
        return True

    @staticmethod
    def _call_intervenable(reft_model, batch):
        # No labels here: every caller computes its own loss on the label
        # rows. With labels, the model also builds a full-vocabulary fp32
        # cross-entropy over B x T x V that is never used; for Gemma's 256K
        # vocabulary at 2048 tokens that is about 8 GB per forward pass.
        keep = ("input_ids", "attention_mask",
                "intervention_locations", "subspaces")
        b = {k: v for k, v in batch.items() if k in keep}
        iv_locs = b.pop("intervention_locations", None)
        b.pop("subspaces", None)
        b["use_cache"] = False
        unit_locations = None
        if iv_locs is not None:
            unit_locations = {"sources->base": (None, iv_locs.permute(1, 0, 2).tolist()
                                                if hasattr(iv_locs, "permute")
                                                else iv_locs)}
        _, outputs = reft_model(base=b, unit_locations=unit_locations)
        return outputs

    @torch.no_grad()
    def compute_kl_on_batch(self, inputs) -> Tuple[torch.Tensor, torch.Tensor]:
        self.shadow_reft.eval()
        outputs = self._call_intervenable(self.shadow_reft, inputs)
        logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
        labels = inputs.get("labels", None)
        if labels is not None:
            mask = _kl_mask(labels, logits.device)
        else:
            mask = torch.ones(logits.shape[:2],
                              dtype=torch.bool, device=logits.device)
        # Keep only the label rows (N x V). The full B x T x V anchor logits
        # would otherwise stay allocated through the live backward pass.
        if mask.dim() == 2 and mask.shape == logits.shape[:2]:
            return logits[mask.bool()].detach(), mask
        return logits.detach(), mask

    def compute_step_kl(self, live_logits: torch.Tensor,
                        anchor_logits: torch.Tensor,
                        label_mask: torch.Tensor) -> torch.Tensor:
        # Same quantity as before (mean per-token KL over label positions),
        # computed on the label rows only. The previous form materialised
        # five B x T x V fp32 tensors before masking; at B=4, T=2048 and a
        # 128K vocabulary that is about 4 GB each and runs out of memory.
        # Same code as the CRAFT-LoRA copy.
        rows = label_mask.to(live_logits.device).bool()
        if rows.dim() == 2 and rows.shape == live_logits.shape[:2]:
            n_rows = int(rows.sum().item())
            if n_rows == 0:
                return live_logits.float().sum() * 0.0
            live_sel   = live_logits[rows]
            anchor_sel = anchor_logits.to(live_logits.device)
            if anchor_sel.dim() == 3:        # full B x T x V; otherwise already N x V
                anchor_sel = anchor_sel[rows]
            live_lp    = F.log_softmax(live_sel.float(),   dim=-1)
            anchor_lp  = F.log_softmax(anchor_sel.float(), dim=-1)
            kl_rows    = (anchor_lp.exp() * (anchor_lp - live_lp)).sum(dim=-1)
            return kl_rows.sum() / float(n_rows)
        live_lp   = F.log_softmax(live_logits.float(),   dim=-1)
        anchor_lp = F.log_softmax(anchor_logits.float(), dim=-1)
        anchor_p  = anchor_lp.exp()
        kl_per_token = (anchor_p * (anchor_lp - live_lp)).sum(dim=-1) 
        mask_f = label_mask.to(kl_per_token.dtype)
        denom  = mask_f.sum().clamp_min(1.0)
        return (kl_per_token * mask_f).sum() / denom
    
    @torch.no_grad()
    def compute_anchor_hidden(self, inputs, layer_indices: List[int]
                              ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        self.shadow_reft.eval()
        b = dict(inputs)
        iv_locs = b.pop("intervention_locations", None)
        b.pop("subspaces", None)
        b["use_cache"] = False
        b["output_hidden_states"] = True
        unit_locations = None
        if iv_locs is not None:
            unit_locations = {"sources->base": (None, iv_locs.permute(1, 0, 2).tolist()
                                                if hasattr(iv_locs, "permute")
                                                else iv_locs)}
        _, outputs = self.shadow_reft(base=b, unit_locations=unit_locations)
        all_hidden = getattr(outputs, "hidden_states", None)
        if all_hidden is None:
            raise RuntimeError("Shadow forward did not return hidden_states; "
                               "check output_hidden_states plumbing.")
        picked = [all_hidden[i].detach() for i in layer_indices]
        labels = inputs.get("labels", None)
        if labels is not None and picked:
            mask = (labels != -100).to(picked[0].device)
        else:
            mask = torch.ones(picked[0].shape[:2],
                              dtype=torch.bool, device=picked[0].device)
        return picked, mask

    def compute_rep_kl(self, live_hidden_list: List[torch.Tensor],
                       anchor_hidden_list: List[torch.Tensor],
                       mask: torch.Tensor) -> torch.Tensor:
        assert len(live_hidden_list) == len(anchor_hidden_list)
        assert len(live_hidden_list) > 0
        total = 0.0
        for live_h, anchor_h in zip(live_hidden_list, anchor_hidden_list):
            diff = (live_h.float() - anchor_h.float())
            sq_per_tok = (diff * diff).sum(dim=-1)
            mask_f = mask.to(sq_per_tok.dtype)
            denom = mask_f.sum().clamp_min(1.0)
            per_layer = 0.5 * (sq_per_tok * mask_f).sum() / denom / live_h.size(-1)
            total = total + per_layer
        return total / float(len(live_hidden_list))

    def record(self, kl_value: float, step_idx: int,
               task_loss: Optional[float] = None):
        k = float(kl_value)
        self.kl_history.append(k)
        self.n_steps_seen += 1
        self.total_kl_sum += k

        alpha = self.running_mean_ema
        if self._kl_running_mean is None:
            self._kl_running_mean = k
        else:
            self._kl_running_mean = (alpha * k
                                     + (1.0 - alpha) * self._kl_running_mean)
        if task_loss is not None:
            tl = float(task_loss)
            if self._task_loss_running_mean is None:
                self._task_loss_running_mean = tl
            else:
                self._task_loss_running_mean = (
                    alpha * tl + (1.0 - alpha) * self._task_loss_running_mean)

        if (self.early_stop and self._threshold is not None
                and not self.stop_requested
                and len(self.kl_history) >= self.rolling_window):
            window = self.kl_history[-self.rolling_window:]
            rolling = float(np.mean(window))
            if rolling > self._threshold:
                self.stop_requested = True
                self.stop_step = int(step_idx)
                # Eviction is only eligible once the beta ramp has finished.
                # Redundant given that set_first_epoch_boundary() already
                # defers past the ramp (so _first_epoch_end >= warmup), but
                # stated explicitly so the invariant cannot be broken by a
                # future change to where the boundary is set.
                ramp_complete = (self.beta_warmup_steps <= 0
                                 or step_idx >= self.beta_warmup_steps)
                is_early = (self._first_epoch_end is not None
                            and ramp_complete
                            and (step_idx - self._first_epoch_end)
                                < self.eviction_early_steps)
                severity = rolling / max(self._threshold, 1e-8)
                is_severe = severity > self.eviction_severity_threshold
                if self.eviction_enabled and is_early and is_severe:
                    self.stop_action = "evict"
                    print(f"  [KL] EVICT at step {step_idx}: rolling mean "
                          f"(last {self.rolling_window}) = {rolling:.4f} "
                          f"> threshold {self._threshold:.4f}  "
                          f"(severity={severity:.2f}×, steps-since-epoch1="
                          f"{step_idx - self._first_epoch_end}).  Task will "
                          f"retrain as own new group.")
                else:
                    self.stop_action = "stop"
                    print(f"  [KL] Early stop at step {step_idx}: rolling "
                          f"mean (last {self.rolling_window}) = {rolling:.4f} "
                          f"> threshold {self._threshold:.4f}.")

    def summary(self) -> Dict[str, float]:
        if self.n_steps_seen == 0:
            return {"mean_kl": float("nan"), "final_kl": float("nan"),
                    "stop_step": None, "threshold": float("nan"),
                    "n_steps": 0, "stop_action": None,
                    "beta_initial": self._beta_initial,
                    "beta_final": self.beta, "kl_mode": self.kl_mode}
        tail = self.kl_history[-self.rolling_window:] if self.kl_history else []
        return {
            "mean_kl":   self.total_kl_sum / self.n_steps_seen,
            "final_kl":  float(np.mean(tail)) if tail else float("nan"),
            "stop_step": self.stop_step,
            "stop_action": self.stop_action,
            "threshold": self._threshold if self._threshold is not None else float("nan"),
            "n_steps":   self.n_steps_seen,
            "beta_initial": self._beta_initial,
            "beta_final":   self.beta,
            "kl_mode": self.kl_mode,
        }

    @torch.no_grad()
    def evaluate_save_time_kl(self, live_reft, eval_loader, max_batches: int = 50,
                              device: str = "cuda") -> float:
        live_reft.eval()
        self.shadow_reft.eval()
        total_kl, total_tokens = 0.0, 0
        n_used = 0
        for batch in eval_loader:
            if n_used >= max_batches:
                break
            batch = {k: (v.to(device) if (hasattr(v, "to") and hasattr(v, "device")) else v)
                     for k, v in batch.items()}
            try:
                live_out    = self._call_intervenable(live_reft,       batch)
                shadow_out  = self._call_intervenable(self.shadow_reft, batch)
            except Exception as e:
                print(f"  [KL] eval batch skipped ({type(e).__name__}: {e})")
                continue
            live_logits   = live_out.logits   if hasattr(live_out,   "logits") else live_out[0]
            anchor_logits = shadow_out.logits if hasattr(shadow_out, "logits") else shadow_out[0]
            labels = batch.get("labels", None)
            if labels is not None:
                mask = _kl_mask(labels, live_logits.device)
            else:
                mask = torch.ones(live_logits.shape[:2],
                                  dtype=torch.bool, device=live_logits.device)
            # Label rows only, as in compute_step_kl: same value, far less memory.
            rows = mask.bool()
            n_rows = int(rows.sum().item())
            if n_rows > 0:
                live_lp   = F.log_softmax(live_logits[rows].float(),   dim=-1)
                anchor_lp = F.log_softmax(anchor_logits[rows].float(), dim=-1)
                kl_rows   = (anchor_lp.exp() * (anchor_lp - live_lp)).sum(dim=-1)
                total_kl     += float(kl_rows.sum().item())
                total_tokens += n_rows
            n_used += 1
            del live_out, shadow_out, live_logits, anchor_logits
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        if total_tokens == 0:
            return float("nan")
        return total_kl / total_tokens

    def teardown(self):
        try:
            _teardown_reft_model(self.shadow_reft)
        except Exception:
            pass
        self.shadow_reft = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class MoIReftTrainer(ReftTrainerForCausalLM):
    def set_custom_optimizer(self, opt):
        self._custom_optimizer = opt
    def create_optimizer(self):
        if getattr(self, '_custom_optimizer', None) is not None:
            self.optimizer = self._custom_optimizer
            return self.optimizer
        super().create_optimizer()
        return self.optimizer

    def set_kl_controller(self, controller: Optional["KLRazorController"]):
        """Attach (or clear) the KL-Razor controller for this Phase-2 run."""
        self._kl_controller = controller
        self._kl_first_epoch_recorded = False

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        kl_ctrl: Optional[KLRazorController] = getattr(self, "_kl_controller", None)
        if kl_ctrl is None:
            return super().compute_loss(
                model, inputs, return_outputs=return_outputs)

        use_rep_kl = (kl_ctrl.kl_mode == "representation"
                      and kl_ctrl.rep_layer_indices)
        
        if use_rep_kl:
            b = dict(inputs)
            iv_locs = b.pop("intervention_locations", None)
            b.pop("subspaces", None)
            b["use_cache"] = False
            b["output_hidden_states"] = True
            unit_locations = None
            if iv_locs is not None:
                unit_locations = {"sources->base": (None, iv_locs.permute(1, 0, 2).tolist()
                                                    if hasattr(iv_locs, "permute")
                                                    else iv_locs)}
            _, live_out = model(base=b, unit_locations=unit_locations)
        else:
            live_out = KLRazorController._call_intervenable(model, inputs)

        task_loss   = live_out.loss
        live_logits = live_out.logits

        labels = inputs.get("labels", None)
        if labels is not None and live_logits is not None:
            # Same value as cross-entropy over every position with
            # ignore_index=-100 (the mean over target tokens), computed on the
            # target rows only instead of an fp32 copy of all B x T x V logits.
            shift_labels = labels[..., 1:].to(live_logits.device)
            tgt = shift_labels != -100
            if bool(tgt.any()):
                task_loss = F.cross_entropy(
                    live_logits[..., :-1, :][tgt].float(), shift_labels[tgt])
            else:
                task_loss = live_logits.float().sum() * 0.0

        total_loss  = task_loss

        try:
            if use_rep_kl:
                with torch.no_grad():
                    anchor_hiddens, label_mask = kl_ctrl.compute_anchor_hidden(
                        inputs, kl_ctrl.rep_layer_indices)
                live_all_hidden = getattr(live_out, "hidden_states", None)
                if live_all_hidden is None:
                    raise RuntimeError("Live forward missing hidden_states.")
                live_hiddens = [live_all_hidden[i]
                                for i in kl_ctrl.rep_layer_indices]
                raw_kl = kl_ctrl.compute_rep_kl(
                    live_hiddens, anchor_hiddens, label_mask)
                del anchor_hiddens, live_hiddens, label_mask
            else:
                with torch.no_grad():
                    anchor_logits, label_mask = kl_ctrl.compute_kl_on_batch(inputs)
                raw_kl = kl_ctrl.compute_step_kl(
                    live_logits, anchor_logits, label_mask)
                del anchor_logits, label_mask

            kl_term_val = float(raw_kl.detach().item())

            if (kl_ctrl.normalize_kl
                    and kl_ctrl._kl_running_mean is not None
                    and kl_ctrl._kl_running_mean > 1e-8):
                kl_for_loss = raw_kl / kl_ctrl._kl_running_mean
            else:
                kl_for_loss = raw_kl

            kl_ctrl.record(
                kl_value=kl_term_val,
                step_idx=self.state.global_step,
                task_loss=float(task_loss.detach().item()),
            )
            beta_eff = kl_ctrl.effective_beta(self.state.global_step)
            total_loss = task_loss + beta_eff * kl_for_loss
        except torch.cuda.OutOfMemoryError:
            # KL_OOM_FATAL=1: stop the job. Otherwise the step silently trains
            # without the anchor, and the task is no longer trained by CRAFT.
            if os.environ.get("KL_OOM_FATAL", "") in ("1", "true", "True"):
                raise
            print("  [KL] OOM on shadow forward; this step uses L_task only.")
            if torch.cuda.is_available(): torch.cuda.empty_cache()
        except Exception as e:
            print(f"  [KL] shadow forward failed ({type(e).__name__}: {e}); "
                  f"this step uses L_task only.")

        if (not self._kl_first_epoch_recorded
                and self.state.epoch is not None
                and self.state.epoch >= 1.0):
            # May defer while the beta ramp is still running; retry each step
            # until it is accepted, so the threshold is measured at full anchor.
            if kl_ctrl.set_first_epoch_boundary(self.state.global_step):
                self._kl_first_epoch_recorded = True

        if return_outputs:
            return total_loss, live_out
        return total_loss


    def set_agem_projector(self, projector: Optional["AGemShadowProjector"]):
        self._agem_projector = projector

    def training_step(self, model, inputs, num_items_in_batch=None):

        agem_prj: Optional[AGemShadowProjector] = getattr(self, "_agem_projector", None)
        if agem_prj is None:
            return super().training_step(model, inputs)

        try:
            prepared_inputs = self._prepare_inputs(inputs)
            anchor_W_grads  = agem_prj.compute_anchor_W_grads(prepared_inputs)
        except torch.cuda.OutOfMemoryError:
            # AGEM_OOM_FATAL=1: stop the job. Otherwise the step silently trains
            # without the anchor, and the task is no longer trained by CRAFT.
            if os.environ.get("AGEM_OOM_FATAL", "") in ("1", "true", "True"):
                raise
            print(f"  [agem] OOM on shadow pass; skipping projection this step.")
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            return super().training_step(model, inputs)
        except Exception as e:
            print(f"  [agem] anchor pass failed ({type(e).__name__}: {e}); "
                  f"skipping projection for this step.")
            return super().training_step(model, inputs)

        loss = super().training_step(model, inputs)
        agem_prj.project_live_W_grads(anchor_W_grads)
        anchor_W_grads.clear()
        del anchor_W_grads
        return loss


class EarlyStopOnKLCallback(TrainerCallback):
    def __init__(self, kl_controller: "KLRazorController"):
        self.ctrl = kl_controller

    def on_step_end(self, args, state, control, **kwargs):
        if self.ctrl is not None and self.ctrl.stop_requested:
            control.should_training_stop = True
        return control

def evaluate_reft_on_task(
    task_name, reft_model, tokenizer, eval_dataset, data_items,
    eval_batch_size, greedy_decoding, temperature, top_p, top_k,
) -> float:
    _, stats = compute_metrics(
        task_name, task_name, reft_model, tokenizer,
        eval_dataset, data_items, trigger_tokens="", run_name="cl_eval",
        batch_size=eval_batch_size, data_collator=None, split="eval",
        greedy_decoding=greedy_decoding, temperature=temperature,
        top_p=top_p, top_k=top_k,
    )
    return stats.get(f"eval/{task_name}", 0.0)



def finetune_cl(args):
    hf_token = os.environ.get("HF_TOKEN", "")
    if hf_token:
        login(token=hf_token)
    reset_logs()
    print("  [logger] All logs reset.")

    global _KL_PRED_MASK
    _KL_PRED_MASK = bool(getattr(args, "kl_pred_mask", False))
    kl_per_task_log = []
    perf_matrix = {}    
    task_to_group_log = {}  

    benchmark = getattr(args, "benchmark", "trace")
    if benchmark == "standard_cl":
        assert getattr(args, "cl_benchmark_dir", None)
        CL_DATA_DIR = args.cl_benchmark_dir
        CL_ORDER = ([t.strip() for t in args.cl_tasks.split(",") if t.strip()]
                    if getattr(args, "cl_tasks", None) else
                    sorted([d for d in os.listdir(CL_DATA_DIR)
                            if os.path.isdir(os.path.join(CL_DATA_DIR, d))
                            and os.path.exists(os.path.join(CL_DATA_DIR, d, "train.json"))]))
        CL_EPOCH_SCHED = {t: TASK_EPOCH_SCHEDULE.get(t, args.epochs) for t in CL_ORDER}
        print(f"  Benchmark: standard_cl  tasks={CL_ORDER}")
    else:
        CL_ORDER       = TRACE_CL_ORDER
        CL_DATA_DIR    = args.data_dir
        CL_EPOCH_SCHED = TASK_EPOCH_SCHEDULE
        print(f"  Benchmark: trace  ({len(CL_ORDER)} tasks)")

    dtype = dtype_mapping[args.dtype]
    set_seed(args.seed)

    # Stream position. --f_positions/--l_positions give the asymmetric form
    # (e.g. f5+l25); with neither set it stays symmetric fN+lN from
    # --n_positions, which is the old behaviour unchanged.
    n_pos = int(getattr(args, "n_positions", 9))
    f_pos = getattr(args, "f_positions", None)
    l_pos = getattr(args, "l_positions", None)
    if f_pos is None and l_pos is None:
        f_pos = l_pos = n_pos
    else:
        f_pos = int(f_pos) if f_pos is not None else n_pos
        l_pos = int(l_pos) if l_pos is not None else n_pos
    assert f_pos > 0 and l_pos > 0, (
        f"--f_positions/--l_positions must both be > 0 (got f={f_pos}, l={l_pos}). "
        f"Interventions are built as base_layers x {LAYER_MULT} (one f-stream and "
        f"one l-stream per layer), so a single-stream position string would not "
        f"match len(task_layers).")
    task_pos = f"f{f_pos}+l{l_pos}"

    def _parse_task_map(spec, cast):
        out = {}
        for item in (spec or "").split(","):
            if ":" in item:
                k, v = item.split(":", 1)
                out[k.strip()] = cast(v.strip())
        return out
    per_task_maxlen = _parse_task_map(getattr(args, "per_task_max_length", None), int)
    per_task_pos    = _parse_task_map(getattr(args, "per_task_positions", None), str)
    for _t, _p in per_task_pos.items():
        _ok = (_p.startswith("f") and "+l" in _p
               and _p[1:].split("+l")[0].isdigit()
               and _p.split("+l")[1].isdigit())
        assert _ok, f"--per_task_positions {_t}:{_p} must look like fN+lM"
    print(f"  Position: {task_pos}  (f={f_pos}, l={l_pos})"
          + (f"  overrides={per_task_pos}" if per_task_pos else ""))
    if per_task_maxlen:
        print(f"  Max length: {args.max_length}  overrides={per_task_maxlen}")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model, model_max_length=args.max_length,
        padding_side="right", use_fast=False)

    _name_lower = (args.model or "").lower()
    _is_llama3 = (tokenizer.unk_token is None and
                  (tokenizer.pad_token is None or
                   tokenizer.pad_token_id == tokenizer.eos_token_id))
    _is_mistral = "mistral" in _name_lower

    if _is_llama3:
        rid = tokenizer.convert_tokens_to_ids("<|reserved_special_token_0|>")
        tokenizer.pad_token = (
            "<|reserved_special_token_0|>"
            if (rid and rid != tokenizer.unk_token_id) else tokenizer.eos_token)
    elif _is_mistral:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"  [pad-token] Mistral detected: pad_token set to eos_token "
              f"(id={tokenizer.pad_token_id}) to avoid unk-id-0 collision.")
        if getattr(args, "mistral_system_first_turn", False):
            tokenizer.chat_template = MISTRAL_CHAT_TEMPLATE
            print("  [chat-template] Mistral: system prompt placed in the first "
                  "user turn (prompt is a prefix of prompt+answer).")
    elif tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token

    question_offset = _find_question_start(tokenizer)

    base_model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype if dtype != "float8" else None,
        load_in_8bit=(dtype == "float8"),
        device_map=device)
    for p in base_model.parameters():
        p.requires_grad = False
    print("Base model FROZEN.")

    base_layers = ([] if args.layers.strip() == "" else
                   [int(l) for l in args.layers.split(";")] if args.layers != "all" else
                   list(range(base_model.config.num_hidden_layers)))
    task_layers = _get_task_layers(base_layers)


    intervention_cls = intervention_mapping[args.intervention_type]
    iv_dtype         = torch.bfloat16 if isinstance(dtype, str) else dtype

    shallow_frac = getattr(args, "shallow_frac", 0.33)
    deep_frac    = getattr(args, "deep_frac",    0.38)
    tier1_layers, tier2_layers, tier3_layers = _stratify_layers(
        base_layers, shallow_frac=shallow_frac, deep_frac=deep_frac)
    tier1_set = set(tier1_layers)
    tier2_set = set(tier2_layers)
    tier3_set = set(tier3_layers)

    assign_thr    = getattr(args, "weight_distance_threshold", 0.30)
    warmup_steps  = getattr(args, "causal_warmup_steps",        50)
    skip_steps    = getattr(args, "warmup_skip_steps",          15)
    tier1_alpha   = getattr(args, "tier1_alpha",                0.30)
    group_alpha   = getattr(args, "group_alpha",                0.50)
    use_moi       = getattr(args, "moi",                        True)
    normalize     = getattr(args, "weight_distance_normalize",  "zscore")
    aggregate     = getattr(args, "weight_distance_tiers",      "mean_all")
    fixed_seed    = getattr(args, "fixed_intervention_seed",    42)
    router_method = getattr(args, "router_method",              "kl_routing")

    manual_routing    = (bool(getattr(args, "manual_routing", False))
                         or router_method == "manual")
    manual_group_map  = _parse_manual_groups_arg(getattr(args, "manual_groups", None))
    if manual_routing:
        print(f"  Manual routing ENABLED — warmup SKIPPED.")
        print(f"  Manual task→group map ({len(manual_group_map)} tasks):")
        for t, g in manual_group_map.items():
            print(f"    {t:<15} → {g}")

    router = None
    flat_routing       = bool(getattr(args, "flat_routing",       True))
    flat_aggregator    = str(getattr(args,  "flat_aggregator",     "vote"))
    flat_vote_quantile = float(getattr(args,"flat_vote_quantile",  0.5))
    vote_max_dist      = float(getattr(args, "vote_max_dist",      0.20))

    group_capacity  = int(getattr(args, "group_capacity", 0) or 0)
    merge_rule      = str(getattr(args, "merge_rule", "overwrite")).lower()
    merge_ema_alpha = float(getattr(args, "merge_ema_alpha", 0.5))
    merge_kl_scale  = float(getattr(args, "merge_kl_scale", 1.0))
    _halflife = getattr(args, "merge_kl_halflife", None)
    if _halflife is not None:
        merge_kl_scale = kl_scale_from_halflife(_halflife)
        print(f"  [merge] kl half-life {_halflife} → merge_kl_scale "
              f"{merge_kl_scale:.4g} (overrides --merge_kl_scale)")
    merge_fisher_batches = int(getattr(args, "merge_fisher_batches", 8))

    memory = WeightDistanceMemory(
        tier1_set, tier2_set, tier3_set,
        group_alpha            = group_alpha,
        assign_threshold       = assign_thr,
        normalize              = normalize,
        aggregate              = aggregate,
        fixed_intervention_seed= fixed_seed,
        flat_routing           = flat_routing,
        flat_aggregator        = flat_aggregator,
        flat_vote_quantile     = flat_vote_quantile,
        vote_max_dist          = vote_max_dist,
        group_capacity         = group_capacity,
        merge_rule             = merge_rule,
        merge_ema_alpha        = merge_ema_alpha,
        merge_kl_scale         = merge_kl_scale,
    )

    # Trainable task-identity router (router_head.py). Lives beside the
    # memory; refit at every commit; never touches the interventions.
    router_head = None
    memory.router_head = None
    if getattr(args, "router_head", False):
        router_head = RouterHead(layer_ids=_unique_layers(task_layers),
                                 n_pos=l_pos, l2=args.router_head_l2,
                                 iters=args.router_head_iters,
                                 prompt_only=bool(getattr(args, "router_head_prompt_only", False)))
        print(f"  [router_head] enabled: layers={router_head.layer_ids} "
              f"n_pos={l_pos} samples/task={args.router_head_samples} "
              f"prompt_only={router_head.prompt_only}")

    memory.all_groupwise = args.all_groupwise

    if not manual_routing and router_method != "manual":
        _init_reft = _build_reft_model(
            base_model, task_layers, args.rank,
            intervention_cls, iv_dtype, args)
        memory.set_init_state(_extract_iv_state(_init_reft))
        del _init_reft
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    else:
        print(f"  [memory] Manual routing: skipping fixed-seed init snapshot "
              f"(delta-cosine not used).")

    print(f"\nCRAFT-CL [Unified Warmup Routers]:")
    print(f"  router     : {router_method}")
    print(f"  flat_route : {flat_routing}  aggregator={flat_aggregator}  "
          f"vote_max_dist={vote_max_dist}")
    print(f"  aggregate  : {aggregate}    (legacy tier-mode; ignored when flat_routing=True)")
    print(f"  normalize  : {normalize}")
    print(f"  threshold  : {assign_thr}   "
          f"(in 'vote' mode this is the minimum vote MARGIN to join, not a distance)")
    print(f"  fixed_seed : {fixed_seed}  (same LoReFT init for every task)")
    print(f"  stratify   : T1={len(tier1_layers)} T2={len(tier2_layers)} T3={len(tier3_layers)}  "
          f"(used only for storage/EMA; routing is per-iv flat)")
    print(f"  position   : {task_pos}  (f={f_pos}, l={l_pos})")
    print(f"  group_cap  : {group_capacity if group_capacity > 0 else 'uncapped'}")
    print(f"  merge_rule : {merge_rule}"
          + (f"  fisher_batches={args.merge_fisher_batches}" if merge_rule == "fisher" else "")
          + (f"  kl_scale={args.merge_kl_scale}" if merge_rule == "kl" else "")
          + (f"  ema_alpha={args.merge_ema_alpha}" if merge_rule == "ema" else ""))

    if getattr(args, "per_task_epochs", None):
        raw = [int(e) for e in args.per_task_epochs.split(",")]
        if len(raw) != len(CL_ORDER):
            raw = (raw + [args.epochs] * len(CL_ORDER))[:len(CL_ORDER)]
        epoch_schedule = {t: e for t, e in zip(CL_ORDER, raw)}
    else:
        epoch_schedule = dict(CL_EPOCH_SCHED)
    print("\nEpoch schedule:", epoch_schedule)

    n_tasks     = len(CL_ORDER)
    perf_matrix = [[None] * n_tasks for _ in range(n_tasks)]
    cl_eval_sets: Dict[str, dict] = {}
    routing_log: List[dict] = []

    resume_upto: Optional[int] = None
    resume_rng = None
    if getattr(args, "resume_from", None):
        _st = _load_resume_state(args.resume_from, CL_ORDER)
        resume_upto       = int(_st["completed_task_idx"])
        perf_matrix       = _st["perf_matrix"]
        kl_per_task_log   = _st["kl_per_task_log"]
        task_to_group_log = _st["task_to_group_log"]
        routing_log       = _st["routing_log"]
        memory            = _st["memory"]
        router_head       = _st["router_head"]
        memory.router_head = router_head
        resume_rng        = _st.get("rng")
        del _st
        print(f"  [resume] restored state after task {resume_upto + 1}/{n_tasks} "
              f"('{CL_ORDER[resume_upto]}'); groups={memory.task_to_group}")

    os.makedirs(args.output_dir, exist_ok=True)
    if resume_upto is not None and (args.skip_eval or args.save_eval_snapshots):
        # A continued run writes into a new output folder. Bring the per-task
        # snapshots of the tasks already done, so this folder holds all of them
        # and can be evaluated later (--eval_only needs one snapshot per task).
        import shutil
        _n = 0
        for _i in range(resume_upto + 1):
            _src = _eval_snapshot_path(args.resume_from, _i, CL_ORDER[_i])
            _dst = _eval_snapshot_path(args.output_dir, _i, CL_ORDER[_i])
            if os.path.isfile(_src) and not os.path.isfile(_dst):
                os.makedirs(os.path.dirname(_dst), exist_ok=True)
                shutil.copy2(_src, _dst)
                _n += 1
        print(f"  [resume] copied {_n} evaluation snapshot(s) of tasks 1-{resume_upto + 1} "
              f"from {args.resume_from}")
    _mode = ("eval_only" if args.eval_only else "train_only" if args.skip_eval else "train_and_eval")
    try:
        with open(os.path.join(args.output_dir,
                               "eval_args.json" if args.eval_only else TRAIN_ARGS_FILE), "w") as f:
            json.dump(vars(args), f, indent=2, default=str)
    except Exception as e:
        print(f"  [args] WARNING: could not write the argument file ({type(e).__name__}: {e})")
    print(f"\n  Run mode : {_mode}" + (f"   (evaluating {args.eval_from})" if args.eval_only else ""))

    def _row_eval(task_idx):
        """Row task_idx of the performance matrix: the task just trained, or every
        task so far with --per_task_bwd_eval. Uses the current memory state."""
        task_name = CL_ORDER[task_idx]
        print(f"\n  Backward evaluation after {task_name}:")
        print(f"  {'Task':<15}  {'Score':>7}  {'Forgetting':>12}")
        print(f"  {'-'*15}  {'-'*7}  {'-'*12}")

        _release_gpu()
        if torch.cuda.is_available():
            _mem_before = torch.cuda.memory_allocated() / (1024**3)
            print(f"  [gpu_mem] before backward-eval: {_mem_before:.2f} GiB allocated")

        with torch.no_grad():
            eval_range = (range(task_idx + 1) if args.per_task_bwd_eval else [task_idx])
            for eval_idx in eval_range:
                eval_task = CL_ORDER[eval_idx]
                ev_info   = cl_eval_sets[eval_task]

                eval_reft = _build_reft_model(
                    base_model, task_layers, args.rank,
                    intervention_cls, iv_dtype, args)

                loaded = route_and_load(
                    eval_reft, eval_task, memory, base_model, tokenizer,
                    args, ev_info["dataset"],
                    build_reft_fn=lambda: _build_reft_model(
                        base_model, task_layers, args.rank,
                        intervention_cls, iv_dtype, args),
                    teardown_fn=_teardown_reft_model)
                if not loaded:
                    perf_matrix[task_idx][eval_idx] = 0.0
                    _teardown_reft_model(eval_reft)
                    del eval_reft
                    _release_gpu()
                    continue

                eval_reft.eval()
                try:
                    score = evaluate_reft_on_task(
                        eval_task, eval_reft, tokenizer,
                        ev_info["dataset"], ev_info["data_items"],
                        args.eval_batch_size, args.greedy_decoding,
                        args.temperature, args.top_p, args.top_k)
                finally:
                    _teardown_reft_model(eval_reft)
                    del eval_reft
                    _release_gpu()
                perf_matrix[task_idx][eval_idx] = score

                fgt_str = ("  (just trained)" if eval_idx == task_idx else
                           f"{perf_matrix[eval_idx][eval_idx]-score:+.4f}"
                           if perf_matrix[eval_idx][eval_idx] is not None
                           else "     N/A")
                print(f"  {eval_task:<15}  {score:>7.4f}  {fgt_str:>12}")

        if torch.cuda.is_available():
            _mem_after = torch.cuda.memory_allocated() / (1024**3)
            print(f"  [gpu_mem] after backward-eval:  {_mem_after:.2f} GiB allocated")

        if args.inference_router not in ("oracle", "litar"):   # litar writes its own log
            agn_dump_report(args.output_dir)

        if task_idx > 0:
            fgt_list = [perf_matrix[j][j] - perf_matrix[task_idx][j]
                        for j in range(task_idx)
                        if (perf_matrix[j][j] is not None and
                            perf_matrix[task_idx][j] is not None)]
            if fgt_list:
                print(f"\n  Mean forgetting: {float(np.mean(fgt_list)):+.4f}")

    for task_idx, task_name in enumerate(CL_ORDER):
        n_epochs = epoch_schedule[task_name]
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        if (resume_upto is not None and task_idx == resume_upto + 1
                and resume_rng is not None):
            import random
            torch.set_rng_state(resume_rng["torch"])
            if resume_rng.get("cuda") is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(resume_rng["cuda"])
            np.random.set_state(resume_rng["numpy"])
            random.setstate(resume_rng["python"])
        ref_type = TRACE_TASK_TYPES.get(task_name, 0)
        ref_name = TASK_TYPE_NAMES.get(ref_type, "?")

        # per-task overrides (default: global values, unchanged behaviour)
        task_pos_t   = per_task_pos.get(task_name, task_pos)
        task_maxlen  = per_task_maxlen.get(task_name, args.max_length)
        tokenizer.model_max_length = task_maxlen   # datasets truncate at build

        print(f"\n{'='*72}")
        print(f"  CL step {task_idx+1}/{n_tasks}:  {task_name}  "
              f"[epochs={n_epochs}  pos={task_pos_t}  max_len={task_maxlen}]  "
              f"(ref={ref_name})")
        print(f"{'='*72}\n")

        train_path = os.path.join(CL_DATA_DIR, task_name, "train.json")
        assert os.path.exists(train_path), f"Not found: {train_path}"

        train_dataset = LoReftTRACEDataset(
            task_name, train_path, tokenizer, data_split="train",
            seed=args.seed, max_n_example=args.max_n_train_example,
            num_interventions=len(task_layers), position=task_pos_t,
            share_weights=args.share_weights, test_split="test",
            question_offset=question_offset)
        print(f"  Training samples: {len(train_dataset)}")

        try:
            _first = train_dataset[0]
            _labels = _first.get("labels", None)
            if _labels is not None:
                if not isinstance(_labels, torch.Tensor):
                    _labels = torch.tensor(_labels)
                _nonmask = int((_labels != -100).sum().item())
                _total   = int(_labels.numel())
                _pct     = (100.0 * _nonmask / _total) if _total > 0 else 0.0
                _live_ids = (_labels[_labels != -100][:20].tolist()
                             if _nonmask > 0 else [])
                print(f"  [label-check] task={task_name}  "
                      f"non-masked={_nonmask}/{_total} ({_pct:.2f}%)  "
                      f"first 20 label-ids={_live_ids}")
                if _pct < 0.5:
                    print(f"  [label-check] *** WARNING: <0.5% of tokens "
                          f"contribute to loss for task '{task_name}'. "
                          f"Training will likely show loss=0 from step 1. "
                          f"Check tokenizer / question_offset / pad_token "
                          f"for this model.")
        except Exception as _e:
            print(f"  [label-check] skipped (could not inspect labels: "
                  f"{type(_e).__name__}: {_e})")
        _ev_file = getattr(args, "cl_eval_file", None)
        ev_path = os.path.join(CL_DATA_DIR, task_name, _ev_file or "eval.json")
        if _ev_file is None and not os.path.exists(ev_path):
            ev_path = os.path.join(CL_DATA_DIR, task_name, "test.json")
        assert os.path.exists(ev_path), f"Eval not found: {ev_path}"
        print(f"  [eval] {task_name}: {os.path.basename(ev_path)}")
        ev_ds = LoReftTRACEDataset(
            task_name, ev_path, tokenizer, data_split="eval", seed=args.seed,
            max_n_example=args.max_n_eval_example,
            num_interventions=len(task_layers), position=task_pos_t,
            share_weights=args.share_weights,
            question_offset=question_offset)
        cl_eval_sets[task_name] = {
            "dataset": ev_ds, "data_items": ev_ds.raw_dataset,
        }
        print(f"  Eval set built: n={len(ev_ds)}")
        tokenizer.model_max_length = args.max_length   # restore global limit

        if args.eval_only:
            # Evaluation only: the state after this task comes from the training
            # run's snapshot; no training happens here.
            _snap = _load_eval_snapshot(args.eval_from, task_idx, CL_ORDER)
            memory            = _snap["memory"]
            router_head       = _snap["router_head"]
            memory.router_head = router_head
            routing_log       = _snap["routing_log"]
            kl_per_task_log   = _snap["kl_per_task_log"]
            task_to_group_log = _snap["task_to_group_log"]
            del _snap
            print(f"  [eval-only] restored the state after '{task_name}' "
                  f"({task_idx + 1}/{n_tasks}); groups={memory.task_to_group}")
            _row_eval(task_idx)
            continue

        if resume_upto is not None and task_idx <= resume_upto:
            # Finished in the resumed run: its evaluation set is rebuilt above
            # (needed for the final row), everything else comes from the state.
            print(f"  [resume] '{task_name}' finished in the resumed run; not retrained.")
            continue

        # Built once per task and reused: the routing phase scores against it,
        # and memory.save() retains it as the group's evidence so a LATER
        # arrival can be scored on this task's data (--kl_routing_direction
        # backward). Without retaining it, incumbent-side routing has nothing
        # to measure against.
        task_probe = _build_current_task_probe(
            train_dataset, tokenizer,
            int(getattr(args, "kl_routing_probe_size", 16)), device)

        warmup_end_state: Dict[str, dict] = {}
        assigned_gid: Optional[int] = None
        is_new_group: bool          = True
        routing_debug: Dict         = {}

        if manual_routing:
            group_name = manual_group_map.get(task_name)
            if group_name is None:
                raise KeyError(
                    f"Task '{task_name}' has no entry in --manual_groups / "
                    f"MANUAL_TASK_GROUPS.  Known tasks: {sorted(manual_group_map)}"
                )
            assigned_gid, is_new_group, routing_debug = memory.route_manual(
                task_name=task_name, group_name=group_name,
            )
            print(f"\n  [Phase 1 skipped] Manual routing: "
                  f"'{task_name}' → '{group_name}' "
                  f"→ G{assigned_gid} "
                  f"({'NEW GROUP' if is_new_group else 'join existing'})")

        elif router_method == "kl_routing":
            warmup_steps_arg   = int(getattr(args, "causal_warmup_steps", 100))
            probe_size_arg     = int(getattr(args, "kl_routing_probe_size", 16))
            kl_threshold_arg   = float(getattr(args, "kl_routing_threshold", 0.5))
            top_k_arg          = int(getattr(args, "kl_routing_top_k", 64))
            temperature_arg    = float(getattr(args, "kl_routing_temperature", 2.0))
            normalize_arg      = str(getattr(args, "kl_routing_normalize", "raw"))
            min_dkg_arg        = float(getattr(args, "kl_routing_min_dkg", 0.01))

            assigned_gid, is_new_group, routing_debug = run_kl_routing_phase(
                base_model=base_model, task_layers=task_layers, args=args,
                intervention_cls=intervention_cls, iv_dtype=iv_dtype,
                train_dataset=train_dataset, tokenizer=tokenizer,
                memory=memory,
                task_name=task_name, task_idx=task_idx,
                warmup_steps=warmup_steps_arg,
                probe_size=probe_size_arg,
                kl_threshold=kl_threshold_arg,
                output_dir=args.output_dir,
                seed=args.seed,
                batch_size=args.batch_size,
                lr=args.lr,
                gradient_accumulation_steps=args.gradient_accumulation_steps,
                top_k=top_k_arg,
                temperature=temperature_arg,
                normalize=normalize_arg,
                min_dkg=min_dkg_arg,
                direction=str(getattr(args, "kl_routing_direction", "forward")),
                base_model_for_bwd=base_model,
                task_probe=task_probe,
            )
            print(f"\n  [Phase 1 complete] kl_routing for '{task_name}':")
            _print_routing_debug(routing_debug)
            _release_gpu()          # also frees the warm-up trainer
            _gpu_mem("after routing")

        else:
            assigned_gid = memory._next_gid
            is_new_group = True
            routing_debug = {
                "task": task_name, "decision": "new_group_no_warmup",
                "assigned_gid": assigned_gid,
            }

        print(f"\n  [Phase 2] Full training ...")
        reft_model = _build_reft_model(
            base_model, task_layers, args.rank, intervention_cls, iv_dtype, args)
        
        if memory.has_task(task_name):
            print(f"  Warm-starting from memory (task seen before).")
            memory.load(reft_model, task_name=task_name)
        elif (not is_new_group) and assigned_gid in memory.groups:
            print(f"  Warm-starting from group G{assigned_gid}.")
            memory.load(reft_model, gid=assigned_gid)
        else:
            print(f"  New group — starting from fresh fixed-seed init.")


        reft_model.print_trainable_parameters()

        optimizer = build_stratified_optimizer(
            reft_model, task_layers, tier1_set, tier2_set, tier3_set,
            base_lr=args.lr, weight_decay=args.weight_decay)

        data_collator = ReftDataCollator(data_collator=DataCollatorForSeq2Seq(
            tokenizer=tokenizer, model=reft_model.model,
            label_pad_token_id=-100, padding="longest"))
        now_run = datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")
        tr_args = TrainingArguments(
            output_dir=os.path.join(
                args.output_dir, f"cl_{task_idx:02d}.{task_name}.{now_run}"),
            run_name=f"cl_{task_idx:02d}.{task_name}",
            num_train_epochs=n_epochs,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.eval_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            eval_strategy="no", save_strategy="no",
            logging_strategy="steps", logging_steps=args.logging_steps,
            lr_scheduler_type=args.schedule,
            learning_rate=args.lr, warmup_ratio=args.warmup_ratio,
            optim="adamw_torch", weight_decay=args.weight_decay,
            report_to="none", use_cpu=(device != "cuda"),
            seed=args.seed, remove_unused_columns=False,
        )
        trainer = MoIReftTrainer(
            model=reft_model, tokenizer=tokenizer,
            args=tr_args, train_dataset=train_dataset,
            eval_dataset=None, data_collator=data_collator)
        trainer.set_custom_optimizer(optimizer)

        kl_controller: Optional[KLRazorController] = None
        agem_projector: Optional[AGemShadowProjector] = None
        shadow_reft_for_mem = None
        use_agem_flag = bool(getattr(args, "use_agem", False))
        use_kl_flag   = bool(getattr(args, "use_kl",   True))

        if (not is_new_group) and assigned_gid in memory.groups:
            groupwise_iv_keys = set(memory.groups[assigned_gid]["iv_state"].keys())
            anchor_iv_state = {
                k: _clone_state(v) for k, v in memory.groups[assigned_gid]["iv_state"].items()
            }
            _release_gpu()
            _gpu_mem("before anchor build")
            shadow_reft_for_mem = _build_shadow_reft(
                base_model, task_layers, args.rank,
                intervention_cls, iv_dtype, args,
                anchor_iv_state=anchor_iv_state,
            )
            if use_agem_flag:
                print(f"  [A-GEM] Building A-GEM projector (anchor = G{assigned_gid}, "
                      f"{len(groupwise_iv_keys)} groupwise iv-keys).")
                agem_projector = AGemShadowProjector(
                    live_reft   = reft_model,
                    shadow_reft = shadow_reft_for_mem,
                    groupwise_iv_keys = groupwise_iv_keys,
                )
                trainer.set_agem_projector(agem_projector)
                trainer.set_kl_controller(None)
            elif use_kl_flag:
                kl_mode = str(getattr(args, "kl_mode", "output"))
                rep_layer_indices: Optional[List[int]] = None
                if kl_mode == "representation":
                    rep_raw = getattr(args, "kl_rep_layers", None)
                    if rep_raw:
                        rep_layer_indices = [int(x) for x in
                                             str(rep_raw).split(",") if x.strip()]
                    else:
                        rep_layer_indices = [int(l) + 1 for l in task_layers
                                             if int(l) in tier3_set]
                        if not rep_layer_indices:
                            rep_layer_indices = [int(task_layers[-1]) + 1]

                print(f"  [KL] Building KL-Razor controller "
                      f"(anchor=G{assigned_gid}, "
                      f"{len(groupwise_iv_keys)} groupwise iv-keys, "
                      f"mode={kl_mode}, "
                      f"β={float(getattr(args, 'kl_beta', 0.1)):.3f}"
                      f"{', adaptive' if getattr(args,'kl_adaptive_beta',False) else ''}"
                      f"{', normalized' if getattr(args,'kl_normalize_kl',False) else ''}"
                      f"{', evict' if getattr(args,'kl_eviction_enabled',False) else ''}"
                      f").")
                if kl_mode == "representation":
                    print(f"  [KL] rep_layer_indices = {rep_layer_indices}")

                kl_controller = KLRazorController(
                    shadow_reft        = shadow_reft_for_mem,
                    groupwise_iv_keys  = groupwise_iv_keys,
                    beta               = float(getattr(args, "kl_beta", 0.1)),
                    early_stop         = bool(getattr(args, "kl_early_stop", True)),
                    rolling_window     = int(getattr(args, "kl_rolling_window", 20)),
                    eviction_enabled   = bool(getattr(args, "kl_eviction_enabled", False)),
                    eviction_severity_threshold = float(
                        getattr(args, "kl_eviction_severity_threshold", 1.3)),
                    eviction_early_steps = int(
                        getattr(args, "kl_eviction_early_steps", 30)),
                    eviction_absolute_threshold = float(
                        getattr(args, "kl_evict_absolute_threshold", 0.0)),
                    adaptive_beta      = bool(getattr(args, "kl_adaptive_beta", False)),
                    normalize_kl       = bool(getattr(args, "kl_normalize_kl",  False)),
                    kl_target_contribution = float(
                        getattr(args, "kl_target_contribution", 0.5)),
                    beta_floor         = float(getattr(args, "kl_beta_floor",   0.01)),
                    beta_ceiling       = float(getattr(args, "kl_beta_ceiling", 1.0)),
                    kl_mode            = kl_mode,
                    rep_layer_indices  = rep_layer_indices,
                    beta_warmup_steps  = int(getattr(args, "kl_beta_warmup_steps", 0)),
                    beta_ramp_shape    = str(getattr(args, "kl_beta_ramp_shape", "linear")),
                )
                trainer.set_kl_controller(kl_controller)
                trainer.set_agem_projector(None)
                trainer.add_callback(EarlyStopOnKLCallback(kl_controller))
            else:
                trainer.set_kl_controller(None)
                trainer.set_agem_projector(None)
        else:
            trainer.set_kl_controller(None)
            trainer.set_agem_projector(None)

        _before_flat = capture_before_flat(reft_model)
        _grad_cb = GradientTrajectoryCallback.make(
            reft_model, task_name, task_idx, log_every=20)
        trainer.add_callback(_grad_cb)
        trainer.train()

        evicted_this_task = False
        if (kl_controller is not None
                and kl_controller.stop_action == "evict"
                and not is_new_group):
            evicted_this_task = True
            previous_gid = assigned_gid
            print(f"\n  [evict] Task '{task_name}' triggered eviction from "
                  f"G{previous_gid}; rerunning Phase 2 as new-group.")

            try:
                kl_controller.teardown()
            except Exception:
                pass
            kl_controller = None
            if agem_projector is not None:
                try:
                    agem_projector.teardown()
                except Exception:
                    pass
                agem_projector = None
            if shadow_reft_for_mem is not None:
                try:
                    _teardown_reft_model(shadow_reft_for_mem)
                except Exception:
                    pass
                shadow_reft_for_mem = None
            try:
                del trainer, reft_model, optimizer, data_collator
            except Exception:
                pass
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            new_gid = memory.evict_task_and_allocate_new_group(
                task_name=task_name, previous_gid=previous_gid)
            assigned_gid = new_gid
            is_new_group = True

            _set_intervention_seed(args.fixed_intervention_seed)
            reft_model = _build_reft_model(
                base_model, task_layers, args.rank,
                intervention_cls, iv_dtype, args)
            print(f"  [evict] Fresh fixed-seed init; Phase 2 restart "
                  f"without KL anchor.")
            reft_model.print_trainable_parameters()

            optimizer = build_stratified_optimizer(
                reft_model, task_layers, tier1_set, tier2_set, tier3_set,
                base_lr=args.lr, weight_decay=args.weight_decay)

            data_collator = ReftDataCollator(data_collator=DataCollatorForSeq2Seq(
                tokenizer=tokenizer, model=reft_model.model,
                label_pad_token_id=-100, padding="longest"))
            now_run2 = datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")
            tr_args2 = TrainingArguments(
                output_dir=os.path.join(
                    args.output_dir,
                    f"cl_{task_idx:02d}.{task_name}.evicted.{now_run2}"),
                run_name=f"cl_{task_idx:02d}.{task_name}.evicted",
                num_train_epochs=n_epochs,
                per_device_train_batch_size=args.batch_size,
                per_device_eval_batch_size=args.eval_batch_size,
                gradient_accumulation_steps=args.gradient_accumulation_steps,
                eval_strategy="no", save_strategy="no",
                logging_strategy="steps", logging_steps=args.logging_steps,
                lr_scheduler_type=args.schedule,
                learning_rate=args.lr, warmup_ratio=args.warmup_ratio,
                optim="adamw_torch", weight_decay=args.weight_decay,
                report_to="none", use_cpu=(device != "cuda"),
                seed=args.seed, remove_unused_columns=False,
            )
            trainer = MoIReftTrainer(
                model=reft_model, tokenizer=tokenizer,
                args=tr_args2, train_dataset=train_dataset,
                eval_dataset=None, data_collator=data_collator)
            trainer.set_custom_optimizer(optimizer)
            trainer.set_kl_controller(None)
            trainer.set_agem_projector(None)

            _before_flat = capture_before_flat(reft_model)
            _grad_cb2 = GradientTrajectoryCallback.make(
                reft_model, task_name, task_idx, log_every=20)
            trainer.add_callback(_grad_cb2)
            trainer.train()


        if agem_projector is not None:
            stats = agem_projector.epoch_stats()
            print(f"  [A-GEM] Phase 2 complete: steps={stats.get('n_steps',0)}  "
                  f"mean_cos={stats.get('mean_cos', float('nan')):.3f}  "
                  f"projection_rate={stats.get('proj_rate', 0.0):.3f}  "
                  f"({stats.get('n_projected',0)}/{stats.get('n_tensors',0)} "
                  f"W-tensor-steps projected due to cos<0)")
            agem_projector.teardown()
            agem_projector = None

        kl_final: Optional[float] = None
        kl_running_mean_final: Optional[float] = None
        kl_history_snapshot: Optional[List[float]] = None
        if kl_controller is not None:
            stats = kl_controller.summary()
            print(f"  [KL] Phase 2 complete: steps={stats['n_steps']}  "
                  f"mean_kl={stats['mean_kl']:.4f}  "
                  f"tail_kl={stats['final_kl']:.4f}  "
                  f"stop_step={stats['stop_step']}  "
                  f"threshold={stats['threshold']:.4f}")

            try:
                _kl_src = str(getattr(args, "merge_kl_source", "eval"))
                eval_ds_for_kl = (train_dataset if _kl_src == "train"
                                  else cl_eval_sets[task_name]["dataset"])
                kl_eval_loader = torch.utils.data.DataLoader(
                    eval_ds_for_kl,
                    batch_size=args.eval_batch_size,
                    collate_fn=data_collator,
                    shuffle=False,
                )
                kl_final = kl_controller.evaluate_save_time_kl(
                    reft_model, kl_eval_loader,
                    max_batches=int(getattr(args, "kl_eval_batches", 50)),
                    device=device,
                )
                print(f"  [KL] Save-time KL_final on {_kl_src} split = {kl_final:.4f}  "
                      f"(merge weight input when --merge_rule kl)")
            except Exception as e:
                # KL_OOM_FATAL=1: stop the job. Otherwise kl_final=None and
                # merge_rule kl silently falls back to overwrite for this task.
                if os.environ.get("KL_OOM_FATAL", "") in ("1", "true", "True"):
                    raise
                print(f"  [KL] Save-time KL evaluation failed "
                      f"({type(e).__name__}: {e}).")
                kl_final = None

            kl_running_mean_final = (float(kl_controller._kl_running_mean)
                                     if kl_controller._kl_running_mean is not None
                                     else None)
            kl_history_snapshot = list(kl_controller.kl_history)

            kl_controller.teardown()
            kl_controller = None

        if shadow_reft_for_mem is not None:
            try:
                _teardown_reft_model(shadow_reft_for_mem)
            except Exception:
                pass
            shadow_reft_for_mem = None
        if torch.cuda.is_available(): torch.cuda.empty_cache()

        kl_per_task_log.append({
            "task_idx": task_idx,
            "task_name": task_name,
            "assigned_gid": assigned_gid,
            "is_new_group": is_new_group,
            "kl_running_mean_final": kl_running_mean_final,
            "kl_history": kl_history_snapshot,
            "kl_final_val_diagnostic": kl_final,
        })
        task_to_group_log[task_name] = assigned_gid

        compute_param_delta(_before_flat, reft_model, task_name, task_idx, ref_type)
        del _before_flat
        compute_fisher(reft_model, task_name, task_idx, ref_type)

        reft_model.eval()
        for iv_list in reft_model.interventions.values():
            (iv_list[0] if isinstance(iv_list, (list, tuple)) else iv_list).eval()

        # merge_rule=fisher needs this task's own parameter importance. Computed
        # for new groups too, so the group carries an F_old into later merges.
        fisher_new = None
        if merge_rule == "fisher" and compute_iv_fisher is not None:
            try:
                fisher_new = compute_iv_fisher(
                    reft_model, train_dataset, data_collator,
                    n_batches=merge_fisher_batches,
                    batch_size=args.batch_size,
                    device=device, seed=args.seed)
            except Exception as e:
                print(f"  [fisher] computation failed "
                      f"({type(e).__name__}: {e}); merge falls back to overwrite.")
                fisher_new = None
            reft_model.eval()

        final_gid, confirm_new = memory.save(
            reft_model, task_layers, task_name, task_idx,
            assigned_gid=assigned_gid, is_new_group=is_new_group,
            kl_final=kl_final, fisher_new=fisher_new,
            task_probe=task_probe)

        # LITAR (utils/litar_router.py): store this task's bank prompts, the same
        # seeded draw as the head. Banks and classifiers are built only when
        # --inference_router / --extra_eval_routers use litar.
        try:
            if getattr(memory, "litar", None) is None:
                memory.litar = LitarRouter(layer_ids=_unique_layers(task_layers), n_pos=l_pos,
                                           n_samples=args.router_head_samples,
                                           l2=args.router_head_l2, iters=args.router_head_iters)
            memory.litar.add_task(task_name, train_dataset, seed=args.seed + task_idx)
        except Exception as e:
            print(f"  [litar] WARNING: could not store bank prompts for {task_name} "
                  f"({type(e).__name__}: {e}); LITAR cannot predict this task.")

        if router_head is not None:
            _dev = next(base_model.parameters()).device
            _feats = features_from_dataset(
                base_model, train_dataset, tokenizer, router_head.layer_ids,
                router_head.n_pos, args.router_head_samples, _dev,
                prompt_only=router_head.prompt_only,
                seed=args.seed + task_idx)
            router_head.add_task(task_name, _feats)
            _rec = router_head.fit(device=_dev)
            memory.router_head = router_head
            print(f"  [router_head] refit after {task_name}: tasks={_rec['n_tasks']} "
                  f"examples={_rec['n_examples']} train_acc={_rec['train_acc']} "
                  f"holdout_acc={_rec['holdout_acc']}")

        routing_entry = {
            "task":         task_name,
            "gid":          int(final_gid),
            "is_new_group": bool(confirm_new),
            "evicted":      bool(evicted_this_task),
            "ref_type":     ref_name,
            "position":     task_pos_t,
            "max_length":   task_maxlen,
            "routing_debug": routing_debug,
        }
        routing_log.append(routing_entry)

        now      = datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")
        ckpt_dir = os.path.join(
            args.output_dir, f"ckpt_{task_idx:02d}.{task_name}.{now}")
        os.makedirs(ckpt_dir, exist_ok=True)
        reft_model.save(ckpt_dir)
        memory.save_to_disk(ckpt_dir)
        if router_head is not None:
            router_head.save(os.path.join(ckpt_dir, "router_head.pt"))
        with open(os.path.join(ckpt_dir, "routing_log.json"), "w") as f:
            json.dump(routing_log, f, indent=2)
        with open(os.path.join(ckpt_dir, "memory_summary.json"), "w") as f:
            json.dump(memory.get_summary(), f, indent=2)
        with open(os.path.join(ckpt_dir, "merge_log.json"), "w") as f:
            json.dump(memory.merge_log, f, indent=2)
        print(f"  Checkpoint saved → {ckpt_dir}")

        if kl_controller is not None:
            try:
                kl_controller.teardown()
            except Exception:
                pass
            kl_controller = None
        if agem_projector is not None:
            try:
                agem_projector.teardown()
            except Exception:
                pass
            agem_projector = None
        if shadow_reft_for_mem is not None:
            try:
                _teardown_reft_model(shadow_reft_for_mem)
            except Exception:
                pass
            shadow_reft_for_mem = None
        try:
            _teardown_reft_model(reft_model)
        except Exception:
            pass
        del reft_model, trainer, optimizer, data_collator
        _release_gpu()

        if args.skip_eval:
            print(f"\n  [train-only] evaluation of '{task_name}' deferred (--skip_eval).")
        else:
            _row_eval(task_idx)

        _save_resume_state(args.output_dir, task_idx, CL_ORDER, perf_matrix,
                           kl_per_task_log, task_to_group_log, routing_log,
                           memory, router_head)
        if args.skip_eval or args.save_eval_snapshots:
            _save_eval_snapshot(args.output_dir, task_idx, CL_ORDER, kl_per_task_log,
                                task_to_group_log, routing_log, memory, router_head)

    if args.skip_eval:
        snaps = [_eval_snapshot_path(args.output_dir, i, CL_ORDER[i]) for i in range(n_tasks)]
        have = [p_ for p_ in snaps if os.path.isfile(p_)]
        print(f"\n{'='*72}")
        print(f"  TRAIN-ONLY RUN COMPLETE  (evaluation deferred)")
        print(f"{'='*72}")
        print(f"  Snapshots : {len(have)}/{n_tasks} in {os.path.join(args.output_dir, EVAL_SNAPSHOT_DIR)}")
        print(f"  Evaluate  : bash scripts/eval_later.sh {args.output_dir}")
        print(f"  Groups Created                 {memory.n_groups()}")
        print(f"\n  Routing log:")
        for e in routing_log:
            tag = "NEW" if e["is_new_group"] else "JOIN"
            print(f"    {e['task']:<15}  G{e['gid']}  {tag:<4}  [ref={e['ref_type']}]")
        print(f"\n  Group contents:")
        for gid, g in memory.groups.items():
            print(f"    G{gid}: {g['task_names']}  (n_merged={g['n_merged']})")
        summary = {
            "run_mode": "train_only",
            "cl_order": list(CL_ORDER),
            "complete": len(have) == n_tasks,
            "snapshots": snaps,
            "routing_log": routing_log,
            "task_to_group": task_to_group_log,
            "kl_per_task": kl_per_task_log,
            "memory_summary": memory.get_summary(),
            "merge_log": getattr(memory, "merge_log", None),
            "n_groups": memory.n_groups(),
        }
        with open(os.path.join(args.output_dir, TRAIN_SUMMARY_FILE), "w") as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"\n  Train summary saved → {os.path.join(args.output_dir, TRAIN_SUMMARY_FILE)}")
        flush_log(args.output_dir)
        return

    if not args.per_task_bwd_eval:
        print(f"\n{'='*72}")
        print(f"  Final backward evaluation across all {n_tasks} tasks")
        print(f"{'='*72}")
        print(f"  {'Task':<15}  {'Score':>7}")
        print(f"  {'-'*15}  {'-'*7}")

        _release_gpu()
        with torch.no_grad():
            for eval_idx in range(n_tasks):
                if perf_matrix[n_tasks - 1][eval_idx] is not None:
                    score_existing = perf_matrix[n_tasks - 1][eval_idx]
                    print(f"  {CL_ORDER[eval_idx]:<15}  {score_existing:>7.4f}")
                    continue

                eval_task = CL_ORDER[eval_idx]
                ev_info   = cl_eval_sets[eval_task]

                eval_reft = _build_reft_model(
                    base_model, task_layers, args.rank,
                    intervention_cls, iv_dtype, args)

                loaded = route_and_load(
                    eval_reft, eval_task, memory, base_model, tokenizer,
                    args, ev_info["dataset"],
                    build_reft_fn=lambda: _build_reft_model(
                        base_model, task_layers, args.rank,
                        intervention_cls, iv_dtype, args),
                    teardown_fn=_teardown_reft_model)
                if not loaded:
                    perf_matrix[n_tasks - 1][eval_idx] = 0.0
                    _teardown_reft_model(eval_reft)
                    del eval_reft
                    _release_gpu()
                    print(f"  {eval_task:<15}  {0.0:>7.4f}  (no state)")
                    continue

                eval_reft.eval()
                try:
                    score = evaluate_reft_on_task(
                        eval_task, eval_reft, tokenizer,
                        ev_info["dataset"], ev_info["data_items"],
                        args.eval_batch_size, args.greedy_decoding,
                        args.temperature, args.top_p, args.top_k)
                finally:
                    _teardown_reft_model(eval_reft)
                    del eval_reft
                    _release_gpu()
                perf_matrix[n_tasks - 1][eval_idx] = score
                print(f"  {eval_task:<15}  {score:>7.4f}")
                
    T        = n_tasks - 1
    diag     = [perf_matrix[j][j] for j in range(n_tasks)]
    last_row = [perf_matrix[T][j] for j in range(n_tasks)]

    def _cl_metrics(final_row):
        """OP  = mean of the final row (all tasks).
        BWT = mean over the first T tasks of (final - diagonal); the last task
              is excluded (its diff is 0 by definition). Negative = forgetting."""
        _op = round(float(np.mean([x for x in final_row if x is not None])), 4)
        _d  = [final_row[j] - diag[j] for j in range(T)
               if diag[j] is not None and final_row[j] is not None]
        _bwt = round(float(np.mean(_d)), 4) if _d else 0.0
        return _op, _bwt

    forgetting = {CL_ORDER[j]: round(diag[j]-last_row[j], 4)
                  for j in range(n_tasks-1)
                  if diag[j] is not None and last_row[j] is not None}
    mf   = round(float(np.mean(list(forgetting.values()))), 4) if forgetting else 0.0
    la   = round(float(np.mean([d for d in diag if d is not None])), 4)
    op, bwt = _cl_metrics(last_row)

    # Extra final rows: replay only the final evaluation row once per extra
    # router, on the same end-of-stream memory. The primary --inference_router
    # still drives the diagonal and the main OP/BWT above. BWT here is
    # measured against that primary diagonal.
    final_row_by_router = {}
    extra_routers = [r for r in (args.extra_eval_routers or [])
                     if r != args.inference_router]
    _primary_router = args.inference_router
    for _rtr in extra_routers:
        print(f"\n{'='*72}")
        print(f"  Extra final row  [router={_rtr}]")
        print(f"{'='*72}")
        args.inference_router = _rtr
        _row = [None] * n_tasks
        _release_gpu()
        try:
            with torch.no_grad():
                for eval_idx in range(n_tasks):
                    eval_task = CL_ORDER[eval_idx]
                    ev_info   = cl_eval_sets[eval_task]
                    eval_reft = _build_reft_model(
                        base_model, task_layers, args.rank,
                        intervention_cls, iv_dtype, args)
                    loaded = route_and_load(
                        eval_reft, eval_task, memory, base_model, tokenizer,
                        args, ev_info["dataset"],
                        build_reft_fn=lambda: _build_reft_model(
                            base_model, task_layers, args.rank,
                            intervention_cls, iv_dtype, args),
                        teardown_fn=_teardown_reft_model)
                    if not loaded:
                        _teardown_reft_model(eval_reft)
                        del eval_reft
                        _release_gpu()
                        _row[eval_idx] = 0.0
                        print(f"  {eval_task:<15}  {0.0:>7.4f}  (no state)")
                        continue
                    eval_reft.eval()
                    try:
                        score = evaluate_reft_on_task(
                            eval_task, eval_reft, tokenizer,
                            ev_info["dataset"], ev_info["data_items"],
                            args.eval_batch_size, args.greedy_decoding,
                            args.temperature, args.top_p, args.top_k)
                    finally:
                        _teardown_reft_model(eval_reft)
                        del eval_reft
                        _release_gpu()
                    _row[eval_idx] = score
                    print(f"  {eval_task:<15}  {score:>7.4f}")
        except Exception as e:
            print(f"  [extra_router] {_rtr} failed ({type(e).__name__}: {e}); "
                  f"skipping it.")
            args.inference_router = _primary_router
            continue
        args.inference_router = _primary_router
        _op, _bwt = _cl_metrics(_row)
        print(f"  [{_rtr}]  OP {_op:.4f}   BWT {_bwt:+.4f}   "
              f"(BWT vs the {_primary_router} diagonal)")
        final_row_by_router[_rtr] = {
            "final_row":           {CL_ORDER[j]: _row[j] for j in range(n_tasks)},
            "overall_performance": _op,
            "backward_transfer":   _bwt,
        }
    args.inference_router = _primary_router
    if final_row_by_router:
        final_row_by_router = {
            _primary_router: {
                "final_row":           {CL_ORDER[j]: last_row[j] for j in range(n_tasks)},
                "overall_performance": op,
                "backward_transfer":   bwt,
            },
            **final_row_by_router,
        }

    col_w = 12
    header = f"{'Trained→':<18}" + "".join(f"{t[:10]:>{col_w}}" for t in CL_ORDER)
    print(f"\n{'='*72}\n  PERFORMANCE MATRIX\n{'='*72}")
    print(header); print("-" * len(header))
    for i in range(n_tasks):
        row = "".join(f"{perf_matrix[i][j]:>{col_w}.4f}"
                      if perf_matrix[i][j] is not None else f"{'---':>{col_w}}"
                      for j in range(n_tasks))
        print(f"{CL_ORDER[i]:<18}{row}")

    print(f"\n  {'Learning Accuracy (LA)':<30} {la:.4f}")
    print(f"  {'Mean Forgetting':<30} {mf:+.4f}")
    print(f"  {'Overall Performance (OP)':<30} {op:.4f}")
    print(f"  {'Backward Transfer (BWT)':<30} {bwt:+.4f}")
    print(f"  {'Groups Created':<30} {memory.n_groups()}")
    print(f"{'='*72}")
    if final_row_by_router:
        print(f"\n  Final row by router (BWT vs {_primary_router} diagonal):")
        print(f"  {'Router':<24}  {'OP':>7}  {'BWT':>8}")
        for _r, _v in final_row_by_router.items():
            print(f"  {_r:<24}  {_v['overall_performance']:>7.4f}  "
                  f"{_v['backward_transfer']:>+8.4f}")
    print(f"\n  Routing log:")
    for e in routing_log:
        tag = "NEW" if e["is_new_group"] else "JOIN"
        print(f"    {e['task']:<15}  G{e['gid']}  {tag:<4}  [ref={e['ref_type']}]")
    print(f"\n  Group contents:")
    for gid, g in memory.groups.items():
        print(f"    G{gid}: {g['task_names']}  (n_merged={g['n_merged']})")

    cl_results = {
        "method": "CRAFT-CL [Weight-Distance Routing]",
        "position": task_pos,
        "routing": {
            "signal":            "cosine(flatten(W)) on 26 groupwise interventions",
            "aggregate":         aggregate,
            "normalize":         normalize,
            "threshold":         assign_thr,
            "warmup_steps":      warmup_steps,
            "skip_steps":        skip_steps,
            "fixed_seed":        fixed_seed,
            "eval_lookup":       "task_to_group[task_name] → group's W + task's T3_l",
        },
        "routing_log":         routing_log,
        "memory_summary":      memory.get_summary(),
        "perf_matrix":         perf_matrix,
        "kl_per_task":         kl_per_task_log,
        "task_to_group":       task_to_group_log,
        "diagonal":            {CL_ORDER[j]: diag[j]     for j in range(n_tasks)},
        "final_row":           {CL_ORDER[j]: last_row[j] for j in range(n_tasks)},
        "forgetting_per_task": forgetting,
        "mean_forgetting":     mf,
        "learning_accuracy":   la,
        "overall_performance": op,
        "backward_transfer":   bwt,
        "n_groups":            memory.n_groups(),
        "inference_router":    _primary_router,
        "final_row_by_router": final_row_by_router,
        "run_mode":            ("eval_only" if args.eval_only else "train_and_eval"),
        "eval_from":           args.eval_from if args.eval_only else None,
        "eval_file":           getattr(args, "cl_eval_file", None),
    }
    
    results_path = os.path.join(args.output_dir, "cl_results.json")

    with open(results_path, "w") as f:
        json.dump(cl_results, f, indent=4)
    print(f"\n Log Results saved → {results_path}")

    try:
        from utils.analyze_cl_results import run_analysis
        run_analysis(args.output_dir)
    except Exception as e:
        print(f"  [analysis] post-hoc analysis failed "
                f"({type(e).__name__}: {e}); cl_results.json is still saved.")

    print(f"\n  [post-analysis] Building pairwise trained-weight distance matrix ...")
    analysis = _post_cl_analysis(
        memory=memory,
        cl_order=CL_ORDER,
        task_layers=task_layers,
        routing_log=routing_log,
        tier_sets=(tier1_set, tier2_set, tier3_set),
        output_dir=args.output_dir,
    )
    print(f"  [post-analysis] Trained-weight ARI vs 4-type labels: {analysis.get('ari', 'n/a')}")
    print(f"  [post-analysis] Routing ARI (assigned groups vs types):  {analysis.get('routing_ari', 'n/a')}")

    flush_log(args.output_dir)


def _post_cl_analysis(
    memory: "WeightDistanceMemory",
    cl_order: List[str],
    task_layers: List[int],
    routing_log: List[dict],
    tier_sets: Tuple[Set[int], Set[int], Set[int]],
    output_dir: str,
) -> Dict:

    tier1_set, tier2_set, tier3_set = tier_sets

    task_vectors: Dict[str, np.ndarray] = {}
    for tname in cl_order:
        if tname not in memory.task_warmup_snapshots:
            continue
        state = memory.task_warmup_snapshots[tname]
        parts = []
        for iv_key in sorted(state.keys()):
            W = _get_w_from_state(state[iv_key])
            if W is None: continue
            parts.append(W.detach().float().flatten().cpu().numpy())
        if parts:
            task_vectors[tname] = np.concatenate(parts)

    present = [t for t in cl_order if t in task_vectors]
    n = len(present)
    if n == 0:
        return {"error": "no task vectors"}

    dist_matrix = np.zeros((n, n), dtype=np.float64)
    for i, ti in enumerate(present):
        vi = task_vectors[ti]
        ni = np.linalg.norm(vi)
        for j, tj in enumerate(present):
            if i == j: continue
            vj = task_vectors[tj]
            nj = np.linalg.norm(vj)
            if ni < 1e-12 or nj < 1e-12:
                dist_matrix[i][j] = 1.0
            else:
                cos = float(np.dot(vi, vj) / (ni * nj))
                dist_matrix[i][j] = max(0.0, 1.0 - cos)

    ari = None
    routing_ari = None
    try:
        from sklearn.cluster import AgglomerativeClustering
        from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

        labels = [TRACE_TASK_TYPES.get(t, 0) for t in present]
        if any(lb > 0 for lb in labels):
            k = min(4, len(set(labels)))
            if k >= 2 and n >= k:
                d_sym = 0.5 * (dist_matrix + dist_matrix.T)
                clust = AgglomerativeClustering(
                    n_clusters=k, metric="precomputed", linkage="average")
                pred = clust.fit_predict(d_sym)
                ari = float(adjusted_rand_score(labels, pred))
                nmi = float(normalized_mutual_info_score(labels, pred))
            else:
                nmi = None

            route_pred = [memory.task_to_group.get(t, -1) for t in present]
            if all(p >= 0 for p in route_pred):
                routing_ari = float(adjusted_rand_score(labels, route_pred))
        else:
            nmi = None
    except ImportError:
        nmi = None

    out = {
        "n_tasks":       n,
        "tasks":         present,
        "distance_matrix": dist_matrix.tolist(),
        "ari":           ari,
        "nmi":           nmi,
        "routing_ari":   routing_ari,
        "task_to_group": memory.task_to_group,
        "task_types":    {t: TASK_TYPE_NAMES.get(TRACE_TASK_TYPES.get(t, 0), "?")
                          for t in present},
    }
    path = os.path.join(output_dir, "post_cl_analysis.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  [post-analysis] Saved pairwise matrix → {path}")
    return out


def main():
    parser = argparse.ArgumentParser(
        description="CRAFT-CL: Weight-Distance Routing")

    parser.add_argument("--task",          type=str, default=None)
    parser.add_argument("--data_dir",      type=str, default="./datasets")
    parser.add_argument("--train_dataset", type=str, default=None)
    parser.add_argument("--eval_dataset",  type=str, default=None)
    parser.add_argument("--model",         type=str, default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--seed",          type=int, default=42)
    parser.add_argument("--dtype",         type=str,
                        default="bfloat16" if device == "cuda" else "float32")
    parser.add_argument("--max_length",    type=int, default=1024)
    parser.add_argument("--layers",            type=str, default="all")
    parser.add_argument("--rank",              type=int, default=8)
    parser.add_argument("--n_positions",       type=int, default=9,
                        help="N in fN+lN position string. Same N for f and l, "
                             "same for all tasks. Default 9 → f9+l9.")
    parser.add_argument("--f_positions",       type=int, default=None,
                        help="Prefix (f) token count. Set together with "
                             "--l_positions for an asymmetric position string "
                             "(e.g. 5 + 25 -> f5+l25). Unset on both = "
                             "symmetric fN+lN from --n_positions.")
    parser.add_argument("--l_positions",       type=int, default=None,
                        help="Suffix (l) token count. See --f_positions.")
    parser.add_argument("--intervention_type", type=str, default="LoreftIntervention")
    parser.add_argument("--share_weights",     action="store_true")
    parser.add_argument("--act_fn",            type=str,   default=None)
    parser.add_argument("--add_bias",          action="store_true")
    parser.add_argument("--dropout",           type=float, default=0.00)

    parser.add_argument("--epochs",                      type=int,   default=1)
    parser.add_argument("--lr",                          type=float, default=5e-4)
    parser.add_argument("--schedule",                    type=str,   default="linear")
    parser.add_argument("--warmup_ratio",                type=float, default=0.10)
    parser.add_argument("--weight_decay",                type=float, default=0.02)
    parser.add_argument("--batch_size",                  type=int,   default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int,   default=1)
    parser.add_argument("--eval_batch_size",             type=int,   default=4)
    parser.add_argument("--max_n_train_example", type=int, default=None)
    parser.add_argument("--max_n_eval_example",  type=int, default=None)
    parser.add_argument("--output_dir",    type=str, default="./results")
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_model",    action="store_true")
    parser.add_argument("--is_wandb",      action="store_true")
    parser.add_argument("--wandb_name",    type=str, default="reft")
    parser.add_argument("--wandb_dir",     type=str, default="wandb")
    parser.add_argument("--wandb_proj",    type=str, default="MyReFT")

    parser.add_argument("--test_split",            type=str,  default="validation")
    parser.add_argument("--greedy_decoding",       action="store_true")
    parser.add_argument("--temperature",           type=float, default=None)
    parser.add_argument("--top_p",                 type=float, default=None)
    parser.add_argument("--top_k",                 type=float, default=None)
    parser.add_argument("--metric_for_best_model", type=str, default="accuracy")
    parser.add_argument("--train_on_inputs",         action="store_true")
    parser.add_argument("--use_normalized_template", action="store_true")
    parser.add_argument("--allow_cls_grad",          action="store_true")
    parser.add_argument("--disable_reft",            action="store_true")
    parser.add_argument("--use_lora",                action="store_true")
    parser.add_argument("--lora_rank",    type=int, default=8)
    parser.add_argument("--lora_alpha",   type=int, default=32)
    parser.add_argument("--lora_modules", type=str, default="o_proj")
    parser.add_argument("--lora_layers",  type=str, default="all")

    parser.add_argument("--continual_learning", action="store_true")
    parser.add_argument("--benchmark",           type=str, default="trace",
                        choices=["trace", "standard_cl"])
    parser.add_argument("--cl_benchmark_dir",    type=str, default=None)
    parser.add_argument("--cl_tasks",            type=str, default=None)
    parser.add_argument("--moi",                 action="store_true", default=True)
    parser.add_argument("--no_moi",              dest="moi", action="store_false")
    parser.add_argument("--per_task_epochs",     type=str, default=None)
    parser.add_argument("--per_task_max_length", type=str, default=None,
                        help="Per-task tokenizer limit, e.g. "
                             "'MeetingBank:2048,Py150:2048'. Tasks not "
                             "listed keep --max_length. Applied to that "
                             "task's train and eval datasets only.")
    parser.add_argument("--per_task_positions",  type=str, default=None,
                        help="Per-task position string, e.g. "
                             "'MeetingBank:f15+l40,Py150:f15+l40'. Tasks "
                             "not listed keep the global fN+lM. Changes only "
                             "which tokens are intervened, not the number "
                             "of parameters.")

    parser.add_argument("--causal_warmup_steps",    type=int, default=100,
                        help="Warmup steps before routing.")
    parser.add_argument("--warmup_skip_steps",      type=int, default=15,
                        help="Skip first N warmup steps.")

    parser.add_argument("--shallow_frac", type=float, default=0.33,
                        help="Fraction of layers in T1 (shallow).")
    parser.add_argument("--deep_frac",    type=float, default=0.38,
                        help="Fraction of layers in T3 (deep).")

    parser.add_argument("--no_all_groupwise", dest="all_groupwise",
                        action="store_false",
                        help="Keep the T3 l-stream task-private (the default "
                             "split). Use to override an earlier "
                             "--all_groupwise on the same command line.")
    parser.add_argument("--all_groupwise", action="store_true", default=False,
                        help="Store T3_l at group level instead of per-task.")

    parser.add_argument("--manual_routing", action="store_true", default=False,
                        help="Oracle routing: assign tasks to human-labelled families.")
    parser.add_argument("--manual_groups", type=str, default=None,
                        help="Override task→group map as 'task:group,...' pairs.")
    parser.add_argument("--router_method", type=str, default="kl_routing",
                        choices=["manual", "kl_routing"],
                        help="Routing strategy: kl_routing or manual.")
    parser.add_argument("--weight_distance_threshold", type=float, default=0.10,
                        help="Threshold for joining a group vs creating a new one.")
    parser.add_argument("--weight_distance_tiers", type=str, default="mean_all",
                        choices=["mean_all", "tiered"],
                        help="Legacy tier aggregation (ignored when flat_routing=True).")
    parser.add_argument("--weight_distance_normalize", type=str, default="none",
                        choices=["none", "minmax", "zscore"],
                        help="Normalize distances before threshold comparison.")
    parser.add_argument("--fixed_intervention_seed", type=int, default=42,
                        help="Seed for consistent LoReFT init across tasks.")
    parser.add_argument("--flat_routing", action="store_true", default=True,
                        help="Route using all groupwise ivs as a flat list.")
    parser.add_argument("--no_flat_routing", dest="flat_routing", action="store_false",
                        help="Revert to legacy 4-tier averaged routing.")
    parser.add_argument("--flat_aggregator", type=str, default="vote",
                        choices=["vote", "mean", "robust_mean"],
                        help="Aggregator for per-iv distances: vote, mean, or robust_mean.")
    parser.add_argument("--flat_vote_quantile", type=float, default=0.5,
                        help="Reserved; currently unused.")
    parser.add_argument("--vote_max_dist", type=float, default=0.20,
                        help="Max median per-iv distance to vote winner before new group.")

    parser.add_argument("--w_cos_per_iv_weights", type=str, default="0.4,0.3,0.2,0.1",
                        help="Weights for w_cosine_per_iv: mean,median,max,iqr.")
    parser.add_argument("--cicm_normalise_by_n", action="store_true", default=True,
                        help="Normalise CICM Frobenius distance by 2N.")
    parser.add_argument("--no_cicm_normalise_by_n", dest="cicm_normalise_by_n",
                        action="store_false",
                        help="Disable CICM Frobenius normalisation.")
    parser.add_argument("--grassmann_k", type=int, default=3,
                        help="Subspace dimensionality for Grassmann/update_pca routers.")
    parser.add_argument("--update_pca_k", type=int, default=3,
                        help="Subspace dimensionality for update_pca router.")
    parser.add_argument("--update_pca_spectrum_w", type=float, default=0.5,
                        help="Mix weight between spectrum-cosine and Grassmann in update_pca.")
    parser.add_argument("--w_r_b_weights", type=str, default="0.7,0.2,0.1",
                        help="Weights for w_r_b router: W, R, b.")

    parser.add_argument("--probe_steps", type=int, default=20,
                        help="Probe training steps per group for multihead_probe.")
    parser.add_argument("--probe_drift_metric", type=str, default="wrb",
                        choices=["w", "r", "b", "wrb"],
                        help="Parameter component(s) for probe drift: w, r, b, or wrb.")
    parser.add_argument("--probe_drift_weights", type=str, default="0.7,0.2,0.1",
                        help="Weights for wrb drift metric: W, R, b.")
    parser.add_argument("--probe_drift_threshold", type=float, default=0.20,
                        help="Join smallest-drift group if below this; else new group.")
    parser.add_argument("--probe_drift_normalize", type=str, default="none",
                        choices=["none", "minmax", "zscore"],
                        help="Normalize probe drifts before threshold comparison.")

    parser.add_argument("--kl_routing_probe_size", type=int, default=16,
                        help="Probe examples from current task for KL routing.")
    parser.add_argument("--kl_routing_threshold", type=float, default=0.5,
                        help="KL distance threshold to join a group.")
    parser.add_argument("--kl_routing_normalize", type=str, default="raw",
                        choices=["raw", "ratio"],
                        help="Normalize D_KG before threshold: raw or ratio.")
    parser.add_argument("--kl_routing_min_dkg", type=float, default=0.01,
                        help="Signal-floor: force new group when closest D_KG < this.")
    parser.add_argument("--kl_routing_top_k", type=int, default=64,
                        help="Top-k union for KL computation.")
    parser.add_argument("--kl_routing_bwd_eps", type=float, default=0.05,
                        help="Denominator floor for the backward routing "
                             "distance. Separate from --kl_routing_min_dkg, "
                             "which is the no-signal floor on raw D_KG.")
    parser.add_argument("--kl_routing_direction", type=str, default="forward",
                        choices=["forward", "backward", "both"],
                        help="Whose data the routing distance is measured on. "
                             "forward = the arriving task's (legacy). backward "
                             "= the group members' own probes, i.e. how much "
                             "the arrival would perturb what the group already "
                             "knows. On the 56-pair interference matrix the "
                             "same measurements read backward rank damage at "
                             "rho=0.517 vs 0.192 forward. 'both' requires the "
                             "join to pass on each side and so raises K.")
    parser.add_argument("--kl_routing_temperature", type=float, default=2.0,
                        help="Temperature for KL softmax smoothing.")
    parser.add_argument("--group_capacity", type=int, default=0,
                        help="Max tasks per group. <=0 = uncapped (default). "
                             "Full groups are dropped from the routing "
                             "candidate set, so a task that would have joined "
                             "one opens a fresh group instead.")
    parser.add_argument("--merge_rule", type=str, default="overwrite",
                        choices=list(MERGE_RULES),
                        help="How a joining task's trained state is combined "
                             "with the existing group state. overwrite = new "
                             "state wins (default, unchanged behaviour); ema, "
                             "prototype, kl = scalar blends; fisher = "
                             "per-parameter blend by diagonal empirical Fisher. "
                             "Anything but overwrite needs merge_rules.py.")
    parser.add_argument("--merge_fisher_batches", type=int, default=8,
                        help="Minibatches of the task's own data used for the "
                             "diagonal empirical Fisher (merge_rule=fisher).")
    parser.add_argument("--merge_kl_scale", type=float, default=1.0,
                        help="merge_rule=kl: weight on the new state is "
                             "1/(1 + scale * kl_final). Low drift -> new state, "
                             "high drift -> preserve the group state.")
    parser.add_argument("--merge_kl_halflife", type=float, default=None,
                        help="merge_rule=kl, interpretable form of "
                             "--merge_kl_scale: the kl_final at which the "
                             "group state and the new state get equal weight "
                             "(scale = 1/halflife). Overrides --merge_kl_scale "
                             "when set. Observed kl_final is ~0.03, so the "
                             "default scale=1.0 (halflife=1.0) makes this rule "
                             "identical to overwrite; try 0.01-0.10.")
    parser.add_argument("--kl_pred_mask", action="store_true", default=False,
                        help="Score the KL at the positions that predict answer "
                             "tokens (shifted mask, as in the CE loss). Applies to "
                             "the training KL, the save-time KL and the routing KL. "
                             "Off = original mask, one position late.")
    parser.add_argument("--merge_kl_source", type=str, default="eval",
                        choices=["eval", "train"],
                        help="Data for the save-time KL that sets the merge weight. "
                             "eval = the task's evaluation file (original); "
                             "train = its training split.")
    parser.add_argument("--resume_from", type=str, default=None,
                        help="Continue a crashed run: the previous run's output "
                             "directory (or its resume_state.pt). Tasks it "
                             "finished are not retrained.")
    parser.add_argument("--cl_eval_file", type=str, default=None,
                        choices=["eval.json", "test.json"],
                        help="Evaluation file per task. Default: eval.json if "
                             "present, else test.json (original).")
    parser.add_argument("--merge_ema_alpha", type=float, default=0.5,
                        help="merge_rule=ema: fixed weight on the new state.")

    parser.add_argument("--tier1_alpha",        type=float, default=0.30)
    parser.add_argument("--group_alpha",        type=float, default=0.50,
                        help="EMA alpha for merging trained weights into group.")

    parser.add_argument("--use_kl", action="store_true", default=True,
                        help="Use KL-Razor regulariser during Phase 2.")
    parser.add_argument("--no_kl", dest="use_kl", action="store_false",
                        help="Disable KL-Razor (plain SFT).")
    parser.add_argument("--use_agem", action="store_true", default=False,
                        help="Use A-GEM gradient projection instead of KL-Razor.")
    parser.add_argument("--kl_beta_warmup_steps", type=int, default=0,
                        help="Ramp the KL anchor linearly over the first N "
                             "optimizer steps of each task "
                             "(beta_eff = beta * min(1, step/N)). 0 = off, "
                             "i.e. full anchor from step 0 (original). Buys "
                             "plasticity on the new task without lowering the "
                             "final beta that protects earlier tasks.")
    parser.add_argument("--kl_beta_ramp_shape", type=str, default="linear",
                        choices=["linear", "cosine"],
                        help="Shape of the --kl_beta_warmup_steps ramp. "
                             "cosine is flat at both ends; linear reaches the "
                             "cap with a slope discontinuity. Only meaningful "
                             "when --kl_beta_warmup_steps > 0.")
    parser.add_argument("--kl_beta", type=float, default=0.1,
                        help="KL regularisation coefficient β.")
    parser.add_argument("--kl_early_stop", action="store_true", default=True,
                        help="Stop when rolling KL exceeds first-epoch threshold.")
    parser.add_argument("--no_kl_early_stop", dest="kl_early_stop",
                        action="store_false",
                        help="Disable KL early stop.")
    parser.add_argument("--kl_rolling_window", type=int, default=20,
                        help="Rolling-mean window size for early-stop rule.")
    parser.add_argument("--kl_eval_batches", type=int, default=50,
                        help="Val batches for save-time KL estimate.")

    parser.add_argument("--kl_eviction_enabled", action="store_true",
                        default=False,
                        help="Evict and retrain on early+severe KL threshold cross.")
    parser.add_argument("--kl_eviction_severity_threshold", type=float,
                        default=1.3,
                        help="Severity multiplier for eviction trigger.")
    parser.add_argument("--kl_eviction_early_steps", type=int, default=30,
                        help="Steps past epoch-1 boundary for eviction eligibility.")
    parser.add_argument("--kl_evict_absolute_threshold", type=float,
                        default=0.0,
                        help="Evict if first-epoch mean KL exceeds this value.")

    parser.add_argument("--kl_adaptive_beta", action="store_true",
                        default=False,
                        help="Rescale β at first-epoch boundary to match task-loss scale.")
    parser.add_argument("--kl_normalize_kl", action="store_true",
                        default=False,
                        help="Divide KL by running mean before applying β.")
    parser.add_argument("--kl_target_contribution", type=float, default=0.5,
                        help="Target fraction of task-loss that β·KL contributes.")
    parser.add_argument("--kl_beta_floor", type=float, default=0.01,
                        help="Lower clip on adaptive β.")
    parser.add_argument("--kl_beta_ceiling", type=float, default=1.0,
                        help="Upper clip on adaptive β.")

    parser.add_argument("--kl_mode", type=str, default="output",
                        choices=["output", "representation"],
                        help="KL space: output (vocab softmax) or representation (hidden states).")
    parser.add_argument("--kl_rep_layers", type=str, default=None,
                        help="Layer indices for representation-level KL.")

    parser.add_argument("--per_task_bwd_eval", action="store_true",
                        help="Evaluate all prior tasks after each training step.")
    parser.add_argument("--mistral_system_first_turn", action="store_true",
                        help="Mistral only: use MISTRAL_CHAT_TEMPLATE so the "
                             "prompt is a prefix of prompt+answer. Off by "
                             "default (original behaviour for every model).")
    parser.add_argument("--inference_router", type=str.lower, default="oracle",
                        choices=["oracle", "litar", "learned_head"],
                        help="Group selection at eval time (case-insensitive). "
                             "'oracle' = task_to_group lookup (default); "
                             "'litar' = task identified through the learned group "
                             "interventions (utils/litar_router.py); "
                             "'learned_head' = task head on frozen-base features "
                             "(utils/router_head.py, needs --router_head).")
    parser.add_argument("--extra_eval_routers", type=str.lower, nargs="*", default=[],
                        choices=["oracle", "litar", "learned_head"],
                        help="After the stream, replay the final evaluation "
                             "row once per router listed here (in addition "
                             "to --inference_router). Results go to "
                             "cl_results.json['final_row_by_router'].")
    parser.add_argument("--router_probe_size", type=int, default=16,
                        help="Eval examples used per task-agnostic routing "
                             "decision.")
    parser.add_argument("--router_head", action="store_true", default=False,
                        help="Train a task-identity router head during the "
                             "stream (router_head.py). At every task commit "
                             "the head is refit on stored frozen-base features "
                             "of all tasks so far, and saved as "
                             "router_head.pt in the checkpoint. Off by default.")
    parser.add_argument("--router_head_samples", type=int, default=64,
                        help="Training examples per task banked for the head (and for LITAR).")
    parser.add_argument("--router_head_l2", type=float, default=1e-2,
                        help="L2 strength for the head's logistic regression (and LITAR's).")
    parser.add_argument("--router_head_prompt_only", action="store_true", default=False,
                        help="Build the head's feature bank from prompt tokens only "
                             "(labels == -100). Saved in router_head.pt, so inference "
                             "follows it. Off = original behaviour.")
    parser.add_argument("--router_head_iters", type=int, default=200,
                        help="LBFGS iterations per refit (head and LITAR).")

    parser.add_argument("--skip_eval", action="store_true", default=False,
                        help="Train only: skip the per-task and final evaluation, keep a "
                             "snapshot after every task (eval_snapshots/) and write "
                             "train_summary.json. Evaluate later with --eval_only.")
    parser.add_argument("--save_eval_snapshots", action="store_true", default=False,
                        help="In a normal train+eval run, also keep the per-task snapshots, "
                             "so the run can be evaluated again later (another split, or "
                             "the full lower triangle).")
    parser.add_argument("--eval_only", action="store_true", default=False,
                        help="Evaluation only, from the snapshots of a training run "
                             "(--eval_from). Training settings come from its "
                             "train_args.json; give only evaluation settings here.")
    parser.add_argument("--eval_from", type=str, default=None,
                        help="Training output directory for --eval_only.")
    parser.add_argument("--no_per_task_bwd_eval", dest="per_task_bwd_eval", action="store_false",
                        default=argparse.SUPPRESS,
                        help="--eval_only: evaluate the diagonal and final row only, even if "
                             "the training run asked for the full lower triangle.")

    args = parser.parse_args()

    if args.eval_only and args.skip_eval:
        raise SystemExit("--eval_only and --skip_eval cannot be combined.")
    if args.eval_only:
        args = _restore_train_args(args)

    if str(args.merge_rule).lower() != "overwrite" and not _HAS_MERGE_RULES:
        raise SystemExit(
            f"--merge_rule={args.merge_rule} needs merge_rules.py alongside "
            f"train.py, but the import failed: {_merge_import_err}")
    if (args.f_positions is None) != (args.l_positions is None):
        print(f"  [warn] only one of --f_positions/--l_positions was given; "
              f"the missing one defaults to --n_positions={args.n_positions}.")

    col = max(len(k) for k in vars(args)) + 2
    print("\n" + "=" * 70)
    print("  CRAFT-CL [Weight-Distance Routing]  --  Config")
    print("=" * 70)
    for k, v in sorted(vars(args).items()):
        print(f"  {k:<{col}} {v}")
    print("=" * 70 + "\n")

    if args.continual_learning:
        try:
            finetune_cl(args)
        except Exception as e:
            if isinstance(e, getattr(torch, "OutOfMemoryError", ())) or \
                    "out of memory" in str(e).lower():
                _print_cuda_census()
            raise

if __name__ == "__main__":
    main()