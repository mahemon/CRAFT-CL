#!/usr/bin/env python3
"""
collect_craft_detail.py -- detailed results of every CRAFT run, from the job logs,
for one or more output_log folders (e.g. CRAFT-CL/output_log and
CRAFT-CL/explore/Tuned_router/output_log, which were written by different
versions of train.py). Standalone: it carries its own copy of the log parser of
scripts/collect_results.py, so it does not import or change any other file, and
it runs the same way from either folder.

For every finished run it reports
  * date (start time from the log name, else the "Started" line), seed, code folder
  * hyperparameters: delta, capacity, beta, h, merge rule, rank, positions,
    warm-up, learning rate, batch x accumulation, epochs and positions per task,
    max length, eval cap, routing direction, router, greedy decoding
  * per task: score right after training (diagonal), final score, change, group, JOIN/NEW
  * LA, OP and BWT three ways (percent, negative = forgetting, higher is better):
        BWT_GEM           (1/(T-1)) sum_{i<T} (final_i - after_i)   Lopez-Paz & Ranzato 2017; train.py log
        BWT_TRACE         (1/T)     sum_{i<T} (final_i - after_i)   Wang et al. 2023; = OP - LA; current tables
        BWT_TreeLoRAcode  (1/T)     sum_i min(final_i - after_i, 0) TreeLoRA repo code (gains clipped to 0)
and, per setting, the mean +- std over seeds (one run per seed: the most recent).
Legacy runs (old KL mask, normalized KL, eviction, ...) are kept but marked and
never averaged with current-method runs.

Usage
    R1=output_log
    R2=<another output_log folder>
    python collect_craft_detail.py --roots $R1 $R2 --out craft_detail
    python collect_craft_detail.py --roots $R1 $R2 --split test --backbone llama3_2_1b gemma_2b
    python collect_craft_detail.py --roots $R1 $R2 --current_only

Writes
    <out>_runs.csv       one row per finished run: setting, date, seed, LA, OP, three BWTs, ...
    <out>_tasks.csv      one row per run and task: right after, final, change, group, decision
    <out>_settings.csv   one row per setting: seeds, mean and std of K, LA, OP and the three BWTs
    <out>_summary.md     per code folder, backbone and setting: hyperparameters, a table of
                         LA / OP / BWTs per seed with the seed mean, and a per-task table
    <out>_unfinished.csv runs without a complete result (crashed, running, trained only)
"""
# ---------------------------------------------------------------------------------
# Log parser: copied unchanged from scripts/collect_results.py (the version that
# also reads EVAL_RTR_* logs), so this file has no dependency on it.
# ---------------------------------------------------------------------------------
import argparse, csv, glob, os, re, statistics
from collections import defaultdict, OrderedDict

DEFAULT_TASKS = ["C-STANCE", "FOMC", "MeetingBank", "Py150", "ScienceQA",
                 "NumGLUE-cm", "NumGLUE-ds", "20Minuten"]
ARG_KEYS = {"model", "seed", "lr", "cl_tasks", "cl_eval_file", "kl_pred_mask", "kl_normalize_kl",
            "kl_eviction_enabled", "kl_early_stop", "use_kl", "kl_beta", "kl_routing_threshold",
            "kl_routing_direction", "merge_rule", "merge_kl_halflife", "merge_kl_source",
            "group_capacity", "all_groupwise", "batch_size", "gradient_accumulation_steps",
            "max_length", "use_agem", "inference_router", "extra_eval_routers", "output_dir",
            "rank", "n_positions", "f_positions", "l_positions", "causal_warmup_steps",
            "skip_eval", "eval_only", "eval_from", "save_eval_snapshots", "per_task_bwd_eval"}
TRACE8 = DEFAULT_TASKS
EXTRA7 = ["dbpedia", "amazon", "yahoo", "agnews", "yelp", "BoolQA", "QQP"]
STREAMS = OrderedDict([
    ("TRACE-8", TRACE8),
    ("TRACE-15", TRACE8 + EXTRA7),
    ("Math-LLM", ["ScienceQA", "NumGLUE-cm", "NumGLUE-ds"]),
    ("O-LoRA order 1", ["dbpedia", "amazon", "yahoo", "agnews"]),
    ("O-LoRA order 2", ["dbpedia", "amazon", "agnews", "yahoo"]),
    ("O-LoRA order 3", ["yahoo", "amazon", "agnews", "dbpedia"]),
    ("O-LoRA order 4", "MNLI,CB,WiC,COPA,QQP,BoolQA,RTE,IMDB,yelp,amazon,SST-2,dbpedia,agnews,MultiRC,yahoo".split(",")),
    ("O-LoRA order 5", "MultiRC,BoolQA,WiC,MNLI,CB,COPA,QQP,RTE,IMDB,SST-2,dbpedia,agnews,yelp,amazon,yahoo".split(",")),
    ("O-LoRA order 6", "yelp,amazon,MNLI,CB,COPA,QQP,RTE,IMDB,SST-2,dbpedia,agnews,yahoo,MultiRC,BoolQA,WiC".split(",")),
])


def stream_label(tasks, bench):
    for name, order in STREAMS.items():
        if list(tasks) == order:
            return f"{name}@{bench}"
    return f"custom-{len(tasks)}@{bench}"


def _hours_between(s):
    """Wall-clock hours between the job's 'Started' and 'Finished at' lines, if both exist."""
    from datetime import datetime
    def _parse(txt):
        txt = re.sub(r"\s+[A-Z]{2,4}\s+(\d{4})$", r" \1", txt.strip())   # drop the time-zone token
        for fmt_ in ("%a %b %d %I:%M:%S %p %Y", "%a %b %d %H:%M:%S %Y"):
            try:
                return datetime.strptime(txt, fmt_)
            except ValueError:
                pass
        return None
    a = re.search(r"Started\s*:\s*(.+)", s)
    b = re.search(r"Finished at (.+?) with exit code", s)
    if not (a and b):
        return None
    t0, t1 = _parse(a.group(1)), _parse(b.group(1))
    return round((t1 - t0).total_seconds() / 3600, 2) if t0 and t1 else None


OLD_TAGS = {"llama8": "llama3_2_1b", "gemma8": "gemma_2b", "mistral8": "mistral_7b",
            "qwen3b8": "qwen2_5_3b", "qwen7b8": "qwen2_5_7b", "r1distill8": "r1distill_llama_8b",
            "r1qwen1p5b8": "r1distill_qwen_1_5b"}


def truthy(v):
    return str(v).strip().lower() in ("true", "1", "yes")


def fnum(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def mean(v):
    v = [x for x in v if x is not None]
    return sum(v) / len(v) if v else None


def parse_log(path, folder_tag):
    s = open(path, errors="ignore").read()
    lines = s.split("\n")
    name = os.path.basename(path)
    is_eval_log = name.startswith("EVAL_")
    if is_eval_log:
        name = name[len("EVAL_"):]
    r = OrderedDict()
    base_tag = re.sub(r"_(old|bak|backup|prev)\d*$", "", folder_tag)
    r["backbone"] = OLD_TAGS.get(base_tag, base_tag)
    r["folder"] = folder_tag
    m = None
    for t in dict.fromkeys([folder_tag, base_tag, r["backbone"]] + list(OLD_TAGS) + list(OLD_TAGS.values())):
        m = re.match(r"RTR_(.+?)_%s_s(\d+)_" % re.escape(t), name)
        if m:
            break
    r["arm"] = m.group(1) if m else "?"
    r["seed"] = int(m.group(2)) if m else None
    mo = re.match(r"o(\d)_(.+)$", r["arm"])
    if mo:                                   # O-LoRA order runs: RTR_o<k>_<arm>_...
        r["arm"] = mo.group(2)
    mb = re.search(r"_b(\d+)_", name)
    r["bench"] = int(mb.group(1)) if mb else None

    # argument dump: first occurrence of each known key before the results section
    cut = s.find("PERFORMANCE MATRIX")
    head = s if cut < 0 else s[:cut]
    args = {}
    for mm in re.finditer(r"^  (\w+)\s{2,}(\S.*?)\s*$", head, re.M):
        k = mm.group(1)
        if k in ARG_KEYS and k not in args:
            args[k] = mm.group(2)
    tasks = args.get("cl_tasks", ",".join(DEFAULT_TASKS)).strip('"').split(",")
    T = len(tasks)
    r["stream"] = stream_label(tasks, r["bench"])
    r["n_tasks"] = T
    r["model"] = args.get("model", "?")
    if r["seed"] is None:
        r["seed"] = int(args["seed"]) if "seed" in args else None
    r["split"] = (args.get("cl_eval_file") or "auto").replace(".json", "").replace("None", "auto")
    r["mode"] = ("eval_only" if (is_eval_log or truthy(args.get("eval_only", "False"))) else
                 "train_only" if truthy(args.get("skip_eval", "False")) else "together")
    r["output_dir"] = os.path.normpath(args["output_dir"]) if args.get("output_dir") else None
    r["eval_from"] = os.path.normpath(args["eval_from"]) if args.get("eval_from") not in (None, "None") else None
    r["train_log"] = path if r["mode"] != "eval_only" else None
    r["delta"] = fnum(args.get("kl_routing_threshold"))
    r["beta"] = fnum(args.get("kl_beta"))
    r["merge"] = args.get("merge_rule", "?")
    r["h"] = fnum(args.get("merge_kl_halflife")) if r["merge"] == "kl" else None
    r["bs_x_gas"] = "%s x %s" % (args.get("batch_size", "?"), args.get("gradient_accumulation_steps", "?"))
    r["cap"] = int(args["group_capacity"]) if str(args.get("group_capacity", "")).isdigit() else None
    r["rank"] = int(args["rank"]) if str(args.get("rank", "")).isdigit() else None
    fp, lp = args.get("f_positions"), args.get("l_positions")
    npos = args.get("n_positions")
    r["positions"] = (f"f{fp}+l{lp}" if str(fp).isdigit() and str(lp).isdigit()
                      else (f"f{npos}+l{npos}" if str(npos).isdigit() else None))
    r["warmup"] = int(args["causal_warmup_steps"]) if str(args.get("causal_warmup_steps", "")).isdigit() else None
    r["lr"] = fnum(args.get("lr"))

    # method: current or legacy (and why)
    use_kl = truthy(args.get("use_kl", "True"))
    why = []
    if not truthy(args.get("kl_pred_mask", "False")):
        why.append("old KL mask")
    if use_kl and truthy(args.get("kl_normalize_kl", "False")):
        why.append("normalized KL")
    if truthy(args.get("kl_eviction_enabled", "False")):
        why.append("eviction on")
    if use_kl and truthy(args.get("kl_early_stop", "False")):
        why.append("KL early stop on")
    if args.get("merge_kl_source", "eval") != "train":
        why.append("merge KL from eval split")
    if r["arm"] == "taskwise":
        # every task opens its own group: no routing, anchor or merge, so these settings never act
        why = []
        r["delta"] = r["beta"] = r["h"] = None
        r["merge"] = "—"
    elif r["arm"].startswith("allinone"):
        r["delta"] = None            # one fixed group: the routing threshold is not used
    r["method"] = "current" if not why else "legacy: " + ", ".join(why)

    # status
    err_path = path[:-4] + ".err"
    err = open(err_path, errors="ignore").read() if os.path.exists(err_path) else ""
    tb = err if "Traceback" in err else (s[s.rfind("Traceback"):] if "Traceback" in s else "")
    finished = "Overall Performance (OP)" in s
    last_task = None
    for mm in re.finditer(r"\[Phase 1 complete\] kl_routing for '([^']+)'", s):
        last_task = mm.group(1)
    if r["mode"] == "train_only" and "TRAIN-ONLY RUN COMPLETE" in s:
        r["status"] = "trained, eval pending"
    elif finished:
        r["status"] = "finished"
    elif tb:
        exc = [l for l in tb.strip().split("\n") if re.match(r"^[\w.]+(Error|Exception)", l)]
        r["status"] = "crashed"
        r["error"] = (exc[-1] if exc else tb.strip().split("\n")[-1])[:160]
    else:
        r["status"] = "running or stopped"
    r["last_task"] = last_task
    r["oom_steps"] = s.count("OOM on shadow")
    rt = [float(x) for x in re.findall(r"'train_runtime': ([0-9.]+)", s)]
    r["train_hours"] = round(sum(rt) / 3600, 2) if rt else None
    r["wall_hours"] = _hours_between(s)

    # performance matrix (last one in the log)
    pm = {}
    if cut >= 0:
        blk = s[s.rfind("PERFORMANCE MATRIX"):].split("\n")
        for ln in blk[1:]:
            tok = ln.split()
            if tok and tok[0] in tasks and tok[0] not in pm and len(tok) >= 1 + T:
                pm[tok[0]] = [None if x in ("---", "-") else fnum(x) for x in tok[1:1 + T]]
            if len(pm) == T:
                break
    diag = [pm[t][i] if t in pm else None for i, t in enumerate(tasks)]
    final = pm.get(tasks[-1], [None] * T)

    # routing log, merges, terminal KL
    groups, decision = {}, {}
    rl = s.rfind("Routing log:")
    if rl >= 0:
        for ln in s[rl:].split("\n")[1:]:
            tok = ln.split()
            if not tok:
                break
            if tok[0] in tasks and len(tok) >= 2:
                groups[tok[0]] = tok[1]
                decision[tok[0]] = tok[2] if len(tok) > 2 else ""
    merge_w = {mm.group(2): float(mm.group(4)) for mm in re.finditer(
        r"\[merge\] G(\d+) ← '([^']+)'\s+rule=(\w+)\s+mean_weight_on_new=([0-9.]+)", s)}
    term_kl, cur = {}, None
    for ln in lines:
        mm = re.search(r"\[Phase 1 complete\] kl_routing for '([^']+)'", ln)
        if mm:
            cur = mm.group(1)
            continue
        mm = re.search(r"Save-time KL_final on \w+ split = ([0-9.]+)", ln)
        if mm and cur:
            term_kl[cur] = float(mm.group(1))

    # Largest group. A run at capacity C whose groups never exceeded 2 tasks routed
    # exactly as it would at capacity 2, so it counts as a capacity-2 run.
    sizes = defaultdict(int)
    for g in groups.values():
        sizes[g] += 1
    r["max_group"] = max(sizes.values()) if sizes else None

    # metrics (paper convention)
    complete = all(x is not None for x in diag) and all(x is not None for x in final)
    if complete:
        fg = [d - f for d, f in zip(diag, final)]
        later_same = [any(groups.get(tasks[u]) == groups.get(tasks[i]) for u in range(i + 1, T))
                      for i in range(T)]
        r["K"] = len(set(groups.values())) if groups else None
        r["LA"] = round(100 * mean(diag), 2)
        r["OP"] = round(100 * mean(final), 2)
        r["BWT"] = round(100 * mean(fg), 2)
        inside = [fg[i] for i in range(T) if later_same[i]] if groups else []
        r["BWT_in_groups"] = round(100 * mean(inside), 2) if inside else None
    else:
        r["K"] = len(set(groups.values())) if groups else None
        r["LA"] = r["OP"] = r["BWT"] = r["BWT_in_groups"] = None

    # learned head: final-row OP and decision accuracy
    fr = s.rfind("Final row by router")
    r["head_OP"] = None
    if fr >= 0:
        for ln in s[fr:].split("\n")[2:]:
            tok = ln.split()
            if not tok:
                break
            if tok[0] == "learned_head" and len(tok) >= 2:
                r["head_OP"] = round(100 * float(tok[1]), 2)
    hits = re.findall(r"\[agnostic:learned_head\]\s+(\S+)\s+pred=(\S+)\s+\(G\d+.*?oracle=G\d+\s+(HIT|MISS)", s)
    r["head_task_acc"] = round(100 * sum(a == b for a, b, _ in hits) / len(hits), 1) if hits else None
    r["head_group_acc"] = round(100 * sum(h == "HIT" for _, _, h in hits) / len(hits), 1) if hits else None
    r["min_merge_w"] = min(merge_w.values()) if merge_w else None
    r["_merge_w"], r["_term_kl"] = merge_w, term_kl
    r.setdefault("error", "")
    r["log"] = path

    # Drop on the earlier members of its group caused by committing task i:
    # mean over earlier same-group tasks j of a[i-1, j] - a[i, j]. Needs the rows
    # before and after task i, i.e. a run with --per_task_bwd_eval.
    caused = {}
    for i, t in enumerate(tasks):
        if i == 0 or decision.get(t) != "JOIN":
            continue
        prev, cur = pm.get(tasks[i - 1]), pm.get(t)
        mem = [j for j in range(i) if groups.get(tasks[j]) == groups.get(t)]
        if not (prev and cur and mem):
            continue
        d = [prev[j] - cur[j] for j in mem if prev[j] is not None and cur[j] is not None]
        if len(d) == len(mem):
            caused[t] = round(100 * mean(d), 2)
    r["full_matrix"] = all(pm.get(tasks[i]) and all(pm[tasks[i]][j] is not None for j in range(i + 1))
                           for i in range(T)) if pm else False

    task_rows = []
    for i, t in enumerate(tasks):
        task_rows.append(OrderedDict(
            backbone=r["backbone"], stream=r["stream"], arm=r["arm"], seed=r["seed"], split=r["split"],
            cap=r["cap"], delta=r["delta"], task=t,
            right_after=round(100 * diag[i], 2) if diag[i] is not None else None,
            final=round(100 * final[i], 2) if final[i] is not None else None,
            forgetting=round(100 * (diag[i] - final[i]), 2) if diag[i] is not None and final[i] is not None else None,
            group=groups.get(t), decision=decision.get(t), merge_w=merge_w.get(t), terminal_kl=term_kl.get(t),
            in_group_drop_caused=caused.get(t), method=r["method"], status=r["status"],
            log=os.path.basename(path)))
    return r, task_rows


def fmt(v, digits=2):
    return "—" if v is None else f"{v:.{digits}f}"


# ---------------------------------------------------------------------------------
# New part: dates, extra hyperparameters, three BWTs, detailed output
# ---------------------------------------------------------------------------------
EXTRA_ARGS = {"max_n_eval_example": "eval_cap", "max_n_train_example": "train_cap",
              "kl_routing_direction": "route_dir", "inference_router": "router",
              "greedy_decoding": "greedy"}
TS = re.compile(r"(20\d\d)[_-]?(\d\d)[_-]?(\d\d)[_-](\d\d)[_-]?(\d\d)[_-]?(\d\d)")
SETTING_KEYS = ["code", "backbone", "stream", "split", "method", "arm", "delta", "cap", "beta", "h", "merge",
                "rank", "positions", "warmup", "lr", "bs_x_gas", "eval_cap", "train_cap", "route_dir",
                "router", "greedy"]
RUN_INFO = ["epochs", "pos_per_task", "max_len"]
METRICS = ["K", "LA", "OP", "BWT_GEM", "BWT_TRACE", "BWT_TreeLoRAcode"]


def log_extras(path):
    s = open(path, errors="ignore").read()
    cut = s.find("PERFORMANCE MATRIX")
    head = s if cut < 0 else s[:cut]
    out = {v: None for v in EXTRA_ARGS.values()}
    for mm in re.finditer(r"^  (\w+)\s{2,}(\S.*?)\s*$", head, re.M):
        k = mm.group(1)
        if k in EXTRA_ARGS and out[EXTRA_ARGS[k]] is None:
            v = mm.group(2).strip()
            out[EXTRA_ARGS[k]] = None if v in ("None", "") else v
    steps = OrderedDict()
    for mm in re.finditer(r"CL step (\d+)/\d+:\s+(\S+)\s+\[epochs=(\d+)\s+pos=(\S+)\s+max_len=(\d+)\]", s):
        steps.setdefault(int(mm.group(1)), (mm.group(3), mm.group(4), mm.group(5)))
    out["epochs"] = ",".join(v[0] for v in steps.values()) or None
    out["pos_per_task"] = ",".join(v[1] for v in steps.values()) or None
    out["max_len"] = ",".join(sorted(set(v[2] for v in steps.values()), key=int)) or None
    m = TS.search(os.path.basename(path))
    if m:
        out["date"] = "%s-%s-%s %s:%s" % m.groups()[:5]
    else:
        st = re.search(r"Started\s*:\s*(.+)", s)
        out["date"] = st.group(1).strip() if st else None
    return out


def code_label(root):
    p = os.path.abspath(root.rstrip("/"))
    return os.path.basename(os.path.dirname(p)) if os.path.basename(p) == "output_log" else os.path.basename(p)


def bwt_three(after, final):
    T = len(after)
    if T < 2 or any(x is None for x in after) or any(x is None for x in final):
        return None
    d = [f_ - a_ for a_, f_ in zip(after, final)]
    return OrderedDict(
        T=T, LA=sum(after) / T, OP=sum(final) / T,
        BWT_GEM=sum(d[:-1]) / (T - 1),
        BWT_TRACE=sum(d[:-1]) / T,
        BWT_TreeLoRAcode=sum(min(x, 0.0) for x in d) / T,
        n_gains=sum(1 for x in d[:-1] if x > 0),
        sum_gains=sum(x for x in d[:-1] if x > 0),
        last_task_gap=d[-1])


def ms(vals, nd=2):
    v = [x for x in vals if x is not None]
    if not v:
        return "—"
    if len(v) == 1:
        return f"{v[0]:.{nd}f}"
    return f"{statistics.mean(v):.{nd}f} ± {statistics.stdev(v):.{nd}f}"


def merged(rs, k):
    vals = sorted({str(r[k]) for r in rs if r.get(k) not in (None, "")})
    return "—" if not vals else (vals[0] if len(vals) == 1 else "differs by run (see _runs.csv)")


def hp_line(r, rs):
    h = fmt(r["h"]) if r["merge"] == "kl" and r["h"] is not None else (r["merge"] or "—")
    lr = "—" if r["lr"] is None else f"{r['lr']:g}"
    return (f"δ={fmt(r['delta'], 1)} · C={'—' if r['cap'] is None else r['cap']} · β={fmt(r['beta'])} · h={h} · "
            f"rank={'—' if r['rank'] is None else r['rank']} · positions={r['positions'] or '—'} · "
            f"warm-up={'—' if r['warmup'] is None else r['warmup']} · lr={lr} · batch={r['bs_x_gas']} · "
            f"epochs per task={merged(rs, 'epochs')} · positions per task={merged(rs, 'pos_per_task')} · "
            f"max_len={merged(rs, 'max_len')} · eval_cap={r['eval_cap'] or 'all'} · train_cap={r['train_cap'] or 'all'} · "
            f"routing={r['route_dir'] or '—'} · router={r['router'] or '—'} · greedy={r['greedy'] or '—'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--roots", nargs="+", required=True, help="output_log folder(s)")
    ap.add_argument("--out", default="craft_detail")
    ap.add_argument("--split", default=None, choices=["eval", "test", "auto"])
    ap.add_argument("--backbone", nargs="*", default=None, help="keep only these backbone tags")
    ap.add_argument("--current_only", action="store_true", help="drop legacy runs")
    a = ap.parse_args()

    runs, tasks_out, unfinished = [], [], []
    for root in a.roots:
        code = code_label(root)
        for d in sorted(glob.glob(os.path.join(root, "logs_*_router_*"))):
            tag = os.path.basename(d).split("_router_", 1)[1]
            for p in sorted(glob.glob(os.path.join(d, "RTR_*.txt")) + glob.glob(os.path.join(d, "EVAL_RTR_*.txt"))):
                try:
                    r, task_rows = parse_log(p, tag)
                    ex = log_extras(p)
                except Exception as e:
                    print(f"skip {p}: {type(e).__name__}: {e}")
                    continue
                if a.split and r["split"] != a.split:
                    continue
                if a.backbone and r["backbone"] not in a.backbone:
                    continue
                if a.current_only and r["method"] != "current":
                    continue
                r = dict(r)
                r.update(ex)
                r["code"] = code
                after = [t["right_after"] for t in task_rows]
                final = [t["final"] for t in task_rows]
                m = bwt_three(after, final) if r["status"] == "finished" else None
                if m is None:
                    unfinished.append(OrderedDict(
                        code=code, backbone=r["backbone"], stream=r.get("stream"), split=r["split"], arm=r["arm"],
                        seed=r["seed"], date=r["date"], method=r["method"], status=r["status"],
                        last_task=r.get("last_task"), error=r.get("error", ""), log=os.path.basename(p),
                        output_dir=r.get("output_dir")))
                    continue
                row = OrderedDict((k, r.get(k)) for k in SETTING_KEYS + RUN_INFO + ["seed", "date", "mode"])
                row["K"] = r.get("K")
                row["max_group"] = r.get("max_group")
                row.update(m)
                row["head_OP"] = r.get("head_OP")
                row["log"] = os.path.basename(p)
                row["path"] = p
                row["eval_from"] = r.get("eval_from")
                runs.append(row)
                for t in task_rows:
                    ch = None if t["right_after"] is None or t["final"] is None else round(t["final"] - t["right_after"], 2)
                    tasks_out.append(OrderedDict(
                        code=code, backbone=r["backbone"], stream=r.get("stream"), split=r["split"],
                        method=r["method"], arm=r["arm"], seed=r["seed"], date=r["date"], task=t["task"],
                        right_after=t["right_after"], final=t["final"], change=ch,
                        group=t.get("group"), decision=t.get("decision"), merge_w=t.get("merge_w"),
                        log=os.path.basename(p)))
    if not runs:
        raise SystemExit("No finished runs found.")
    # train-only runs whose scores come from a later EVAL_ log (eval_later.sh): mark them, the EVAL_ log is the row
    evaluated = {os.path.normpath(r["eval_from"]) for r in runs if r.get("eval_from")}
    for u in unfinished:
        if u["status"] == "trained, eval pending" and u.get("output_dir") and os.path.normpath(u["output_dir"]) in evaluated:
            u["status"] = "trained only; scored by its EVAL_ log (in _runs.csv)"

    def cfg(r):
        return tuple(r[k] for k in SETTING_KEYS)

    def when(r):
        return (r["date"] or "", r["log"])

    runs.sort(key=lambda r: tuple(str(x) for x in cfg(r)) + (str(r["seed"]),) + when(r))

    def write(path, rows):
        if not rows:
            return
        keys = list(rows[0].keys())
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for x in rows:
                w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in x.items()})

    write(a.out + "_runs.csv", runs)
    write(a.out + "_tasks.csv", tasks_out)
    write(a.out + "_unfinished.csv", unfinished)

    tasks_by_log = defaultdict(list)
    for t in tasks_out:
        tasks_by_log[t["log"]].append(t)
    groups = defaultdict(list)
    for r in runs:
        groups[cfg(r)].append(r)

    settings, md = [], [
        "# CRAFT runs: detailed results", "",
        "Percent. d_i = final score - score right after training task i. All BWT: negative = forgetting, higher is better.", "",
        "- **BWT GEM**: (1/(T-1)) sum over the first T-1 tasks of d_i (Lopez-Paz & Ranzato 2017; what train.py logs)",
        "- **BWT TRACE**: (1/T) sum over the first T-1 tasks of d_i = OP - LA (Wang et al. 2023; current tables)",
        "- **BWT TreeLoRA code**: (1/T) sum over all tasks of min(d_i, 0) (TreeLoRA repo code; gains clipped to 0)", "",
        "Seed mean: one run per seed (the most recent if a seed was run more than once). "
        "Per-task cells: final (change). Legacy runs are marked and never averaged with current ones.", ""]
    tree = defaultdict(lambda: defaultdict(list))
    for key, rs in groups.items():
        tree[key[0]][key[1]].append((key, rs))

    for code in sorted(tree):
        md += [f"# Code folder: {code}", ""]
        for bb in sorted(tree[code]):
            md += [f"## {code} / {bb}", ""]
            for key, rs in sorted(tree[code][bb], key=lambda kv: tuple(str(x) for x in kv[0])):
                latest = {}
                for r in sorted(rs, key=when):
                    latest[r["seed"]] = r
                used = [latest[s] for s in sorted(latest, key=lambda s: (s is None, s))]
                first = rs[0]
                tag = "" if first["method"] == "current" else f"  *(legacy: {first['method'][8:]})*"
                md += [f"### {first['stream']} · {first['split']} · {first['arm']}{tag}", "",
                       hp_line(first, rs), "",
                       "| Seed | Date | Mode | K | LA | OP | BWT GEM | BWT TRACE | BWT TreeLoRA code | Gains clipped | Log |",
                       "|---|---|---|---|---|---|---|---|---|---|---|"]
                for r in sorted(rs, key=lambda x: (str(x["seed"]),) + when(x)):
                    mark = "" if r is latest.get(r["seed"]) else " (older, not averaged)"
                    md.append(f"| {r['seed']}{mark} | {r['date'] or '—'} | {r['mode']} | {fmt(r['K'], 0)} | {fmt(r['LA'])} | "
                              f"{fmt(r['OP'])} | {fmt(r['BWT_GEM'])} | {fmt(r['BWT_TRACE'])} | {fmt(r['BWT_TreeLoRAcode'])} | "
                              f"{r['n_gains']} ({r['sum_gains']:+.2f}) | {r['log']} |")
                if len(used) > 1:
                    md.append(f"| **mean ({len(used)} seeds)** | | | {ms([r['K'] for r in used], 1)} | "
                              f"{ms([r['LA'] for r in used])} | {ms([r['OP'] for r in used])} | "
                              f"{ms([r['BWT_GEM'] for r in used])} | {ms([r['BWT_TRACE'] for r in used])} | "
                              f"{ms([r['BWT_TreeLoRAcode'] for r in used])} | | |")
                # per-task table: final (change) per seed, then the mean
                tnames = [t["task"] for t in tasks_by_log[used[0]["log"]]]
                md += ["", "| Seed | " + " | ".join(tnames) + " |", "|---|" + "---|" * len(tnames)]
                per_seed = []
                for r in used:
                    tr = {t["task"]: t for t in tasks_by_log[r["log"]]}
                    per_seed.append(tr)
                    md.append(f"| {r['seed']} | " + " | ".join(
                        f"{fmt(tr[t]['final'])} ({tr[t]['change']:+.2f})" if t in tr and tr[t]["change"] is not None else "—"
                        for t in tnames) + " |")
                if len(used) > 1:
                    cells = []
                    for t in tnames:
                        fv = [tr[t]["final"] for tr in per_seed if t in tr and tr[t]["final"] is not None]
                        cv = [tr[t]["change"] for tr in per_seed if t in tr and tr[t]["change"] is not None]
                        cells.append(f"{statistics.mean(fv):.2f} ({statistics.mean(cv):+.2f})" if fv and cv else "—")
                    md.append("| **mean** | " + " | ".join(cells) + " |")
                md.append("")
                srow = OrderedDict(zip(SETTING_KEYS, key))
                srow["epochs"] = merged(rs, "epochs")
                srow["max_len"] = merged(rs, "max_len")
                srow["seeds"] = ",".join(str(r["seed"]) for r in used)
                srow["dates"] = ",".join(str(r["date"]) for r in used)
                srow["n_seeds"] = len(used)
                srow["n_older_runs_not_averaged"] = len(rs) - len(used)
                srow["T"] = first["T"]
                for mname in METRICS:
                    vals = [r[mname] for r in used if r[mname] is not None]
                    srow[mname + "_mean"] = round(statistics.mean(vals), 4) if vals else None
                    srow[mname + "_std"] = round(statistics.stdev(vals), 4) if len(vals) > 1 else None
                settings.append(srow)
    write(a.out + "_settings.csv", settings)

    gap = [r for r in runs if abs(r["last_task_gap"]) > 1e-6]
    md += ["# Checks", "",
           f"- Finished runs: {len(runs)} ({sum(r['method'] == 'current' for r in runs)} current, "
           f"{sum(r['method'] != 'current' for r in runs)} legacy). Not finished or incomplete: {len(unfinished)} "
           f"(listed in {a.out}_unfinished.csv).",
           "- Code folders: " + ", ".join(f"{c} ({sum(r['code'] == c for r in runs)} runs)" for c in sorted(tree)) + ".",
           f"- Last task: right-after score differs from final score in {len(gap)} run(s)"
           + (": " + ", ".join(r["log"] for r in gap) if gap else " (BWT TRACE = OP - LA in every run)."),
           f"- Runs where clipping changes BWT: {sum(1 for r in runs if r['n_gains'] > 0)} of {len(runs)}."]
    open(a.out + "_summary.md", "w").write("\n".join(md) + "\n")
    print("\n".join(md[-6:]))
    print(f"\nWrote {a.out}_runs.csv ({len(runs)} runs), {a.out}_tasks.csv, {a.out}_settings.csv "
          f"({len(settings)} settings), {a.out}_unfinished.csv, {a.out}_summary.md")


if __name__ == "__main__":
    main()



# python collect_craft_detail.py --roots output_log <another output_log folder> --out craft_detail_all