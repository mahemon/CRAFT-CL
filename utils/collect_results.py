#!/usr/bin/env python3
"""
collect_results.py -- collect and summarize every CRAFT run from its job logs.

Scans  <root>/logs_*_router_*/*.txt  (one job log per run) and writes:

  <out>_runs.csv     one row per run: backbone, arm, seed, split, status,
                     method (current / legacy and why), delta, beta, h,
                     groups, LA, OP, BWT, BWT inside groups, learned-head OP
                     and accuracy, smallest merge weight, OOM steps, error
  <out>_tasks.csv    one row per run and task: score right after training,
                     final score, forgetting, group, JOIN/NEW, merge weight,
                     terminal KL
  <out>_summary.md   per backbone: finished runs of the current method grouped
                     by configuration (mean +- std over seeds), then legacy
                     runs, then unfinished or crashed runs

Metrics use the paper's convention, recomputed from the performance matrix:
  LA  = mean of the scores right after each task's training (diagonal)
  OP  = mean of the final row
  BWT = mean over all T tasks of (right after training - final); positive = forgetting
(The job log prints BWT with the opposite sign, and "Mean Forgetting" over T-1 tasks.)

"current" = corrected KL mask, plain beta*KL (no normalization), no eviction,
no KL early stopping, merge KL from the training split. Anything else is
"legacy" and the reason is listed.

Usage (standard library only):
  python collect_results.py --roots /path/to/Tuned_router/output_log
  python collect_results.py --roots OLD/output_log NEW/output_log --out results
  python collect_results.py --roots ... --split test          # only test.json runs
"""
import argparse, csv, glob, os, re, statistics
from collections import defaultdict, OrderedDict

DEFAULT_TASKS = ["C-STANCE", "FOMC", "MeetingBank", "Py150", "ScienceQA",
                 "NumGLUE-cm", "NumGLUE-ds", "20Minuten"]
ARG_KEYS = {"model", "seed", "lr", "cl_tasks", "cl_eval_file", "kl_pred_mask", "kl_normalize_kl",
            "kl_eviction_enabled", "kl_early_stop", "use_kl", "kl_beta", "kl_routing_threshold",
            "kl_routing_direction", "merge_rule", "merge_kl_halflife", "merge_kl_source",
            "group_capacity", "all_groupwise", "batch_size", "gradient_accumulation_steps",
            "max_length", "use_agem", "inference_router", "extra_eval_routers", "output_dir"}
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

    # argument dump: first occurrence of each known key before the results section
    cut = s.find("PERFORMANCE MATRIX")
    head = s if cut < 0 else s[:cut]
    args = {}
    for mm in re.finditer(r"^  (\w+)\s{2,}(\S.*?)\s*$", head, re.M):
        k = mm.group(1)
        if k in ARG_KEYS and k not in args:
            args[k] = mm.group(2)
    tasks = args.get("cl_tasks", ",".join(DEFAULT_TASKS)).split(",")
    T = len(tasks)
    r["model"] = args.get("model", "?")
    if r["seed"] is None:
        r["seed"] = int(args["seed"]) if "seed" in args else None
    r["split"] = (args.get("cl_eval_file") or "auto").replace(".json", "").replace("None", "auto")
    r["delta"] = fnum(args.get("kl_routing_threshold"))
    r["beta"] = fnum(args.get("kl_beta"))
    r["merge"] = args.get("merge_rule", "?")
    r["h"] = fnum(args.get("merge_kl_halflife")) if r["merge"] == "kl" else None
    r["lr"] = args.get("lr")
    r["bs_x_gas"] = "%s x %s" % (args.get("batch_size", "?"), args.get("gradient_accumulation_steps", "?"))

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
    if finished:
        r["status"] = "finished"
    elif tb:
        exc = [l for l in tb.strip().split("\n") if re.match(r"^[\w.]+(Error|Exception)", l)]
        r["status"] = "crashed"
        r["error"] = (exc[-1] if exc else tb.strip().split("\n")[-1])[:160]
    else:
        r["status"] = "running or stopped"
    r["last_task"] = last_task
    r["oom_steps"] = s.count("OOM on shadow")

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
    r.setdefault("error", "")
    r["log"] = path

    task_rows = []
    for i, t in enumerate(tasks):
        task_rows.append(OrderedDict(
            backbone=r["backbone"], arm=r["arm"], seed=r["seed"], split=r["split"], task=t,
            right_after=round(100 * diag[i], 2) if diag[i] is not None else None,
            final=round(100 * final[i], 2) if final[i] is not None else None,
            forgetting=round(100 * (diag[i] - final[i]), 2) if diag[i] is not None and final[i] is not None else None,
            group=groups.get(t), decision=decision.get(t), merge_w=merge_w.get(t), terminal_kl=term_kl.get(t),
            log=os.path.basename(path)))
    return r, task_rows


def fmt(v, digits=2):
    return "—" if v is None else f"{v:.{digits}f}"


def ms(vals, digits=2):
    v = [x for x in vals if x is not None]
    if not v:
        return "—"
    if len(v) == 1:
        return f"{v[0]:.{digits}f}"
    return f"{statistics.mean(v):.{digits}f} ± {statistics.stdev(v):.{digits}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="+", required=True, help="output_log folder(s)")
    ap.add_argument("--out", default="results")
    ap.add_argument("--split", default=None, choices=["eval", "test", "auto"], help="keep only this split")
    a = ap.parse_args()

    runs, task_rows = [], []
    for root in a.roots:
        for d in sorted(glob.glob(os.path.join(root, "logs_*_router_*"))):
            tag = os.path.basename(d).split("_router_", 1)[1]
            for p in sorted(glob.glob(os.path.join(d, "RTR_*.txt"))):
                try:
                    r, tr = parse_log(p, tag)
                except Exception as e:
                    print(f"skip {p}: {type(e).__name__}: {e}")
                    continue
                if a.split and r["split"] != a.split:
                    continue
                runs.append(r)
                task_rows.extend(tr)
    if not runs:
        raise SystemExit("No job logs found under " + ", ".join(a.roots))

    keys = list(runs[0].keys())
    for r in runs:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(a.out + "_runs.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(runs)
    with open(a.out + "_tasks.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(task_rows[0].keys()))
        w.writeheader()
        w.writerows(task_rows)

    md = ["# CRAFT results", "",
          "Scores in percent. BWT is positive for forgetting (paper convention), over all tasks. "
          "OP of the learned head is task-agnostic; everything else uses the task label (oracle).", ""]
    by_bb = defaultdict(list)
    for r in runs:
        by_bb[r["backbone"]].append(r)
    for bb in sorted(by_bb):
        rs = by_bb[bb]
        md += [f"## {bb}", ""]
        cur = [r for r in rs if r["status"] == "finished" and r["method"] == "current"]
        if cur:
            md += ["**Current method, finished runs**", "",
                   "| Arm | Split | δ | β | h | Seeds | Groups | LA | OP | BWT | BWT in groups | Head OP | Head task acc |",
                   "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
            cfg = defaultdict(list)
            for r in cur:
                cfg[(r["arm"], r["split"], r["delta"], r["beta"], r["h"] if r["merge"] == "kl" else r["merge"])].append(r)
            for (arm, split, d, b, h), g in sorted(cfg.items(), key=lambda kv: (kv[0][1], kv[0][0])):
                seeds = ",".join(str(x["seed"]) for x in sorted(g, key=lambda x: x["seed"] or 0))
                hs = fmt(h) if isinstance(h, float) else str(h)
                md.append(f"| {arm} | {split} | {fmt(d, 1)} | {fmt(b)} | {hs} | {seeds} | "
                          f"{ms([x['K'] for x in g], 0)} | {ms([x['LA'] for x in g])} | {ms([x['OP'] for x in g])} | "
                          f"{ms([x['BWT'] for x in g])} | {ms([x['BWT_in_groups'] for x in g])} | "
                          f"{ms([x['head_OP'] for x in g])} | {ms([x['head_task_acc'] for x in g])} |")
            md.append("")
        leg = [r for r in rs if r["status"] == "finished" and r["method"] != "current"]
        if leg:
            md += ["**Legacy runs (old code or settings; not for the paper)**", "",
                   "| Arm | Seed | Split | OP | BWT | Why legacy |", "|---|---|---|---|---|---|"]
            for r in sorted(leg, key=lambda x: (x["arm"], x["seed"] or 0)):
                md.append(f"| {r['arm']} | {r['seed']} | {r['split']} | {fmt(r['OP'])} | {fmt(r['BWT'])} | {r['method'][8:]} |")
            md.append("")
        other = [r for r in rs if r["status"] != "finished"]
        if other:
            md += ["**Not finished**", "", "| Arm | Seed | Split | Method | Status | Last task reached | Error |",
                   "|---|---|---|---|---|---|---|"]
            for r in sorted(other, key=lambda x: (x["arm"], x["seed"] or 0)):
                md.append(f"| {r['arm']} | {r['seed']} | {r['split']} | {r['method']} | {r['status']} | {r['last_task'] or '—'} | "
                          f"{(r['error'] or '').replace('|', '/')} |")
            md.append("")
    open(a.out + "_summary.md", "w").write("\n".join(md))
    print("\n".join(md))
    print(f"\nWrote {a.out}_runs.csv ({len(runs)} runs), {a.out}_tasks.csv, {a.out}_summary.md")


if __name__ == "__main__":
    main()

# Run it on the cluster:
# python collect_results.py --roots output_log

# Once runs exist under the new root, add it too, and both trees go into one summary:
# python collect_results.py --roots output_log <another output_log folder>