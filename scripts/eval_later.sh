#!/bin/bash
# =============================================================================
# eval_later.sh -- evaluate runs that were trained with EVAL_MODE=train.
#
# From the repository root:
#   bash scripts/eval_later.sh --pending                 # every trained run not yet evaluated
#   bash scripts/eval_later.sh <train_output_dir> [...]  # these runs only
#   DRY=1 bash scripts/eval_later.sh --pending           # print, submit nothing
#
# Each evaluation is one GPU job running   train.py --eval_only --eval_from <run>.
# train.py restores the training configuration from <run>/train_args.json and
# the state after every task from <run>/eval_snapshots/, so nothing about the
# training run has to be repeated here.
#
# Where things go (the training job name is the link between them):
#   training log : output_log/logs_<bench>_router_<backbone>/RTR_..._tronly_<ts>_<jobid>.txt
#   eval log     : output_log/logs_<bench>_router_<backbone>/EVAL_RTR_..._tronly_<ts>_<tag>_<jobid>.txt
#   eval results : <run>/eval_<tag>/cl_results.json         tag = test | val, plus _bwd
#
# Settings (environment):
#   EVAL_FILE     test.json or eval.json   (default: the split the run was trained for)
#   BWD           1 = full lower triangle, 0 = diagonal + final row (default: as trained)
#   INFER_ROUTER  e.g. "oracle learned_head" (default: as trained)
#   ROOTS         directories --pending searches (default: output_log)
#   WALLTIME      default from model size (1B 8h, 2-3B 12h, 7-8B 20h, 13B 36h; x2 with BWD)
#   GRES MEM      default gpu:a100:1, 64G
#   PARTITION ACCOUNT VENV_ACTIVATE CUDA_MODULE EXCLUDE_NODES   cluster settings; also read
#                 from scripts/cluster.env if it exists (see cluster.env.example)
#   FORCE=1       evaluate again even if results exist
#   MAX_JOBS      submit at most this many in one call
# =============================================================================
set -e
STUDY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${STUDY_DIR}"
[ -f train.py ] || { echo "FATAL: train.py not found in ${STUDY_DIR}" >&2; exit 1; }
grep -q -- '"--eval_only"' train.py || { echo "FATAL: this train.py has no --eval_only." >&2; exit 1; }

ROOT_DIR="${ROOT_DIR:-${STUDY_DIR}}"      # the repository root (holds pyvene/ and pyreft/)
[ -f "${STUDY_DIR}/scripts/cluster.env" ] && source "${STUDY_DIR}/scripts/cluster.env"
VENV_ACTIVATE="${VENV_ACTIVATE:-}"
CUDA_MODULE="${CUDA_MODULE:-}"
EXCLUDE_NODES="${EXCLUDE_NODES:-}"
PARTITION="${PARTITION:-}"
ACCOUNT="${ACCOUNT:-}"
SB_SITE=""
[ -n "${ACCOUNT}" ]   && SB_SITE+="#SBATCH --account=${ACCOUNT}"$'\n'
[ -n "${PARTITION}" ] && SB_SITE+="#SBATCH --partition=${PARTITION}"$'\n'
SB_EXCL=""; [ -n "${EXCLUDE_NODES}" ] && SB_EXCL="#SBATCH --exclude=${EXCLUDE_NODES}"$'\n'
MODULE_LINE=""; [ -n "${CUDA_MODULE}" ]   && MODULE_LINE="module load ${CUDA_MODULE}"
VENV_LINE="";   [ -n "${VENV_ACTIVATE}" ] && VENV_LINE="source \"${VENV_ACTIVATE}\""
GRES="${GRES:-gpu:a100:1}"
MEM="${MEM:-64G}"
ROOTS="${ROOTS:-output_log}"
MAX_JOBS="${MAX_JOBS:-0}"

# ---- which runs ---------------------------------------------------------------
RUNS=()
if [ "$1" = "--pending" ]; then
    for root in ${ROOTS}; do
        [ -d "${root}" ] || continue
        while IFS= read -r a; do RUNS+=("$(cd "$(dirname "$a")" && pwd)"); done \
            < <(find "${root}" -mindepth 2 -maxdepth 3 -name train_args.json -path "*outputs_*" | sort)
    done
elif [ $# -gt 0 ]; then
    for d in "$@"; do RUNS+=("$(cd "$d" && pwd)"); done
else
    echo "usage: [DRY=1] bash scripts/eval_later.sh {--pending | <train_output_dir> ...}" >&2
    exit 1
fi
[ ${#RUNS[@]} -eq 0 ] && { echo "No trained runs found."; exit 0; }

QUEUED=""
command -v squeue >/dev/null 2>&1 && QUEUED="$(squeue -h -u "${USER}" -o %j 2>/dev/null || true)"

n_sub=0; n_skip=0
for RUN in "${RUNS[@]}"; do
    JOB="$(basename "${RUN}")"
    # everything needed from the training run, in one python call
    INFO="$(python3 - "${RUN}" <<'PY'
import json, os, sys
run = sys.argv[1]
try:
    a = json.load(open(os.path.join(run, "train_args.json")))
except Exception as e:
    print("ERR cannot read train_args.json:", e); sys.exit()
tasks = [t for t in str(a.get("cl_tasks", "")).strip('"').split(",") if t]
snap = os.path.join(run, "eval_snapshots")
have = len([f for f in os.listdir(snap) if f.endswith(".pt")]) if os.path.isdir(snap) else 0
model = str(a.get("model", "")).lower()
size = "13" if "13b" in model else "7" if ("7b" in model or "8b" in model) else \
       "2" if any(x in model for x in ("2b", "1.5b", "1_5b", "3b")) else "1"
print("OK", len(tasks), have, a.get("cl_eval_file") or "test.json",
      1 if a.get("per_task_bwd_eval") else 0, size)
PY
)"
    set -- ${INFO}
    if [ "$1" != "OK" ]; then echo "skip ${JOB}: ${INFO}"; n_skip=$((n_skip+1)); continue; fi
    NT=$2; NS=$3; TRAIN_EF=$4; TRAIN_BWD=$5; SIZE=$6
    if [ "${NS}" -eq 0 ] && [ -f "${RUN}/cl_results.json" ]; then
        n_skip=$((n_skip+1)); continue        # an ordinary train+eval run: nothing to do
    fi
    if [ "${NS}" -lt "${NT}" ]; then
        echo "skip ${JOB}: ${NS}/${NT} snapshots (training not finished, or trained without snapshots)"
        n_skip=$((n_skip+1)); continue
    fi
    EF="${EVAL_FILE:-${TRAIN_EF}}"
    B="${BWD:-${TRAIN_BWD}}"
    SPLIT="test"; [ "${EF}" = "eval.json" ] && SPLIT="val"
    TAG="${SPLIT}"; [ "${B}" = "1" ] && TAG="${TAG}_bwd"
    OUT="${RUN}/eval_${TAG}"
    NAME="EVAL_${JOB}_${TAG}"
    if [ -f "${OUT}/cl_results.json" ] && [ "${FORCE:-0}" != "1" ]; then
        echo "done ${JOB}  (${TAG})"; n_skip=$((n_skip+1)); continue
    fi
    if printf '%s\n' "${QUEUED}" | grep -qx "${NAME}"; then
        echo "queued ${JOB}  (${TAG})"; n_skip=$((n_skip+1)); continue
    fi
    if [ "${MAX_JOBS}" -gt 0 ] && [ "${n_sub}" -ge "${MAX_JOBS}" ]; then
        echo "MAX_JOBS=${MAX_JOBS} reached; run again for the rest."; break
    fi

    # logs next to the training log: outputs_<...>/<job> -> logs_<...>/
    PARENT="$(dirname "${RUN}")"
    LOG_ROOT="$(dirname "${PARENT}")/$(basename "${PARENT}" | sed 's/^outputs_/logs_/')"
    [ "${LOG_ROOT}" = "${PARENT}" ] && LOG_ROOT="${PARENT}/logs_eval"
    mkdir -p "${LOG_ROOT}"

    case "${SIZE}" in 13) H=36 ;; 7) H=20 ;; 2) H=12 ;; *) H=8 ;; esac
    [ "${B}" = "1" ] && H=$((H * 2))
    [ ${H} -gt 71 ] && H=71
    TIME="${WALLTIME:-$(printf '%d-%02d:00:00' $((H / 24)) $((H % 24)))}"

    # the python command, built as one line (no line continuations in the job script)
    PY_ARGS=(--eval_only --eval_from "${RUN}" --output_dir "${OUT}" --cl_eval_file "${EF}")
    [ -n "${BWD}" ] && { [ "${BWD}" = "1" ] && PY_ARGS+=(--per_task_bwd_eval) || PY_ARGS+=(--no_per_task_bwd_eval); }
    if [ -n "${INFER_ROUTER}" ]; then
        PRIM="${INFER_ROUTER%% *}"; EXTRA=""
        [ "${INFER_ROUTER}" != "${PRIM}" ] && EXTRA="${INFER_ROUTER#* }"
        PY_ARGS+=(--inference_router "${PRIM}")
        [ -n "${EXTRA}" ] && PY_ARGS+=(--extra_eval_routers ${EXTRA})
    fi
    PY_CMD="$(printf '%q ' python -u "${STUDY_DIR}/train.py" "${PY_ARGS[@]}")"

    if [ -n "${DRY}" ]; then
        echo "[dry] ${NAME}   split=${EF} bwd=${B} time=${TIME}"
        echo "      log    ${LOG_ROOT}/${NAME}_<jobid>.txt"
        echo "      result ${OUT}/cl_results.json"
        n_sub=$((n_sub+1)); continue
    fi

    JID=$(sbatch --parsable <<EOF
#!/bin/bash
${SB_SITE}#SBATCH --time=${TIME}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=6
${SB_EXCL}#SBATCH --mem=${MEM}
#SBATCH --gres=${GRES}
#SBATCH --job-name=${NAME}
#SBATCH --output=${LOG_ROOT}/${NAME}_%j.txt
#SBATCH --error=${LOG_ROOT}/${NAME}_%j.err

cd "${STUDY_DIR}" || exit 1
${MODULE_LINE}
${VENV_LINE}
export PYTHONPATH="${ROOT_DIR}:\${PYTHONPATH}"
if [ -f "\$HOME/.hf_token" ]; then export HF_TOKEN="\$(cat "\$HOME/.hf_token")"; fi
: "\${HF_TOKEN:?HF_TOKEN is not set.}"
huggingface-cli login --token "\$HF_TOKEN" >/dev/null 2>&1 || true
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export ROUTER_STRICT=1
echo "  Node     : \$(hostname)"
if ! python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)"; then
    echo "FATAL: torch sees no CUDA device on \$(hostname)." >&2; exit 3
fi
echo "===================================================================="
echo "  EVALUATION-ONLY RUN"
echo "  Train run: ${RUN}"
echo "  Train log: ${LOG_ROOT}/${JOB}_*.txt"
echo "  Split    : ${EF}   full lower triangle: ${B}"
echo "  Output   : ${OUT}"
echo "  Started  : \$(date)"
echo "  Command  : ${PY_CMD}"
echo "===================================================================="
${PY_CMD}
EXIT_CODE=\$?
echo ""
echo "  Finished at \$(date) with exit code \$EXIT_CODE"
exit \$EXIT_CODE
EOF
) || { echo "FATAL: sbatch rejected ${NAME}" >&2; exit 1; }
    echo "Submitted: ${NAME}  jobid=${JID}"
    n_sub=$((n_sub+1))
done
echo ""
echo "submitted ${n_sub}, skipped ${n_skip}"
