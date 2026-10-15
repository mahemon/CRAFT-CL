#!/bin/bash
# ============================================================================
# run_craft.sh -- CRAFT on any backbone, one SLURM job per run (TRACE-8 by default).
#
#   MODEL=llama3_2_1b bash scripts/run_craft.sh                      # TRACE-8, seed 42
#   bash scripts/run_craft.sh gemma_2b                               # the model can also be the first argument
#   PARTITION=gpu ACCOUNT=myproject MODEL=llama2_7b bash scripts/run_craft.sh  # another partition
#   SEEDS="42 1234" MODEL=mistral_7b bash scripts/run_craft.sh       # two seeds
#   DRY=1 MODEL=llama3_2_1b bash scripts/run_craft.sh                # print the jobs, submit nothing
#
# Models (settings per model are in the table below):
#   llama3_2_1b  gemma_2b  llama2_7b  llama2_13b  mistral_7b
#   r1distill_llama_8b  r1distill_qwen_1_5b  qwen2_5_3b  qwen2_5_7b
#
# Common settings (environment):
#   SEEDS="42"            seeds, one job each
#   ARMS="v20260828"      method arms (see ARM_ENV below); ARMS=all = + taskwise + allinone
#   INFER_ROUTER="oracle litar learned_head"   routers: oracle, litar, learned_head; the first
#                         drives the stream, the others replay the final row
#   GROUP_CAPACITY=N      largest number of tasks in one group (0 = no cap)
#   ROUTING_THRESHOLD=d   KL routing threshold delta
#   LR_LIST / RANK_LIST   lr x rank grid      EVAL_FILE=eval.json   validation split
#   BENCH=500             500-per-task benchmark   TASKS=... EPOCHS=...  another task stream
#   LONG=0|1              longer context and l-stream for MeetingBank, 20Minuten, Py150
#   EVAL_MODE=train       train only (evaluate later with scripts/eval_later.sh)
#   RESUME_FROM=<dir>     continue a stopped run from its resume_state.pt
#   MAXPAR=2              jobs running at the same time (0 = all at once)
#   WALLTIME / MEM        override the model defaults
#
# Cluster settings: PARTITION, ACCOUNT, VENV_ACTIVATE, CUDA_MODULE, EXCLUDE_NODES and
# SHORT_PARTITIONS (partitions with a shorter time limit). Set them in the environment or
# in scripts/cluster.env, which is read first if it exists (see cluster.env.example).
#
# Task orders 1-6, TRACE-15 and the math stream: scripts/run_streams.sh
# ============================================================================

VERSION="20260828"

# ---- paths: the repository root is the folder that holds train.py -------------
STUDY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[ -f "${STUDY_DIR}/train.py" ] || STUDY_DIR="$(dirname "${STUDY_DIR}")"
ROOT_DIR="${ROOT_DIR:-${STUDY_DIR}}"                      # holds pyvene/ and pyreft/
DATA_DIR="${DATA_DIR:-${ROOT_DIR}/dataset/TRACE_Benchmark}"
# Site settings (account, partition, venv, CUDA module): scripts/cluster.env if present.
[ -f "${STUDY_DIR}/scripts/cluster.env" ] && source "${STUDY_DIR}/scripts/cluster.env"
VENV_ACTIVATE="${VENV_ACTIVATE:-}"
CUDA_MODULE="${CUDA_MODULE:-}"
EXCLUDE_NODES="${EXCLUDE_NODES:-}"
SHORT_PARTITIONS="${SHORT_PARTITIONS:-}"
PARTITION="${PARTITION:-}"
ACCOUNT="${ACCOUNT:-}"

# ---- model presets --------------------------------------------------------------
MODEL_TAG="${MODEL:-${1:-}}"
set_p() {
  P_ID="$1"; P_LR="$2"; P_BS="$3"; P_GAS="$4"; P_BETA="$5"; P_H="$6"; P_CAP="$7"; P_DELTA="$8"
  P_MEM="$9"; P_EBS="${10}"; P_MAXLEN="${11}"; P_LONG="${12}"; P_WALL="${13}"; P_WALL_SHORT="${14}"
  P_ROUTERS="${15}"
}
case "${MODEL_TAG}" in
  #                                  HF model id                             lr   bs gas beta  h  cap delta mem  ebs maxlen long wall time   wall(short)  routers
  llama3_2_1b)         set_p meta-llama/Llama-3.2-1B-Instruct        2e-4 4 1 0.3 0.6 3 0.7 64G  4 2048 1 23:00:00   18:00:00   "oracle litar learned_head" ;;
  gemma_2b)            set_p google/gemma-2b-it                      2e-4 4 1 0.2 1.0 3 0.7 64G  2 1024 1 1-12:00:00 1-12:00:00 "oracle litar learned_head" ;;
  llama2_7b)           set_p meta-llama/Llama-2-7b-chat-hf           1e-4 4 1 0.3 0.6 3 0.7 128G 4 1024 0 2-23:00:00 1-23:00:00 "oracle learned_head" ;;
  llama2_13b)          set_p meta-llama/Llama-2-13b-chat-hf          1e-4 2 2 0.3 0.6 2 0.7 64G  4 1024 0 15:00:00   15:00:00   "oracle learned_head" ;;
  mistral_7b)          set_p mistralai/Mistral-7B-Instruct-v0.3      5e-4 4 1 0.3 0.6 2 0.7 128G 4 2048 1 2-23:00:00 2-23:00:00 "oracle learned_head" ;;
  r1distill_llama_8b)  set_p deepseek-ai/DeepSeek-R1-Distill-Llama-8B  2e-4 2 2 0.3 0.6 2 0.7 64G  4 2048 1 2-23:00:00 2-23:00:00 "oracle learned_head" ;;
  r1distill_qwen_1_5b) set_p deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B 2e-4 4 1 0.3 0.6 2 0.7 64G  4 2048 1 23:00:00   10:00:00   "oracle learned_head" ;;
  qwen2_5_3b)          set_p Qwen/Qwen2.5-3B-Instruct                2e-4 4 1 0.3 0.6 3 0.7 64G  4 2048 1 18:00:00   18:00:00   "oracle learned_head" ;;
  qwen2_5_7b)          set_p Qwen/Qwen2.5-7B-Instruct                2e-4 2 2 0.3 0.6 3 1.5 64G  4 2048 1 23:00:00   23:00:00   "oracle learned_head" ;;
  *) echo "usage: MODEL=<model> bash scripts/run_craft.sh   (models: llama3_2_1b gemma_2b llama2_7b llama2_13b" >&2
     echo "       mistral_7b r1distill_llama_8b r1distill_qwen_1_5b qwen2_5_3b qwen2_5_7b)" >&2
     exit 1 ;;
esac
# pyvene must know the model class (checked inside the job), and model-specific train.py flags
PYVENE_CLASS=""; EXTRA_TRAIN_FLAGS=""
case "${MODEL_TAG}" in
  mistral_7b)          PYVENE_CLASS="mistral.modeling_mistral.MistralForCausalLM"
                       EXTRA_TRAIN_FLAGS="--mistral_system_first_turn" ;;
  r1distill_llama_8b)  PYVENE_CLASS="llama.modeling_llama.LlamaForCausalLM" ;;
  r1distill_qwen_1_5b|qwen2_5_*) PYVENE_CLASS="qwen2.modeling_qwen2.Qwen2ForCausalLM" ;;
esac

MODEL_ID="${MODEL_ID:-${P_ID}}"
BACKBONE_TAG="${MODEL_TAG}"
BENCH="${BENCH:-5000}"
# STREAM_TAG and NAME_TAG are set by run_streams.sh (e.g. olora_ and o1_); empty here.
STREAM_TAG="${STREAM_TAG:-}"
NAME_TAG="${NAME_TAG:-}"
LOG_ROOT="${STUDY_DIR}/output_log/logs_${BENCH}_${STREAM_TAG}router_${BACKBONE_TAG}"
OUT_ROOT="${STUDY_DIR}/output_log/outputs_${BENCH}_${STREAM_TAG}router_${BACKBONE_TAG}"
mkdir -p "${LOG_ROOT}"

# Wall time from the model table; a partition listed in SHORT_PARTITIONS uses the
# second (shorter) value. WALLTIME and MEM override them.
case " ${SHORT_PARTITIONS} " in
  *" ${PARTITION:-none} "*) WALLTIME="${WALLTIME:-${P_WALL_SHORT}}" ;;
  *)                        WALLTIME="${WALLTIME:-${P_WALL}}" ;;
esac
MEM="${MEM:-${P_MEM}}"
MAX_LENGTH="${MAX_LENGTH:-${P_MAXLEN}}"
EVAL_BS="${EVAL_BS:-${P_EBS}}"

LR="${LR:-${P_LR}}"
RANK="${RANK:-8}"
# Sweep lists default to the single values above; LR_LIST/RANK_LIST turn one call into a grid.
LR_LIST="${LR_LIST:-${LR}}"
RANK_LIST="${RANK_LIST:-${RANK}}"
TASKS="${TASKS:-C-STANCE,FOMC,MeetingBank,Py150,ScienceQA,NumGLUE-cm,NumGLUE-ds,20Minuten}"
EPOCHS="${EPOCHS:-5,4,7,5,4,5,5,7}"
if [ "$(echo "${TASKS}" | tr ',' '\n' | wc -l)" != "$(echo "${EPOCHS}" | tr ',' '\n' | wc -l)" ]; then
    echo "FATAL: TASKS and EPOCHS have different lengths." >&2
    exit 1
fi
DEFAULT_THRESHOLD="${ROUTING_THRESHOLD:-${P_DELTA}}"
DEFAULT_MIN_DKG="${ROUTING_MIN_DKG:-0.01}"
BENCH_DIR="${DATA_DIR}/LLM-CL-Benchmark_${BENCH}"
# Reported runs use test.json; any sweep that chooses a setting must use EVAL_FILE=eval.json.
EVAL_FILE="${EVAL_FILE:-test.json}"
# Continue a stopped run: RESUME_FROM=<output dir of that run> (holds resume_state.pt).
RESUME_FROM="${RESUME_FROM:-}"
if [ -n "${RESUME_FROM}" ] && [ ! -f "${RESUME_FROM}/resume_state.pt" ]; then
    echo "FATAL: ${RESUME_FROM}/resume_state.pt not found." >&2
    exit 1
fi
# Micro-batch (BS) and gradient accumulation (GAS). BS x GAS must stay 4.
BS="${BS:-${P_BS}}"
GAS="${GAS:-${P_GAS}}"
if [ $((BS * GAS)) -ne 4 ]; then
    echo "FATAL: BS x GAS = $((BS * GAS)); it must be 4." >&2
    exit 1
fi

# KL-regulariser switches. Defaults are the current method.
#   KL_PRED_MASK=1  KL at the positions that predict answer tokens (--kl_pred_mask)
#   KL_NORMALIZE=1  penalty is beta * KL / running mean of KL (--kl_normalize_kl)
KL_PRED_MASK="${KL_PRED_MASK:-1}"
KL_NORMALIZE="${KL_NORMALIZE:-0}"
KL_BETA="${KL_BETA:-${P_BETA}}"
KL_BETA_DEFAULT="${KL_BETA}"
GROUP_CAPACITY_DEFAULT="${GROUP_CAPACITY:-${P_CAP}}"
MERGE_KL_HALFLIFE_DEFAULT="${MERGE_KL_HALFLIFE:-${P_H}}"
KL_PRED_FLAG="";  [ "${KL_PRED_MASK}" = "1" ] && KL_PRED_FLAG="--kl_pred_mask"
KL_NORM_FLAG="";  [ "${KL_NORMALIZE}" = "1" ] && KL_NORM_FLAG="--kl_normalize_kl"
# Non-default KL settings are added to the job name so arms never share a folder.
KL_TAG=""
if [ "${KL_PRED_MASK}${KL_NORMALIZE}" != "10" ] || [ "${KL_BETA}" != "${P_BETA}" ]; then
    KL_TAG="_pm${KL_PRED_MASK}_nm${KL_NORMALIZE}_b${KL_BETA}"
fi
EVAL_TAG=""; [ "${EVAL_FILE}" = "eval.json" ] && EVAL_TAG="_val"
# EVAL_MODE=together  train and evaluate in one job (default)
# EVAL_MODE=train     train only, a snapshot after every task; evaluate later with
#                     scripts/eval_later.sh (job names end in _tronly)
# SAVE_SNAPSHOTS=1    in a together run, also keep the snapshots for a later re-evaluation
EVAL_MODE="${EVAL_MODE:-together}"
case "${EVAL_MODE}" in
  together) MODE_FLAGS=""; MODE_TAG=""
            [ "${SAVE_SNAPSHOTS:-0}" = "1" ] && MODE_FLAGS="--save_eval_snapshots" ;;
  train)    MODE_FLAGS="--skip_eval"; MODE_TAG="_tronly" ;;
  *)        echo "FATAL: EVAL_MODE must be 'together' or 'train' (got '${EVAL_MODE}')." >&2; exit 1 ;;
esac
TRAIN_SCRIPT="${STUDY_DIR}/train.py"

# Task-type partition of TRACE-8, used by the fixed-partition arms.
TYPE_MAP="C-STANCE:cls,FOMC:cls,ScienceQA:cls,MeetingBank:gen,20Minuten:gen,Py150:code,NumGLUE-cm:math,NumGLUE-ds:math"
declare -A ARM_ENV=(
  ["v20260828"]=""
  ["taskwise"]="TASKWISE=1"
  ["allinone"]="ALLINONE=1"
  # routing direction
  ["dir_forward"]="ROUTING_DIRECTION=forward"
  # private stream (shared = no private deep l-stream)
  ["shared"]="ALL_GROUPWISE=1"
  ["shared_fixed"]="ALL_GROUPWISE=1 MANUAL_GROUPS=\${TYPE_MAP}"
  ["private_fixed"]="MANUAL_GROUPS=\${TYPE_MAP}"
  # merge rule
  ["merge_overwrite"]="MERGE_RULE=overwrite"
  ["merge_h0.2"]="MERGE_KL_HALFLIFE=0.2"
  ["merge_h2.0"]="MERGE_KL_HALFLIFE=2.0"
  ["merge_h5.0"]="MERGE_KL_HALFLIFE=5.0"
  # routing threshold
  ["delta_0.3"]="ROUTING_THRESHOLD=0.3"
  ["delta_0.5"]="ROUTING_THRESHOLD=0.5"
  ["delta_0.7"]="ROUTING_THRESHOLD=0.7"
  ["delta_1.0"]="ROUTING_THRESHOLD=1.0"
  ["delta_1.5"]="ROUTING_THRESHOLD=1.5"
  ["delta_2.0"]="ROUTING_THRESHOLD=2.0"
  ["delta_3.0"]="ROUTING_THRESHOLD=3.0"
  # group capacity
  ["cap_2"]="GROUP_CAPACITY=2"
  ["cap_3"]="GROUP_CAPACITY=3"
  ["cap_4"]="GROUP_CAPACITY=4"
  # intervention positions per stream (default 15 = f15 + l15)
  ["pos_9"]="N_POSITIONS=9"
  ["pos_21"]="N_POSITIONS=21"
  # routing warm-up steps (default 100)
  ["wu_50"]="WARMUP_STEPS=50"
  ["wu_200"]="WARMUP_STEPS=200"
  ["wu_400"]="WARMUP_STEPS=400"
  # anchor strength
  ["beta_0.1"]="KL_BETA=0.1"
  ["beta_1.0"]="KL_BETA=1.0"
  ["beta_3.0"]="KL_BETA=3.0"
  # beta x h grid at the KL scale of larger-vocabulary backbones
  ["b0.3_h1.0"]="MERGE_KL_HALFLIFE=1.0"
  ["b0.2_h0.6"]="KL_BETA=0.2"
  ["b0.2_h1.0"]="KL_BETA=0.2 MERGE_KL_HALFLIFE=1.0"
  # regularizer table: routed CRAFT without KL, and with A-GEM
  ["noreg"]="KL_OFF=1"
  ["agem"]="KL_OFF=1 AGEM=1"
)
ALL_ARMS="v20260828 taskwise allinone"
# All-in-One regulariser: 1 = KL anchor to the previous shared state, 0 = --no_kl.
ALLINONE_KL="${ALLINONE_KL:-1}"

ARMS="${ARMS:-v20260828}"
[ "${ARMS}" = "all" ] && ARMS="${ALL_ARMS}"
SEEDS="${SEEDS:-42}"
MAXPAR="${MAXPAR:-2}"
DEP_TYPE="${DEP_TYPE:-afterany}"

# Eval scope. Empty (default) = diagonal during the stream plus one final row.
# --per_task_bwd_eval gives the full lower triangle.
BWD_EVAL_FLAG="${BWD_EVAL_FLAG:-}"

# Trainable router head. ROUTER_HEAD=0 disables it.
ROUTER_HEAD="${ROUTER_HEAD:-1}"
ROUTER_HEAD_SAMPLES="${ROUTER_HEAD_SAMPLES:-64}"
ROUTER_HEAD_L2="${ROUTER_HEAD_L2:-1e-2}"
# Routers evaluated in the same job. The first (oracle) drives the stream:
# diagonal and main OP/BWT. Each other one (litar, learned_head) replays the
# final row at the end of the job on the same stored states and reports its own OP.
INFER_ROUTER="${INFER_ROUTER:-${P_ROUTERS}}"
PRIMARY_ROUTER="${INFER_ROUTER%% *}"
EXTRA_ROUTERS=""
[ "${INFER_ROUTER}" != "${PRIMARY_ROUTER}" ] && EXTRA_ROUTERS="${INFER_ROUTER#* }"
EXTRA_ROUTER_FLAGS="${EXTRA_ROUTERS:+--extra_eval_routers ${EXTRA_ROUTERS}}"
HEAD_FLAGS=""
if [ "${ROUTER_HEAD}" = "1" ]; then
    HEAD_FLAGS="--router_head --router_head_samples ${ROUTER_HEAD_SAMPLES} --router_head_l2 ${ROUTER_HEAD_L2} --router_head_prompt_only"
fi
case " ${INFER_ROUTER,,} " in
  *" learned_head "*)
    if [ "${ROUTER_HEAD}" != "1" ]; then
        echo "FATAL: INFER_ROUTER includes learned_head but ROUTER_HEAD=${ROUTER_HEAD}." >&2
        exit 1
    fi ;;
esac

# Per-task overrides for the long-output tasks (LONG=1). Only these three tasks change.
LONG="${LONG:-${P_LONG}}"
PER_TASK_MAXLEN="${PER_TASK_MAXLEN:-MeetingBank:2048,20Minuten:1536,Py150:2048}"
PER_TASK_POS="${PER_TASK_POS:-MeetingBank:f15+l30,20Minuten:f15+l30,Py150:f15+l30}"
LONG_FLAGS=""
[ "${LONG}" = "1" ] && LONG_FLAGS="--per_task_max_length ${PER_TASK_MAXLEN} --per_task_positions ${PER_TASK_POS}"

# ---------------------------------------------------------------------------
# Preflight: refuse to run against stale code, since the old defects produced
# plausible numbers for the wrong configuration.
# ---------------------------------------------------------------------------
for f in train.py utils/merge_rules.py utils/agnostic_router.py utils/router_head.py utils/litar_router.py; do
    if [ ! -f "${STUDY_DIR}/${f}" ]; then
        echo "FATAL: ${STUDY_DIR}/${f} not found." >&2
        exit 1
    fi
done
for flag in router_head router_head_samples per_task_max_length per_task_positions extra_eval_routers \
            skip_eval eval_only save_eval_snapshots kl_routing_direction no_all_groupwise \
            merge_kl_halflife kl_beta_warmup_steps kl_routing_normalize manual_groups \
            kl_pred_mask merge_kl_source cl_eval_file router_head_prompt_only; do
    if ! grep -q -- "--${flag}" "${STUDY_DIR}/train.py"; then
        echo "FATAL: ${STUDY_DIR}/train.py has no --${flag}; not the current train.py." >&2
        exit 1
    fi
done
if grep -q -- '"--position"' "${STUDY_DIR}/train.py"; then
    echo "FATAL: ${STUDY_DIR}/train.py still defines the dead --position flag." >&2
    exit 1
fi
if grep -q "labels_slice != -100" "${STUDY_DIR}/train.py"; then
    echo "FATAL: ${STUDY_DIR}/train.py still has an unshifted routing mask." >&2
    exit 1
fi
if ! grep -q "learned_head" "${STUDY_DIR}/utils/agnostic_router.py" \
   || ! grep -q "ROUTER_STRICT" "${STUDY_DIR}/utils/agnostic_router.py" \
   || ! grep -q "prompt_only" "${STUDY_DIR}/utils/router_head.py"; then
    echo "FATAL: utils/agnostic_router.py or utils/router_head.py is not the current copy." >&2
    exit 1
fi
if ! grep -q "_remap_to_state_keys" "${STUDY_DIR}/utils/merge_rules.py"; then
    echo "FATAL: utils/merge_rules.py predates the Fisher key fix (merge_rule=fisher would be a silent overwrite)." >&2
    exit 1
fi
if [ "${MODEL_TAG}" = "mistral_7b" ] && ! grep -q "_is_mistral" "${STUDY_DIR}/train.py"; then
    echo "FATAL: ${STUDY_DIR}/train.py has no Mistral pad-token branch (_is_mistral)." >&2
    exit 1
fi
if [[ "${MODEL_TAG}" == qwen2_5_* ]] && ! grep -q "_n_special" "${STUDY_DIR}/train.py"; then
    echo "FATAL: ${STUDY_DIR}/train.py lacks the BOS-aware _find_question_start." >&2
    exit 1
fi
if [ ! -f "$HOME/.hf_token" ]; then
    echo "FATAL: ~/.hf_token not found." >&2
    exit 1
fi
if [ ! -d "${BENCH_DIR}" ]; then
    echo "FATAL: benchmark not found at ${BENCH_DIR} (set DATA_DIR)." >&2
    exit 1
fi
for t in ${TASKS//,/ }; do
    if [ ! -f "${BENCH_DIR}/${t}/${EVAL_FILE}" ]; then
        echo "FATAL: ${BENCH_DIR}/${t}/${EVAL_FILE} not found." >&2
        exit 1
    fi
done

# Job set-up shared by every run (module, venv, HF login, GPU check).
PYVENE_CHECK=""
if [ -n "${PYVENE_CLASS}" ]; then
    PYVENE_CHECK="if ! python -c \"import sys, pyvene.models.intervenable_modelcard as m, transformers.models.${PYVENE_CLASS%.*} as mm; sys.exit(0 if mm.${PYVENE_CLASS##*.} in m.type_to_module_mapping else 1)\"; then
    echo \"FATAL: installed pyvene has no ${PYVENE_CLASS##*.} mapping.\" >&2
    exit 4
fi"
fi
# #SBATCH lines that depend on the cluster; left out when not set.
SB_SITE=""
[ -n "${ACCOUNT}" ]   && SB_SITE+="#SBATCH --account=${ACCOUNT}"$'\n'
[ -n "${PARTITION}" ] && SB_SITE+="#SBATCH --partition=${PARTITION}"$'\n'
SB_EXCL=""; [ -n "${EXCLUDE_NODES}" ] && SB_EXCL="#SBATCH --exclude=${EXCLUDE_NODES}"$'\n'
MODULE_LINE=""; [ -n "${CUDA_MODULE}" ]   && MODULE_LINE="module load ${CUDA_MODULE}"
VENV_LINE="";   [ -n "${VENV_ACTIVATE}" ] && VENV_LINE="source \"${VENV_ACTIVATE}\""
IFS= read -r -d '' SETUP <<EOF || true
${MODULE_LINE}
${VENV_LINE}
export PYTHONPATH="${ROOT_DIR}:\${PYTHONPATH}"
if [ -f "\$HOME/.hf_token" ]; then
    export HF_TOKEN="\$(cat "\$HOME/.hf_token")"
fi
: "\${HF_TOKEN:?HF_TOKEN is not set.}"
huggingface-cli login --token "\$HF_TOKEN" >/dev/null 2>&1 || true
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export ROUTER_STRICT=1             # a router that cannot decide fails; no oracle fallback
export KL_OOM_FATAL=1              # an OOM in the KL step stops the job instead of dropping the KL term
# GPU sanity check: fail fast with the node name so it can be excluded.
echo "  Node     : \$(hostname)   CUDA_VISIBLE_DEVICES=\${CUDA_VISIBLE_DEVICES:-<unset>}"
if ! python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)"; then
    echo "FATAL: torch sees no CUDA device on \$(hostname)." >&2
    exit 3
fi
${PYVENE_CHECK}
EOF

declare -a JIDS; N=0

for ARM in ${ARMS}; do
  ENVSET="${ARM_ENV[$ARM]}"
  if [ -z "${ARM_ENV[$ARM]+x}" ]; then      # a key that is not in ARM_ENV (a typo) is skipped
      echo "skip: unknown arm '${ARM}'" >&2; continue
  fi

  for SEED in ${SEEDS}; do
   for LR in ${LR_LIST}; do
    for RANK in ${RANK_LIST}; do

    # ---- resolve this arm's configuration ------------------------------
    N_POSITIONS=15; F_POSITIONS=""; L_POSITIONS=""
    ROUTING_DIRECTION="backward"
    ROUTING_NORMALIZE="ratio"
    ROUTING_THRESHOLD="${DEFAULT_THRESHOLD}"
    ROUTING_MIN_DKG="${DEFAULT_MIN_DKG}"
    MERGE_RULE="kl"; MERGE_KL_HALFLIFE="${MERGE_KL_HALFLIFE_DEFAULT}"; MERGE_FISHER_BATCHES="8"
    ALL_GROUPWISE="0"
    BETA_WARMUP_STEPS="0"
    WARMUP_STEPS="100"
    TASKWISE="0"
    ALLINONE="0"
    MANUAL_GROUPS=""
    KL_OFF="0"
    AGEM="0"
    KL_BETA="${KL_BETA_DEFAULT}"
    GROUP_CAPACITY="${GROUP_CAPACITY_DEFAULT}"
    eval "${ENVSET}"

    KL_ON_FLAGS="--use_kl \
  --kl_beta ${KL_BETA} \
  --kl_beta_warmup_steps ${BETA_WARMUP_STEPS} \
  --no_kl_early_stop \
  --kl_rolling_window 20 \
  --kl_eval_batches 50 \
  ${KL_NORM_FLAG} \
  --kl_mode output"
    MERGE_FLAGS="--merge_rule ${MERGE_RULE} \
  --merge_kl_halflife ${MERGE_KL_HALFLIFE} \
  --merge_fisher_batches ${MERGE_FISHER_BATCHES}"

    # Task-wise isolation: a unique group per task via manual routing, and no
    # KL anchor since there is never a prior group to anchor to.
    if [ "${TASKWISE}" = "1" ]; then
        MANUAL_MAP=""
        for t in ${TASKS//,/ }; do
            MANUAL_MAP="${MANUAL_MAP:+${MANUAL_MAP},}${t}:solo_${t}"
        done
        ROUTER_FLAGS="--router_method manual --manual_routing --manual_groups ${MANUAL_MAP}"
        KL_FLAGS=""
        MERGE_FLAGS=""
        GROUPWISE_FLAG="--no_all_groupwise"
        GROUPWISE_TAG="task-wise isolation (no sharing)"
        ROUTE_TAG="manual (solo per task)"
        ARM_TAG="TASKWISE"
    # All-in-One: every task routed to the same group by name, all tiers
    # shared (no private T3-l), no routing. The KL anchor applies to tasks
    # 2..T (anchor = previous shared state) unless ALLINONE_KL=0.
    elif [ "${ALLINONE}" = "1" ]; then
        MANUAL_MAP=""
        for t in ${TASKS//,/ }; do
            MANUAL_MAP="${MANUAL_MAP:+${MANUAL_MAP},}${t}:shared"
        done
        ROUTER_FLAGS="--router_method manual --manual_routing --manual_groups ${MANUAL_MAP}"
        if [ "${ALLINONE_KL}" = "1" ]; then
            KL_FLAGS="${KL_ON_FLAGS}"; AIO_REG="KL"
        else
            KL_FLAGS="--no_kl"; AIO_REG="noreg"
        fi
        GROUPWISE_FLAG="--all_groupwise"
        GROUPWISE_TAG="all-in-one (single shared intervention, no private params)"
        ROUTE_TAG="manual (all tasks -> one group)  regulariser=${AIO_REG}"
        ARM_TAG="ALLINONE_${AIO_REG}"
    else
        ARM_TAG="KLROUTING"
        if [ -n "${MANUAL_GROUPS}" ]; then
          ROUTER_FLAGS="--router_method manual --manual_routing --manual_groups ${MANUAL_GROUPS}"
          ARM_TAG="FIXED_PARTITION"
        else
        ROUTER_FLAGS="--router_method kl_routing \
  --kl_routing_direction ${ROUTING_DIRECTION} \
  --kl_routing_threshold ${ROUTING_THRESHOLD} \
  --kl_routing_normalize ${ROUTING_NORMALIZE} \
  --kl_routing_min_dkg ${ROUTING_MIN_DKG} \
  --kl_routing_top_k 512 \
  --kl_routing_temperature 2.0 \
  --kl_routing_probe_size 16 \
  --group_capacity ${GROUP_CAPACITY}"
        fi
        KL_FLAGS="${KL_ON_FLAGS}"
        if [ "${KL_OFF}" = "1" ]; then
          KL_FLAGS="--no_kl"
          [ "${AGEM}" = "1" ] && KL_FLAGS="--no_kl --use_agem"
        fi
        ROUTE_TAG="kl_routing dir=${ROUTING_DIRECTION} thr=${ROUTING_THRESHOLD} floor=${ROUTING_MIN_DKG} cap=${GROUP_CAPACITY}"
    fi

    if [ -n "${F_POSITIONS}" ] && [ -n "${L_POSITIONS}" ]; then
        POSITION_FLAGS="--f_positions ${F_POSITIONS} --l_positions ${L_POSITIONS}"
        POSITION_TAG="f${F_POSITIONS}+l${L_POSITIONS}"
    else
        POSITION_FLAGS="--n_positions ${N_POSITIONS}"
        POSITION_TAG="f${N_POSITIONS}+l${N_POSITIONS}"
    fi
    if [ "${TASKWISE}" != "1" ] && [ "${ALLINONE}" != "1" ]; then
      if [ "${ALL_GROUPWISE}" = "1" ]; then
        GROUPWISE_FLAG="--all_groupwise";    GROUPWISE_TAG="shared (no private params)"
      else
        GROUPWISE_FLAG="--no_all_groupwise"; GROUPWISE_TAG="private T3-l retained"
      fi
    fi

    ARM_LABEL="${ARM}"
    TS=$(date +%Y_%m_%d_%H_%M_%S)
    [ "${ALLINONE}" = "1" ] && ARM_LABEL="allinone_${AIO_REG}"
    CAP_TAG=""; [ "${GROUP_CAPACITY}" != "3" ] && [ "${ARM}" != "cap_${GROUP_CAPACITY}" ] && CAP_TAG="_cap${GROUP_CAPACITY}"
    JOB_NAME="RTR_${NAME_TAG}${ARM_LABEL}_${BACKBONE_TAG}_s${SEED}_lr${LR}_r${RANK}_b${BENCH}${LONG_FLAGS:+_long}${HEAD_FLAGS:+_head}${CAP_TAG}${KL_TAG}${EVAL_TAG}${MODE_TAG}_${TS}"

    if [ -n "${DRY}" ]; then
      echo "[dry] ${JOB_NAME}   <<< ${ARM_TAG}"
      echo "        model=${MODEL_ID}  tasks=${TASKS}  epochs=${EPOCHS}"
      echo "        pos=${POSITION_TAG}  ${ROUTE_TAG}  beta=${KL_BETA}  h=${MERGE_KL_HALFLIFE}"
      echo "        ${GROUPWISE_TAG}  bwd_eval='${BWD_EVAL_FLAG:-diagonal+final}'"
      [ -n "${LONG_FLAGS}" ] && echo "        long: maxlen=${PER_TASK_MAXLEN}  pos=${PER_TASK_POS}"
      [ -z "${PACK_LIST}" ] && echo "        sbatch: partition=${PARTITION}  account=${ACCOUNT}  time=${WALLTIME}  mem=${MEM}"
      echo "        router_head=${ROUTER_HEAD} (samples=${ROUTER_HEAD_SAMPLES})  eval routers=${INFER_ROUTER}"
      [ -n "${PACK_LIST}" ] && echo "${JOB_NAME}" >> "${PACK_LIST}"
      continue
    fi

    # A run started inside a packed job (run_streams.sh orders) logs to <job>.txt.
    if [ -n "${PACK_LIST}" ]; then LOGF="${LOG_ROOT}/${JOB_NAME}.txt"
    else LOGF="${LOG_ROOT}/${JOB_NAME}_\${SLURM_JOB_ID}.txt"; fi

IFS= read -r -d '' RUN <<EOF || true
cd "${STUDY_DIR}" || exit 1
OUTPUT_DIR="${OUT_ROOT}/${JOB_NAME}"
mkdir -p "\${OUTPUT_DIR}"

echo "===================================================================="
echo "  Study    : CRAFT (v${VERSION} + router head)"
echo "  Arm      : ${ARM_LABEL}   [${ARM_TAG}]      seed: ${SEED}"
echo "  Model    : ${MODEL_ID}"
echo "  Tasks    : ${TASKS}"
echo "  Epochs   : ${EPOCHS}"
echo "  Sweep    : lr=${LR}   rank=${RANK}   bench=${BENCH}/task   bs=${BS} gas=${GAS}"
echo "  Positions: ${POSITION_TAG}"
echo "  Routing  : ${ROUTE_TAG}"
echo "  KL       : beta=${KL_BETA}  pred_mask=${KL_PRED_MASK}  normalize=${KL_NORMALIZE}  eviction=off"
echo "  Storage  : ${GROUPWISE_TAG}"
echo "  BWD eval : ${BWD_EVAL_FLAG:-diagonal + final row only}"
echo "  Eval mode: ${EVAL_MODE}${MODE_FLAGS:+  (${MODE_FLAGS})}"
echo "  Long     : ${LONG_FLAGS:-off (all tasks at --max_length ${MAX_LENGTH}, ${POSITION_TAG})}"
echo "  Head     : ${HEAD_FLAGS:-off}"
echo "  Eval rtr : ${INFER_ROUTER}   eval file: ${EVAL_FILE}"
echo "  train.py : ${TRAIN_SCRIPT}"
echo "  Output   : \${OUTPUT_DIR}"
echo "  Started  : \$(date)"
echo "===================================================================="

python -u "${TRAIN_SCRIPT}" \
  --continual_learning \
  --benchmark standard_cl \
  --cl_benchmark_dir "${BENCH_DIR}" \
  --cl_tasks "${TASKS}" \
  --per_task_epochs "${EPOCHS}" \
  --model ${MODEL_ID} \
  --layers all \
  --rank ${RANK} \
  ${POSITION_FLAGS} \
  --intervention_type LoreftIntervention \
  --batch_size ${BS} \
  --gradient_accumulation_steps ${GAS} \
  --eval_batch_size ${EVAL_BS} \
  --lr ${LR} \
  --warmup_ratio 0.05 \
  --weight_decay 0.00 \
  --dropout 0.00 \
  --max_length ${MAX_LENGTH} \
  --logging_steps 10 \
  --output_dir "\${OUTPUT_DIR}" \
  --dtype bfloat16 \
  --seed ${SEED} \
  --shallow_frac 0.33 \
  --deep_frac 0.38 \
  --causal_warmup_steps ${WARMUP_STEPS} \
  --fixed_intervention_seed 42 \
  --tier1_alpha 0.30 \
  --group_alpha 0.50 \
  ${ROUTER_FLAGS} \
  ${KL_FLAGS} \
  ${MERGE_FLAGS} \
  ${GROUPWISE_FLAG} \
  ${BWD_EVAL_FLAG} \
  ${LONG_FLAGS} \
  ${HEAD_FLAGS} \
  ${KL_PRED_FLAG} \
  --merge_kl_source train \
  --cl_eval_file ${EVAL_FILE} \
  ${MODE_FLAGS} \
  ${RESUME_FROM:+--resume_from ${RESUME_FROM}} \
  --inference_router ${PRIMARY_ROUTER} \
  ${EXTRA_ROUTER_FLAGS} \
  ${EXTRA_TRAIN_FLAGS}

EXIT_CODE=\$?
echo ""
echo "  Finished at \$(date) with exit code \$EXIT_CODE"
echo "  Arm type : ${ARM_TAG}   (${GROUPWISE_TAG})"
LOGF="${LOGF}"
if grep -q "Position: ${POSITION_TAG}" "\${LOGF}" 2>/dev/null; then
    echo "  Position : verified ${POSITION_TAG}"
else
    echo "  WARNING  : expected 'Position: ${POSITION_TAG}' not found in \${LOGF}"
fi
if [ -n "${HEAD_FLAGS}" ]; then
    NREFIT=\$(grep -c "\[router_head\] refit after" "\${LOGF}" 2>/dev/null); NREFIT=\${NREFIT:-0}
    echo "  Head     : \${NREFIT} refits logged (expect one per task)"
    grep "\[router_head\] refit after" "\${LOGF}" 2>/dev/null | tail -1
fi
exit \$EXIT_CODE
EOF

    # Packed mode (run_streams.sh): write the run, the packed GPU job starts it.
    if [ -n "${PACK_LIST}" ]; then
        printf '#!/bin/bash\n%s\n' "${RUN}" > "${LOG_ROOT}/${JOB_NAME}.run.sh"
        printf '%s\n' "${SETUP}" > "${PACK_LIST}.setup"
        echo "${JOB_NAME}" >> "${PACK_LIST}"
        continue
    fi

    DEP=""
    if [ "${MAXPAR}" -gt 0 ] && [ $((N - MAXPAR)) -ge 0 ]; then
        PREV="${JIDS[$((N - MAXPAR))]}"
        [ -n "${PREV}" ] && DEP="--dependency=${DEP_TYPE}:${PREV}"
    fi

JID=$(sbatch --parsable ${DEP} <<EOF
#!/bin/bash
${SB_SITE}#SBATCH --time=${WALLTIME}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=6
${SB_EXCL}#SBATCH --mem=${MEM}
#SBATCH --gres=gpu:a100:1
#SBATCH --job-name=${JOB_NAME}
#SBATCH --output=${LOG_ROOT}/${JOB_NAME}_%j.txt
#SBATCH --error=${LOG_ROOT}/${JOB_NAME}_%j.err

cd "${STUDY_DIR}" || exit 1
${SETUP}

${RUN}
EOF
)
    if [ -z "${JID}" ]; then
        echo "FATAL: sbatch rejected ${JOB_NAME}; nothing was submitted." >&2
        exit 1
    fi
    JIDS[$N]="${JID}"
    N=$((N + 1))
    echo "Submitted: ${JOB_NAME}  jobid=${JID}  [${PARTITION}/${ACCOUNT}]${DEP:+  (${DEP})}"
    done
   done
  done
done

if [ -z "${PACK_LIST}" ]; then
    echo
    echo "Logs:    ${LOG_ROOT}"
    echo "Outputs: ${OUT_ROOT}"
    echo "Results: python scripts/collect_results.py --roots output_log --out results"
fi
