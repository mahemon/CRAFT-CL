#!/bin/bash
# ============================================================================
# run_streams.sh -- CRAFT on the other task streams. Same method and settings as
# scripts/run_craft.sh, which builds every run; only the task stream changes.
#
#   STREAM=orders  bash scripts/run_streams.sh   # O-LoRA task orders 1-6 (Llama-3.2-1B, 500/task)
#   STREAM=trace15 bash scripts/run_streams.sh   # TRACE-15 long sequence (Llama-3.2-1B, 500/task)
#   STREAM=math    bash scripts/run_streams.sh   # ScienceQA, NumGLUE-cm, NumGLUE-ds (Llama-2-7B, 5000/task)
#
#   DRY=1 STREAM=orders bash scripts/run_streams.sh                   # print, submit nothing
#   PARTITION=gpu STREAM=math SEEDS=1234 bash scripts/run_streams.sh  # another partition
#   STREAM=orders JOB_GROUPS="1,2,3" bash scripts/run_streams.sh      # orders 1-3 only
#   STREAM=orders JOB_GROUPS="4 5 6" bash scripts/run_streams.sh      # one GPU job per order
#   STREAM=trace15 GROUP_CAPACITY=0 INFER_ROUTER="oracle litar learned_head" bash scripts/run_streams.sh
#
# Orders 1-3 are the 4-task standard CL benchmark, orders 4-6 the 15-task long
# sequence (O-LoRA, Wang et al., 2023, Table 7). Orders are packed into GPU jobs:
# each group in JOB_GROUPS is ONE SLURM job on one A100 that runs its orders as
# parallel processes (default: orders 1-3 in one job, orders 4-6 in another).
#
# MODEL= picks another backbone for any stream, and every run_craft.sh setting
# (SEEDS, ARMS, GROUP_CAPACITY, INFER_ROUTER, PARTITION, WALLTIME, ...) works here too.
# ============================================================================
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Site settings (account, partition, venv, CUDA module): scripts/cluster.env if present.
[ -f "${HERE}/cluster.env" ] && source "${HERE}/cluster.env"
is_short() { case " ${SHORT_PARTITIONS:-} " in *" ${PARTITION:-none} "*) return 0 ;; esac; return 1; }
STREAM="${STREAM:-${1:-}}"

case "${STREAM}" in
  trace15)
    export MODEL="${MODEL:-llama3_2_1b}"
    export BENCH="${BENCH:-500}"
    export TASKS="${TASKS:-C-STANCE,FOMC,MeetingBank,Py150,ScienceQA,NumGLUE-cm,NumGLUE-ds,20Minuten,dbpedia,amazon,yahoo,agnews,yelp,BoolQA,QQP}"
    export EPOCHS="${EPOCHS:-5,4,7,5,4,5,5,7,5,5,5,5,5,5,5}"
    exec bash "${HERE}/run_craft.sh"
    ;;
  math)
    export MODEL="${MODEL:-llama2_7b}"
    export STREAM_TAG="mathllm_"
    export TASKS="${TASKS:-ScienceQA,NumGLUE-cm,NumGLUE-ds}"
    export EPOCHS="${EPOCHS:-4,5,5}"            # the TRACE epochs of these three tasks
    export GROUP_CAPACITY="${GROUP_CAPACITY:-2}"
    export MAX_LENGTH="${MAX_LENGTH:-2048}"
    is_short && export WALLTIME="${WALLTIME:-23:00:00}"
    exec bash "${HERE}/run_craft.sh"
    ;;
  orders) ;;
  *) sed -n '2,24p' "$0"; exit 1 ;;
esac

# ---- orders: one packed GPU job per group in JOB_GROUPS -----------------------
export MODEL="${MODEL:-llama3_2_1b}"
export BENCH="${BENCH:-500}"
export STREAM_TAG="olora_"
export GROUP_CAPACITY="${GROUP_CAPACITY:-2}"
export LONG="${LONG:-0}"
declare -A ORDER_TASKS=(
  [1]="dbpedia,amazon,yahoo,agnews"
  [2]="dbpedia,amazon,agnews,yahoo"
  [3]="yahoo,amazon,agnews,dbpedia"
  [4]="MNLI,CB,WiC,COPA,QQP,BoolQA,RTE,IMDB,yelp,amazon,SST-2,dbpedia,agnews,MultiRC,yahoo"
  [5]="MultiRC,BoolQA,WiC,MNLI,CB,COPA,QQP,RTE,IMDB,SST-2,dbpedia,agnews,yelp,amazon,yahoo"
  [6]="yelp,amazon,MNLI,CB,COPA,QQP,RTE,IMDB,SST-2,dbpedia,agnews,yahoo,MultiRC,BoolQA,WiC"
)
JOB_GROUPS="${JOB_GROUPS:-1,2,3 4,5,6}"
EPOCHS_PER_TASK="${EPOCHS_PER_TASK:-5}"
for o in ${JOB_GROUPS//,/ }; do
    [ -n "${ORDER_TASKS[$o]+x}" ] || { echo "FATAL: unknown order '${o}' (known: 1 2 3 4 5 6)." >&2; exit 1; }
done

STUDY_DIR="$(dirname "${HERE}")"
LOG_ROOT="${STUDY_DIR}/output_log/logs_${BENCH}_olora_router_${MODEL}"
mkdir -p "${LOG_ROOT}"
if is_short; then GWALL="${WALLTIME:-18:00:00}"; else GWALL="${WALLTIME:-23:00:00}"; fi
SB_SITE=""
[ -n "${ACCOUNT:-}" ]   && SB_SITE+="#SBATCH --account=${ACCOUNT}"$'\n'
[ -n "${PARTITION:-}" ] && SB_SITE+="#SBATCH --partition=${PARTITION}"$'\n'
SB_EXCL=""; [ -n "${EXCLUDE_NODES:-}" ] && SB_EXCL="#SBATCH --exclude=${EXCLUDE_NODES}"$'\n'
GMEM="${MEM:-192G}"
ARMS_ALL="${ARMS:-v20260828}"
SEEDS_ALL="${SEEDS:-42}"
MAXPAR="${MAXPAR:-2}"
DEP_TYPE="${DEP_TYPE:-afterany}"

declare -a JIDS; N=0
for ARM in ${ARMS_ALL}; do
 for SEED in ${SEEDS_ALL}; do
  for GROUP in ${JOB_GROUPS}; do
    TS=$(date +%Y_%m_%d_%H_%M_%S)
    PACK_LIST="${LOG_ROOT}/pack_g${GROUP//,/}_${ARM}_s${SEED}_${TS}.list"; : > "${PACK_LIST}"
    for ORDER in ${GROUP//,/ }; do
        TASKS="${ORDER_TASKS[$ORDER]}"
        EPOCHS=$(echo "${TASKS}" | tr ',' '\n' | sed "s/.*/${EPOCHS_PER_TASK}/" | paste -sd, -)
        TASKS="${TASKS}" EPOCHS="${EPOCHS}" NAME_TAG="o${ORDER}_" ARMS="${ARM}" SEEDS="${SEED}" \
          PACK_LIST="${PACK_LIST}" bash "${HERE}/run_craft.sh" || exit 1
    done
    mapfile -t RUN_NAMES < "${PACK_LIST}"
    # GPU job name: as the runs, e.g. RTR_olora_g123_v20260828_llama3_2_1b_s42_lr2e-4_r8_b500_<ts>
    SWEEP="$(echo "${RUN_NAMES[0]}" | grep -oE '_lr[^_]+_r[0-9]+_b[0-9]+' | head -1)"
    GJOB="RTR_olora_g${GROUP//,/}_${ARM}_${MODEL}_s${SEED}${SWEEP}_${TS}"
    [ -n "${DRY}" ] && echo "[dry] GPU job ${GJOB}  (orders ${GROUP//,/ } share one A100)"
    if [ -n "${DRY}" ] || [ ${#RUN_NAMES[@]} -eq 0 ]; then rm -f "${PACK_LIST}" "${PACK_LIST}.setup"; continue; fi

    DEP=""
    if [ "${MAXPAR}" -gt 0 ] && [ $((N - MAXPAR)) -ge 0 ]; then
        PREV="${JIDS[$((N - MAXPAR))]}"
        [ -n "${PREV}" ] && DEP="--dependency=${DEP_TYPE}:${PREV}"
    fi
    SETUP="$(cat "${PACK_LIST}.setup")"

JID=$(sbatch --parsable ${DEP} <<EOF
#!/bin/bash
${SB_SITE}#SBATCH --time=${GWALL}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=12
${SB_EXCL}#SBATCH --mem=${GMEM}
#SBATCH --gres=gpu:a100:1
#SBATCH --job-name=${GJOB}
#SBATCH --output=${LOG_ROOT}/${GJOB}_%j.txt
#SBATCH --error=${LOG_ROOT}/${GJOB}_%j.err

cd "${STUDY_DIR}" || exit 1
${SETUP}
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4

PIDS=(); NAMES=()
for R in ${RUN_NAMES[*]}; do
    bash "${LOG_ROOT}/\${R}.run.sh" > "${LOG_ROOT}/\${R}.txt" 2> "${LOG_ROOT}/\${R}.err" &
    PIDS+=(\$!); NAMES+=("\${R}")
    echo "  Started  : \${R}  pid=\$!  \$(date)"
    sleep 30
done
FAIL=0
for j in \${!PIDS[@]}; do
    wait \${PIDS[\$j]}; RC=\$?
    echo "  Finished : \${NAMES[\$j]}  exit=\${RC}  \$(date)"
    [ \${RC} -ne 0 ] && FAIL=1
done
exit \${FAIL}
EOF
)
    rm -f "${PACK_LIST}" "${PACK_LIST}.setup"
    if [ -z "${JID}" ]; then
        echo "FATAL: sbatch rejected ${GJOB}; nothing was submitted." >&2
        exit 1
    fi
    JIDS[$N]="${JID}"; N=$((N + 1))
    echo "Submitted: ${GJOB}  jobid=${JID}  [${PARTITION}/${ACCOUNT}]${DEP:+  (${DEP})}"
    for R in "${RUN_NAMES[@]}"; do echo "    run: ${R}"; done
  done
 done
done

echo
echo "Logs:    ${LOG_ROOT}"
echo "Outputs: ${STUDY_DIR}/output_log/outputs_${BENCH}_olora_router_${MODEL}"
