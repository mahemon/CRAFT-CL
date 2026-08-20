#!/bin/bash
# ============================================================================
# CRAFT-CL — ANCHOR DIRECTION ABLATION (ido5 W2)
# Training-time KL penalty direction: forward / reverse / symmetric.
# 15-task stream, Llama-3.2-1B-Instruct, NCSA DELTA.
#
#   forward   : KL(anchor||live)   (paper default, mode-covering, LwF-style)
#   reverse   : KL(live||anchor)   (mode-seeking; the direction implicit
#                                    in the cited RL-forgetting mechanism)
#   symmetric : mean of both
#
# Everything else pinned to the R3 config (ratio routing thr=0.7 floor=0.01,
# KL anchor beta=0.3 normalized, eviction enabled, all-groupwise, per-task
# backward eval, oracle inference, seed 42). The ONLY varied factor is
# --kl_direction. The save-time terminal-KL diagnostic used for the beta
# dose-response study is intentionally NOT changed by this flag (always
# forward), so terminal-KL readings stay comparable across arms.
#
# Submit from a Delta login node:
#   export HF_TOKEN=hf_...
#   bash explore/Anchor_direction/submit_anchor_direction_llama15.sh
# Single arm:
#   ARMS="reverse" bash explore/Anchor_direction/submit_anchor_direction_llama15.sh
# ============================================================================

ROOT_DIR="/projects/bfwa/mhossen1/CRAFT-CL"
VENV="/projects/bdpm/mhossen1/myenv"
CACHE_ROOT="/projects/bdpm/mhossen1"

MODEL="/projects/bdpm/mhossen1/models/Llama-3.2-1B-Instruct"

# VERIFY: must contain all 15 tasks; match whatever the paper's 15-task
# runs used.
BENCH_DIR_REL="dataset/TRACE_Benchmark/LLM-CL-Benchmark_500"
TASKS="C-STANCE,FOMC,MeetingBank,Py150,ScienceQA,NumGLUE-cm,NumGLUE-ds,20Minuten,dbpedia,amazon,yahoo,agnews,yelp,BoolQA,QQP"
EPOCHS2="5,4,7,5,4,5,5,7,5,5,5,5,5,5,5"

ROUTING_NORMALIZE="ratio"
ROUTING_THRESHOLD="0.7"
ROUTING_MIN_DKG="0.01"
BETA="0.3"

EVICT_SEVERITY="1.3"
EVICT_EARLY_STEPS="30"
EVICT_ABSOLUTE_THRESHOLD="1.0"

ALL_ARMS="forward reverse symmetric"
ARMS="${ARMS:-$ALL_ARMS}"

TRAIN_SCRIPT="explore/Anchor_direction/train.py"

LOG_ROOT="${ROOT_DIR}/explore/Anchor_direction/logs/llama15"
mkdir -p "${LOG_ROOT}"

if [ -z "${HF_TOKEN}" ]; then
    echo "FATAL: HF_TOKEN is not set. Run: export HF_TOKEN=hf_..." >&2
    exit 1
fi

submit_job() {
    local lr=$1
    local epochs=$2
    local direction=$3
    local job_name=$4

    sbatch <<EOF
#!/bin/bash
#SBATCH --account=bhot-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --time=1-23:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --job-name=${job_name}
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

ROOT_DIR="${ROOT_DIR}"
cd "\${ROOT_DIR}" || exit 1
source "${VENV}/bin/activate"
export PYTHONPATH="\${ROOT_DIR}:\${PYTHONPATH}"

export XDG_CACHE_HOME="${CACHE_ROOT}/.cache"
export HF_HOME="\${XDG_CACHE_HOME}/huggingface"
export TORCH_HOME="\${XDG_CACHE_HOME}/torch"
export MPLCONFIGDIR="\${XDG_CACHE_HOME}/matplotlib"
export TRITON_CACHE_DIR="\${XDG_CACHE_HOME}/triton"
export NLTK_DATA="\${XDG_CACHE_HOME}/nltk_data"
mkdir -p "\$HF_HOME" "\$TORCH_HOME" "\$MPLCONFIGDIR" "\$TRITON_CACHE_DIR" "\$NLTK_DATA"

if [ -z "\${HF_TOKEN}" ]; then
    echo "FATAL: HF_TOKEN did not reach the job environment." >&2
    exit 1
fi
huggingface-cli login --token "\$HF_TOKEN" >/dev/null 2>&1 || true

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

rank=8
timestamp=\$(date +%F_%H-%M-%S)
out_file="${LOG_ROOT}/llama32_1b_15task_dir-${direction}_job\${SLURM_JOB_ID}_\${timestamp}.txt"
err_file="${LOG_ROOT}/llama32_1b_15task_dir-${direction}_job\${SLURM_JOB_ID}_\${timestamp}.err.txt"
exec > >(tee -a "\$out_file")
exec 2> >(tee -a "\$err_file" >&2)

OUTPUT_DIR="\${ROOT_DIR}/explore/Anchor_direction/outputs/${job_name}_\${timestamp}"
mkdir -p "\${OUTPUT_DIR}"

echo "════════════════════════════════════════════════════════════════════"
echo "  Job: ${job_name}   (SLURM id \${SLURM_JOB_ID}, node \$(hostname))"
echo "  KL DIRECTION : ${direction}"
echo "  Routing  : kl_routing  normalize=${ROUTING_NORMALIZE}  thr=${ROUTING_THRESHOLD}  floor=${ROUTING_MIN_DKG}"
echo "  KL anchor: β = ${BETA}  (normalized KL)"
echo "  Eviction : enabled  severity=${EVICT_SEVERITY}  early_steps=${EVICT_EARLY_STEPS}  abs_thresh=${EVICT_ABSOLUTE_THRESHOLD}"
echo "  Storage  : ALL groupwise (no private T3_l)"
echo "  Stream   : 15-task"
echo "  Output   : \${OUTPUT_DIR}"
echo "  Started  : \$(date)"
echo "════════════════════════════════════════════════════════════════════"

python -u ${TRAIN_SCRIPT} \\
  --continual_learning \\
  --benchmark standard_cl \\
  --cl_benchmark_dir "\${ROOT_DIR}/${BENCH_DIR_REL}" \\
  --cl_tasks "${TASKS}" \\
  --per_task_epochs "${epochs}" \\
  --model ${MODEL} \\
  --layers all \\
  --rank \$rank \\
  --n_positions 15 \\
  --intervention_type LoreftIntervention \\
  --batch_size 4 \\
  --gradient_accumulation_steps 1 \\
  --eval_batch_size 4 \\
  --lr ${lr} \\
  --warmup_ratio 0.05 \\
  --weight_decay 0.00 \\
  --dropout 0.00 \\
  --max_length 1024 \\
  --logging_steps 10 \\
  --output_dir "\${OUTPUT_DIR}" \\
  --dtype bfloat16 \\
  --seed 42 \\
  --shallow_frac 0.33 \\
  --deep_frac 0.38 \\
  \\
  --router_method kl_routing \\
  --kl_routing_threshold ${ROUTING_THRESHOLD} \\
  --kl_routing_normalize ${ROUTING_NORMALIZE} \\
  --kl_routing_min_dkg ${ROUTING_MIN_DKG} \\
  --kl_routing_top_k 64 \\
  --kl_routing_temperature 2.0 \\
  --kl_routing_probe_size 16 \\
  --causal_warmup_steps 100 \\
  --warmup_skip_steps 15 \\
  --fixed_intervention_seed 42 \\
  --tier1_alpha 0.30 \\
  --group_alpha 0.50 \\
  \\
  --use_kl \\
  --kl_beta ${BETA} \\
  --kl_early_stop \\
  --kl_rolling_window 20 \\
  --kl_eval_batches 50 \\
  --kl_normalize_kl \\
  --kl_mode output \\
  \\
  `# ── Anchor direction arm ───────────────────────────────────────────` \\
  --kl_direction ${direction} \\
  \\
  --kl_eviction_enabled \\
  --kl_eviction_severity_threshold ${EVICT_SEVERITY} \\
  --kl_eviction_early_steps ${EVICT_EARLY_STEPS} \\
  --kl_evict_absolute_threshold ${EVICT_ABSOLUTE_THRESHOLD} \\
  \\
  --all_groupwise \\
  \\
  --per_task_bwd_eval \\
  \\
  --inference_router oracle \\
  --router_probe_size 16

EXIT_CODE=\$?
echo ""
echo "════════════════════════════════════════════════════════════════════"
echo "  Finished at \$(date) with exit code \$EXIT_CODE"
echo "  Results: \${OUTPUT_DIR}/cl_results.json"
echo "════════════════════════════════════════════════════════════════════"
EOF
}

for DIR in ${ARMS}; do
    JOB_NAME="CRAFT_CL_direction_${DIR}_llama32_1b_15task"
    echo "Submitting kl_direction=${DIR}  as job ${JOB_NAME}"
    submit_job "2e-4" "$EPOCHS2" "$DIR" "$JOB_NAME"
done

echo ""
echo "Submitted anchor-direction arms on the 15-task stream (Llama-1B)."
echo "Monitor:        squeue -u \$USER"
echo "stdout logs:    ${LOG_ROOT}"
echo "Run outputs:    ${ROOT_DIR}/explore/Anchor_direction/outputs/CRAFT_CL_direction_*"
