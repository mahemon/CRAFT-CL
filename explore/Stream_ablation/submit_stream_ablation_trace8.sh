#!/bin/bash
# ============================================================================
# CRAFT-CL — STREAM ABLATION (jvrh W1: what does each prompt stream
# contribute, and does the split interact with prompt length?)
#
# Arms on TRACE-8, Llama-3.2-1B-Instruct, NCSA DELTA. All arms hold the
# TOTAL intervention position budget fixed at 30 (matching the paper's
# default f15+l15) and vary only how that budget is split between the
# prefix (f) and suffix (l) streams:
#
#   f15l15  : f=15, l=15   (paper default; reference arm)
#   f30     : f=30, l=0    (prefix stream only)
#   l30     : f=0,  l=30   (suffix stream only)
#   f5l25   : f=5,  l=25   (asymmetric budget)
#
# Everything else pinned to the R3 config (ratio routing thr=0.7
# floor=0.01, KL anchor beta=0.3 normalized, eviction enabled,
# all-groupwise, per-task backward eval, oracle inference, seed 42).
# The ONLY varied factors are --f_positions / --l_positions.
#
# Submit from a Delta login node:
#   export HF_TOKEN=hf_...
#   bash explore/Stream_ablation/submit_stream_ablation_trace8.sh
# Single arm:
#   ARMS="l30" bash explore/Stream_ablation/submit_stream_ablation_trace8.sh
# ============================================================================

ROOT_DIR="/projects/bfwa/mhossen1/CRAFT-CL"
VENV="/projects/bdpm/mhossen1/myenv"
CACHE_ROOT="/projects/bdpm/mhossen1"

MODEL="/projects/bdpm/mhossen1/models/Llama-3.2-1B-Instruct"

# TRACE-8, matching the main llama runs.
BENCH_DIR_REL="dataset/TRACE_Benchmark/LLM-CL-Benchmark_5000"
TASKS="C-STANCE,FOMC,MeetingBank,Py150,ScienceQA,NumGLUE-cm,NumGLUE-ds,20Minuten"
EPOCHS2="4,4,5,5,4,5,5,5"

ROUTING_NORMALIZE="ratio"
ROUTING_THRESHOLD="0.7"
ROUTING_MIN_DKG="0.01"
BETA="0.3"

EVICT_SEVERITY="1.3"
EVICT_EARLY_STEPS="30"
EVICT_ABSOLUTE_THRESHOLD="1.0"

# arm = name:f_positions:l_positions   (total held at 30 across all arms)
ALL_ARMS="f15l15:15:15 f30:30:0 l30:0:30 f5l25:5:25"
ARMS="${ARMS:-$ALL_ARMS}"

TRAIN_SCRIPT="explore/Stream_ablation/train.py"

LOG_ROOT="${ROOT_DIR}/explore/Stream_ablation/logs/trace8_llama1b"
mkdir -p "${LOG_ROOT}"

if [ -z "${HF_TOKEN}" ]; then
    echo "FATAL: HF_TOKEN is not set. Run: export HF_TOKEN=hf_..." >&2
    exit 1
fi

submit_job() {
    local lr=$1
    local epochs=$2
    local arm_name=$3
    local f_pos=$4
    local l_pos=$5
    local job_name=$6

    sbatch <<EOF
#!/bin/bash
#SBATCH --account=bdpm-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --time=23:00:00
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
out_file="${LOG_ROOT}/llama32_1b_trace8_${arm_name}_f${f_pos}l${l_pos}_job\${SLURM_JOB_ID}_\${timestamp}.txt"
err_file="${LOG_ROOT}/llama32_1b_trace8_${arm_name}_f${f_pos}l${l_pos}_job\${SLURM_JOB_ID}_\${timestamp}.err.txt"
exec > >(tee -a "\$out_file")
exec 2> >(tee -a "\$err_file" >&2)

OUTPUT_DIR="\${ROOT_DIR}/explore/Stream_ablation/outputs/${job_name}_\${timestamp}"
mkdir -p "\${OUTPUT_DIR}"

echo "════════════════════════════════════════════════════════════════════"
echo "  Job: ${job_name}   (SLURM id \${SLURM_JOB_ID}, node \$(hostname))"
echo "  ARM      : ${arm_name}   f_positions=${f_pos}   l_positions=${l_pos}   total=$((f_pos + l_pos))"
echo "  Routing  : kl_routing  normalize=${ROUTING_NORMALIZE}  thr=${ROUTING_THRESHOLD}  floor=${ROUTING_MIN_DKG}"
echo "  KL anchor: β = ${BETA}  (normalized KL)"
echo "  Eviction : enabled  severity=${EVICT_SEVERITY}  early_steps=${EVICT_EARLY_STEPS}  abs_thresh=${EVICT_ABSOLUTE_THRESHOLD}"
echo "  Storage  : ALL groupwise (no private T3_l)"
echo "  Stream   : TRACE-8"
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
  \\
  `# ── Stream ablation arm ────────────────────────────────────────────` \\
  --f_positions ${f_pos} \\
  --l_positions ${l_pos} \\
  \\
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

for ARM in ${ARMS}; do
    NAME=$(echo "$ARM" | cut -d: -f1)
    FPOS=$(echo "$ARM" | cut -d: -f2)
    LPOS=$(echo "$ARM" | cut -d: -f3)
    JOB_NAME="CRAFT_CL_stream_${NAME}_llama32_1b_trace8"
    echo "Submitting arm=${NAME}  f=${FPOS}  l=${LPOS}  as job ${JOB_NAME}"
    submit_job "2e-4" "$EPOCHS2" "$NAME" "$FPOS" "$LPOS" "$JOB_NAME"
done

echo ""
echo "Submitted stream-ablation arms on TRACE-8 (Llama-1B)."
echo "Monitor:        squeue -u \$USER"
echo "stdout logs:    ${LOG_ROOT}"
echo "Run outputs:    ${ROOT_DIR}/explore/Stream_ablation/outputs/CRAFT_CL_stream_*"
