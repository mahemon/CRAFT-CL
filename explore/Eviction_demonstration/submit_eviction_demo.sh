#!/bin/bash
# ============================================================================
# CRAFT-CL — EVICTION DEMONSTRATION OR DELETION (C3: eviction never fires
# in any reported run; is the mechanism real or dead code?)
# NCSA DELTA, Llama-3.2-1B-Instruct.
#
# Reproduces the paper's OWN adversarial-separation unit test (Table
# "FOMC -> ScienceQA routing decision across delta"): a minimal 2-task
# stream (FOMC, then ScienceQA) at the routing threshold (delta=1.0)
# where the paper's own delta-sweep documents a spurious merge ("JOIN
# G0"). This is the paper's characterized incompatible pair, forced to
# merge on purpose, at the exact configuration the paper already showed
# produces a bad merge -- the correct place to test whether eviction
# ever rescues a known-bad routing decision.
#
# 4 arms, increasing eviction sensitivity:
#   no_evict     : eviction disabled entirely (control -- confirms the
#                  bad merge happens and stays merged without eviction)
#   evict_default: eviction enabled at the paper's shipped thresholds
#                  (severity=1.3, abs=1.0) -- per Tables 15/16, these
#                  never fire under normal routing; this checks whether
#                  they fire even under intentionally forced mis-routing
#   evict_sens   : more sensitive (severity=1.1, abs=0.3)
#   evict_vsens  : maximally sensitive short of trivial (severity=1.05,
#                  abs=0.1) -- if eviction NEVER fires even here, that
#                  is itself the deletion-worthy finding
#
# All other config pinned to R3 (ratio routing floor=0.01, KL anchor
# beta=0.3 normalized, all-groupwise, per-task backward eval, oracle
# inference, seed 42). ONLY the eviction thresholds vary; routing
# threshold is fixed at 1.0 in every arm (the forcing mechanism).
#
# NOTE: eviction firing is NOT reliably visible in cl_results.json --
# grep the stdout log for "[KL] EVICT" instead (analyze_eviction.py
# does this for you).
#
# Submit from a Delta login node:
#   export HF_TOKEN=hf_...
#   bash explore/Eviction_demonstration/submit_eviction_demo.sh
# Single arm:
#   ARMS="evict_vsens" bash explore/Eviction_demonstration/submit_eviction_demo.sh
# ============================================================================

ROOT_DIR="/projects/bfwa/mhossen1/CRAFT-CL"
VENV="/projects/bdpm/mhossen1/myenv"
CACHE_ROOT="/projects/bdpm/mhossen1"

MODEL="/projects/bdpm/mhossen1/models/Llama-3.2-1B-Instruct"

BENCH_DIR_REL="dataset/TRACE_Benchmark/LLM-CL-Benchmark_5000"
TASKS="FOMC,ScienceQA"
EPOCHS2="4,4"

ROUTING_NORMALIZE="ratio"
ROUTING_THRESHOLD="1.0"   # forces the spurious merge (paper's own table)
ROUTING_MIN_DKG="0.01"
BETA="0.3"

EVICT_EARLY_STEPS="30"

# arm = name:eviction_enabled:severity:abs_threshold
ALL_ARMS="no_evict:0:1.3:1.0 evict_default:1:1.3:1.0 evict_sens:1:1.1:0.3 evict_vsens:1:1.05:0.1"
ARMS="${ARMS:-$ALL_ARMS}"

TRAIN_SCRIPT="explore/Eviction_demonstration/train.py"

LOG_ROOT="${ROOT_DIR}/explore/Eviction_demonstration/logs/fomc_scienceqa"
mkdir -p "${LOG_ROOT}"

if [ -z "${HF_TOKEN}" ]; then
    echo "FATAL: HF_TOKEN is not set. Run: export HF_TOKEN=hf_..." >&2
    exit 1
fi

submit_job() {
    local arm_name=$1
    local evict_enabled=$2
    local severity=$3
    local abs_thresh=$4
    local job_name=$5

    local evict_flag=""
    if [ "${evict_enabled}" = "1" ]; then
        evict_flag="--kl_eviction_enabled"
    fi

    sbatch <<EOF
#!/bin/bash
#SBATCH --account=bdpm-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
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
out_file="${LOG_ROOT}/llama32_1b_${arm_name}_job\${SLURM_JOB_ID}_\${timestamp}.txt"
err_file="${LOG_ROOT}/llama32_1b_${arm_name}_job\${SLURM_JOB_ID}_\${timestamp}.err.txt"
exec > >(tee -a "\$out_file")
exec 2> >(tee -a "\$err_file" >&2)

OUTPUT_DIR="\${ROOT_DIR}/explore/Eviction_demonstration/outputs/${job_name}_\${timestamp}"
mkdir -p "\${OUTPUT_DIR}"

echo "════════════════════════════════════════════════════════════════════"
echo "  Job: ${job_name}   (SLURM id \${SLURM_JOB_ID}, node \$(hostname))"
echo "  ARM : ${arm_name}   eviction_enabled=${evict_enabled}  severity=${severity}  abs_threshold=${abs_thresh}"
echo "  Forced mis-routing: FOMC then ScienceQA at kl_routing_threshold=${ROUTING_THRESHOLD}"
echo "  (paper's own delta-sweep table documents a spurious merge here)"
echo "  Output   : \${OUTPUT_DIR}"
echo "  Started  : \$(date)"
echo "════════════════════════════════════════════════════════════════════"

python -u ${TRAIN_SCRIPT} \\
  --continual_learning \\
  --benchmark standard_cl \\
  --cl_benchmark_dir "\${ROOT_DIR}/${BENCH_DIR_REL}" \\
  --cl_tasks "${TASKS}" \\
  --per_task_epochs "${EPOCHS2}" \\
  --model ${MODEL} \\
  --layers all \\
  --rank \$rank \\
  --n_positions 15 \\
  --intervention_type LoreftIntervention \\
  --batch_size 4 \\
  --gradient_accumulation_steps 1 \\
  --eval_batch_size 4 \\
  --lr 2e-4 \\
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
  `# ── Eviction sensitivity arm ───────────────────────────────────────` \\
  ${evict_flag} \\
  --kl_eviction_severity_threshold ${severity} \\
  --kl_eviction_early_steps ${EVICT_EARLY_STEPS} \\
  --kl_evict_absolute_threshold ${abs_thresh} \\
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
echo "  Grep this log for '[KL] EVICT' to check whether eviction fired."
echo "════════════════════════════════════════════════════════════════════"
EOF
}

for ARM in ${ARMS}; do
    NAME=$(echo "$ARM" | cut -d: -f1)
    EN=$(echo "$ARM" | cut -d: -f2)
    SEV=$(echo "$ARM" | cut -d: -f3)
    ABS=$(echo "$ARM" | cut -d: -f4)
    JOB_NAME="CRAFT_CL_evict_${NAME}_llama32_1b_fomc_scienceqa"
    echo "Submitting arm=${NAME}  eviction_enabled=${EN}  severity=${SEV}  abs=${ABS}  as job ${JOB_NAME}"
    submit_job "$NAME" "$EN" "$SEV" "$ABS" "$JOB_NAME"
done

echo ""
echo "Submitted eviction-demonstration arms (forced FOMC->ScienceQA mis-routing)."
echo "Monitor:        squeue -u \$USER"
echo "stdout logs:    ${LOG_ROOT}"
echo "Run outputs:    ${ROOT_DIR}/explore/Eviction_demonstration/outputs/CRAFT_CL_evict_*"
