#!/bin/bash
# ============================================================================
# CRAFT-CL — SCIENCEQA-A/B DUPLICATE-TASK UNIT TEST (same top_k grid,
# partition an artifact of the top-64 union-support approximation?)
# on the 2-task ScienceQA-A/ScienceQA-B split. Run make_scienceqa_split.py
# runs warm-up + routing decision per task, SKIPS Phase 2 training/eval
# entirely, registers the warm-up state as each group's representative so
# multi-task partitions still form. Cost is warm-up + probe forwards only.
#
# IMPORTANT COST NOTE: the union-KL match step is O(k^2) per token
# position (match tensor of shape [..., 2k, k]). This rules out a literal
# "full vocabulary" arm (k ~ 128k would need multi-GB temporaries per
# token). The sweep below tops out at k=512, an 8x increase over the
# paper's k=64 default -- large enough to test truncation sensitivity,
# NOT "full vocab". Do not describe the k512 arm as full-vocabulary
# in the paper; call it a large-k arm.
#
# first to build the split dir. The unit test: does ScienceQA-A ever JOIN
# ScienceQA-B (or vice versa)? At small k this should FAIL to merge
# (near-orthogonal top-k support after truncation, the saturation failure
# documented in Limitations); at larger k the two halves should route
# into the SAME group, since they are genuinely near-identical data.
#
# Arms (name : capture_top_k : kl_routing_top_k : probe_size):
#   k64_p16  (reference, matches the paper default) : 64  : 64  : 16
#   k128_p16                                          : 128 : 128 : 16
#   k256_p16                                          : 256 : 256 : 16
#   k512_p16 (largest feasible, NOT full vocab)        : 512 : 512 : 16
#   k64_p8                                             : 64  : 64  : 8
#   k64_p64                                            : 64  : 64  : 64
#
# Everything else pinned to the R3 config (ratio routing thr=0.7
# floor=0.01). KL anchor / eviction / eval flags are irrelevant here
# (Phase 2 never runs) and are omitted.
#
# Submit from a Delta login node:
#   export HF_TOKEN=hf_...
#   bash explore/Routing_KL_truncation/submit_duplicate_unit_test.sh
# Single arm:
#   ARMS="k512_p16" bash explore/Routing_KL_truncation/submit_duplicate_unit_test.sh
# ============================================================================

ROOT_DIR="/projects/bfwa/mhossen1/CRAFT-CL"
VENV="/projects/bdpm/mhossen1/myenv"
CACHE_ROOT="/projects/bdpm/mhossen1"

MODEL="/projects/bdpm/mhossen1/models/Llama-3.2-1B-Instruct"

BENCH_DIR_REL="dataset/TRACE_Benchmark/LLM-CL-Benchmark_ScienceQA_split"  # build with make_scienceqa_split.py first
TASKS="ScienceQA-A,ScienceQA-B"
EPOCHS2="4,4"

ROUTING_NORMALIZE="ratio"
ROUTING_THRESHOLD="0.7"
ROUTING_MIN_DKG="0.01"

# arm = name:capture_top_k:kl_routing_top_k:probe_size
ALL_ARMS="k64_p16:64:64:16 k128_p16:128:128:16 k256_p16:256:256:16 k512_p16:512:512:16 k64_p8:64:64:8 k64_p64:64:64:64"
ARMS="${ARMS:-$ALL_ARMS}"

TRAIN_SCRIPT="explore/Routing_KL_truncation/train.py"

LOG_ROOT="${ROOT_DIR}/explore/Routing_KL_truncation/logs/scienceqa_dup_test"
mkdir -p "${LOG_ROOT}"

if [ -z "${HF_TOKEN}" ]; then
    echo "FATAL: HF_TOKEN is not set. Run: export HF_TOKEN=hf_..." >&2
    exit 1
fi

submit_job() {
    local arm_name=$1
    local capture_k=$2
    local top_k=$3
    local probe_size=$4
    local job_name=$5

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
out_file="${LOG_ROOT}/llama32_1b_dup_${arm_name}_job\${SLURM_JOB_ID}_\${timestamp}.txt"
err_file="${LOG_ROOT}/llama32_1b_dup_${arm_name}_job\${SLURM_JOB_ID}_\${timestamp}.err.txt"
exec > >(tee -a "\$out_file")
exec 2> >(tee -a "\$err_file" >&2)

OUTPUT_DIR="\${ROOT_DIR}/explore/Routing_KL_truncation/outputs/${job_name}_\${timestamp}"
mkdir -p "\${OUTPUT_DIR}"

echo "════════════════════════════════════════════════════════════════════"
echo "  Job: ${job_name}   (SLURM id \${SLURM_JOB_ID}, node \$(hostname))"
echo "  ARM : ${arm_name}   capture_top_k=${capture_k}  kl_routing_top_k=${top_k}  probe_size=${probe_size}"
echo "  routing_diagnostic_only : TRUE (Phase 2 skipped for every task)"
echo "  Routing  : kl_routing  normalize=${ROUTING_NORMALIZE}  thr=${ROUTING_THRESHOLD}  floor=${ROUTING_MIN_DKG}"
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
  --kl_routing_top_k ${top_k} \\
  --kl_routing_capture_top_k ${capture_k} \\
  --kl_routing_temperature 2.0 \\
  --kl_routing_probe_size ${probe_size} \\
  --causal_warmup_steps 100 \\
  --warmup_skip_steps 15 \\
  --fixed_intervention_seed 42 \\
  --tier1_alpha 0.30 \\
  --group_alpha 0.50 \\
  \\
  --all_groupwise \\
  \\
  `# ── Truncation-sweep arm: diagnostic-only, no Phase 2 ──────────────` \\
  --routing_diagnostic_only \\
  \\
  --inference_router oracle \\
  --router_probe_size 16

EXIT_CODE=\$?
echo ""
echo "════════════════════════════════════════════════════════════════════"
echo "  Finished at \$(date) with exit code \$EXIT_CODE"
echo "  Results: \${OUTPUT_DIR}/cl_results.json  (OP/BWT meaningless here;"
echo "            read task_to_group / routing_log only)"
echo "════════════════════════════════════════════════════════════════════"
EOF
}

for ARM in ${ARMS}; do
    NAME=$(echo "$ARM" | cut -d: -f1)
    CAPK=$(echo "$ARM" | cut -d: -f2)
    TOPK=$(echo "$ARM" | cut -d: -f3)
    PROBE=$(echo "$ARM" | cut -d: -f4)
    JOB_NAME="CRAFT_CL_trunc_dup_${NAME}_llama32_1b"
    echo "Submitting arm=${NAME}  capture_top_k=${CAPK}  top_k=${TOPK}  probe_size=${PROBE}  as job ${JOB_NAME}"
    submit_job "$NAME" "$CAPK" "$TOPK" "$PROBE" "$JOB_NAME"
done

echo ""
echo "Submitted ScienceQA-A/B duplicate unit test (Llama-1B), diagnostic-only."
echo "Monitor:        squeue -u \$USER"
echo "stdout logs:    ${LOG_ROOT}"
echo "Run outputs:    ${ROOT_DIR}/explore/Routing_KL_truncation/outputs/CRAFT_CL_trunc_*"
