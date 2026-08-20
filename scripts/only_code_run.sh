#!/bin/bash

# Default to current directory if not provided
ROOT_DIR="${1:-$(pwd)}" 
LOG_ROOT="${ROOT_DIR}/logs"
mkdir -p "${LOG_ROOT}"

# MODEL="meta-llama/Llama-3.2-1B-Instruct"
MODEL="/projects/bdpm/mhossen1/models/Llama-3.2-1B-Instruct"
TASKS="ScienceQA,NumGLUE-cm,NumGLUE-ds,20Minuten,C-STANCE,FOMC,MeetingBank,Py150"
# EPOCHS="4,4,5,5,4,5,5,5"

EPOCHS="2,2,2,2,2,2,2,2"

ROUTING_NORMALIZE="ratio"
ROUTING_THRESHOLD="0.7"
ROUTING_MIN_DKG="0.01"

BETA_LIST=(0.3)

EVICT_SEVERITY="1.3"
EVICT_EARLY_STEPS="30"
EVICT_ABSOLUTE_THRESHOLD="1.0"

TRAIN_SCRIPT="train.py"

run_job() {
    local lr=$1
    local epochs=$2
    local beta=$3
    local job_name=$4

    cd "${ROOT_DIR}" || exit 1
    
    # Assuming you already activated your env in the interactive session, 
    # but leaving this here just in case. Update path if needed.
    source "source /projects/bdpm/mhossen1/myenv/bin/activate" 2>/dev/null || true

    # Ensure HuggingFace token is set in your terminal before running this script
    export HF_TOKEN="${HF_TOKEN:?HF_TOKEN is not set. Please run: export HF_TOKEN='your_token'}"
    huggingface-cli login --token "$HF_TOKEN" >/dev/null 2>&1 || true

    export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

    local rank=8
    local timestamp=$(date +%F_%H-%M-%S)
    local OUTPUT_DIR="${ROOT_DIR}/outputs/${job_name}_${timestamp}"
    local LOG_FILE="${LOG_ROOT}/${job_name}_${timestamp}.log"
    mkdir -p "${OUTPUT_DIR}"

    echo "════════════════════════════════════════════════════════════════════"
    echo "  Job     : ${job_name}"
    echo "  Routing : kl_routing  normalize=${ROUTING_NORMALIZE}  thr=${ROUTING_THRESHOLD}  floor=${ROUTING_MIN_DKG}"
    echo "  KL anchor: β=${beta}  (normalized)"
    echo "  Eviction: severity=${EVICT_SEVERITY}  early_steps=${EVICT_EARLY_STEPS}  abs_thresh=${EVICT_ABSOLUTE_THRESHOLD}"
    echo "  rank    : ${rank}"
    echo "  Output  : ${OUTPUT_DIR}"
    echo "  Log File: ${LOG_FILE}"
    echo "  Started : $(date)"
    echo "════════════════════════════════════════════════════════════════════"

    python -u ${TRAIN_SCRIPT} \
      --continual_learning \
      --benchmark standard_cl \
      --cl_benchmark_dir "${ROOT_DIR}/dataset/TRACE_Benchmark/LLM-CL-Benchmark_500" \
      --cl_tasks "${TASKS}" \
      --per_task_epochs "${epochs}" \
      --model ${MODEL} \
      --layers all \
      --rank $rank \
      --n_positions 15 \
      --intervention_type LoreftIntervention \
      --batch_size 4 \
      --gradient_accumulation_steps 1 \
      --eval_batch_size 4 \
      --lr ${lr} \
      --warmup_ratio 0.05 \
      --weight_decay 0.00 \
      --dropout 0.00 \
      --max_length 1024 \
      --logging_steps 10 \
      --output_dir "${OUTPUT_DIR}" \
      --dtype bfloat16 \
      --seed 42 \
      --shallow_frac 0.33 \
      --deep_frac 0.38 \
      --router_method kl_routing \
      --kl_routing_threshold ${ROUTING_THRESHOLD} \
      --kl_routing_normalize ${ROUTING_NORMALIZE} \
      --kl_routing_min_dkg ${ROUTING_MIN_DKG} \
      --kl_routing_top_k 64 \
      --kl_routing_temperature 2.0 \
      --kl_routing_probe_size 16 \
      --causal_warmup_steps 100 \
      --warmup_skip_steps 15 \
      --fixed_intervention_seed 42 \
      --tier1_alpha 0.30 \
      --group_alpha 0.50 \
      --use_kl \
      --kl_beta ${beta} \
      --kl_early_stop \
      --kl_rolling_window 20 \
      --kl_eval_batches 50 \
      --kl_normalize_kl \
      --kl_mode output \
      --kl_eviction_enabled \
      --kl_eviction_severity_threshold ${EVICT_SEVERITY} \
      --kl_eviction_early_steps ${EVICT_EARLY_STEPS} \
      --kl_evict_absolute_threshold ${EVICT_ABSOLUTE_THRESHOLD} \
      --all_groupwise \
      --per_task_bwd_eval 2>&1 | tee "${LOG_FILE}"

    EXIT_CODE=${PIPESTATUS[0]}
    echo "════════════════════════════════════════════════════════════════════"
    echo "Job finished with exit code ${EXIT_CODE} at $(date)"
}

for BETA in "${BETA_LIST[@]}"; do
    BETA_TAG=$(echo "$BETA" | tr '.' 'p')
    JOB_NAME="CRAFT_llama32_1b"
    echo "Starting β=${BETA} as job ${JOB_NAME}"
    run_job "1e-4" "$EPOCHS" "$BETA" "$JOB_NAME"
done