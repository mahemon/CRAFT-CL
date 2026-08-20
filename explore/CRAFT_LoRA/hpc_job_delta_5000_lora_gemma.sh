#!/bin/bash
# ============================================================================
# CRAFT-CL — kl_routing (R3: ratio/thr=0.7/floor=0.01)
#          + KL anchor (β=0.3, normalized)
#          + eviction (relative + absolute trigger)
#          + all-groupwise LoReFT (no private T3_l)
#          + per-task backward eval
#          + CRAFT-LoRA BACKEND (LoRA adapters replace LoReFT;
#            oracle/default inference)
# on the 8-task TRACE benchmark (LLM-CL-Benchmark_5000) with
# google/gemma-2b-it — NCSA DELTA VERSION.
#
# All training hyperparameters identical to the Nova 5000 script.
#
# Fully self-contained: train.py, craft_lora.py, this bash, logs/ and
# outputs/ all live in explore/CRAFT_LoRA/.  Submit from a Delta login node:
#   export HF_TOKEN=hf_...
#   bash /projects/bfwa/mhossen1/CRAFT-CL/explore/CRAFT_LoRA/hpc_job_delta_5000_lora_gemma.sh
#
# Logs   → explore/CRAFT_LoRA/logs/
# Output → explore/CRAFT_LoRA/outputs/
# ============================================================================

ROOT_DIR="/projects/bfwa/mhossen1/CRAFT-CL"
LORA_DIR="${ROOT_DIR}/explore/CRAFT_LoRA"  # code, logs, outputs all here
VENV="/projects/bdpm/mhossen1/myenv"
CACHE_ROOT="/projects/bdpm/mhossen1"

MODEL="/projects/bdpm/mhossen1/models/gemma-2b-it"
TASKS="C-STANCE,FOMC,MeetingBank,Py150,ScienceQA,NumGLUE-cm,NumGLUE-ds,20Minuten"
EPOCHS2="5,4,7,5,4,5,5,7"

# ─── KL routing config (R3 from the floor sweep) ──────────────────────────────
ROUTING_NORMALIZE="ratio"
ROUTING_THRESHOLD="0.7"
ROUTING_MIN_DKG="0.01"

# ─── KL regularization β values to sweep ──────────────────────────────────────
BETA_LIST=(0.3)

# ─── Eviction config ──────────────────────────────────────────────────────────
EVICT_SEVERITY="1.3"
EVICT_EARLY_STEPS="30"
EVICT_ABSOLUTE_THRESHOLD="1.0"

# ─── Inference router ablation (oracle = original behaviour) ──────────────────
INFERENCE_ROUTER="oracle"   # CRAFT-LoRA runs use the default (oracle) inference
ROUTER_PROBE_SIZE="${ROUTER_PROBE_SIZE:-16}"

# ─── Training script (Delta patched file with router support) ─────────────────
TRAIN_SCRIPT="train.py"

LOG_ROOT="${LORA_DIR}/logs"
mkdir -p "${LOG_ROOT}"

if [ -z "${HF_TOKEN}" ]; then
    echo "FATAL: HF_TOKEN is not set. Run: export HF_TOKEN=hf_..." >&2
    exit 1
fi

submit_job() {
    local lr=$1
    local epochs=$2
    local beta=$3
    local job_name=$4

    sbatch <<EOF
#!/bin/bash
#SBATCH --account=bdpm-delta-gpu
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
cd "${LORA_DIR}" || exit 1
source "${VENV}/bin/activate"

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

export PYTHONPATH="${LORA_DIR}:${ROOT_DIR}:\${PYTHONPATH}"

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

rank=8
timestamp=\$(date +%F_%H-%M-%S)
beta_tag=\$(echo "${beta}" | tr '.' 'p')
out_file="${LOG_ROOT}/craftlora_gemma2b_klroute_R3_evict_beta\${beta_tag}_rank\${rank}_lr${lr}_CRAFT-LoRA_job\${SLURM_JOB_ID}_\${timestamp}_5000.txt"
err_file="${LOG_ROOT}/craftlora_gemma2b_klroute_R3_evict_beta\${beta_tag}_rank\${rank}_lr${lr}_CRAFT-LoRA_job\${SLURM_JOB_ID}_\${timestamp}_5000.err.txt"
exec > >(tee -a "\$out_file")
exec 2> >(tee -a "\$err_file" >&2)

OUTPUT_DIR="${LORA_DIR}/outputs/${job_name}_\${timestamp}"
mkdir -p "\${OUTPUT_DIR}"

echo "════════════════════════════════════════════════════════════════════"
echo "  Job: ${job_name}   (SLURM id \${SLURM_JOB_ID}, node \$(hostname))"
echo "  Routing  : kl_routing  normalize=${ROUTING_NORMALIZE}  thr=${ROUTING_THRESHOLD}  floor=${ROUTING_MIN_DKG}"
echo "  KL anchor: β = ${beta}  (normalized KL: KL / running_mean before β)"
echo "  Eviction : enabled  severity=${EVICT_SEVERITY}  early_steps=${EVICT_EARLY_STEPS}  abs_thresh=${EVICT_ABSOLUTE_THRESHOLD}"
echo "  Storage  : ALL groupwise (no private T3_l)"
echo "  BWD eval : per-task (full perf_matrix)"
echo "  Backend  : CRAFT-LoRA (rank=8 alpha=32 target=o_proj)"
echo "  Inference: router=${INFERENCE_ROUTER}  probe=${ROUTER_PROBE_SIZE}"
echo "  rank     : \${rank}"
echo "  Output   : \${OUTPUT_DIR}"
echo "  Started  : \$(date)"
echo "════════════════════════════════════════════════════════════════════"

python -u ${TRAIN_SCRIPT} \\
  --continual_learning \\
  --benchmark standard_cl \\
  --cl_benchmark_dir "\${ROOT_DIR}/dataset/TRACE_Benchmark/LLM-CL-Benchmark_5000" \\
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
  `# ── Routing (kl_routing, R3 config) ────────────────────────────────` \\
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
  `# ── KL regularization ──────────────────────────────────────────────` \\
  --use_kl \\
  --kl_beta ${beta} \\
  --kl_early_stop \\
  --kl_rolling_window 20 \\
  --kl_eval_batches 50 \\
  --kl_normalize_kl \\
  --kl_mode output \\
  \\
  `# ── Eviction ───────────────────────────────────────────────────────` \\
  --kl_eviction_enabled \\
  --kl_eviction_severity_threshold ${EVICT_SEVERITY} \\
  --kl_eviction_early_steps ${EVICT_EARLY_STEPS} \\
  --kl_evict_absolute_threshold ${EVICT_ABSOLUTE_THRESHOLD} \\
  \\
  `# ── Storage: all-groupwise ─────────────────────────────────────────` \\
  --all_groupwise \\
  \\
  `# ── Per-task backward eval ─────────────────────────────────────────` \\
  --per_task_bwd_eval \\
  \\
  `# ── Inference router (CRAFT-LoRA: oracle/default) ─────────────────` \\
  --inference_router ${INFERENCE_ROUTER} \\
  --router_probe_size ${ROUTER_PROBE_SIZE} \\
  \\
  `# ── CRAFT-LoRA backend (LoRA adapters instead of LoReFT) ──────────` \\
  --use_lora \\
  --lora_rank 8 \\
  --lora_alpha 32 \\
  --lora_modules o_proj \\
  --lora_layers all

EXIT_CODE=\$?
echo ""
echo "════════════════════════════════════════════════════════════════════"
echo "  Finished at \$(date) with exit code \$EXIT_CODE"
if [ \$EXIT_CODE -eq 0 ] && [ "${INFERENCE_ROUTER}" != "oracle" ]; then
    echo "  Routing report: \${OUTPUT_DIR}/agnostic_routing_log.json"
fi
echo "════════════════════════════════════════════════════════════════════"
EOF
}

for BETA in "${BETA_LIST[@]}"; do
    BETA_TAG=$(echo "$BETA" | tr '.' 'p')
    JOB_NAME="CRAFT_LoRA_KLroute_R3_evict_beta${BETA_TAG}_gemma2b"
    echo "Submitting β=${BETA}  router=${INFERENCE_ROUTER}  as job ${JOB_NAME}"
    submit_job "1e-4" "$EPOCHS2" "$BETA" "$JOB_NAME"
done

echo ""
echo "Submitted kl_routing R3 + KL anchor + eviction + all-groupwise job(s)"
echo "on 8-task TRACE (5000)   inference_router=${INFERENCE_ROUTER}"
echo ""
echo "Monitor:        squeue -u \$USER"
echo "stdout logs:    ${LOG_ROOT}"
echo "Run outputs:    ${LORA_DIR}/outputs/"
