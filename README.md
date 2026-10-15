# CRAFT: Forgetting-Aware Intervention-Based Adaptation for Continual Learning

CRAFT is a continual-learning method for large language models. The backbone stays frozen: each task
is learned as a small low-rank intervention on hidden representations (LoReFT). Similar tasks share an
intervention, and one KL-based signal decides how tasks are grouped, trained and merged. At test time,
CRAFT can identify the task from the prompt alone (LITAR) and load the matching intervention.

## Installation

Install PyReFT from pip (https://github.com/stanfordnlp/pyreft):
```bash
pip install pyreft
```
or the latest version from git:
```bash
pip install git+https://github.com/stanfordnlp/pyreft.git
```

Gated models (Llama, Gemma, Mistral) need a Hugging Face token in `~/.hf_token`.

## Dataset

TRACE benchmark, `LLM-CL-Benchmark_5000` (https://github.com/BeyonderXX/TRACE).

Task-order datasets from O-LoRA (https://github.com/cmnfriend/O-LoRA), used for orders 1-6 and the
15-task stream at 500 examples per task.

Place the data as:
```
dataset/TRACE_Benchmark/LLM-CL-Benchmark_5000/<task>/{train,eval,test}.json
dataset/TRACE_Benchmark/LLM-CL-Benchmark_500/<task>/{train,eval,test}.json
```
or set `DATA_DIR` to another location.

## Scripts

The scripts submit SLURM jobs. Copy `scripts/cluster.env.example` to `scripts/cluster.env` and set your
partition, account, virtual environment and CUDA module.

```bash
MODEL=llama3_2_1b bash scripts/run_craft.sh                 # TRACE-8 on one backbone
SEEDS="42 1234" MODEL=llama2_7b bash scripts/run_craft.sh   # two seeds
STREAM=orders  bash scripts/run_streams.sh                  # task orders 1-6
STREAM=trace15 bash scripts/run_streams.sh                  # 15-task stream
STREAM=math    bash scripts/run_streams.sh                  # ScienceQA, NumGLUE-cm, NumGLUE-ds
DRY=1 MODEL=gemma_2b bash scripts/run_craft.sh              # print the job, submit nothing
```

Models: `llama3_2_1b`, `gemma_2b`, `llama2_7b`, `llama2_13b`, `mistral_7b`, `r1distill_llama_8b`,
`r1distill_qwen_1_5b`, `qwen2_5_3b`, `qwen2_5_7b`. Routers at test time are set with
`INFER_ROUTER="oracle litar learned_head"`. Other settings are listed at the top of each script.

## Results

Each run writes `cl_results.json` (score matrix, OP and BWT per router) to
`output_log/outputs_<bench>_router_<model>/<run>/`. To summarize all runs:
```bash
python scripts/collect_results.py --roots output_log --out results
```
