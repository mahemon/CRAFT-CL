from templates import *

task_config = {
 
    # ------------------------------------------------------------------ #
    #  TRACE Benchmark datasets                                           #
    #  Format: {"prompt": "...", "answer": "..."}                        #
    #  Splits: train.json / eval.json / test.json                        #
    # ------------------------------------------------------------------ #
 
    "C-STANCE": {
        "train_datasets": ["C-STANCE/train.json"],
        "eval_datasets":  ["C-STANCE/test.json"],
        "task_prompt_template": (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n%s\n\n### Response:\n"
        ),
        "trigger_tokens": "",
    },
    "FOMC": {
        "train_datasets": ["FOMC/train.json"],
        "eval_datasets":  ["FOMC/test.json"],
        "task_prompt_template": (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n%s\n\n### Response:\n"
        ),
        "trigger_tokens": "",
    },
    "MeetingBank": {
        "train_datasets": ["MeetingBank/train.json"],
        "eval_datasets":  ["MeetingBank/test.json"],
        "task_prompt_template": (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n%s\n\n### Response:\n"
        ),
        "trigger_tokens": "",
    },
    "Py150": {
        "train_datasets": ["Py150/train.json"],
        "eval_datasets":  ["Py150/test.json"],
        "task_prompt_template": (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n%s\n\n### Response:\n"
        ),
        "trigger_tokens": "",
    },
    "ScienceQA": {
        "train_datasets": ["ScienceQA/train.json"],
        "eval_datasets":  ["ScienceQA/test.json"],
        "task_prompt_template": (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n%s\n\n### Response:\n"
        ),
        "trigger_tokens": "",
    },
    "NumGLUE-cm": {
        "train_datasets": ["NumGLUE-cm/train.json"],
        "eval_datasets":  ["NumGLUE-cm/test.json"],
        "task_prompt_template": (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n%s\n\n### Response:\n"
        ),
        "trigger_tokens": "",
    },
    "NumGLUE-ds": {
        "train_datasets": ["NumGLUE-ds/train.json"],
        "eval_datasets":  ["NumGLUE-ds/test.json"],
        "task_prompt_template": (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n%s\n\n### Response:\n"
        ),
        "trigger_tokens": "",
    },
    "20Minuten": {
        "train_datasets": ["20Minuten/train.json"],
        "eval_datasets":  ["20Minuten/test.json"],
        "task_prompt_template": (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n%s\n\n### Response:\n"
        ),
        "trigger_tokens": "",
    },
    "Lima": {
        "train_datasets": ["Lima/train.json"],
        "eval_datasets":  ["Lima/test.json"],
        "task_prompt_template": (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n%s\n\n### Response:\n"
        ),
        "trigger_tokens": "",
    },
 
    # Convenience alias — resolves to C-STANCE (first TRACE dataset)
    "trace": {
        "train_datasets": ["C-STANCE/train.json"],
        "eval_datasets":  ["C-STANCE/test.json"],
        "task_prompt_template": (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n%s\n\n### Response:\n"
        ),
        "trigger_tokens": "",
    },
    "commonsense": {
        "train_datasets": [
            # "commonsense_170k"
            "boolq", "piqa", "hellaswag", "social_i_qa", "winogrande", "ARC-Easy", "ARC-Challenge", "openbookqa"
        ],
        "eval_datasets": [
            "boolq", "piqa", "hellaswag", "social_i_qa", "winogrande", "ARC-Easy", "ARC-Challenge", "openbookqa"
        ],
        "server_eval_datasets": [
            "boolq", "piqa"
        ],
        "split_train_datasets": [
          "commonsense_170k"
        ],
        "task_prompt_template": "%s\n",
        "trigger_tokens": "the correct answer is ",
        "generation_args": {
            # align with https://github.com/AGI-Edgerunners/LLM-Adapters
            True: {
                "max_new_tokens": 32,
                "do_sample": False,
            },
            False: {
                "max_new_tokens": 32,
                "temperature": 0.1,
                "top_p": 0.75,
                "top_k": 40,
                "num_beams": 4,
                "do_sample": True,
            }
        }
    },
    "math": {
        "train_datasets": [
            # "math_10k"
            "AQuA", "SVAMP", "mawps", "AddSub", "MultiArith", "SingleEq",
        ],
        "eval_datasets": [
            "AQuA", "SVAMP", "mawps", "AddSub", "MultiArith", "SingleEq",
        ],
        "server_eval_datasets": [
            "gsm8k"
        ],
        "split_train_datasets": [
          "math_10k"
        ],
        "task_prompt_template": alpaca_prompt_no_input_template,
        "trigger_tokens": "### Response:",
        "generation_args": {
            # slightly changed to optimize our performance on top of
            # https://github.com/AGI-Edgerunners/LLM-Adapters
            True: {
                "max_new_tokens": 512,
                "do_sample": False,
            },
            False: {
                "max_new_tokens": 512,
                "temperature": 0.3,
                "top_p": 0.75,
                "top_k": 40,
                "num_beams": 4,
                "do_sample": True,
            }
        }
    },
    "alpaca": {
        "train_datasets": ["alpaca_data_cleaned"],
        "eval_datasets": ["alpaca_eval"],
        "task_prompt_template": alpaca_prompt_template,
        "trigger_tokens": "### Response:",
        "generation_args": {
            # align with https://arxiv.org/abs/2402.15179
            True: {
                "max_length": 2048,
                "do_sample": False,
            },
            False: {
                "max_length": 2048,
                "no_repeat_ngram_size": 5,
                "repetition_penalty": 1.1,
                "do_sample": False,
            }
        }
    },
    "instruct": {
        "train_datasets": ["instruct"],
        "eval_datasets": ["alpaca_eval"],
        "task_prompt_template": alpaca_prompt_template,
        "trigger_tokens": "### Response:",
        "generation_args": {
            # align with https://arxiv.org/abs/2402.15179
            True: {
                "max_length": 2048,
                "do_sample": False,
            },
            False: {
                "max_length": 2048,
                "no_repeat_ngram_size": 5,
                "repetition_penalty": 1.1,
                "do_sample": False,
            }
        }
    },
    "ultrafeedback": {
        "train_datasets": ["ultrafeedback"],
        "eval_datasets": ["alpaca_eval"],
        "task_prompt_template": alpaca_prompt_template,
        "trigger_tokens": "### Response:",
        "generation_args": {
            # align with https://arxiv.org/abs/2402.15179
            True: {
                "max_length": 2048,
                "do_sample": False,
            },
            False: {
                "max_length": 2048,
                "no_repeat_ngram_size": 5,
                "repetition_penalty": 1.1,
                "do_sample": False,
            }
        }
    },
    "ultrafeedback_pair": {
        "train_datasets": ["argilla/ultrafeedback-binarized-preferences-cleaned"],
        "eval_datasets": ["alpaca_eval"],
        "task_prompt_template": alpaca_prompt_template,
        "trigger_tokens": "### Response:",
        "generation_args": {
            # align with https://arxiv.org/abs/2402.15179
            True: {
                "max_length": 2048,
                "do_sample": False,
            },
            False: {
                "max_length": 2048,
                "no_repeat_ngram_size": 5,
                "repetition_penalty": 1.1,
                "do_sample": False,
            }
        }
    },
    "glue": {
        "train_datasets": None,
        "eval_datasets": None,
        "task_prompt_template": None,
        "trigger_tokens": None
    },
    "gsm8k": {
        "train_datasets": ["gsm8k"],
        "eval_datasets": ["gsm8k"],
        "task_prompt_template": gsm8k_template,
        "trigger_tokens": "First think step by step and then answer the final number.\n",
        "generation_args": {
            # default values are from LoftQ
            # https://arxiv.org/pdf/2310.08659.pdf
            True: {
                "max_new_tokens": 256,
                "do_sample": False,
            },
            False: {
                "max_new_tokens": 256,
                "temperature": 0.1,
                "top_p": 0.95,
                "top_k": 40,
                "do_sample": True
            }
        }
    },

    "metamathqa": {
    "train_datasets": ["MetaMathQA"],
    "eval_datasets": ["gsm8k"],  #  evaluate on GSM8K
    "server_eval_datasets": ["gsm8k"],
    "split_train_datasets": ["MetaMathQA"],
    "task_prompt_template": alpaca_prompt_no_input_template,
    "trigger_tokens": "### Response:",
    "generation_args": {
        True: {
            "max_new_tokens": 512,
            "do_sample": False,
        },
        False: {
            "max_new_tokens": 512,
            "temperature": 0.3,
            "top_p": 0.75,
            "top_k": 40,
            "num_beams": 4,
            "do_sample": True,
        }
    }
},

}


# All TRACE dataset names in continual-learning order (Lima = replay only)
TRACE_DATASETS = [
    "C-STANCE", "FOMC", "MeetingBank", "Py150",
    "ScienceQA", "NumGLUE-cm", "NumGLUE-ds", "20Minuten",
]
TRACE_REPLAY_DATASET = "Lima"
TRACE_ALL_DATASETS   = TRACE_DATASETS + [TRACE_REPLAY_DATASET]