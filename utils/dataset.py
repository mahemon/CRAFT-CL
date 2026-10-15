import os
from copy import deepcopy

import torch
import random
import transformers
from datasets import load_dataset
from collections import defaultdict
from copy import deepcopy
from utils.task_config import task_config
from utils.templates import *

from pyreft import ReftDataset

glue_task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

def parse_positions(positions: str):
    first_n, last_n = 0, 0
    if "+" in positions:
        first_n = int(positions.split("+")[0].strip("f"))
        last_n = int(positions.split("+")[1].strip("l"))
    else:
        if "f" in positions:
            first_n = int(positions.strip("f"))
        elif "l" in positions:
            last_n = int(positions.strip("l"))
    return first_n, last_n


class LoReftGLUEDataset(ReftDataset):

    def preprocess(self, kwargs):
        self.raw_dataset, self.trigger_tokens, self.num_labels = None, None, None
        self.pad_mode = "last" 
        self.fields_to_pad = ["input_ids"]

        self.sentence1_key, self.sentence2_key = glue_task_to_keys[self.data_path]
    
    def postprocess(self, kwargs):
        is_regression = self.data_path == "stsb"
        if not is_regression:
            label_list = self.task_dataset.features["label"].names
            num_labels = len(label_list)
        else:
            num_labels = 1
        self.num_labels = num_labels

    def tokenize(self, data_item):
        result = {}

        args = ((data_item[self.sentence1_key],)
                if self.sentence2_key is None
                else (data_item[self.sentence1_key], data_item[self.sentence2_key]))
        base_input_ids = self.tokenizer(
            *args, max_length=self.tokenizer.model_max_length, truncation=True,
            return_tensors="pt"
        )["input_ids"][0]
        output_ids = data_item["label"]
        last_position = len(base_input_ids)

        result["input_ids"] = base_input_ids
        result["labels"] = output_ids

        return result, last_position
    

class LoReftSupervisedDataset(ReftDataset):

    def preprocess(self, kwargs):
        print(kwargs)
        self.raw_dataset, self.trigger_tokens, self.num_labels = None, None, None
        dataset_config = task_config[self.task]
        self.task_prompt_template = dataset_config["task_prompt_template"]
        self.trigger_tokens = dataset_config["trigger_tokens"]
        self.original_data_split = self.data_split
        self.test_split = kwargs["test_split"] if "test_split" in kwargs else None
        
        if self.task in ["alpaca", "instruct", "ultrafeedback", "ultrafeedback_pair"] and self.data_split != "train":
            self.task = "tatsu-lab/alpaca_eval"
            self.data_path = "alpaca_eval"
            self.data_split = "eval"
        if self.task in ["gsm8k"]:
            self.data_path = "main" 
            if self.data_split != "test":
                self.data_split = "train" 
        elif self.task in ["math", "commonsense", "ultrafeedback", "alpaca"]:
            self.data_path = os.path.join(self.data_path, self.data_split + ".json")

    def postprocess(self, kwargs):
        original_dataset_size = len(self.task_dataset)
        if self.task in ["gsm8k"] and \
            self.original_data_split == "train" and self.test_split == "validation":
            self.task_dataset = self.task_dataset.select(
                range(original_dataset_size - 300))
        if self.task in ["gsm8k"] and self.original_data_split == "validation":
            self.task_dataset = self.task_dataset.select(
                range(original_dataset_size - 300, original_dataset_size))
        self.raw_dataset = self.task_dataset 
        return
    
    def tokenize(self, data_item):
        result = {}

        if self.task == "commonsense":
            base_prompt = self.task_prompt_template % (data_item['instruction'])
            base_input = base_prompt + self.trigger_tokens + data_item["answer"] + self.tokenizer.eos_token
        elif self.task == "math": 
            base_prompt = self.task_prompt_template % (data_item['instruction'])
            base_input = base_prompt + data_item["output"] + self.tokenizer.eos_token
        elif self.task in ["alpaca", "instruct", "ultrafeedback", "ultrafeedback_pair", "tatsu-lab/alpaca_eval"]:
            if 'input' not in data_item or data_item['input'] == "":
                base_prompt = alpaca_prompt_no_input_template % (data_item['instruction'])
            else:
                base_prompt = self.task_prompt_template % (data_item['instruction'], data_item['input'])
            if self.task == "ultrafeedback_pair" and self.data_split == "train":

                base_input = base_prompt + data_item["rejected_output"] + self.tokenizer.eos_token
            else:
                base_input = base_prompt + data_item["output"] + self.tokenizer.eos_token
        elif self.task == "gsm8k": 
            if "Meta-Llama-3-8B-Instruct" in self.tokenizer.name_or_path: 
                system_prompt = "You are a helpful assistant."
                base_prompt = self.tokenizer.apply_chat_template(
                    [{"role": "system", "content": system_prompt}, {"role": "user", "content": data_item['question']}], 
                    tokenize=False,
                )[len("<|begin_of_text|>"):]
                base_input = self.tokenizer.apply_chat_template(
                    [{"role": "system", "content": system_prompt}, {"role": "user", "content": data_item['question']},
                     {"role": "assistant", "content": data_item["answer"]}], 
                    tokenize=False,
                )[len("<|begin_of_text|>"):] + self.tokenizer.eos_token
            else: # setup is from https://github.com/yxli2123/LoftQ/
                base_prompt = f"{data_item['question']}{QUESTION_PROMPT}"
                base_input = base_prompt + f"{data_item['answer']}{self.tokenizer.eos_token}".replace("####", "The final answer is: ")
        else:
            raise ValueError(f"Unrecognized task: {self.task}")
        
        base_prompt_ids = self.tokenizer(
            base_prompt, max_length=self.tokenizer.model_max_length, truncation=True, return_tensors="pt")["input_ids"][0]
        base_prompt_length = len(base_prompt_ids)
        if self.original_data_split == "train":
            base_input_ids = self.tokenizer(
                base_input, max_length=self.tokenizer.model_max_length, truncation=True, return_tensors="pt")["input_ids"][0]

            if self.task == "ultrafeedback_pair" and self.data_split == "train":
                base_output = base_prompt + data_item["chosen_output"] + self.tokenizer.eos_token
                
                base_output_ids = self.tokenizer(
                    base_output, max_length=self.tokenizer.model_max_length, truncation=True, return_tensors="pt")["input_ids"][0]
                output_ids = base_output_ids
                output_ids[:base_prompt_length] = IGNORE_INDEX

                max_length = max(base_input_ids.size(0), output_ids.size(0))
                input_pad_length = max_length - base_input_ids.size(0)
                output_pad_length = max_length - output_ids.size(0)

                input_pad_tensor = torch.full((input_pad_length,), self.tokenizer.pad_token_id, dtype=torch.long)
                output_pad_tensor = torch.full((output_pad_length,), IGNORE_INDEX, dtype=torch.long)

                base_input_ids = torch.cat((base_input_ids, input_pad_tensor), dim=0)
                output_ids = torch.cat((output_ids, output_pad_tensor), dim=0)
            else:
                output_ids = deepcopy(base_input_ids)
                output_ids[:base_prompt_length] = IGNORE_INDEX
                
            result["input_ids"] = base_input_ids
            result["labels"] = output_ids
        else:
            result["input_ids"] = base_prompt_ids
        last_position = base_prompt_length

        base_prompt_length = min(base_prompt_length, len(result["input_ids"]))
        
        return result, last_position

class LoReftTRACEDataset(ReftDataset):

    def preprocess(self, kwargs):
        self.raw_dataset = None
        self.num_labels = None
        self.original_data_split = self.data_split
        self.test_split = kwargs.get("test_split", "test")

        dataset_config = task_config.get(self.task, task_config.get("trace", {}))
        self.task_prompt_template = dataset_config.get("task_prompt_template", "%s")
        self.trigger_tokens = dataset_config.get("trigger_tokens", "")

        split_file_map = {
            "train":      "train.json",
            "validation": "eval.json",
            "eval":       "eval.json",
            "test":       "test.json",
        }
        split_file = split_file_map.get(self.data_split, f"{self.data_split}.json")

        if not self.data_path.endswith(".json"):
            self.data_path = os.path.join(self.data_path, split_file)
        assert os.path.exists(self.data_path), (
            f"TRACE data file not found: {self.data_path}\n"
            f"Expected one of: train.json, eval.json, test.json "
            f"inside the task directory."
        )

    def postprocess(self, kwargs):
        self.raw_dataset = self.task_dataset

    def _is_chat_model(self) -> bool:
        name = self.tokenizer.name_or_path.lower()
        return "instruct" in name or "chat" in name

    def _apply_chat_template(self, messages: list) -> str:

        text = self.tokenizer.apply_chat_template(messages, tokenize=False)
        bos = self.tokenizer.bos_token
        if bos and text.startswith(bos):
            text = text[len(bos):]
        return text

    def tokenize(self, data_item):
        result = {}
        prompt = data_item.get("prompt", "")
        answer = data_item.get("answer", "")

        TRACE_PLACEHOLDER_REPLACEMENTS = [
            ("<EOL>", "\n"),      
            ("</s>",  "<eos>"),   
            ("<s>",   "<bos>"),   
        ]
        for special_str, replacement in TRACE_PLACEHOLDER_REPLACEMENTS:
            prompt = prompt.replace(special_str, replacement)
            answer = answer.replace(special_str, replacement)
      
        if self._is_chat_model():
            
            system_prompt = "You are a helpful assistant."
            user_messages = [
                {"role": "system",    "content": system_prompt},
                {"role": "user",      "content": prompt},
            ]
            full_messages = user_messages + [
                {"role": "assistant", "content": answer},
            ]

            base_prompt_ids = self.tokenizer.apply_chat_template(
                user_messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
                truncation=True,
                max_length=self.tokenizer.model_max_length,
            )[0]

            base_input_ids = self.tokenizer.apply_chat_template(
                full_messages,
                tokenize=True,
                add_generation_prompt=False,
                return_tensors="pt",
                truncation=True,
                max_length=self.tokenizer.model_max_length,
            )[0]

            base_prompt = self.tokenizer.decode(base_prompt_ids,
                                                skip_special_tokens=False)
            base_input  = self.tokenizer.decode(base_input_ids,
                                                skip_special_tokens=False)
            base_prompt_length = len(base_prompt_ids)

            _chat_path_ids_ready = True
        else:
            base_prompt = prompt

            _name = self.tokenizer.name_or_path.lower()
            if "gemma" in _name and "-it" in _name:
                _end_token = "<end_of_turn>"
            else:
                _end_token = self.tokenizer.eos_token
            base_input = base_prompt + answer + _end_token
            _chat_path_ids_ready = False
                        
        if not _chat_path_ids_ready:
            base_prompt_ids = self.tokenizer(
                base_prompt,
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )["input_ids"][0]
            base_prompt_length = len(base_prompt_ids)

        if self.original_data_split == "train":
            if not _chat_path_ids_ready:
                base_input_ids = self.tokenizer(
                    base_input,
                    max_length=self.tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                )["input_ids"][0]
            actual_prompt_length = 0
            min_len = min(len(base_input_ids), len(base_prompt_ids))
            for i in range(min_len):
                if int(base_input_ids[i]) == int(base_prompt_ids[i]):
                    actual_prompt_length = i + 1
                else:
                    break
            if actual_prompt_length == 0:
                actual_prompt_length = min(base_prompt_length,
                                           len(base_input_ids) - 1)
            if actual_prompt_length >= len(base_input_ids):
                actual_prompt_length = len(base_input_ids) - 1

            output_ids = deepcopy(base_input_ids)
            output_ids[:actual_prompt_length] = IGNORE_INDEX

            result["input_ids"] = base_input_ids
            result["labels"]    = output_ids
            base_prompt_length  = actual_prompt_length
        else:
            result["input_ids"] = base_prompt_ids
        base_prompt_length = min(base_prompt_length, len(result["input_ids"]))

        return result, base_prompt_length