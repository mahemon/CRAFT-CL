import pyvene as pv
import torch
from torch.utils.data import DataLoader
from transformers import DataCollatorForSeq2Seq, AutoTokenizer
from typing import Optional
from task_config import task_config
from tqdm import tqdm
import re
import difflib
import numpy as np
import evaluate as hf_evaluate
from transformers.utils import logging
from pyreft import ReftDataCollator

device = "cuda" if torch.cuda.is_available() else "cpu"
logger = logging.get_logger(__name__)

TRACE_TASKS = {
    "trace", "C-STANCE", "FOMC", "MeetingBank", "Py150",
    "ScienceQA", "NumGLUE-cm", "NumGLUE-ds", "20Minuten", "Lima",
}

# Letter-based exact match  (A / B / C / D)
# These TRACE tasks use prompts that instruct the model to output a letter.
TRACE_CLASSIFICATION = {"C-STANCE", "FOMC", "ScienceQA"}

# Numeric tolerance match
TRACE_NUMBER = {"NumGLUE-cm", "NumGLUE-ds"}

# Abstractive summarisation → ROUGE-L
TRACE_ROUGE = {"MeetingBank"}

# Text simplification → SARI
TRACE_SARI = {"20Minuten"}

# Code generation → token-level F1
TRACE_CODE = {"Py150"}

STRING_LABEL_TASKS = {
    "ag_news", "agnews",
    "amazon",
    "yelp",
    "dbpedia",
    "yahoo",
    "BoolQA",
    "CB",
    "COPA",
    "IMDB",
    "MNLI",
    "MultiRC",
    "QQP",
    "RTE",
    "SST-2",
    "WiC",
}

TRACE_MAX_NEW_TOKENS = {
    "C-STANCE":    10,
    "FOMC":        10,
    "ScienceQA":   10,
    "NumGLUE-cm": 128,
    "NumGLUE-ds": 128,
    "MeetingBank": 512,
    "20Minuten":   384,
    "Py150":       128,
    "Lima":        256,
    "ag_news":    20,
    "agnews":     20,
    "amazon":     20,
    "yelp":       20,
    "dbpedia":    20,
    "yahoo":      20,
    "BoolQA":     10,
    "CB":         10,
    "COPA":       20,
    "IMDB":       10,
    "MNLI":       10,
    "MultiRC":    10,
    "QQP":        10,
    "RTE":        10,
    "SST-2":      10,
    "WiC":        10,
}

def is_float(value) -> bool:
    if value is None:
        return False
    try:
        float(value)
        return True
    except (ValueError, TypeError):
        return False


def extract_answer_number(sentence: str) -> float:
    sentence = sentence.replace(",", "")

    # Try answer-marker patterns first
    marker_patterns = [
        r"answer\s*(?:is|=|:)\s*(-?\d+\.?\d*)",
        r"####\s*(-?\d+\.?\d*)",
        r"=\s*(-?\d+\.?\d*)\s*[.\n]",  # last "= X" before period or newline
    ]
    for pattern in marker_patterns:
        matches = re.findall(pattern, sentence, flags=re.IGNORECASE)
        if matches:
            try:
                return float(matches[-1])
            except ValueError:
                pass
    matches = re.findall(r"-?\d+\.?\d*", sentence)
    if not matches:
        return float("inf")
    try:
        return float(matches[-1])
    except ValueError:
        return float("inf")


def extract_answer_letter(sentence: str) -> str:
    """Return the first A/B/C/D/E found in sentence (case-insensitive)."""
    for ch in (sentence or "").strip().upper():
        if ch in ("A", "B", "C", "D", "E"):
            return ch
    return ""


def _string_label_match(pred: str, gold: str) -> bool:
    p = pred.strip().lower()
    g = gold.strip().lower()
    if not p or not g:
        return False
    if p == g:
        return True
    if p.startswith(g):
        return True
    first_word = p.split()[0] if p.split() else ""
    if first_word and g.startswith(first_word):
        return True
    return False

def extract_output(pred: str, trigger: str) -> str:

    if not trigger:
        return pred
    idx = pred.find(trigger)
    if idx < 0:
        return ""
    return pred[idx + len(trigger):].lstrip()


def make_data_collator(tokenizer, model) -> ReftDataCollator:
    return ReftDataCollator(
        data_collator=DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            model=model,
            label_pad_token_id=-100,
            padding="longest",
            max_length=2048,
        )
    )


def make_dataloader(dataset, batch_size: int, collate_fn, shuffle: bool) -> DataLoader:
    return DataLoader(dataset, shuffle=shuffle, batch_size=batch_size, collate_fn=collate_fn)

def compute_metrics(
    task: str,
    dataset_name: str,
    intervenable: pv.IntervenableModel,
    tokenizer: AutoTokenizer,
    eval_dataset,
    data_items,
    trigger_tokens: str,
    run_name: str,
    batch_size: int = 4,
    data_collator=None,
    split: Optional[str] = None,
    greedy_decoding: bool = False,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    top_k: Optional[float] = None,
    max_eval_steps: Optional[int] = None,
):
    is_glue = task == "glue"

    if not is_glue:
        tokenizer.padding_side = "left"
        num_beams = 4 if task in ("commonsense", "math") and not greedy_decoding else 1

    data_collator = data_collator or make_data_collator(tokenizer, intervenable.model)
    eval_dataloader = make_dataloader(eval_dataset, batch_size, data_collator, shuffle=False)

    correct_count = 0.0
    total_count   = 0
    generations   = []
    all_preds     = []
    all_labels    = []

    # Accumulation lists for ROUGE / SARI (computed after all batches)
    all_generated:  list = []
    all_references: list = []
    all_sources:    list = []  # needed for SARI (source = original prompt)

    _name_lower = tokenizer.name_or_path.lower()
    is_llama3_instruct = "instruct" in _name_lower and "llama" in _name_lower

    is_llama3_any = (
        is_llama3_instruct
        or tokenizer.convert_tokens_to_ids("<|eot_id|>") != tokenizer.unk_token_id
    )

    _is_gemma_it = "gemma" in _name_lower and "-it" in _name_lower

    if _is_gemma_it:
        eot_id      = tokenizer.convert_tokens_to_ids("<end_of_turn>")
        terminators = [tokenizer.eos_token_id, eot_id]  

    elif is_llama3_any:
        eot_id      = tokenizer.convert_tokens_to_ids("<|eot_id|>")
        terminators = [tokenizer.eos_token_id, eot_id]
    else:
        terminators = tokenizer.eos_token_id

    max_new_tokens = TRACE_MAX_NEW_TOKENS.get(dataset_name, 128)
    eval_iterator  = tqdm(eval_dataloader, desc=f"eval [{dataset_name}]",
                          position=0, leave=True)

    with torch.no_grad():
        for step, inputs in enumerate(eval_iterator):
            if max_eval_steps is not None and step >= max_eval_steps:
                break

            for k, v in inputs.items():
                if isinstance(v, torch.Tensor):
                    inputs[k] = v.to(device)

            # intervention_locations: permute from [layers, batch, pos] → [batch, layers, pos]
            if inputs["intervention_locations"].dim() == 3:
                intervention_locations = inputs["intervention_locations"].permute(1, 0, 2)
            else:
                intervention_locations = None

            if is_glue:
                _, cf_outputs = intervenable(
                    {"input_ids": inputs["input_ids"],
                     "attention_mask": inputs["attention_mask"]},
                    unit_locations={"sources->base": (None, intervention_locations.tolist())},
                )
                preds = (cf_outputs.logits.argmax(dim=-1)
                         if dataset_name != "stsb"
                         else cf_outputs.logits.squeeze(dim=1))
                all_preds  += preds.tolist()
                all_labels += inputs["labels"].tolist()
                continue

            n_input_tokens = inputs["input_ids"].shape[1]

            if intervention_locations is not None:
                seq_len = n_input_tokens

                bos_pos = (inputs["input_ids"] == tokenizer.bos_token_id
                           ).nonzero(as_tuple=True)[1]

                if bos_pos.numel() > 0:
                    actual_batch   = intervention_locations.shape[1]
                    bos_offsets    = bos_pos[:actual_batch].reshape(1, -1, 1).to(device)

                    intervention_locations = (intervention_locations + bos_offsets - 1)
                else:
                    logger.warning(
                        "No BOS token found in batch; skipping left-pad adjustment. "
                        "Intervention positions may be slightly off."
                    )

                intervention_locations = intervention_locations.clamp(0, seq_len - 1)

                intervention_locations = intervention_locations.repeat_interleave(
                    num_beams, dim=1).tolist()
            else:
                intervention_locations = 0 

            generation_args = {
                "base": {
                    "input_ids":      inputs["input_ids"],
                    "attention_mask": inputs["attention_mask"],
                },
                "unit_locations":       {"sources->base": (None, intervention_locations)},
                "intervene_on_prompt":  True,
                "eos_token_id":         terminators,
                "early_stopping":       True,
                "max_new_tokens":       max_new_tokens,
            }

            if task in task_config and "generation_args" in task_config[task]:
                generation_args.update(task_config[task]["generation_args"][greedy_decoding])
            if temperature is not None:
                generation_args["temperature"] = temperature
            if top_p is not None:
                generation_args["top_p"] = top_p
            if top_k is not None:
                generation_args["top_k"] = top_k

            _, steered_response = intervenable.generate(**generation_args)

            generated_only = steered_response[:, n_input_tokens:]
            actual_preds   = tokenizer.batch_decode(generated_only, skip_special_tokens=True)

            for sample_id, pred in zip(inputs["id"].tolist(), actual_preds):
                example = data_items[sample_id]

                try:
                    raw_generation = extract_output(pred, trigger_tokens)
                except Exception:
                    raw_generation = pred

                answer     = example.get("answer", "")
                generation = raw_generation  

                if task == "commonsense":
                    generation = raw_generation
                    if generation.strip() == answer.strip():
                        correct_count += 1

                elif task in ("math", "gsm8k"):
                    if not is_float(answer):
                        generation = extract_answer_letter(raw_generation)
                        if generation.strip() == answer.strip():
                            correct_count += 1
                    else:
                        ref = (answer.split("####")[-1].strip()
                               if task == "gsm8k" else answer.strip())
                        pred_num = extract_answer_number(raw_generation)
                        tol      = 0.5 if task == "gsm8k" else 0.001
                        if pred_num != float("inf") and abs(float(ref) - pred_num) <= tol:
                            correct_count += 1
                        generation = str(pred_num)

                elif task in TRACE_TASKS:

                    if dataset_name in TRACE_CLASSIFICATION:
                        pred_letter = extract_answer_letter(raw_generation)
                        gold_letter = extract_answer_letter(answer)
                        generation  = pred_letter

                        if total_count < 5:
                            logger.info(
                                f"[DBG {dataset_name}] "
                                f"gold={answer[:40]!r} gold_l={gold_letter} "
                                f"gen={raw_generation[:40]!r} pred_l={pred_letter}"
                            )

                        if pred_letter and gold_letter and pred_letter == gold_letter:
                            correct_count += 1

                    elif dataset_name in TRACE_NUMBER:
                        pred_num   = extract_answer_number(raw_generation)
                        gold_num   = extract_answer_number(answer)
                        generation = str(pred_num)

                        if total_count < 5:
                            logger.info(
                                f"[DBG {dataset_name}] "
                                f"gold={answer[:80]!r} gold_n={gold_num} "
                                f"gen={raw_generation[:80]!r} pred_n={pred_num}"
                            )

                        if (pred_num != float("inf") and gold_num != float("inf") and
                                abs(pred_num - gold_num) <= 0.001):
                            correct_count += 1

                    elif dataset_name in TRACE_ROUGE:
                        # MeetingBank: accumulate for ROUGE-L computed at end
                        generation = raw_generation
                        all_generated.append(generation)
                        all_references.append(answer)

                    elif dataset_name in TRACE_SARI:
                        generation = raw_generation
                        all_generated.append(generation)
                        all_references.append(answer)
                        all_sources.append(example.get("prompt", ""))

                    elif dataset_name in TRACE_CODE:
                        gen  = raw_generation.strip()
                        gold = answer.strip()
                        if gen and gold:
                            sim = difflib.SequenceMatcher(None, gen, gold).ratio()
                        else:
                            sim = 0.0
                        generation = gen
                        correct_count += sim  # accumulate continuous score

                        if total_count < 5:
                            logger.info(
                                f"[DBG Py150] sim={sim:.3f} "
                                f"gold={gold[:60]!r} gen={gen[:60]!r}"
                            )
                    else:
                        generation = raw_generation

                elif dataset_name in STRING_LABEL_TASKS:
                    generation = raw_generation.strip()
                    if _string_label_match(raw_generation, answer):
                        correct_count += 1

                    if total_count < 5:
                        logger.info(
                            f"[DBG {dataset_name}] "
                            f"gold={answer!r} "
                            f"gen={raw_generation[:40]!r} "
                            f"match={_string_label_match(raw_generation, answer)}"
                        )

                total_count += 1
                generations.append({
                    "prompt":         example.get("prompt", ""),
                    "raw_generation": raw_generation,
                    "generation":     generation,
                    "answer":         answer,
                    "dataset":        dataset_name,
                })

                if dataset_name not in (TRACE_ROUGE | TRACE_SARI) and total_count > 0:
                    eval_iterator.set_postfix(
                        {"score": f"{correct_count / total_count:.3f}"}
                    )

    if is_glue:
        metric = hf_evaluate.load("glue", dataset_name)
        report = metric.compute(predictions=all_preds, references=all_labels)
        if len(report) > 1:
            report["combined_score"] = float(np.mean(list(report.values())))
        if split:
            report = {f"{split}_{k}": v for k, v in report.items()}
        print(report)
        return [], report

    if task in ("alpaca", "instruct", "ultrafeedback", "ultrafeedback_pair"):
        return generations, {}

    if task in TRACE_TASKS:

        if dataset_name in TRACE_ROUGE and all_generated:
            rouge = hf_evaluate.load("rouge", experiment_id=f"{dataset_name}_{id(all_generated)}")
            result = rouge.compute(
                predictions=all_generated,
                references=all_references,
                use_stemmer=False,
            )
            score = round(float(result["rougeL"]), 4)
            print(f"  {dataset_name} ROUGE-L: {score:.4f}")

        elif dataset_name in TRACE_SARI and all_generated:
            try:
                sari_metric = hf_evaluate.load("sari", experiment_id=f"{dataset_name}_{id(all_generated)}")
                result = sari_metric.compute(
                    sources=all_sources,
                    predictions=all_generated,
                    references=[[r] for r in all_references],
                )
                score = round(float(result["sari"]) / 100.0, 4)
                print(f"  {dataset_name} SARI: {score:.4f}")
            except Exception as err:
                logger.warning(
                    f"SARI metric failed ({err}); falling back to ROUGE-L for {dataset_name}."
                )
                rouge = hf_evaluate.load("rouge", experiment_id=f"{dataset_name}_{id(all_generated)}")
                result = rouge.compute(
                    predictions=all_generated,
                    references=all_references,
                    use_stemmer=False,
                )
                score = round(float(result["rougeL"]), 4)
                print(f"  {dataset_name} ROUGE-L (SARI fallback): {score:.4f}")

        elif total_count > 0:
            score = round(correct_count / total_count, 4)
            print(f"  {dataset_name} score: {score:.4f}")
        else:
            score = 0.0

        return generations, {f"eval/{dataset_name}": score}

    if dataset_name in STRING_LABEL_TASKS or task in STRING_LABEL_TASKS:
        score = round(correct_count / total_count, 4) if total_count > 0 else 0.0
        print(f"  {dataset_name} accuracy: {score:.4f}")
        return generations, {f"eval/{dataset_name}": score}
    
    score = correct_count / total_count if total_count > 0 else 0.0
    return generations, {f"eval/{dataset_name}": score}