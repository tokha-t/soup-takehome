"""Reproduce the earlier SQL probe and retain the actual generations."""
from dataclasses import replace
import json
import logging
import os
from pathlib import Path
import random
import re
import sqlite3
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from probe_data import MODEL_ID, MODEL_REVISION, encode_example, load_candidate, normalize_prompt, prepare_candidate
from probe_runtime import GENERATION_BATCH_SIZE, _left_pad, load_model, train_peft
from sql_metrics import score, sql_error
import torch
from transformers import AutoTokenizer, GenerationConfig

LOG = logging.getLogger("rescore")
ROOT = Path(__file__).resolve().parent.parent
MAIN_SYSTEM = "You translate questions into SQL. Use the given table schema. Reply with one SQL query only."
HARDER = re.compile(r"\bJOIN\b|\bGROUP\s+BY\b|\bORDER\s+BY\b|\bHAVING\b|\(\s*SELECT\b", re.IGNORECASE)


@torch.inference_mode()
def generate(model, tokenizer, rows, cap):
    results = [None] * len(rows)
    eos = tokenizer.eos_token_id
    config = GenerationConfig(
        do_sample=False, repetition_penalty=1.0, max_new_tokens=cap,
        eos_token_id=eos, pad_token_id=tokenizer.pad_token_id, use_cache=True,
    )
    order = sorted(range(len(rows)), key=lambda index: len(rows[index].prompt_ids))
    for offset in range(0, len(order), GENERATION_BATCH_SIZE):
        indices = order[offset:offset + GENERATION_BATCH_SIZE]
        ids, mask = _left_pad([rows[index].prompt_ids for index in indices], tokenizer.pad_token_id, model.device)
        for index, output_ids in zip(indices, model.generate(input_ids=ids, attention_mask=mask, generation_config=config)[:, ids.shape[1]:].tolist(), strict=True):
            stop = output_ids.index(eos) if eos in output_ids else None
            text = tokenizer.decode(output_ids[:stop], skip_special_tokens=False, clean_up_tokenization_spaces=False)
            row = rows[index]
            results[index] = {
                "source_index": row.source_index, "output": text, "gold": row.answer,
                "stopped_before_cap": stop is not None and stop + 1 < cap,
                **score(row.context, text, row.answer),
            }
    return results


def rerender(tokenizer, rows):
    corrected = []
    for row in rows:
        prefix, question = row.prompt.split("\n\nQuestion:\n", 1)
        copy = replace(row, system=MAIN_SYSTEM, prompt=prefix + "\n\nQuestion: " + question)
        encode_example(tokenizer, copy)
        corrected.append(copy)
    return corrected


def summarize(rows):
    fields = ("strict", "normalized", "lenient", "valid_sql", "valid_raw_sql", "stopped_before_cap")
    return {field: sum(row[field] for row in rows) / len(rows) for field in fields}


def calibration(rows):
    failures = []
    for row in rows:
        error = sql_error(row.context, row.answer)
        if error:
            failures.append({"source_index": row.source_index, "gold": row.answer, "error": error})
    rate = 1 - len(failures) / len(rows)
    LOG.info("SQLite %s gold validity %s/%s failures=%s", sqlite3.sqlite_version, len(rows) - len(failures), len(rows), failures)
    return {"sqlite_version": sqlite3.sqlite_version, "rate": rate, "failures": failures}


def save(result):
    destination = ROOT / "results/rescore.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(destination)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s.%(msecs)03d %(levelname)s %(message)s", datefmt="%Y-%m-%dT%H:%M:%S")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    loaded = load_candidate("A")
    original = prepare_candidate("A", tokenizer, loaded=loaded)
    gold = calibration(original.heldout)
    result = {"model": MODEL_ID, "revision": MODEL_REVISION, "seed": 42, "gold_calibration": gold, "generation_cap": original.max_new_tokens}
    save(result)
    if gold["rate"] < 0.99:
        raise RuntimeError("Gold validation did not meet the 99% threshold.")
    model = load_model(MODEL_ID, MODEL_REVISION)
    base = generate(model, tokenizer, original.heldout, original.max_new_tokens)
    result.update(base=summarize(base), base_outputs=base, examples=base[:10])
    for row in base[:10]:
        LOG.info("Raw base output and gold %s", json.dumps(row, ensure_ascii=False))
    rate = result["base"]["lenient"]
    LOG.info("Original base scores %s", result["base"])
    alternative = None
    if rate > 0.80:
        used = {normalize_prompt(row.prompt) for row in original.train}
        unique = {}
        for row in loaded[0]:
            key = normalize_prompt(row.prompt)
            if key not in used and HARDER.search(row.answer):
                unique.setdefault(key, row)
        alternative = random.Random(42).sample(list(unique.values()), 100)
        for row in alternative:
            encode_example(tokenizer, row)
        result["alternative"] = {"type": "harder_slice", "gold_calibration": calibration(alternative)}
    elif rate < 0.10:
        alternative = rerender(tokenizer, original.heldout)
        result["alternative"] = {"type": "required_main_prompt", "gold_calibration": gold}
    if alternative is not None:
        outputs = generate(model, tokenizer, alternative, original.max_new_tokens)
        result["alternative"].update(base=summarize(outputs), base_outputs=outputs)
        LOG.info("Alternative base scores %s", result["alternative"]["base"])
    save(result)
    LOG.info("Previous bake-off saved correctness flags only. Reproducing its 500-row probe adapter to regenerate outputs. This is separate from the main training runs.")
    model, stats = train_peft(model, tokenizer, original.train, LOG.info)
    predictions = generate(model, tokenizer, original.heldout, original.max_new_tokens)
    result.update(probe=summarize(predictions), probe_outputs=predictions, probe_training=stats)
    if alternative is not None:
        predictions = generate(model, tokenizer, alternative, original.max_new_tokens)
        result["alternative"].update(probe=summarize(predictions), probe_outputs=predictions)
    decision_rate = result.get("alternative", {}).get("base", result["base"])["lenient"]
    result["gate"] = "confirmed" if 0.10 <= decision_rate <= 0.80 else "unresolved"
    result["decision_base_lenient"] = decision_rate
    LOG.info("Probe scores %s; gate %s", result["probe"], result["gate"])
    save(result)


if __name__ == "__main__":
    main()
