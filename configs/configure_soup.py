"""Fill the generated Soup template using its installed configuration schema."""
import json
import math
from pathlib import Path
import sys

import yaml
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from probe_data import Example, MODEL_ID, MODEL_REVISION, encode_example
from soup_probe import MASKED_TEMPLATE

generated = ROOT / "configs/soup_generated.yaml"
stats = json.loads((ROOT / "results/data_stats.json").read_bytes())
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)


def tokenized(name, *, include_eos):
    rows = []
    for text in (ROOT / f"data/{name}.jsonl").read_text(encoding="utf-8").splitlines():
        raw = json.loads(text)
        message = raw["messages"]
        example = Example(raw["source_index"], message[0]["content"], message[1]["content"], raw["answer"])
        encode_example(tokenizer, example)
        if len(example.input_ids) > stats["max_length"]:
            raise RuntimeError(f"{name} exceeds the preregistered context window at row {example.source_index}")
        if not include_eos:
            last = max(i for i, label in enumerate(example.labels) if label != -100)
            example.labels[last] = -100
        rows.append({"input_ids": example.input_ids, "attention_mask": [1] * len(example.input_ids), "labels": example.labels})
    return Dataset.from_list(rows)


eval_data = tokenized("eval", include_eos=False)
for kind, filename in (("real", "train"), ("placebo", "train"), ("shuffled", "train_shuffled")):
    cache_path = ROOT / "data" / f"soup_tokens_{kind}"
    DatasetDict(train=tokenized(filename, include_eos=True), validation=eval_data).save_to_disk(cache_path)
    config = yaml.safe_load(generated.read_text(encoding="utf-8"))
    config.update(base=MODEL_ID, backend="transformers", output=f"runs/{kind}/adapter")
    config["data"].update(
        train=f"data/{filename}.jsonl", format="pre_tokenized", tokenized_path=str(cache_path.relative_to(ROOT)),
        val_split=0.0, max_length=stats["max_length"], train_on_responses_only=True,
        chat_template=MASKED_TEMPLATE,
    )
    steps = 2 * math.ceil(stats[filename]["rows"] / 8)
    config["training"].update(
        epochs=2, lr=0 if kind == "placebo" else 2e-4, batch_size=8, gradient_accumulation_steps=1,
        quantization="none", optimizer="adamw_torch", scheduler="cosine",
        warmup_ratio=9.999 / steps, gradient_checkpointing="full", train_on_eot=True,
        seed=42, data_seed=42, weight_decay=0.0, max_grad_norm=1.0, logging_steps=1,
        eval_steps=25, rewind_log=False,
    )
    config["training"]["lora"].update(
        r=16, alpha=32, dropout=0.0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    name = "soup.yaml" if kind == "real" else f"soup_{kind}.yaml"
    (ROOT / "configs" / name).write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
