"""Run the preregistered real, placebo, or shuffled fp16 LoRA experiment."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from peft import LoraConfig, TaskType, get_peft_model
from safetensors import safe_open
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

from instruments import LOG, ROOT, atomic_json, phase, setup_logging
from probe_data import Example, MODEL_ID, MODEL_REVISION, encode_example
from probe_runtime import _answer_nll, _answer_sequence, load_model, seed_everything

TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
TRAIN_BATCH = 8
EPOCHS = 2
WARMUP = 10


def read_rows(name, tokenizer, max_length):
    rows = []
    path = ROOT / f"data/{name}.jsonl"
    for line in path.read_text(encoding="utf-8").splitlines():
        raw = json.loads(line)
        row = Example(raw["source_index"], raw["messages"][0]["content"], raw["messages"][1]["content"], raw["answer"], context=raw["context"], category=raw["category"])
        encode_example(tokenizer, row)
        if len(row.input_ids) > max_length:
            raise RuntimeError(f"{name} source row {row.source_index} is longer than registered {max_length}")
        rows.append(row)
    return rows


@torch.inference_mode()
def eval_answer_loss(model, tokenizer, rows):
    model.eval()
    sequences = sorted(
        (_answer_sequence(row, exclude_stop_id=tokenizer.eos_token_id) for row in rows),
        key=lambda row: len(row[0]),
    )
    total_loss, tokens = 0.0, 0
    for start in range(0, len(sequences), TRAIN_BATCH):
        loss, count = _answer_nll(model, tokenizer, sequences[start:start + TRAIN_BATCH])
        total_loss += loss.item()
        tokens += count
    model.train()
    return total_loss / tokens


def assert_registration():
    prereg = ROOT / "prereg.md"
    expected = (ROOT / "prereg.sha256").read_text().split()[0]
    actual = hashlib.sha256(prereg.read_bytes()).hexdigest()
    if actual != expected:
        raise RuntimeError("Preregistration hash differs from the pre-training value.")
    stats = json.loads((ROOT / "results/data_stats.json").read_bytes())
    for name in ("train", "train_shuffled", "eval", "eval_leaky"):
        path = ROOT / f"data/{name}.jsonl"
        if hashlib.sha256(path.read_bytes()).hexdigest() != stats[name]["sha256"]:
            raise RuntimeError(f"Preregistered split changed: {name}")
    LOG.info("Verified preregistration sha256 %s", actual)
    return stats


def train(kind):
    stats = assert_registration()
    max_length = stats["max_length"]
    seed_everything()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    training = read_rows("train_shuffled" if kind == "shuffled" else "train", tokenizer, max_length)
    heldout = read_rows("eval", tokenizer, max_length)
    with phase(f"{kind}_model_load"):
        base = load_model(MODEL_ID, MODEL_REVISION)
        model = get_peft_model(base, LoraConfig(
            task_type=TaskType.CAUSAL_LM, r=16, lora_alpha=32, lora_dropout=0.0,
            target_modules=TARGETS, bias="none", base_model_name_or_path=MODEL_ID,
            revision=MODEL_REVISION,
        )).train()
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=0.0 if kind == "placebo" else 2e-4, weight_decay=0.0)
    steps_total = EPOCHS * math.ceil(len(training) / TRAIN_BATCH)
    scheduler = get_cosine_schedule_with_warmup(optimizer, WARMUP, steps_total)
    scaler = torch.amp.GradScaler("cuda", init_scale=256.0)
    sequences = [_answer_sequence(row) for row in training]
    steps, evaluations = [], []
    rng = random.Random(42)
    skipped = 0
    started = time.monotonic()
    with phase(f"{kind}_training"):
        for epoch in range(1, EPOCHS + 1):
            indices = list(range(len(training)))
            rng.shuffle(indices)
            for start in range(0, len(indices), TRAIN_BATCH):
                step_started = time.monotonic()
                batch = [sequences[index] for index in indices[start:start + TRAIN_BATCH]]
                non_pad = sum(len(ids) for ids, _ in batch)
                total = len(batch) * max(len(ids) for ids, _ in batch)
                optimizer.zero_grad(set_to_none=True)
                lr = optimizer.param_groups[0]["lr"]
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    loss, tokens = _answer_nll(model, tokenizer, batch)
                    loss = loss / tokens
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                norm = torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
                old_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                new_scale = scaler.get_scale()
                overflow = new_scale < old_scale
                if overflow:
                    skipped += 1
                else:
                    scheduler.step()
                loss_value, grad_norm = loss.item(), norm.item()
                metrics = {
                    "step": len(steps) + 1, "epoch": epoch,
                    "loss": loss_value if math.isfinite(loss_value) else None,
                    "grad_norm": grad_norm if math.isfinite(grad_norm) else None,
                    "lr": lr, "scale_before": old_scale, "scale": new_scale,
                    "overflow": overflow, "non_pad_tokens": non_pad, "padded_tokens": total,
                    "answer_tokens": tokens, "step_seconds": time.monotonic() - step_started,
                }
                steps.append(metrics)
                LOG.info("%s optimizer %s", kind, json.dumps(metrics))
                if len(steps) % 25 == 0:
                    eval_started = time.monotonic()
                    evaluation = {"step": len(steps), "answer_nll": eval_answer_loss(model, tokenizer, heldout), "seconds": time.monotonic() - eval_started}
                    evaluations.append(evaluation)
                    LOG.info("%s heldout %s", kind, json.dumps(evaluation))
        optimizer.zero_grad(set_to_none=True)
    wall = time.monotonic() - started
    path = ROOT / "runs" / kind / "adapter"
    with phase(f"{kind}_adapter_save"):
        model.save_pretrained(path)
        tokenizer.save_pretrained(path)
        adapter = path / "adapter_model.safetensors"
        with safe_open(adapter, framework="pt", device="cpu") as handle:
            keys = list(handle.keys())
            dtypes = sorted({str(handle.get_slice(name).get_dtype()) for name in keys})
        if len(keys) != 392:
            raise RuntimeError(f"Expected 392 LoRA tensors and got {len(keys)}")
    timed = steps[5:]
    result = {
        "run": kind, "model": MODEL_ID, "model_revision": MODEL_REVISION,
        "max_length": max_length, "epochs": EPOCHS, "batch_size": TRAIN_BATCH,
        "optimizer_steps_attempted": len(steps), "optimizer_steps_taken": len(steps) - skipped,
        "skipped_steps": skipped, "step_metrics": steps, "heldout": evaluations,
        "wall_seconds": wall, "non_pad_tokens_per_second_excluding_warmup_and_eval": sum(s["non_pad_tokens"] for s in timed) / sum(s["step_seconds"] for s in timed),
        "trainer_non_pad_tokens_per_second_including_eval": sum(s["non_pad_tokens"] for s in steps) / wall,
        "padding_share": 1 - sum(s["non_pad_tokens"] for s in steps) / sum(s["padded_tokens"] for s in steps),
        "adapter_tensor_count": len(keys), "adapter_dtypes": dtypes,
        "adapter_sha256": hashlib.sha256(adapter.read_bytes()).hexdigest(),
        "adapter_parameters": sum(parameter.numel() for parameter in params),
    }
    atomic_json(ROOT / "results" / f"train_{kind}.json", result)
    LOG.info("Training %s %s", kind, json.dumps({key: value for key, value in result.items() if key not in {"step_metrics", "heldout"}}))


if __name__ == "__main__":
    setup_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("run", choices=("real", "placebo", "shuffled"))
    train(parser.parse_args().run)
