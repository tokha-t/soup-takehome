"""Measure the fp16 vocabulary-loss scaling grid on one loaded base model."""
import gc
import inspect
import json
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoTokenizer
from transformers.loss import loss_utils

from configs.train_peft import TARGETS, read_rows
from instruments import LOG, ROOT, atomic_json, phase, setup_logging
from probe_data import MODEL_ID, MODEL_REVISION
from probe_runtime import load_model, seed_everything

POINTS = {"on": [(1, 256), (4, 256), (8, 256)], "off": [(1, 256), (2, 256)]}
GIB, MIB = 2**30, 2**20
BASE_FIXED = 2.875 + .275


def slopes(mode):
    activation = .17 if mode == "on" else 2.6
    return [logits + activation for logits in (1.45, 2.03)]


def main():
    setup_logging()
    seed_everything()
    for function in (loss_utils.ForCausalLMLoss, loss_utils.fixed_cross_entropy):
        for line in inspect.getsource(function).splitlines():
            LOG.info("transformers loss source %s", line)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    rows = read_rows("train", tokenizer, 512)
    with phase("memory_grid_model_load"):
        base = load_model(MODEL_ID, MODEL_REVISION)
    result = {"points": [], "fits": {}, "loss_source": {
        "ForCausalLMLoss": inspect.getsource(loss_utils.ForCausalLMLoss),
        "fixed_cross_entropy": inspect.getsource(loss_utils.fixed_cross_entropy),
        "tensors": "fp16 model logits, fp32 cast, cross-entropy internal log-softmax and backward buffers. Shifted labels are contiguous. The sliced PEFT implementation moves the vocabulary projection to the answer suffix.",
    }}
    for mode, points in POINTS.items():
        for batch, length in points:
            torch.cuda.empty_cache()
            seed_everything()
            model = get_peft_model(base, LoraConfig(
                task_type=TaskType.CAUSAL_LM, r=16, lora_alpha=32, lora_dropout=0,
                target_modules=TARGETS, bias="none",
            )).train()
            if mode == "on":
                model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            params = [p for p in model.parameters() if p.requires_grad]
            optim = torch.optim.AdamW(params, lr=2e-4, weight_decay=0.0)
            scaler = torch.amp.GradScaler("cuda", init_scale=256.0)
            chosen = [row for row in rows if len(row.input_ids) <= length][:batch]
            inputs = torch.tensor([row.input_ids + [tokenizer.pad_token_id]*(length-len(row.input_ids)) for row in chosen], device="cuda")
            masks = torch.tensor([[1]*len(row.input_ids)+[0]*(length-len(row.input_ids)) for row in chosen], device="cuda")
            labels = torch.tensor([row.labels+[-100]*(length-len(row.labels)) for row in chosen], device="cuda")

            def step():
                optim.zero_grad(set_to_none=True)
                with torch.autocast("cuda", torch.float16):
                    loss = model(input_ids=inputs, attention_mask=masks, labels=labels, use_cache=False).loss
                scaler.scale(loss).backward()
                old = scaler.get_scale()
                scaler.step(optim)
                scaler.update()
                if scaler.get_scale() < old:
                    raise FloatingPointError("The grid optimizer skipped a step.")
                torch.cuda.synchronize()
                return loss.item()

            record = {"checkpointing": mode, "batch": batch, "length": length, "tokens": batch*length,
                      "formula_peak_gib": [BASE_FIXED + batch * length * s / 1024 for s in slopes(mode)]}
            try:
                first = step()
                with phase(f"grid_{mode}_{batch}_{length}_steps2_3"):
                    losses = [first, step(), step()]
                    record.update(measured_allocated_gib=torch.cuda.max_memory_allocated()/GIB, loss=losses)
            except Exception as exc:
                LOG.exception("Memory grid %s %s %s failed", mode, batch, length)
                record["error"] = f"{type(exc).__name__}: {exc}"
            finally:
                model.gradient_checkpointing_disable()
                base = model.unload()
                del model, params, optim, scaler, inputs, masks, labels
                gc.collect()
                torch.cuda.empty_cache()
            result["points"].append(record)
            atomic_json(ROOT / "results/memory_grid.json", result)
    for mode in POINTS:
        valid = [x for x in result["points"] if x["checkpointing"] == mode and "measured_allocated_gib" in x]
        if len(valid) >= 2:
            design = [[1, point["tokens"]] for point in valid]
            fixed, slope = np.linalg.lstsq(design, [point["measured_allocated_gib"] for point in valid], rcond=None)[0]
            result["fits"][mode] = {
                "fixed_gib": fixed, "slope_mib_per_token": slope * 1024, "points": len(valid),
                "predicted_fixed_gib": BASE_FIXED, "predicted_slope_mib_per_token": slopes(mode),
            }
    atomic_json(ROOT / "results/memory_grid.json", result)


if __name__ == "__main__":
    main()
