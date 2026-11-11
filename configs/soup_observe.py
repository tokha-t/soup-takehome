"""Observe Soup's upstream HF Trainer without changing its forward or optimizer."""
from contextlib import ExitStack
import json
import os
from pathlib import Path
import runpy
import sys
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from transformers import Trainer, TrainerCallback
from safetensors import safe_open
from instruments import ROOT, LOG, atomic_json, phase, setup_logging
from configs.train_peft import assert_registration

setup_logging()
assert_registration()
kind = sys.argv[1]
entrypoint = sys.argv[2]
sys.argv = ["soup", *sys.argv[3:]]
steps, heldout = [], []
train_start = time.monotonic()
trainer_init = Trainer.__init__
training_step = Trainer.training_step
active = ExitStack()


class Observer(TrainerCallback):
    def __init__(self, trainer):
        self.trainer = trainer
        self.timer = 0.0
        self.non_pad = 0
        self.padded = 0
        self.scale_before = None
        self.lr = None

    def on_train_begin(self, args, state, control, **kwargs):
        if not args.fp16 or args.bf16:
            raise RuntimeError(f"Soup trainer precision differs from preregistration: fp16={args.fp16}, bf16={args.bf16}")
        if args.get_warmup_steps(state.max_steps) != 10:
            raise RuntimeError("Soup did not resolve to ten warmup steps.")
        active.enter_context(phase(f"{kind}_training"))

    def on_step_begin(self, args, state, control, **kwargs):
        self.timer = time.monotonic()
        self.non_pad = self.padded = 0
        scaler = self.trainer.accelerator.scaler
        self.scale_before = scaler.get_scale() if scaler else None
        self.lr = self.trainer.optimizer.param_groups[0]["lr"]

    def on_step_end(self, args, state, control, **kwargs):
        scaler = self.trainer.accelerator.scaler
        steps.append({
            "step": state.global_step, "non_pad_tokens": self.non_pad, "padded_tokens": self.padded,
            "scale_before": self.scale_before, "scale": scaler.get_scale() if scaler else None,
            "overflow": self.trainer.accelerator.optimizer_step_was_skipped,
            "lr": self.lr, "step_seconds": time.monotonic() - self.timer,
        })

    def on_log(self, args, state, control, logs=None, **kwargs):
        if "loss" in (logs or {}) and steps:
            steps[-1].update(loss=logs["loss"], grad_norm=logs.get("grad_norm"))
            LOG.info("%s optimizer %s", kind, json.dumps(steps[-1]))
        if "eval_loss" in (logs or {}):
            heldout.append({"step": state.global_step, "answer_nll": logs["eval_loss"], "seconds": logs.get("eval_runtime")})
        if "train_runtime" in (logs or {}):
            LOG.info("Soup trainer throughput %s", json.dumps(logs))

    def on_train_end(self, args, state, control, **kwargs):
        active.close()


def observe_init(self, *args, **kwargs):
    with phase(f"{kind}_trainer_initialize"):
        trainer_init(self, *args, **kwargs)
    self.soup_takehome_observer = Observer(self)
    self.add_callback(self.soup_takehome_observer)


def observe_step(self, model, inputs, *args, **kwargs):
    callback = getattr(self, "soup_takehome_observer", None)
    if callback is not None:
        callback.non_pad += int(inputs["attention_mask"].sum())
        callback.padded += inputs["attention_mask"].numel()
    return training_step(self, model, inputs, *args, **kwargs)


Trainer.__init__ = observe_init
Trainer.training_step = observe_step
try:
    try:
        runpy.run_path(entrypoint, run_name="__main__")
    except SystemExit as exc:
        if exc.code:
            raise
finally:
    active.close()

if len(steps) < 375:
    raise RuntimeError(f"Soup produced {len(steps)} observed optimizer steps")
adapter = ROOT / f"runs/{kind}/adapter/adapter_model.safetensors"
with safe_open(adapter, framework="pt", device="cpu") as weights:
    keys = list(weights.keys())
    dtypes = sorted({str(weights.get_slice(name).get_dtype()) for name in keys})
timed = steps[5:]
atomic_json(ROOT / f"results/train_{kind}.json", {
    "run": kind, "backend": "Soup", "adapter_tensor_count": len(keys), "adapter_dtypes": dtypes,
    "step_metrics": steps, "heldout": heldout,
    "skipped_steps": sum(s["overflow"] for s in steps),
    "wall_seconds": time.monotonic() - train_start,
    "non_pad_tokens_per_second_excluding_warmup_and_eval": sum(s["non_pad_tokens"] for s in timed) / sum(s["step_seconds"] for s in timed),
    "padding_share": 1 - sum(s["non_pad_tokens"] for s in steps) / sum(s["padded_tokens"] for s in steps),
})
