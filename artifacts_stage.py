"""Merge, save NF4 artifacts, and verify the production directory in a new process."""
import argparse
import json
import os
from pathlib import Path
import runpy
import shutil
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from instruments import ROOT, LOG, atomic_json, phase, setup_logging

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"


def soup_merge(args):
    entrypoint, tag, adapter, base, output, dtype, save_format = args
    sys.argv = [
        "soup", "merge", "--adapter", adapter, "--base", base, "--output", output,
        "--dtype", dtype, "--save-format", save_format,
    ]
    with phase(tag, device="CPU" if save_format == "fp16" else "CPU merge, CUDA NF4"):
        try:
            runpy.run_path(entrypoint, run_name="__main__")
        except SystemExit as exc:
            if exc.code:
                raise


def peft_merge(args):
    tag, adapter, base, output, dtype = args
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    with phase(tag, device="CPU"):
        model = AutoModelForCausalLM.from_pretrained(base, dtype=getattr(torch, dtype), device_map="cpu")
        model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
        model.save_pretrained(output)
        AutoTokenizer.from_pretrained(base).save_pretrained(output)


def bnb_quantize(args):
    tag, source, destination = args
    import bitsandbytes as bnb
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    LOG.info("bitsandbytes %s configuration %s", bnb.__version__, config.to_json_string())
    with phase(tag):
        model = AutoModelForCausalLM.from_pretrained(source, dtype=torch.float16, quantization_config=config, device_map={"": 0})
        model.save_pretrained(destination)
        AutoTokenizer.from_pretrained(source).save_pretrained(destination)


def standalone(args):
    source, base = map(Path, args)
    import bitsandbytes as bnb
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

    missing = []
    for filename in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "generation_config.json"):
        if not (source / filename).exists() and (base / filename).exists():
            shutil.copy2(base / filename, source / filename)
            missing.append(filename)
    for filename in ("tokenizer.json", "tokenizer_config.json", "generation_config.json"):
        if not (source / filename).is_file():
            raise FileNotFoundError(source / filename)
    config = json.loads((source / "generation_config.json").read_bytes())
    config.update(do_sample=True, temperature=0.7, top_p=0.8, top_k=20, repetition_penalty=1.1)
    atomic_json(source / "generation_config.json", config)
    tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True)
    if not tokenizer.chat_template:
        raise RuntimeError("The quantized directory has no standalone chat template.")
    if "quantization_config" not in json.loads((source / "config.json").read_bytes()):
        raise RuntimeError("The production directory is not a quantized checkpoint.")
    eval_path = ROOT / "data/eval_amended.jsonl"
    if not eval_path.exists():
        eval_path = ROOT / "data/eval.jsonl"
    row = json.loads(eval_path.read_text(encoding="utf-8").splitlines()[0])
    cap = json.loads((ROOT / "results/data_stats.json").read_bytes())["max_new_tokens"]
    with phase("production_standalone_load_generate"):
        model = AutoModelForCausalLM.from_pretrained(source, dtype=torch.float16, device_map={"": 0}, local_files_only=True).eval()
        count = sum(isinstance(module, bnb.nn.Linear4bit) for module in model.modules())
        if count != 196:
            raise RuntimeError(f"Expected 196 4-bit projections, got {count}.")
        prompt = tokenizer.apply_chat_template(row["messages"][:-1], tokenize=True, add_generation_prompt=True, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            result = model.generate(prompt, use_model_defaults=False, do_sample=False, generation_config=GenerationConfig(
                max_new_tokens=cap, do_sample=False, pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
            ))[0, prompt.shape[1]:]
        answer = tokenizer.decode(result, skip_special_tokens=True)
    evidence = {
        "source_index": row["source_index"], "standalone": True, "model_path": str(source),
        "linear4bit_count": count, "bitsandbytes": bnb.__version__, "files": sorted(p.name for p in source.iterdir() if p.is_file()),
        "copied_files": missing, "generation_config": config, "answer": answer,
    }
    LOG.info("Artifact completeness %s", json.dumps(evidence))
    atomic_json(ROOT / "results/standalone.json", evidence)


if __name__ == "__main__":
    setup_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("soup_merge", "peft_merge", "bnb_quantize", "standalone"))
    parser.add_argument("args", nargs="+")
    opts = parser.parse_args()
    globals()[opts.mode](opts.args)
