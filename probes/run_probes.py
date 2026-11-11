"""Run SQL rendering, decoding, validation and control probes."""
from collections import Counter
import json
import os
from pathlib import Path
import sqlite3
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from transformers import AutoTokenizer

from configs.train_peft import read_rows
from instruments import LOG, ROOT, atomic_json, phase, setup_logging
from sql_metrics import extract_sql
from verify import generate, generation_summary, kl, load, logprob, pinned_base, read_eval

RESULTS = ROOT / "results"


def read(name):
    return json.loads((RESULTS / name).read_bytes())


def save(name, result):
    atomic_json(RESULTS / f"probe_{name}.json", result)
    LOG.info("Probe %s recorded", name)


def gold_mask_probes(tokenizer, base):
    data = read("data_stats.json")
    train = read_rows("train", tokenizer, data["max_length"])
    cutoff = data["p96_full_train"]
    answer_cut, eos_cut, either = [], [], []
    for row in train:
        cut = [target for target in row.labels[cutoff:] if target != -100]
        if any(target != tokenizer.eos_token_id for target in cut):
            answer_cut.append(row.source_index)
        if tokenizer.eos_token_id in cut:
            eos_cut.append(row.source_index)
        if cut:
            either.append(row.source_index)
    save("P1", {"max_length": cutoff, "rows": len(train), "answer_cut_rows": answer_cut,
                "stop_cut_rows": eos_cut, "either_cut": len(either), "either_rate": len(either)/len(train),
                "soup_gate_log": "logs/probe_P1_doctor.log"})
    eos_copy = AutoTokenizer.from_pretrained(base)
    eos_copy.pad_token = eos_copy.eos_token
    path = ROOT / "artifacts/pad_equals_eos_tokenizer"
    eos_copy.save_pretrained(path)
    for config in ("config.json", "generation_config.json"):
        import shutil
        if (base / config).is_file():
            shutil.copy2(base / config, path / config)
    lost = sum(row.labels.count(eos_copy.pad_token_id) > 0 for row in train)
    save("P2", {"rows": len(train), "eos_supervised_before": lost,
                "eos_masked_by_token_id": lost, "eos_masked_by_attention_mask": 0,
                "soup_gate_log": "logs/probe_P2_doctor.log"})


def decoding(tokenizer):
    stats = read("data_stats.json")
    real = read("verify_real.json")
    path = ROOT / "artifacts/merged_4bit"
    eval_path = ROOT / "data/eval.jsonl"
    eval_rows = read_eval(eval_path, tokenizer)
    cap = stats["max_new_tokens"]
    with phase("probe_decoding_4bit"):
        model = load(path)
        wrong_prompt = read_eval(eval_path, tokenizer, with_system=False)
        without = generate(model, tokenizer, wrong_prompt, cap)
        greedy = real["generations"]["merged4"]
        save("P3", {"with_system": greedy, "without_system": generation_summary(without),
                    "default_render": tokenizer.decode(wrong_prompt[0].prompt_ids), "soup_gate": None})
        shipped = json.loads((path / "generation_config.json").read_bytes())
        # Preserve the saved sampling parameters, with the registered cap and seed.
        keys = ("temperature", "top_p", "top_k", "repetition_penalty")
        defaults = {"temperature": .7, "top_p": .8, "top_k": 20, "repetition_penalty": 1.1}
        actual = {key: shipped.get(key, defaults[key]) for key in keys}
        outputs = {str(seed): generate(model, tokenizer, eval_rows, cap, sampled_config=actual, seed=seed) for seed in (42, 43, 44)}
        pooled = [row for rows in outputs.values() for row in rows]
        sample_summary = generation_summary(pooled)
        rep = generation_summary(generate(model, tokenizer, eval_rows, cap, rep=1.1))
        save("P4", {"shipped_generation_config": shipped, "effective_sampling": actual,
                    "sampling_by_seed": {key: generation_summary(value) for key, value in outputs.items()},
                    "sampling_pooled": sample_summary, "greedy_1_0": greedy, "greedy_1_1": rep,
                    "shipped_match_cost": greedy["all"]["lenient"] - sample_summary["all"]["lenient"],
                    "soup_gate": None, "sampling_outputs": outputs})
        leaky = read_eval(ROOT / "data/eval_leaky.jsonl", tokenizer)
        save("P10", {"schema_split": greedy, "leaky": generation_summary(generate(model, tokenizer, leaky, cap)),
                     "leaky_schema_overlap": stats["leaky_schema_overlap"], "leaky_question_overlap": stats["leaky_question_overlap"],
                     "schema_split_overlap": stats["train_eval_schema_overlap"], "soup_gate": None,
                     "caveat": "The leaky cohort's category mix differs from the schema-disjoint cohort."})
    dtype_stats = {
        name: {"rounded_away": evidence["rounded_to_zero_share"], "changed_count": evidence["changed_count"],
               "answer_loss": real["bootstrap"]["loss"][name if name != "fp16" else "merged"]["estimate"]}
        for name, evidence in [("fp16", real["merge_fp16"]), *[("merged_"+name, value) for name,value in real["merge_variants"].items()]]
    }
    save("P5", {"dtypes": dtype_stats, "bf16_forward": real["bf16_eval_compute"], "soup_gate": None})


@torch.inference_mode()
def precision_probe(tokenizer, base):
    rows = read_eval(ROOT / "data/eval_amended.jsonl", tokenizer)
    with phase("probe_P9_precision"):
        fp16, fp32 = load(base), load(base, dtype=torch.float32)
        nonfinite, maximum, kl_total, count = 0, 0.0, 0.0, 0
        for row in rows:
            ids = torch.tensor([row.input_ids], device="cuda")
            half = fp16(ids, use_cache=False).logits
            nonfinite += int(not torch.isfinite(half).all())
        for row in rows[:20]:
            ids = torch.tensor([row.input_ids], device="cuda")
            last = len(row.input_ids) - len(row.prompt_ids)
            full = fp32(ids, logits_to_keep=last, use_cache=False).logits.float()
            half = fp16(ids, logits_to_keep=last, use_cache=False).logits.float()
            maximum = max(maximum, (full - half).abs().max().item())
            kl_total += kl(full.log_softmax(-1), half.log_softmax(-1)).sum().item()
            count += full.shape[-2]
    save("P9", {"eval_rows": len(rows), "nonfinite_rows": nonfinite, "paired_prompts": 20,
                "max_abs_logit_difference": maximum, "kl_fp32_to_fp16": kl_total/count,
                "scaler_skips": {kind: read(f"train_{kind}.json")["skipped_steps"] for kind in ("real","placebo","shuffled")},
                "soup_gate": {"url": "https://trysoup.dev/docs/soup-ship",
                              "coverage": "The two release legs compare task accuracy and benchmark scores; neither compares raw fp16 and fp32 logits."}})


def main():
    setup_logging()
    base = pinned_base("Qwen/Qwen2.5-1.5B-Instruct")
    tokenizer = AutoTokenizer.from_pretrained(base)
    mode = sys.argv[1]
    if mode == "mask":
        gold_mask_probes(tokenizer, base)
    elif mode == "decode":
        decoding(tokenizer)
    else:
        precision_probe(tokenizer, base)


if __name__ == "__main__":
    main()
