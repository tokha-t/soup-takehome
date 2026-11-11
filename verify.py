"""Paired evaluation of training, format controls and the deployed NF4 artifact."""
import argparse
from collections import defaultdict
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import sqlite3

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from instruments import LOG, ROOT, atomic_json, phase, setup_logging
from probe_data import Example, MODEL_REVISION, encode_example
from probe_runtime import _left_pad, seed_everything
from sql_metrics import score, sql_error
from weight_checks import merge_identity

DISCLAIM = (
    "Memorization or leakage; behaviour outside this held-out SQL task; regressions in other skills "
    "(covered only by soup ship leg 2); single-seed variance; other runtimes (llama.cpp, vLLM); "
    "inputs longer than max_length. String matching rewards the dataset's quoting and aliases, "
    "so a correct query written differently can score wrong. Execution match (P11) addresses that."
)
HASHED = {}


def log_file(path):
    path = Path(path).resolve()
    if path not in HASHED:
        with path.open("rb") as stream:
            HASHED[path] = hashlib.file_digest(stream, "sha256").hexdigest()
        LOG.info("SHA256 %s  %s", HASHED[path], path)
    return HASHED[path]


def log_artifact(path):
    for file in sorted(Path(path).rglob("*")):
        if file.is_file():
            log_file(file)


def read_eval(path, tokenizer, with_system=True):
    records = []
    log_file(path)
    for text in Path(path).read_text(encoding="utf-8").splitlines():
        raw = json.loads(text)
        system, user, _ = raw["messages"]
        row = Example(raw["source_index"], system["content"], user["content"], raw["answer"], context=raw["context"], category=raw["category"])
        if with_system:
            encode_example(tokenizer, row)
        else:
            row.prompt_ids = tokenizer.apply_chat_template([user], tokenize=True, add_generation_prompt=True)
        records.append(row)
    return records


def load(path, *, dtype=torch.float16):
    log_artifact(path)
    return AutoModelForCausalLM.from_pretrained(path, dtype=dtype, device_map={"": 0}, attn_implementation="sdpa").eval()


def dispose(*objects):
    for obj in objects:
        del obj
    gc.collect()
    torch.cuda.empty_cache()


@torch.inference_mode()
def generate(model, tokenizer, rows, cap, *, rep=1.0, sampled_config=None, seed=42, audit=None):
    seed_everything()
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    settings = dict(sampled_config or {})
    settings.update(
        do_sample=bool(sampled_config),
        max_new_tokens=cap, pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id, repetition_penalty=settings.get("repetition_penalty", rep), use_cache=True,
    )
    settings.pop("max_length", None)
    config = GenerationConfig(**settings)
    LOG.info("Effective decoding request: do_sample=%s beams=1 repetition_penalty=%s use_model_defaults=False cap=%s",
             config.do_sample, config.repetition_penalty, cap)
    def decode(indices, batch_size):
        outputs = {}
        for start in range(0, len(indices), batch_size):
            subset = indices[start:start + batch_size]
            ids, mask = _left_pad([rows[index].prompt_ids for index in subset], tokenizer.pad_token_id, model.device)
            tokens = model.generate(
                input_ids=ids, attention_mask=mask, generation_config=config, use_model_defaults=False,
                do_sample=config.do_sample, num_beams=1, repetition_penalty=config.repetition_penalty,
            )[:, ids.shape[1]:].tolist()
            for index, generated in zip(subset, tokens, strict=True):
                if tokenizer.eos_token_id in generated:
                    generated = generated[:generated.index(tokenizer.eos_token_id) + 1]
                outputs[index] = generated
        return outputs

    batch_size = 8
    if not sampled_config:
        prefix = list(range(min(10, len(rows))))
        batched, unbatched = decode(prefix, 8), decode(prefix, 1)
        mismatches = [rows[index].source_index for index in prefix if batched[index] != unbatched[index]]
        try:
            assert batched == unbatched, f"Left-padding generation mismatches: {mismatches}"
        except AssertionError:
            batch_size = 1
        parity = {"rows": len(prefix), "equal": not mismatches, "mismatch_ids": mismatches, "scoring_batch_size": batch_size}
        LOG.info("Left-padding parity before scoring %s", parity)
        if audit is not None:
            audit.append(parity)
    order = sorted(range(len(rows)), key=lambda index: len(rows[index].prompt_ids))
    results = [None] * len(rows)
    for index, generated in decode(order, batch_size).items():
        stop = generated.index(tokenizer.eos_token_id) if tokenizer.eos_token_id in generated else None
        row = rows[index]
        answer = tokenizer.decode(generated[:stop], skip_special_tokens=False, clean_up_tokenization_spaces=False)
        results[index] = {
            "id": row.source_index, "category": row.category, "raw": answer,
            "stop": stop is not None and stop + 1 < cap,
            **score(row.context, answer, row.answer),
        }
    return results


def generation_summary(results):
    groups = defaultdict(list)
    for row in results:
        groups[row["category"]].append(row)
    groups["all"] = results
    return {name: {
        "rows": len(items),
        **{key: sum(x[key] for x in items) / len(items) for key in ("strict", "normalized", "lenient", "valid_sql", "valid_raw_sql", "stop")},
    } for name, items in groups.items()}


def answer_positions(row):
    # Keep the fixed chat prefix, but score the answer content only.
    end = len(row.prompt_ids) + sum(label != -100 for label in row.labels) - 1
    start = len(row.prompt_ids)
    if end <= start:
        raise ValueError(f"No answer tokens for {row.source_index}")
    return row.input_ids[:end - 1], row.input_ids[start:end]


@torch.inference_mode()
def logprob(model, row):
    prefix, targets = answer_positions(row)
    logits = model(
        torch.tensor([prefix], device=model.device),
        logits_to_keep=len(targets), use_cache=False,
    ).logits[0].float()
    if not torch.isfinite(logits).all():
        raise FloatingPointError(f"Non-finite logits on row {row.source_index}")
    return logits.log_softmax(dim=-1), torch.tensor(targets, device=model.device)


def kl(logp, logq):
    return (logp.exp() * (logp - logq)).sum(-1).clamp_min(0)


@torch.inference_mode()
def distribution(models, rows):
    records = []
    for start in range(0, len(rows), 8):
        batch = rows[start:start+8]
        prefixes, answers = zip(*(answer_positions(row) for row in batch), strict=True)
        exemplar = next(iter(models.values()))
        ids, mask = _left_pad(prefixes, exemplar.config.pad_token_id or 151643, exemplar.device)
        positions = (mask.cumsum(-1) - 1).clamp_min(0)
        width = max(map(len, answers))
        logs = {}
        for name, model in models.items():
            logits = model(ids, attention_mask=mask, position_ids=positions, logits_to_keep=width, use_cache=False).logits
            logits = logits.float()
            if not torch.isfinite(logits).all():
                raise FloatingPointError(f"Non-finite logits in {name} at row {start}")
            logs[name] = logits.log_softmax(dim=-1)
        for index, (row, answer) in enumerate(zip(batch, answers, strict=True)):
            count = len(answer)
            target = torch.tensor(answer, device=exemplar.device)
            distributions = {name: prob[index, -count:] for name, prob in logs.items()}
            record = {"id": row.source_index, "tokens": count,
                      "nll": {name: -prob.gather(-1, target[:, None]).sum().item() for name, prob in distributions.items()}}
            if {"base", "merged", "merged4"} <= distributions.keys():
                b, m, q = (distributions[name] for name in ("base", "merged", "merged4"))
                record.update(
                    kl_training=kl(m, b).sum().item(), kl_quantization=kl(m, q).sum().item(),
                    top1_train=(m.argmax(-1) == b.argmax(-1)).sum().item(),
                    top1_quant=(m.argmax(-1) == q.argmax(-1)).sum().item(),
                    fingerprint_to_merged=kl(q, m).mean().item(),
                    fingerprint_to_base=kl(q, b).mean().item(),
                )
            if {"base", "merged", "base4"} <= distributions.keys():
                b, m, q = (distributions[name] for name in ("base", "merged", "base4"))
                record["base_to_merged_kl"] = kl(b, m).sum().item()
                record["base_to_base4_kl"] = kl(b, q).sum().item()
            records.append(record)
        if start % 48 == 0:
            LOG.info("Teacher-forced rows %s/%s", len(records), len(rows))
    return records


def bootstrap(records):
    n = len(records)
    rng = np.random.default_rng(42)
    weights = np.array([row["tokens"] for row in records], dtype=np.float64)
    take = rng.integers(0, n, size=(10000, n))
    point, samples = {}, {}
    for name in records[0]["nll"]:
        loss = np.array([row["nll"][name] for row in records], dtype=np.float64)
        point[name] = loss.sum() / weights.sum()
        samples[name] = loss[take].sum(axis=1) / weights[take].sum(axis=1)

    def describe(value, distribution):
        usable = distribution[np.isfinite(distribution)]
        return {"estimate": float(value) if np.isfinite(value) else None,
                "ci95": np.quantile(usable, [0.025, 0.975]).tolist() if len(usable) else None,
                "finite_resamples": int(len(usable))}

    result = {"loss": {name: describe(mean, samples[name]) for name, mean in point.items()}}
    for label, left, right in (
        ("training", "base", "merged"), ("training4", "base4", "merged4"),
        ("format_fp16", "base", "shuffled"), ("content_fp16", "shuffled", "merged"),
        ("format_4bit", "base4", "shuffled4"), ("content_4bit", "shuffled4", "merged4"),
    ):
        if left in point and right in point:
            result[label] = describe(point[left] - point[right], samples[left] - samples[right])
    if "training4" in result:
        numerator = samples["base4"] - samples["merged4"]
        denominator = samples["base"] - samples["merged"]
        retention = np.divide(numerator, denominator, out=np.full(10000, np.nan), where=np.abs(denominator) > 1e-12)
        base_effect = result["training"]["estimate"]
        result["retention"] = describe(result["training4"]["estimate"] / base_effect if abs(base_effect) > 1e-12 else np.nan, retention)
    if "training" in result:
        loss = samples["base"]
        relative = np.divide(loss - samples["merged"], loss, out=np.full(10000, np.nan), where=loss > 1e-12)
        result["relative_answer_loss_reduction"] = describe(
            (point["base"] - point["merged"]) / point["base"] if point["base"] > 1e-12 else np.nan,
            relative,
        )
    if "unmerged" in point:
        result["merge_only_nll_change"] = describe(
            point["merged"] - point["unmerged"], samples["merged"] - samples["unmerged"],
        )
    return result


def paired(before, after, field="lenient"):
    pairs = list(zip(before, after, strict=True))
    if any(a["id"] != b["id"] for a, b in pairs):
        raise ValueError("Paired generations refer to different rows.")
    plus = sum(not a[field] and b[field] for a, b in pairs)
    minus = sum(a[field] and not b[field] for a, b in pairs)
    n = plus + minus
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(min(plus, minus) + 1)) / (2 ** n))
    return {"wrong_to_right": plus, "right_to_wrong": minus, "exact_mcnemar_p": p, "gain": (plus - minus) / len(pairs)}


def positive(metric):
    return metric is not None and metric["ci95"] is not None and metric["ci95"][0] > 0


def classify(weights, stats, distances, task_delta):
    loss_ci = stats["relative_answer_loss_reduction"]["ci95"]
    min_loss = bool(loss_ci and loss_ci[0] >= .10)
    min_task = task_delta["gain"] >= .10 and task_delta["exact_mcnemar_p"] < .01
    if not weights["changed_count"] or not min_loss:
        trained = "NOT_TRAINED"
    elif min_task:
        trained = "TRAINED"
    else:
        trained = "TRAINED_NOT_ON_TASK"
    retention = stats.get("retention")
    retained = retention and retention["estimate"] is not None and retention["estimate"] >= .9 and retention["ci95"] and retention["ci95"][0] >= .8 and distances["fingerprint_share"] >= .95
    return {"training": trained, "MIN_LOSS": min_loss, "MIN_TASK": min_task,
            "quantization": ("RETAINED" if retained else "LOST") if retention else None}


def distances(rows):
    n = sum(row["tokens"] for row in rows)
    base_to_merged = sum(row["base_to_merged_kl"] for row in rows) / n
    base_to_base4 = sum(row["base_to_base4_kl"] for row in rows) / n
    ratio = base_to_merged / base_to_base4 if base_to_base4 > 1e-12 else None
    result = {"base_to_merged_kl": base_to_merged, "base_to_base4_kl": base_to_base4,
              "quantization_noise_ratio": ratio, "ratio_below_3": ratio < 3 if ratio is not None else None}
    if "kl_training" not in rows[0]:
        return result
    training = sum(row["kl_training"] for row in rows) / n
    quantization = sum(row["kl_quantization"] for row in rows) / n
    fingerprint = [row["fingerprint_to_merged"] < row["fingerprint_to_base"] for row in rows]
    return {**result,
        "kl_training": training, "kl_quantization": quantization,
        "quantization_to_training_ratio": quantization / training if training > 1e-12 else None,
        "top1_training_agreement": sum(row["top1_train"] for row in rows) / n,
        "top1_quantization_agreement": sum(row["top1_quant"] for row in rows) / n,
        "fingerprint_share": sum(fingerprint) / len(rows), "fingerprint_rows": fingerprint,
    }


def pinned_base(path):
    return Path(path) if Path(path).is_dir() else Path(snapshot_download(path, revision=MODEL_REVISION, allow_patterns=[
        "*.safetensors", "*.safetensors.index.json", "config.json", "generation_config.json",
        "tokenizer*", "vocab.json", "merges.txt", "chat_template.jinja",
    ]))


def attach(records, addition, aliases=None):
    for row, extra in zip(records, addition, strict=True):
        if row["id"] != extra["id"] or row["tokens"] != extra["tokens"]:
            raise RuntimeError("Evaluation rows or answer token counts changed")
        row["nll"].update({(aliases or {}).get(name,name): loss for name,loss in extra["nll"].items()})


def main(args):
    from peft import PeftModel

    seed_everything()
    base = pinned_base(args.base)
    log_artifact(base)
    tokenizer = AutoTokenizer.from_pretrained(base)
    rows = read_eval(args.eval, tokenizer)
    stats_path = ROOT / "results/data_stats.json"
    log_file(stats_path)
    cap = json.loads(stats_path.read_bytes())["max_new_tokens"]
    gold_errors = [{"id": row.source_index, "error": error, "gold": row.answer}
                   for row in rows if (error := sql_error(row.context, row.answer))]
    calibration = {"sqlite_version": sqlite3.sqlite_version, "rate": 1-len(gold_errors)/len(rows), "failures": gold_errors}
    LOG.info("Gold SQLite calibration %s", calibration)
    with phase("core_load_base_models"):
        base_model, base4_model = load(base), load(args.base4)
    parity = {}
    def generated(name, model):
        parity[name] = []
        return generate(model, tokenizer, rows, cap, audit=parity[name])

    with phase("core_eval_base"):
        baseline = generated("base", base_model)
        again = generated("base_repeat", base_model)
        base4 = generated("base4", base4_model)
    nondeterministic = sum(a["raw"] != b["raw"] for a,b in zip(baseline,again,strict=True))
    outputs = {"base": baseline, "base4": base4}

    controls = dict(item.split("=", 1) for item in args.extra)
    if args.placebo_merged:
        controls["placebo"] = args.placebo_merged
    if args.shuffled_merged:
        controls["shuffled"] = args.shuffled_merged
    if args.case != "real" and controls:
        raise ValueError("Control merges can only be supplied on the real run")

    def evaluate_merge(case, merged, quantized=None, count_cast=False):
        log_artifact(merged)
        weights = merge_identity(base, merged, count_bf16_cast=count_cast)
        with phase(f"{case}_model_load"):
            model = load(merged)
            comparison = {"base":base_model, "base4":base4_model, "merged":model}
            if quantized:
                comparison["merged4"] = base4_model if Path(quantized).resolve() == Path(args.base4).resolve() else load(quantized)
        with phase(f"{case}_eval_core"):
            measured = distribution(comparison, rows)
            prediction = {"merged": generated(case, model)}
            if quantized:
                prediction["merged4"] = base4 if comparison["merged4"] is base4_model else generated(case+"_4bit", comparison["merged4"])
        del comparison, model
        dispose()
        return weights, measured, prediction

    def result(case, weights, records, prediction):
        metrics = bootstrap(records)
        shifts = distances(records)
        d_fp16 = paired(baseline, prediction["merged"])
        verdict = classify(weights, metrics, shifts, d_fp16)
        bundle = {
            "case":case, "base":str(args.base), "eval_rows":len(rows), "generation_cap":cap,
            "gold_sqlite_calibration":calibration, "merge_fp16":weights, "bootstrap":metrics, "distances":shifts,
            "generations":{name:generation_summary(items) for name,items in prediction.items()},
            "paired_fp16":d_fp16, "instrument_noise_outputs_different":nondeterministic,
            "per_row":records, "raw_outputs":prediction, "verdict":verdict,
            "limitations":DISCLAIM, "padding_parity":parity,
        }
        if "merged4" in prediction:
            bundle["paired_flips"] = paired(baseline, prediction["merged4"])
        return bundle

    control_rows = {}
    for name,path in controls.items():
        weights, records, prediction = evaluate_merge(name, path, args.shuffled4 if name=="shuffled" else None)
        control_rows[name] = (records,prediction)
        bundle = result(name, weights, records, {**outputs, **prediction})
        atomic_json(Path(args.out).with_name(f"verify_{name}.json"), bundle)
        LOG.info("Control %s verdict %s", name, bundle["verdict"])

    log_artifact(args.adapter)
    weights, records, prediction = evaluate_merge(args.case, args.merged, args.merged4, count_cast=True)
    outputs.update(prediction)
    if args.case == "real":
        if "shuffled" in control_rows:
            shuffled_records, shuffled_predictions = control_rows["shuffled"]
            attach(records, [{"id":item["id"], "tokens":item["tokens"],
                "nll":{name:item["nll"][name] for name in ("merged","merged4") if name in item["nll"]}}
                for item in shuffled_records], {"merged":"shuffled", "merged4":"shuffled4"})
            outputs.update({"shuffled" if key=="merged" else "shuffled4":value for key,value in shuffled_predictions.items()})
        with phase("real_eval_unmerged"):
            unmerged = PeftModel.from_pretrained(base_model, args.adapter, autocast_adapter_dtype=False).eval()
            attach(records, distribution({"unmerged":unmerged}, rows))
            base_model = unmerged.unload()
            del unmerged
    evidence = result(args.case, weights, records, outputs)
    atomic_json(args.out, evidence)
    LOG.info("%s verdict %s", args.case, evidence["verdict"])
    LOG.info("What this check does not detect: %s", DISCLAIM)


if __name__ == "__main__":
    setup_logging()
    parser = argparse.ArgumentParser()
    for flag in ("base", "adapter", "merged", "base4", "merged4", "eval", "out"):
        parser.add_argument("--" + flag, required=True)
    parser.add_argument("--placebo-merged")
    parser.add_argument("--shuffled-merged", "--shuffled", dest="shuffled_merged")
    parser.add_argument("--shuffled4")
    parser.add_argument("--extra", action="append", default=[], metavar="NAME=PATH")
    parser.add_argument("--case", default="real")
    main(parser.parse_args())
