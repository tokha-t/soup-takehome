"""POST-HOC S5 generation and paired compatibility scoring."""
from datetime import datetime
from math import comb
from pathlib import Path
import re
import sqlite3
import sys
import time

from instruments import LOG, ROOT, atomic_json, gpu_monitor, phase, setup_logging
import patch_common
from patch_common import PEFT, install, mounted, read_json, sha, step, use
from sql_metrics import extract_sql, sql_error

LABEL = "POST-HOC"
SEEDS = (42, 43, 44)
RESULTS = ROOT / "results"
EVAL = ROOT / "data/eval_amended.jsonl"
PREVIOUS = RESULTS / "s5_posthoc.json"
MODEL_SHA = "7fae1b5b6b9477c163f3011f20583e7b7707ac34899c2bf42692bb5c2730e44a"
TOKEN = re.compile(
    r"""--[^\r\n]*|/\*.*?\*/|'(?:''|[^'])*'|"(?:""|[^"])*"|"""
    r"""\x60(?:\x60\x60|[^\x60])*\x60|\[[^\]]*\]|\$[A-Za-z_][\w$]*|[^\W\d][\w$]*|[0-9]+|\S""",
    re.S,
)
ALIAS_ERROR = re.compile(r'near "(\$[A-Za-z_][\w$]*)": syntax error', re.ASCII)


def compatibility_sql(schema, query):
    error = sql_error(schema, query)
    if error is None:
        return True
    tokens = [match for match in TOKEN.finditer(query) if not match[0].startswith(("--", "/*"))]
    if not tokens or tokens[0][0].upper() != "SELECT":
        return False
    if error == 'near "OFFSET": syntax error':
        depth = 0
        top = []
        for token in tokens:
            if token[0] == "(":
                depth += 1
            elif token[0] == ")":
                depth -= 1
            elif depth == 0:
                top.append(token)
        words = [token[0].upper() for token in top]
        if depth == 0 and words.count("OFFSET") == 1 and "LIMIT" not in words:
            offset = top[words.index("OFFSET")].start()
            if re.fullmatch(r"OFFSET\s+[0-9]+\s*;?\s*", query[offset:], re.I):
                return sql_error(schema, query[:offset] + "LIMIT -1 " + query[offset:]) is None
    match = ALIAS_ERROR.fullmatch(error)
    if match:
        for index, token in enumerate(tokens[1:], 1):
            next_word = tokens[index + 1][0].upper() if index + 1 < len(tokens) else None
            if token[0] == match[1] and tokens[index - 1][0].upper() == "AS" and next_word in {None, ",", "FROM", ";"}:
                quoted = query[:token.start()] + '"' + token[0] + '"' + query[token.end():]
                return sql_error(schema, quoted) is None
    return False


def evaluation():
    import json

    prior = read_json(PREVIOUS)
    assert sha(EVAL) == prior["method"]["input_sha256"]["data/eval_amended.jsonl"]
    records = {row["source_index"]: row for row in map(json.loads, EVAL.read_text().splitlines())}
    repairs = {row["id"]: row["compatibility_check"] for row in prior["gold_failures"]}
    schemas = {
        row_id: repairs.get(row_id, {}).get("schema_after_repair") or row["context"]
        for row_id, row in records.items()
    }
    valid = {
        row_id for row_id, row in records.items()
        if compatibility_sql(schemas[row_id], extract_sql(row["answer"]))
    }
    assert len(records) == 192 and len(valid) == 191 and set(records) - valid == {2961}
    return records, schemas, valid


def flags(saved, records, schemas, original_flag_disagreements=None):
    assert len(saved) == len(records) and {row["id"] for row in saved} == set(records)
    output = {}
    for row in saved:
        row_id = row["id"]
        query = extract_sql(row["raw"])
        original_error = sql_error(records[row_id]["context"], query)
        original = original_error is None
        if original != row["valid_sql"]:
            if original_flag_disagreements is None:
                raise ValueError(f"Saved and local validity flags differ for row {row_id}")
            original_flag_disagreements.append({
                "id": row_id, "sql": query, "local_sqlite_error": original_error,
                "local_original_valid": original, "kaggle_original_valid": row["valid_sql"],
            })
        output[row_id] = compatibility_sql(schemas[row_id], query)
    return output


def rate(count, total):
    return {"count": count, "total": total, "rate": count / total}


def scores(saved, records, schemas, ids, original_flag_disagreements=None):
    valid_flags = flags(saved, records, schemas, original_flag_disagreements)
    return {
        "stop": rate(sum(row["stop"] for row in saved if row["id"] in ids), len(ids)),
        "valid_sql": rate(sum(valid_flags[row_id] for row_id in ids), len(ids)),
    }


def mcnemar(before, after, ids):
    both_valid = sum(before[i] and after[i] for i in ids)
    valid_to_invalid = sum(before[i] and not after[i] for i in ids)
    invalid_to_valid = sum(not before[i] and after[i] for i in ids)
    discordant = valid_to_invalid + invalid_to_valid
    p = min(1.0, 2 * sum(comb(discordant, k) for k in range(min(valid_to_invalid, invalid_to_valid) + 1)) / 2**discordant)
    return {
        "before_valid": rate(both_valid + valid_to_invalid, len(ids)),
        "after_valid": rate(both_valid + invalid_to_valid, len(ids)),
        "both_valid": both_valid, "both_invalid": len(ids) - both_valid - discordant,
        "valid_to_invalid": valid_to_invalid, "invalid_to_valid": invalid_to_valid,
        "exact_two_sided_p": p,
    }


def local_report(run_url=None):
    records, schemas, ids = evaluation()
    raw = read_json(RESULTS / "verify_real.json")["raw_outputs"]
    without = read_json(RESULTS / "probe_P3.json")["outputs_without_system"]
    comparisons = {
        "base_fp16_vs_base_4bit": (raw["base"], raw["base4"]),
        "merged_fp16_vs_merged_4bit": (raw["merged"], raw["merged4"]),
        "merged_4bit_with_vs_without_system": (raw["merged4"], without),
    }
    tests = {}
    for name, (before, after) in comparisons.items():
        tests[name] = {
            "saved_original_flags_192": mcnemar(
                {row["id"]: row["valid_sql"] for row in before},
                {row["id"]: row["valid_sql"] for row in after}, records,
            ),
            "compatibility_flags_191": mcnemar(
                flags(before, records, schemas), flags(after, records, schemas), ids,
            ),
        }
    prior_diagnosis = read_json(RESULTS / "s5_diagnosis.json")["conditions"]
    for test, lhs, rhs in (
        ("base_fp16_vs_base_4bit", "base_greedy", "base_4bit_greedy"),
        ("merged_fp16_vs_merged_4bit", "merged_fp16_greedy", "merged_4bit_greedy"),
        ("merged_4bit_with_vs_without_system", "merged_4bit_greedy", "merged_4bit_without_system_prompt"),
    ):
        pair = tests[test]["compatibility_flags_191"]
        assert pair["before_valid"]["count"] == prior_diagnosis[lhs]["valid_sql"]["count"]
        assert pair["after_valid"]["count"] == prior_diagnosis[rhs]["valid_sql"]["count"]
    output = {
        "label": LABEL,
        "official_verdict": read_json(RESULTS / "summary.json")["verdict"],
        "metric": "valid SQL",
        "mcnemar": tests,
        "test_method": "Two-sided exact binomial test of the discordant pairs, reported without multiplicity adjustment.",
        "prompt_comparison": "Removing the task system prompt caused the Qwen chat template to supply its default system message.",
        "compatibility_scoring": {
            "sqlite_version": sqlite3.sqlite_version, "rows": len(ids), "excluded_gold_row": 2961,
            "validator_reference": "results/s5_posthoc.json",
        },
    }
    p4 = read_json(RESULTS / "probe_P4.json")
    sampling = {}
    for seed in SEEDS:
        path = RESULTS / f"posthoc_s5_fp16_seed_{seed}.json"
        if path.is_file():
            payload = read_json(path)
            assert payload["label"] == LABEL and payload["seed"] == seed and payload["model_sha256"] == MODEL_SHA
            assert payload["settings"] == p4["effective_sampling"] and payload["max_new_tokens"] == p4["max_new_tokens"]
            sampling[str(seed)] = scores(payload["outputs"], records, schemas, ids)
            assert sampling[str(seed)] == payload["scores_191"]
    output["fp16_shipped"] = {
        "per_seed": sampling, "requested_seeds": list(SEEDS),
        "generated_rows_per_seed": len(records), "scored_rows_per_seed": len(ids),
        "settings": p4["effective_sampling"], "max_new_tokens": p4["max_new_tokens"],
        "batching": "Generate all 192 prompts in the Stage 5 length-sorted batches of eight, then score the 191-row compatibility cohort.",
    }
    if len(sampling) == len(SEEDS):
        output["fp16_shipped"]["pooled"] = {
            metric: rate(sum(row[metric]["count"] for row in sampling.values()), len(ids) * len(SEEDS))
            for metric in ("stop", "valid_sql")
        }
    int8 = RESULTS / "posthoc_s5_int8_greedy.json"
    if int8.is_file():
        payload = read_json(int8)
        assert payload["label"] == LABEL and payload["model_sha256"] == MODEL_SHA
        assert payload["linear_8bit_modules"] > 0 and payload["settings"] == {"do_sample": False, "repetition_penalty": 1.0}
        disagreements = []
        local_scores = scores(payload["outputs"], records, schemas, ids, disagreements)
        kaggle_scores = payload["scores_191"]
        assert local_scores["stop"] == kaggle_scores["stop"]
        output["int8_greedy"] = {
            "scores_191": local_scores, "kaggle_scores_191": kaggle_scores,
            "linear_8bit_modules": payload["linear_8bit_modules"], "batching": payload["batching"],
            "original_flag_disagreements": disagreements,
        }
        if disagreements:
            assert [row["id"] for row in disagreements] == [54393]
            assert local_scores["valid_sql"]["count"] == kaggle_scores["valid_sql"]["count"] + 1
            output["int8_greedy"]["sqlite_version_note"] = (
                "Local SQLite accepts the numeric literal 2007_08 in row 54393, which Kaggle rejected. "
                "SQLite added underscores between digits in numeric literals in version 3.46.0."
            )
            output["int8_greedy"]["sqlite_reference"] = "https://www.sqlite.org/releaselog/3_46_0.html"
    stage = RESULTS / "stage_posthoc_s5_followup.json"
    if stage.is_file():
        output["kaggle_stage"] = read_json(stage)
    if run_url:
        output["kaggle_run_url"] = run_url
    sources = [
        EVAL, PREVIOUS, RESULTS / "s5_diagnosis.json", RESULTS / "summary.json",
        RESULTS / "verify_real.json", RESULTS / "probe_P3.json", RESULTS / "probe_P4.json",
        stage, int8, *(RESULTS / f"posthoc_s5_fp16_seed_{seed}.json" for seed in SEEDS),
    ]
    output["input_sha256"] = {str(path.relative_to(ROOT)): sha(path) for path in sources if path.is_file()}
    atomic_json(RESULTS / "s5_followup.json", output)
    print(output)


def gpu_generate(mode):
    import bitsandbytes as bnb
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from verify import generate, load, log_file, read_eval

    setup_logging()
    records, schemas, ids = evaluation()
    model_path = mounted("artifacts/merged_fp16/config.json").parent
    assert log_file(model_path / "model.safetensors") == MODEL_SHA
    p4 = read_json(RESULTS / "probe_P4.json")
    cap = p4["max_new_tokens"]
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    rows = read_eval(EVAL, tokenizer)
    assert len(rows) == 192
    assert torch.cuda.device_count() == 1
    LOG.info("%s %s on %s, bitsandbytes %s", LABEL, mode, torch.cuda.get_device_name(0), bnb.__version__)
    metadata = {"label": LABEL, "model_sha256": MODEL_SHA, "max_new_tokens": cap}
    if mode == "fp16":
        settings = p4["effective_sampling"]
        model = load(model_path)
        assert model.dtype == torch.float16
        for seed in SEEDS:
            with phase(f"posthoc_s5_fp16_seed_{seed}"):
                saved = generate(model, tokenizer, rows, cap, sampled_config=settings, seed=seed)
            measured = scores(saved, records, schemas, ids)
            atomic_json(RESULTS / f"posthoc_s5_fp16_seed_{seed}.json", {
                **metadata, "seed": seed, "settings": settings, "outputs": saved, "scores_191": measured,
            })
            LOG.info("%s fp16 seed %s %s", LABEL, seed, measured)
    else:
        config = BitsAndBytesConfig(load_in_8bit=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path, dtype=torch.float16, device_map={"": 0}, attn_implementation="sdpa",
            quantization_config=config, local_files_only=True,
        ).eval()
        count = sum(isinstance(module, bnb.nn.Linear8bitLt) for module in model.modules())
        assert model.is_loaded_in_8bit and count > 0
        batching = []
        with phase("posthoc_s5_int8_greedy"):
            saved = generate(model, tokenizer, rows, cap, seed=42, audit=batching)
        measured = scores(saved, records, schemas, ids)
        atomic_json(RESULTS / "posthoc_s5_int8_greedy.json", {
            **metadata, "seed": 42, "settings": {"do_sample": False, "repetition_penalty": 1.0},
            "linear_8bit_modules": count, "outputs": saved, "scores_191": measured, "batching": batching,
            "quantization_config": config.to_dict(),
        })
        LOG.info("%s 8-bit greedy %s", LABEL, measured)


def remote(deadline):
    patch_common.DEADLINE = datetime.fromisoformat(deadline).timestamp()
    stage = {"label": LABEL, "deadline": deadline}
    try:
        use("data/eval_amended.jsonl")
        use("results/probe_P4.json")
        python = install("posthoc-s5", PEFT)
        with gpu_monitor():
            stage["fp16_exit"] = step([python, Path(__file__), "fp16"], "posthoc_s5_fp16")
            if stage["fp16_exit"] == 0 and time.time() + 120 < patch_common.DEADLINE:
                stage["int8_exit"] = step([python, Path(__file__), "int8"], "posthoc_s5_int8")
            else:
                stage["int8_status"] = "Skipped because fp16 was incomplete or fewer than 120 seconds remained."
    except Exception as exc:
        stage["error"] = f"{type(exc).__name__}: {exc}"
        LOG.exception("%s S5 notebook setup failed", LABEL)
    finally:
        stage["finished"] = patch_common.stamp()
        atomic_json(RESULTS / "stage_posthoc_s5_followup.json", stage)
        print(f"{LABEL} S5 notebook finished: {stage}", flush=True)


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "local":
        local_report(sys.argv[2] if len(sys.argv) > 2 else None)
    elif mode in {"fp16", "int8"}:
        gpu_generate(mode)
    else:
        remote(sys.argv[2])
