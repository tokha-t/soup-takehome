"""Build the POST-HOC consistency report from saved CPU-readable evidence."""
from collections import Counter
import json
from pathlib import Path
import re
import shlex
import sys

from instruments import ROOT, atomic_json
from patch_common import read_json, sha
from s5_followup import evaluation, rate

RESULTS = ROOT / "results"
WORKSPACE = ROOT.parents[1]
PACKAGE = WORKSPACE / "work/soup_src/testenv/lib/python3.13/site-packages/soup_cli"
COMMAND = re.compile(r"^(\S+) INFO \$ ((?:\S*/)?soup ship(?:\s|$).*)$")
EXIT = re.compile(r" INFO Exit status (-?\d+)$")
JUDGE = re.compile(r"│ (Judge numerics: [^│]+)│")
LEG_1 = re.compile(r"Leg 1 task win \(([^)]+)\): ([0-9.]+) -> ([0-9.]+)")
NAMES = {
    "ship_a": ("base plus real LoRA adapter", "runs/real/adapter/adapter_model.safetensors"),
    "ship_b": ("production NF4 merged checkpoint with portable tokenizer", "artifacts/merged_4bit/model.safetensors"),
    "ship_c": ("base plus zero-learning-rate placebo adapter", "runs/placebo/adapter/adapter_model.safetensors"),
    "ship_d": ("base NF4 checkpoint supplied as the tuned artifact", "artifacts/base_4bit/model.safetensors"),
    "ship_shuffled": ("base plus shuffled-answer LoRA adapter", "runs/shuffled/adapter/adapter_model.safetensors"),
    "ship_weak_001": ("base plus weak_001 scaled LoRA adapter", "runs/weak_001/adapter_model.safetensors"),
    "ship_noise_floor_3": ("base plus real LoRA adapter", "runs/real/adapter/adapter_model.safetensors"),
}


def recorded_artifact_hash(local_path):
    _, name, *parts = Path(local_path).parts
    if local_path.startswith("runs/") and name != "weak_001":
        return read_json(RESULTS / "training_adapter_hashes.json")[name][parts[-1]]["sha256"]
    source = {"weak_001": "artifact_hashes_2b.json", "merged_4bit": "artifact_hashes_4.json",
              "base_4bit": "artifact_hashes_3.json"}[name]
    if name == "merged_4bit":
        name = "merged_4bit_portable"
    return read_json(RESULTS / source)[name][parts[-1]]["sha256"]


def ship_runs():
    task_file = ROOT / "data/ship_task_amended.jsonl"
    tasks = list(map(json.loads, task_file.read_text().splitlines()))
    assert len(tasks) == 100 and all(row["scoring"] == "exact" for row in tasks)
    task_sha = sha(task_file)
    invocations = {}
    for path in sorted((ROOT / "logs").rglob("*.log")):
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            match = COMMAND.fullmatch(line)
            if match:
                invocations.setdefault(match.groups(), []).append(path)
    runs = []
    for (timestamp, exact_command), paths in sorted(invocations.items()):
        primary = min(paths, key=lambda p: len(p.parts))
        log = primary.read_text(encoding="utf-8")
        argv = shlex.split(exact_command)
        value = lambda flag: argv[argv.index(flag) + 1] if flag in argv else None
        result_path = RESULTS / Path(value("--output")).name
        key = result_path.stem
        verdict = read_json(result_path) if result_path.exists() else None
        leg = LEG_1.search(log)
        judge = JUDGE.search(log)
        status = EXIT.findall(log)
        scored = verdict is not None and verdict["failed_rule"] != "adapter_noop"
        artifact, local_path = NAMES[key]
        leg_1 = {
            "measured": scored,
            "base": verdict["task_win"]["base"] if scored else None,
            "tuned": verdict["task_win"]["tuned"] if scored else None,
            "reported_console_scores": {"base": float(leg[2]), "tuned": float(leg[3])} if leg else None,
        }
        if scored:
            assert leg and leg_1["reported_console_scores"] == {k: leg_1[k] for k in ("base", "tuned")}
        leg_2 = verdict["benchmark_deltas"] if verdict else []
        if leg_2:
            evidence = read_json(RESULTS / f"{key}_evidence.json")
            assert evidence["provenance"]["binding"]["task_eval_sha256"] == task_sha
            assert evidence["numerics"] == verdict["numerics"]
            for row in leg_2:
                assert all(evidence["benchmarks"][row["name"]][side] == row[side] for side in ("base", "tuned"))
                assert round(row["tuned"] - row["base"], 6) == row["delta"]
        quantization = read_json((ROOT / local_path).with_name("config.json")).get("quantization_config") if value("--tuned") else None
        runs.append({
            "run": key, "started_utc": timestamp, "command": exact_command,
            "artifact": {
                "description": artifact, "base_argument": value("--base"),
                "tuned_argument": value("--tuned"), "adapter_argument": value("--adapter"),
                "local_weight_file": local_path, "recorded_weight_sha256": recorded_artifact_hash(local_path),
                "task_scored": scored, "checkpoint_quantization_config": quantization,
            },
            "leg_1": leg_1, "leg_2_per_suite": leg_2,
            "judge_numerics_line": judge[1].strip() if judge else None,
            "numerics_stamp": verdict.get("numerics") if verdict else None,
            "loader_line": next((line.split(" INFO ", 1)[1].strip() for line in log.splitlines() if "Live eval: loading" in line), None),
            "verdict": verdict["decision"] if verdict else None,
            "failed_rule": verdict["failed_rule"] if verdict else None,
            "status": ("scored" if scored else "adapter_noop_before_model_load" if verdict else "timed_out_before_verdict"),
            "exit_status": int(status[-1]) if status else None,
            "log": str(primary.relative_to(ROOT)),
            "duplicate_log_copies": [str(path.relative_to(ROOT)) for path in paths if path != primary],
            "verdict_file": str(result_path.relative_to(ROOT)) if verdict else None,
        })
    assert len(runs) == 8 and Counter(row["status"] for row in runs) == {
        "scored": 5, "adapter_noop_before_model_load": 1, "timed_out_before_verdict": 2,
    }
    return {
        "attempts": runs, "task_file": str(task_file.relative_to(ROOT)),
        "task_file_sha256": task_sha, "task_rows": len(tasks), "leg_1_scoring": "Soup exact match",
        "leg_2_delta_direction": "tuned minus base",
        "numerics_note": "Judge numerics records the CLI's live evaluation setting. The stored precision of a prequantized --tuned checkpoint is recorded in its quantization config.",
        "placebo_note": "The adapter_noop verdict prints 0.0000 on both sides. The log states that no model was loaded or evaluated.",
    }


def token_agreement(rows):
    tokens = sum(row["tokens"] for row in rows)
    length = tokens / len(rows)
    matches = sum(row["top1_quant"] for row in rows)
    agreement = matches / tokens
    return {
        "rows": len(rows), "answer_tokens": tokens, "mean_gold_answer_tokens": length,
        "merged_fp16_vs_4bit": {
            "agreement": rate(matches, tokens), "one_minus_agreement_to_mean_length": 1 - agreement**length,
        },
        "base_fp16_vs_4bit": {
            "agreement": None, "one_minus_agreement_to_mean_length": None,
            "reason": "The archived base fp16 versus base 4-bit record contains KL only.",
        },
    }


def verify_results():
    data = read_json(RESULTS / "verify_real.json")
    _, _, ids = evaluation()
    assert ids == set(read_json(RESULTS / "s5_diagnosis.json")["cohort"]["row_ids"])
    rows = data["per_row"]
    all_rows = token_agreement(rows)
    assert all_rows["merged_fp16_vs_4bit"]["agreement"]["rate"] == data["distances"]["top1_quantization_agreement"]
    generations = {}
    for name, key, parity in (
        ("base_fp16", "base", "base"), ("base_4bit", "base4", "base4"),
        ("merged_fp16", "merged", "real"), ("merged_4bit", "merged4", "real_4bit"),
    ):
        saved = data["raw_outputs"][key]
        assert len(saved) == 192 and len({r["id"] for r in saved if r["id"] in ids}) == len(ids)
        batch = data["padding_parity"][parity][0]
        generations[name] = {
            "lenient_match_191": rate(sum(row["lenient"] for row in saved if row["id"] in ids), len(ids)),
            "generation_batch_size": batch["scoring_batch_size"], "padding_side": "left",
            "max_new_tokens": data["generation_cap"], "do_sample": False, "repetition_penalty": 1.0,
            "parity_check": batch,
        }
    return {
        "source": "results/verify_real.json", "greedy": generations,
        "teacher_forced_agreement": {"all_192": all_rows, "compatibility_gold_valid_191": token_agreement(
            [row for row in rows if row["id"] in ids]
        )},
        "answer_length_definition": "Mean of verify.py per-row gold answer-token counts, excluding the stop token.",
        "probability_definition": "1 - agreement ** mean_answer_length. This estimates the chance of at least one top-1 disagreement under independent token decisions and a fixed mean answer length.",
        "source_lines": {
            "token_counts_and_top1": "verify.py:187-205",
            "generation_batch_and_left_padding": "verify.py:91-122 and probe_runtime.py:42-49",
        },
    }


def source_quote(path, first, last):
    return {
        "file": str(path.relative_to(WORKSPACE)), "first_line": first, "last_line": last,
        "quote": "\n".join(path.read_text().splitlines()[first - 1:last]),
    }


def prompt_source():
    render = source_quote(PACKAGE / "utils/live_eval.py", 171, 176)
    assert '[{"role": "user", "content": prompt}]' in render["quote"]
    default = source_quote(ROOT / "artifacts/merged_4bit/chat_template.jinja", 18, 18)
    assert "You are Qwen, created by Alibaba Cloud." in default["quote"]
    return {
        "installed_soup_cli_version": "0.77.2",
        "cli_supplies_system_message": False,
        "effective_qwen_system_message": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
        "rendering": render,
        "leg_1_call_site": source_quote(PACKAGE / "commands/ship.py", 649, 655),
        "task_prompt_dispatch": source_quote(PACKAGE / "eval/custom.py", 440, 444),
        "generator_encoding_call": source_quote(PACKAGE / "utils/live_eval.py", 333, 336),
        "qwen_template_default": default,
        "source_answer": "The live generator hands the chat template one user message. The Qwen template supplies its own default system turn.",
    }


def main():
    data = {
        "label": "POST-HOC",
        "official_verdict": read_json(RESULTS / "summary.json")["verdict"],
        "soup_ship": ship_runs(), "verify": verify_results(), "soup_ship_prompt_source": prompt_source(),
        "optional_gpu": {"status": "pending"},
    }
    if len(sys.argv) > 1:
        data["optional_gpu"] = read_json(Path(sys.argv[1]))
        assert data["optional_gpu"]["label"] == "POST-HOC"
    atomic_json(RESULTS / "consistency.json", data)
    print("POST-HOC report: 8 Soup ship attempts, 4 greedy models, 2 teacher-forced cohorts, prompt source.")


if __name__ == "__main__":
    main()
