"""Run the remaining patched probes before the 08:30 cutoff."""
import json
from pathlib import Path
import random
import shutil
import sys
from tempfile import TemporaryDirectory
import time

from instruments import ROOT, atomic_json, gpu_monitor, phase, setup_logging
from patch_common import CUTOFF, MODEL, PEFT, SOUP, assert_amendment, base_path, install, read_json, stage_result, stamp, step, use
from patch_stage_core import checkpoint, eval_path
from patch_stage_diagnostics import adapter, mounted_optional
from patch_stage_ship import ship
from patch_soup_config import make

RESULTS = ROOT / "results"


def log_json(name):
    raw = (ROOT / f"logs/{name}.log").read_text(encoding="utf-8")
    lines = [line.partition(" INFO ")[2] for line in raw.splitlines()]
    for index, line in enumerate(lines):
        if line.startswith("{"):
            try:
                return json.JSONDecoder().raw_decode("\n".join(lines[index:]))[0]
            except json.JSONDecodeError:
                continue
    return None


def gpu_probes(base):
    from transformers import AutoTokenizer
    from configs.train_peft import read_rows
    from probe_runtime import _answer_sequence
    from verify import generate, generation_summary, load, log_artifact, read_eval
    from weight_checks import nf4_checks

    setup_logging()
    stats = read_json(RESULTS / "data_stats.json")
    tokenizer = AutoTokenizer.from_pretrained(base)
    if time.time() < CUTOFF:
        with phase("nf4_delta_A3"):
            before, after = load(checkpoint("base_4bit")), load(checkpoint("merged_4bit"))
            log_artifact(adapter("real"))
            evidence = nf4_checks(before, after, adapter("real"))
        atomic_json(RESULTS / "weights_A3_nf4.json", evidence)
    if time.time() < CUTOFF:
        with phase("probe_P10_leaky_split"):
            leaky = read_eval(ROOT / "data/eval_leaky.jsonl", tokenizer)
            audit = []
            output = generate(after, tokenizer, leaky, stats["max_new_tokens"], audit=audit)
        atomic_json(RESULTS / "probe_P10.json", {
            "schema_disjoint": read_json(RESULTS / "verify_real.json")["generations"]["merged4"],
            "deliberately_leaky": generation_summary(output), "batching": audit, "outputs": output,
            "leaky_schema_overlap": stats["leaky_schema_overlap"], "leaky_question_overlap": stats["leaky_question_overlap"],
            "limitation": "The leaky and schema-disjoint cohorts also differ in category proportions.",
            "soup_gate": {"url": "https://trysoup.dev/docs/soup-ship", "coverage": "Soup ship scores the supplied task rows and bundled suites; it does not compare their SQL schemas with the training split."},
        })
    if time.time() < CUTOFF:
        train = read_rows("train", tokenizer, stats["max_length"])
        pad_eos = AutoTokenizer.from_pretrained(base)
        pad_eos.pad_token = pad_eos.eos_token
        path = ROOT / "artifacts/pad_equals_eos_tokenizer"
        pad_eos.save_pretrained(path)
        shutil.copy2(base / "config.json", path / "config.json")
        lost = sum(row.labels.count(pad_eos.pad_token_id) > 0 for row in train)
        atomic_json(RESULTS / "probe_P2_mask.json", {"rows": len(train), "eos_supervised_before": lost,
                    "eos_masked_by_token_id": lost, "eos_masked_by_attention_mask": 0})
        replayed = {}
        for label in ("real", "placebo", "shuffled"):
            sequences = [_answer_sequence(row) for row in read_rows("train_shuffled" if label == "shuffled" else "train", tokenizer, stats["max_length"])]
            rng, batches = random.Random(42), []
            for _ in range(2):
                order = list(range(len(sequences)))
                rng.shuffle(order)
                for start in range(0, len(order), 8):
                    ids = [sequences[index][0] for index in order[start:start+8]]
                    batches.append({"non_pad_tokens": sum(map(len, ids)), "padded_tokens": len(ids)*max(map(len, ids))})
            logged = read_json(RESULTS / f"train_{label}.json")["step_metrics"]
            assert len(batches) == len(logged)
            replayed[label] = {"steps": batches, "matches_original_logged_counts":
                               all(all(observed.get(key) == value for key,value in row.items()) for row,observed in zip(batches, logged, strict=True))}
        atomic_json(RESULTS / "replayed_training_padding.json", {"model_forward_performed": False, "runs": replayed})


def main():
    if time.time() >= CUTOFF:
        stage_result("8", "skipped", reason="08:30 cutoff")
        return
    amendment = assert_amendment()
    eval_path()
    for suffix in ("data/train.jsonl", "data/train_shuffled.jsonl", "data/train_soup.jsonl", "data/eval_leaky.jsonl",
                   "data/ship_task_amended.jsonl", "results/verify_real.json", "results/data_stats.json",
                   "results/train_real.json", "results/train_placebo.json", "results/train_shuffled.json"):
        use(suffix)
    python = install("optional-peft", PEFT)
    soup_python = install("optional-soup", SOUP)
    soup = soup_python.with_name("soup")
    base = base_path(python)
    config, _ = make(soup, base=base)
    task = ROOT / "data/ship_task_amended.jsonl"
    statuses = {}

    def run(label, args, *, success=(0,), limit=1200):
        result = step(args, label, success=success, optional=True, limit=limit)
        statuses[label] = {"exit_code": result, "finished": stamp(), "log": f"logs/stages/8/{label}.log"}
        atomic_json(RESULTS / "optional_commands.json", statuses)
        return result

    with gpu_monitor():
        run("memory_grid", [python, ROOT / "mem_grid.py"])
        if (RESULTS / "memory_grid.json").is_file():
            grid = read_json(RESULTS / "memory_grid.json")
            run("soup_profile_help", [soup, "profile", "--help"])
            for point in grid["points"]:
                mode, batch, length = (point[k] for k in ("checkpointing", "batch", "length"))
                label = f"profile_{mode}_{batch}_{length}"
                status = run(label, [soup, "profile", "--model", MODEL, "--seq", length, "--batch", batch,
                                    "--method", "lora", "--lora-r", 16, "--gradient-checkpointing",
                                    "full" if mode == "on" else "off", "--json"], limit=120)
                profile = log_json(label) if status == 0 else None
                if profile and profile.get("total_memory_gb") is not None:
                    point["soup_estimate_gib"] = profile["total_memory_gb"] * 1_000_000_000 / 2**30
                point["soup_profile"] = profile
                point["soup_profile_log"] = statuses[label]["log"]
            atomic_json(RESULTS / "memory_grid.json", grid)
        for name, artifact in (("real", "merged_4bit"), ("wrong", "base_4bit")):
            label = f"quant_check_{name}"
            status = run(label, [soup, "eval", "quant-check", "--before", checkpoint("merged_fp16"),
                                 "--after", checkpoint(artifact), "--tasks", task, "--format", "json"], success=(0, 2))
            if status in (0, 2) and (parsed := log_json(label)):
                atomic_json(RESULTS / f"{label}.json", parsed)
        baseline = use("results/soup_baseline.json") if mounted_optional("results/soup_baseline.json") else None
        extra = ["--baseline", baseline] if baseline else []
        statuses["ship_d"] = ship(soup, base, config, task, "d", "--tuned", checkpoint("base_4bit"), extra=extra, optional=True)
        with TemporaryDirectory(prefix="ship-control-", dir=ROOT) as directory:
            location = Path(directory)
            weak = Path(shutil.copytree(adapter("weak_001"), location / "weak_001"))
            statuses["ship_weak_001"] = ship(soup, base, config, task, "weak_001", "--adapter", weak, extra=extra, optional=True)
            atomic_json(RESULTS / "optional_commands.json", statuses)
            run("A3_and_secondary_probes", [python, __file__, "gpu_probes", base])
            if (RESULTS / "probe_P2_mask.json").is_file():
                path = ROOT / "artifacts/pad_equals_eos_tokenizer"
                run("probe_P2_doctor", [soup, "data", "doctor", ROOT / "data/train_soup.jsonl", "--model", path,
                                        "--sample", 1500, "--max-length", read_json(RESULTS / "data_stats.json")["max_length"],
                                        "--train-on-eot", "--output", RESULTS / "probe_P2_soup.json"], success=(0, 2), limit=300)
                actual = read_json(RESULTS / "probe_P2_soup.json") if (RESULTS / "probe_P2_soup.json").is_file() else statuses["probe_P2_doctor"]
                atomic_json(RESULTS / "probe_P2.json", {**read_json(RESULTS / "probe_P2_mask.json"), "soup_gate": actual})
            run("probe_P9_precision", [python, ROOT / "probes/run_probes.py", "precision"])
            run("probe_P11_execution", [python, ROOT / "probes/execution_match.py"])
            if time.time() < CUTOFF:
                tuned = Path(shutil.copytree(adapter("real"), location / "real"))
                statuses["ship_noise_floor_3"] = ship(soup, base, config, task, "noise_floor_3", "--adapter", tuned,
                                                     extra=[*extra, "--noise-floor", 3], optional=True)
    atomic_json(RESULTS / "optional_commands.json", statuses)
    stage_result("8", "complete" if time.time() < CUTOFF else "partial", amendment=amendment, commands=statuses)


if __name__ == "__main__":
    gpu_probes(Path(sys.argv[2])) if sys.argv[1] == "gpu_probes" else main()
