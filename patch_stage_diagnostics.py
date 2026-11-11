"""Measure merge reconstruction, wrong artifact identity, and Soup diagnostics."""
import gc
from pathlib import Path
import runpy
import shutil
import sys
from tempfile import TemporaryDirectory
import time

from instruments import LOG, ROOT, atomic_json, gpu_monitor, phase, setup_logging
from patch_common import CUTOFF, PEFT, SOUP, assert_amendment, base_path, install, mounted, read_json, stage_result, stamp, step, use
from patch_stage_core import checkpoint, eval_path
from patch_stage_ship import ship
from patch_soup_config import bench_config, make

RESULTS = ROOT / "results"
KINDS = ("real", "placebo", "shuffled", "weak_001", "weak_01")


def adapter(label):
    path = f"runs/{label}/" + ("" if label.startswith("weak_") else "adapter/")
    return mounted(path + "adapter_model.safetensors").parent


def compute(base):
    from transformers import AutoTokenizer
    import torch

    from configs.train_peft import read_rows
    from verify import HASHED, bootstrap, distribution, load, log_artifact, log_file, read_eval
    from weight_checks import merge_checks

    setup_logging()
    tokenizer = AutoTokenizer.from_pretrained(base)
    rows = read_eval(ROOT / "data/eval_amended.jsonl", tokenizer)
    stats = read_json(RESULTS / "data_stats.json")
    train = read_rows("train", tokenizer, stats["max_length"])
    cutoff = stats["p96_full_train"]
    answer_cut, eos_cut, either = [], [], []
    for row in train:
        cut = [target for target in row.labels[cutoff:] if target != -100]
        if cut:
            either.append(row.source_index)
        if any(target != tokenizer.eos_token_id for target in cut):
            answer_cut.append(row.source_index)
        if tokenizer.eos_token_id in cut:
            eos_cut.append(row.source_index)
    atomic_json(RESULTS / "probe_P1_mask.json", {"max_length": cutoff, "rows": len(train), "answer_cut_rows": answer_cut,
                "stop_cut_rows": eos_cut, "either_cut": len(either), "either_rate": len(either)/len(train)})

    log_artifact(base)
    log_file(RESULTS / "data_stats.json")
    for label in ("merged_4bit", "base_4bit", "shuffled_4bit"):
        log_artifact(checkpoint(label))
    dtypes = {}
    native_bf16 = torch.cuda.is_bf16_supported(including_emulation=False)
    for name, dtype in (("fp16", torch.float16), ("bf16", torch.bfloat16), ("fp32", torch.float32)):
        if time.time() >= CUTOFF:
            break
        path = checkpoint("merged_" + name)
        log_artifact(path)
        log_artifact(adapter("real"))
        with phase("reconstruct_" + name, device="CPU safetensors, CUDA delta"):
            evidence = merge_checks(base, path, adapter("real"))
        with phase("eval_" + name):
            actual_dtype = dtype if name != "bf16" or native_bf16 else torch.float32
            model = load(path, dtype=actual_dtype)
            evidence["answer_loss"] = bootstrap(distribution({name: model}, rows))["loss"][name]
            evidence["forward_dtype"] = str(actual_dtype)
            del model
            gc.collect()
            torch.cuda.empty_cache()
        dtypes[name] = evidence
        atomic_json(RESULTS / "probe_P5.json", {
            "dtypes": dtypes, "finished": stamp(), "native_bf16": native_bf16,
            "soup_gate": {"log": "logs/stages/2a/soup_merge_help.log",
                          "coverage": "Soup reports requested compute and save format; the saved dtype, reconstruction and answer losses are verified independently."},
        })

    for name in ("weak_001", "weak_01", "placebo", "shuffled"):
        if time.time() >= CUTOFF:
            break
        path = checkpoint(f"{name}_fp16")
        log_artifact(path)
        log_artifact(adapter(name))
        with phase("reconstruct_" + name, device="CPU safetensors, CUDA delta"):
            evidence = merge_checks(base, path, adapter(name))
        atomic_json(RESULTS / f"weights_{name}.json", evidence)
    atomic_json(RESULTS / "verification_input_hashes.json", {
        "purpose": "Re-emit hashes through verify.py after its INFO records were suppressed in the core Kaggle stdout.",
        "sha256": {str(path): value for path, value in HASHED.items()},
        "finished": stamp(),
    })

def validate(soup):
    source = ROOT / "data/train_soup.jsonl"
    lines = source.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1500
    broken = ROOT / "data/train_soup_half_broken.jsonl"
    broken.write_text("\n".join(lines[:750] + ['{"messages":[]}'] * 750) + "\n", encoding="utf-8")
    codes = {}
    for label, path, extra in (
        ("clean_strict", source, ["--min-valid-fraction", "1.0"]), ("half_default", broken, []),
        ("half_strict", broken, ["--min-valid-fraction", "1.0"]),
    ):
        codes[label] = step([soup, "data", "validate", path, *extra], f"validate_{label}", success=(0, 1, 2), optional=True, limit=120)
    atomic_json(RESULTS / "probe_P12.json", {"total_rows": 1500, "corrupt_rows": 750, "exit_codes": codes,
                "gate_logs": {name: f"logs/stages/7/validate_{name}.log" for name in codes}, "finished": stamp()})


def recover_merge_metadata():
    """Preserve the committed merge logs separately from the very large model downloads."""
    for group in ("training", "2a", "2b"):
        stamp_file = "results/train_real.json" if group == "training" else f"results/stage_{group}.json"
        source = mounted(stamp_file).parent.parent
        directories = ("artifacts", "logs", "results", "runs") if group != "training" else (
            "logs", "results", *[f"runs/{label}/adapter" for label in KINDS[:3]],
        )
        for directory in directories:
            for file in (source / directory).rglob("*"):
                if file.is_file() and file.stat().st_size <= 128 * 2**20:
                    destination = ROOT / "recovery" / group / file.relative_to(source)
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(file, destination)


def bench(entry, config, label, steps):
    setup_logging()
    sys.argv = [entry, "bench", "train", "--config", config, "--steps", steps, "--warmup", "3",
                "--output", str(RESULTS / f"soup_bench_{label}.json")]
    LOG.info("Soup benchmark argv: %s", sys.argv)
    with phase(f"soup_bench_{label}", device="CUDA"):
        try:
            runpy.run_path(entry, run_name="__main__")
        except SystemExit as exc:
            if exc.code:
                raise


def main():
    if time.time() >= CUTOFF:
        stage_result("7", "skipped", reason="08:30 cutoff")
        return
    amendment = assert_amendment()
    eval_path()
    for path in ("results/data_stats.json", "data/train.jsonl", "data/train_soup.jsonl", "data/ship_task_amended.jsonl"):
        use(path)
    for label in KINDS[:3]:
        use(f"results/train_{label}.json")
    recover_merge_metadata()
    python = install("diagnostic-peft", PEFT)
    soup_python = install("diagnostic-soup", SOUP)
    soup = soup_python.with_name("soup")
    base = base_path(python)
    real, placebo = make(soup, base=base)
    commands = {}

    def run(name, args, *, success=(0, 2), limit=600):
        code = step(args, name, success=success, optional=True, attempts=2, limit=limit)
        commands[name] = {"exit_code": code, "finished": stamp(), "log": f"logs/stages/7/{name}.log"}
        atomic_json(RESULTS / "diagnostic_commands.json", commands)
        return code

    with gpu_monitor():
        run("diagnostic_weights", [python, __file__, "compute", base], success=(0,), limit=1800)
        wrong = RESULTS / "verify_wrong.json"
        run("verify_wrong", [python, ROOT / "verify.py", "--base", base, "--adapter", adapter("real"),
                             "--merged", checkpoint("merged_fp16"), "--base4", checkpoint("base_4bit"),
                             "--merged4", checkpoint("base_4bit"), "--eval", ROOT / "data/eval_amended.jsonl",
                             "--out", wrong, "--case", "wrong"], success=(0,), limit=1800)
        if wrong.is_file():
            atomic_json(RESULTS / "probe_P6.json", {
                "wrong": read_json(wrong)["distances"], "real": read_json(mounted("results/verify_real.json"))["distances"],
                "soup_gate": {"real": "results/ship_b.json", "adapter": "results/ship_a.json",
                              "optional_wrong": "results/ship_d.json", "optional_quant_check": "results/quant_check_wrong.json"},
                "finished": stamp(),
            })
        with TemporaryDirectory(prefix="check-local-", dir=ROOT) as root:
            root = Path(root)
            local = {}
            for label in KINDS:
                local[label] = Path(shutil.copytree(adapter(label), root / label))
                run(f"adapters_check_{label}", [soup, "adapters", "check", local[label], "--json"])
            run("adapters_audit_real", [soup, "adapters", "audit", local["real"], "--config", real, "--json"])
            for label, config in (("real", real), ("placebo", placebo)):
                steps = 20
                run(f"bench_{label}", [soup_python, __file__, "bench", soup, bench_config(config, steps), label, steps],
                    success=(0, 1, 2, 3) if label == "placebo" else (0, 2), limit=1800)
            task = ROOT / "data/ship_task_amended.jsonl"
            baseline = use("results/soup_baseline.json") if mounted_optional("results/soup_baseline.json") else None
            extra = ["--baseline", baseline] if baseline else []
            shuffled = ship(soup, base, real, task, "shuffled", "--adapter", local["shuffled"], extra=extra, optional=True)
            atomic_json(RESULTS / "probe_P8.json", {"ship": shuffled, "adapters_check": commands["adapters_check_shuffled"],
                         "verify": "results/verify_shuffled.json", "part_C": "results/verify_real.json"})
        run("probe_P1_doctor", [soup, "data", "doctor", ROOT / "data/train_soup.jsonl", "--model", base, "--sample", 1500,
                                "--max-length", read_json(RESULTS / "data_stats.json")["p96_full_train"], "--train-on-eot",
                                "--output", RESULTS / "probe_P1_soup.json"])
        if (RESULTS / "probe_P1_soup.json").is_file() and (RESULTS / "probe_P1_mask.json").is_file():
            doctor = read_json(RESULTS / "probe_P1_soup.json")
            measured = read_json(RESULTS / "probe_P1_mask.json")
            truncation = next((c for c in doctor.get("checks", []) if c["name"] == "truncation_risk"), {})
            confirmed = doctor.get("rows_scanned") == 1500 and truncation.get("verdict") == "OK" and measured["either_cut"] > 0
            atomic_json(RESULTS / "probe_P1.json", {**measured, "soup_gate": doctor,
                        "assessment": "documented behaviour confirmed" if confirmed else "not confirmed",
                        "documentation": "https://trysoup.dev/docs/cli-reference", "finished": stamp()})
        validate(soup)
    atomic_json(RESULTS / "probe_P7.json", {"adapters_check": commands["adapters_check_placebo"],
                 "verify": "results/verify_placebo.json", "soup_ship": "results/ship_c.json"})
    stage_result("7", "complete" if all(v["exit_code"] is not None for v in commands.values()) and time.time() < CUTOFF else "partial",
                 amendment=amendment, commands=commands)


def mounted_optional(path):
    try:
        return mounted(path)
    except FileNotFoundError:
        return None


if __name__ == "__main__":
    if sys.argv[1] == "compute":
        compute(Path(sys.argv[2]))
    elif sys.argv[1] == "bench":
        bench(*sys.argv[2:])
    else:
        main()
