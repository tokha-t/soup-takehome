"""Retry Soup quant-check with verified checkpoint copies under its cwd."""
from hashlib import sha256
import json
from pathlib import Path
import random
import runpy
import shutil
import sys
from tempfile import TemporaryDirectory
import time

from instruments import LOG, ROOT, atomic_json, gpu_monitor, phase, setup_logging
from patch_common import CUTOFF, PEFT, SOUP, assert_amendment, install, mounted, read_json, sha, stage_result, stamp, step, use

RESULTS = ROOT / "results"


def checked_copy(source, target):
    digest = sha256()
    with source.open("rb") as original, target.open("xb") as copied:
        while True:
            if time.time() >= CUTOFF:
                raise TimeoutError("08:30 cutoff during checkpoint staging")
            block = original.read(8 * 1024 * 1024)
            if not block:
                break
            copied.write(block)
            digest.update(block)
    return digest.hexdigest()


def copy_checkpoint(source, target, manifest, tokenizer_from=None):
    target.mkdir()
    metadata = {}
    tokenizer_files = ("tokenizer.json", "tokenizer_config.json", "merges.txt", "vocab.json", "chat_template.jinja")
    for name, entry in manifest.items():
        if tokenizer_from is not None and name in tokenizer_files:
            continue
        digest = checked_copy(source / name, target / name)
        if digest != entry["sha256"]:
            raise RuntimeError(f"Copy does not match the committed {source.name}/{name} hash")
        metadata[name] = digest
    if tokenizer_from is not None:
        for name in tokenizer_files:
            metadata[name] = checked_copy(tokenizer_from / name, target / name)
    return metadata


def result_from_log(label):
    lines = [line.partition(" INFO ")[2] for line in (ROOT / f"logs/{label}.log").read_text().splitlines()]
    for index, line in enumerate(lines):
        if line.startswith("{"):
            try:
                return json.JSONDecoder().raw_decode("\n".join(lines[index:]))[0]
            except json.JSONDecodeError:
                continue
    return None


def invoke(entry, label, before, after, tasks):
    import torch
    setup_logging()
    if torch.cuda.device_count() != 1 or "T4" not in torch.cuda.get_device_name(0):
        raise RuntimeError("Expected exactly one visible Tesla T4")
    random.seed(42)
    torch.manual_seed(42)
    sys.argv = [entry, "eval", "quant-check", "--before", before, "--after", after,
                "--tasks", tasks, "--format", "json"]
    LOG.info("Soup command argv: %s", sys.argv)
    # Soup's generator leaves placement to Transformers and does not move prompt
    # tensors. Keep both models and inputs on the same visible T4.
    status = 0
    with phase(label, device="CUDA"), torch.device("cuda:0"):
        try:
            runpy.run_path(entry, run_name="__main__")
        except SystemExit as exc:
            status = exc.code or 0
            if status not in (0, 2):
                raise
    raise SystemExit(status)


def export_fp32(source, target):
    import bitsandbytes
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from patch_stage_merge import inspect_artifact
    setup_logging()
    random.seed(42)
    torch.manual_seed(42)
    with phase("quant_merged_4bit_from_fp32"):
        config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                    bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16)
        LOG.info("Optional fp32-reference NF4 export with bitsandbytes %s and %s",
                 bitsandbytes.__version__, config.to_json_string())
        model = AutoModelForCausalLM.from_pretrained(source, dtype=torch.float32,
                                                      quantization_config=config, device_map={"": 0})
        model.save_pretrained(target)
        AutoTokenizer.from_pretrained(source).save_pretrained(target)
    atomic_json(RESULTS / "artifact_hashes_8q.json", {Path(target).name: inspect_artifact(target)})


def main():
    if time.time() >= CUTOFF:
        stage_result("8q", "skipped", reason="08:30 cutoff")
        return
    amendment = assert_amendment()
    task = use("data/ship_task_amended.jsonl")
    manifests = {name: read_json(mounted(f"results/artifact_hashes_{name}.json")) for name in ("2a", "3", "4")}
    statuses = {}
    soup_python = install("quant-check-soup", SOUP, optional=True)
    soup = soup_python.with_name("soup")
    with gpu_monitor(), TemporaryDirectory(prefix="quant-check-", dir=ROOT) as folder:
        folder = Path(folder)
        before = folder / "merged_fp16"
        portable = mounted("artifacts/merged_4bit_portable/config.json").parent
        bindings = {"task_sha256": sha(task), "checkpoints": {},
                    "wrong_artifact_tokenizer": "Pinned Qwen files from the verified portable package; NF4 weight bytes unchanged."}
        bindings["checkpoints"][before.name] = copy_checkpoint(
            mounted("artifacts/merged_fp16/config.json").parent, before, manifests["2a"][before.name])
        for name, artifact, source, manifest in (
            ("real", "merged_4bit_portable", portable, manifests["4"]),
            ("wrong", "base_4bit", mounted("artifacts/base_4bit/config.json").parent, manifests["3"]),
        ):
            if time.time() >= CUTOFF:
                break
            after = folder / artifact
            bindings["checkpoints"][artifact] = copy_checkpoint(source, after, manifest[artifact],
                                                                  tokenizer_from=portable if name == "wrong" else None)
            atomic_json(RESULTS / "quant_check_bindings.json", bindings)
            label = f"quant_check_{name}"
            status = step([soup_python, __file__, "invoke", soup, label, before, after, task],
                          label, success=(0, 2), optional=True, limit=1800)
            statuses[label] = {"exit_code": status, "finished": stamp(), "log": f"logs/stages/8q/{label}.log"}
            if status in (0, 2) and (parsed := result_from_log(label)):
                atomic_json(RESULTS / f"{label}.json", parsed)
            atomic_json(RESULTS / "quant_check_commands.json", statuses)
            shutil.rmtree(after)
        if time.time() + 180 < CUTOFF:
            peft = install("quant-reference-peft", PEFT, optional=True)
            source = mounted("artifacts/merged_fp32/config.json").parent
            target = ROOT / "artifacts/merged_4bit_from_fp32"
            label = "quant_merged_4bit_from_fp32"
            status = step([peft, __file__, "export_fp32", source, target], label, optional=True, limit=900)
            statuses[label] = {"exit_code": status, "finished": stamp(), "log": f"logs/stages/8q/{label}.log"}
            if status != 0 and target.exists():
                shutil.move(target, Path("/kaggle/temp") / target.name)
            atomic_json(RESULTS / "quant_check_commands.json", statuses)
    completed = all((RESULTS / f"quant_check_{name}.json").is_file() for name in ("real", "wrong"))
    stage_result("8q", "complete" if completed else "partial", amendment=amendment, commands=statuses)


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "invoke":
        invoke(*sys.argv[2:])
    elif mode == "export_fp32":
        export_fp32(*sys.argv[2:])
    else:
        try:
            main()
        except Exception as exc:
            setup_logging()
            LOG.exception("Quant-check retry stopped before all steps finished")
            stage_result("8q", "partial", error=f"{type(exc).__name__}: {exc}")
