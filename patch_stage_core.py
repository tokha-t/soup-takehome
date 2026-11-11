"""Core verification on the amended split and every saved control."""
from pathlib import Path
import shutil

from instruments import ROOT, atomic_json, gpu_monitor
from patch_common import PEFT, assert_amendment, base_path, install, mounted, read_json, sha, stage_result, step, use


def checkpoint(label):
    if label == "merged_4bit":
        portable = ROOT / "artifacts/merged_4bit_portable"
        if (portable / "config.json").is_file():
            return portable
        try:
            return mounted("artifacts/merged_4bit_portable/config.json").parent
        except FileNotFoundError:
            pass
    return mounted(f"artifacts/{label}/config.json").parent


def eval_path():
    path = use("data/eval_amended.jsonl")
    expected = read_json(mounted("results/stage_1.json"))["clean_eval_sha256"]
    if sha(path) != expected:
        raise RuntimeError("Clean evaluation set hash mismatch")
    return path


def main():
    amendment = assert_amendment()
    data = eval_path()
    use("results/data_stats.json")
    export = read_json(mounted("results/stage_3.json"))
    if not export["hashes_verified"] or not export["procedure"].startswith("Soup merge"):
        raise RuntimeError("Verification must use the committed Soup NF4 export and its hash manifest")
    adapter_hashes = {}
    for label in ("real", "placebo", "shuffled"):
        adapter = mounted(f"runs/{label}/adapter/adapter_model.safetensors").parent
        adapter_hashes[label] = {
            path.name: {"sha256": sha(path), "bytes": path.stat().st_size}
            for path in sorted(adapter.rglob("*")) if path.is_file()
        }
    atomic_json(ROOT / "results/training_adapter_hashes.json", adapter_hashes)
    python = install("core", PEFT)
    base = base_path(python)
    # The stage 2 uploads succeeded but the hash display failed on a string path.
    groups = {
        "2a": ("merged_fp16", "merged_fp32", "merged_bf16"),
        "2b": ("placebo_fp16", "shuffled_fp16", "weak_001_fp16", "weak_01_fp16"),
        "3": ("merged_4bit", "base_4bit", "shuffled_4bit"),
    }
    for label, names in groups.items():
        arguments = [checkpoint(name) for name in names]
        if label == "2b":
            arguments.extend(mounted(f"runs/{name}/adapter_model.safetensors").parent for name in ("weak_001", "weak_01"))
        status = step([python, ROOT / "patch_stage_merge.py", "inspect", label, *arguments], f"saved_artifact_hashes_{label}")
        if status != 0:
            raise RuntimeError(f"Could not validate the contents and saved dtypes for {label}")
    raw = mounted("artifacts/merged_4bit/config.json").parent
    portable = ROOT / "artifacts/merged_4bit_portable"
    shutil.copytree(raw, portable, dirs_exist_ok=True)
    replaced = []
    for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "vocab.json", "merges.txt", "special_tokens_map.json"):
        original = base / name
        if original.is_file():
            shutil.copy2(original, portable / name)
            replaced.append(name)
    for file in portable.glob("*.safetensors"):
        assert sha(file) == sha(raw / file.name), file.name
    atomic_json(ROOT / "results/production_packaging.json", {
        "source": str(raw), "portable": str(portable), "restored_from_pinned_base": replaced,
        "reason": "Soup's Transformers 5 export wrote list-form extra_special_tokens, which the pinned Transformers 4 loader expects as a mapping.",
        "nf4_weights_identical": True,
    })
    with gpu_monitor():
        standalone = step([python, ROOT / "artifacts_stage.py", "standalone", portable, base], "standalone_portable",
                          attempts=2, limit=240) == 0
        if step([python, ROOT / "patch_stage_merge.py", "inspect", "4", portable], "portable_artifact_hashes") != 0:
            raise RuntimeError("Failed to hash portable artifact")
        status = step([
            python, ROOT / "verify.py", "--base", base,
            "--adapter", mounted("runs/real/adapter/adapter_model.safetensors").parent,
            "--merged", checkpoint("merged_fp16"), "--base4", checkpoint("base_4bit"),
            "--merged4", checkpoint("merged_4bit"), "--eval", data,
            "--out", ROOT / "results/verify_real.json",
            "--placebo-merged", checkpoint("placebo_fp16"), "--shuffled-merged", checkpoint("shuffled_fp16"),
            "--shuffled4", checkpoint("shuffled_4bit"),
            "--extra", f"weak_001={checkpoint('weak_001_fp16')}",
            "--extra", f"weak_01={checkpoint('weak_01_fp16')}",
        ], "verify_core", limit=2400, attempts=2)
    cases = ("real", "placebo", "shuffled", "weak_001", "weak_01")
    evidence = {case: read_json(ROOT / f"results/verify_{case}.json").get("verdict")
                for case in cases if (ROOT / f"results/verify_{case}.json").exists()}
    stage_result("4", "complete" if status == 0 and len(evidence) == 5 else "partial",
                 amendment=amendment, eval_sha256=sha(data), exit_code=status, verdicts=evidence, standalone=standalone)


if __name__ == "__main__":
    main()
