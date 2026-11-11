"""Commit CPU fp16, bf16 and fp32 merges and the two scaled adapters."""
import json
import os
from pathlib import Path
import shutil
import sys

from patch_common import PEFT, SOUP, ROOT, assert_amendment, base_path, install, mounted, read_json, sha, stage_result, step
from instruments import atomic_json


def inspect_artifact(path):
    from collections import Counter
    from safetensors import safe_open
    path = Path(path)
    output = {}
    for file in sorted(path.iterdir()):
        if file.is_file():
            meta = {"sha256": sha(file), "bytes": file.stat().st_size}
            if file.suffix == ".safetensors":
                with safe_open(file, framework="pt", device="cpu") as handle:
                    meta["dtype_counts"] = dict(Counter(handle.get_slice(key).get_dtype() for key in handle.keys()))
            output[file.name] = meta
    if not any(name.endswith(".safetensors") for name in output):
        raise RuntimeError(f"No model tensors in {path}")
    print(f"Saved artifact {path.name}: {json.dumps(output)}", flush=True)
    return output


def weak(source, output, scale):
    from safetensors.torch import load_file, save_file
    source, output = Path(source), Path(output)
    shutil.copytree(source, output, dirs_exist_ok=True, ignore=shutil.ignore_patterns("checkpoint-*", "optimizer.pt", "trainer_state.json"))
    file = output / "adapter_model.safetensors"
    weights = load_file(file)
    keys = [name for name in weights if ".lora_B" in name]
    if len(keys) != 196:
        raise ValueError(f"Expected 196 LoRA B tensors, found {len(keys)}")
    for key in keys:
        weights[key] = weights[key] * float(scale)
    save_file(weights, file, metadata={"format": "pt"})
    return inspect_artifact(output)


def main(group):
    amendment = assert_amendment()
    peft = install("merge-peft", PEFT)
    soup_python = install("merge-soup", SOUP)
    soup = soup_python.with_name("soup")
    step([soup, "merge", "--help"], "soup_merge_help")
    base = base_path(peft)
    script = ROOT / "artifacts_stage.py"
    adapters = {kind: mounted(f"runs/{kind}/adapter/adapter_model.safetensors").parent for kind in ("real","placebo","shuffled")}
    done, commands = {}, {}
    if group == "2a":
        merges = [("merged_fp16", "real", "float16"), ("merged_fp32", "real", "float32"), ("merged_bf16", "real", "bfloat16")]
    else:
        for kind, ratio in (("weak_001", .01), ("weak_01", .1)):
            path = ROOT / f"runs/{kind}"
            if step([peft, __file__, "weak", adapters["real"], path, ratio], f"scale_{kind}") != 0:
                raise RuntimeError(f"Could not scale {kind}")
            adapters[kind] = path
        merges = [(f"{kind}_fp16", kind, "float16") for kind in ("placebo","shuffled","weak_001","weak_01")]
    for name, adapter, dtype in merges:
        target = ROOT / f"artifacts/{name}"
        code = 1
        if dtype != "float32":
            code = step([soup_python, script, "soup_merge", soup, f"merge_{name}", adapters[adapter], base, target, dtype, "fp16"], f"soup_{name}", limit=1200)
            commands[name] = "Soup"
        if code != 0:
            code = step([peft, script, "peft_merge", f"merge_{name}", adapters[adapter], base, target, dtype], f"peft_{name}", limit=1200)
            commands[name] = "PEFT fp32 reference" if dtype == "float32" else "PEFT fallback"
        if code == 0:
            done[name] = str(target.relative_to(ROOT))
    inspect = [peft, __file__, "inspect", group, *[ROOT / path for path in done.values()]]
    if group == "2b":
        inspect += [adapters[key] for key in ("weak_001","weak_01")]
    step(inspect, f"inspect_{group}")
    stage_result(group, "complete" if len(done) == len(merges) else "partial", amendment=amendment, artifacts=done, procedures=commands)


if __name__ == "__main__":
    if sys.argv[1] == "weak":
        weak(*sys.argv[2:])
    elif sys.argv[1] == "inspect":
        group, *paths = sys.argv[2:]
        atomic_json(ROOT / f"results/artifact_hashes_{group}.json", {Path(path).name: inspect_artifact(path) for path in paths})
    else:
        main(sys.argv[1])
