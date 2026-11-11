"""Orchestrate merge, quantization, verifiers, Soup gates and silent-failure probes."""
import json
import os
from pathlib import Path
import shutil
import sys

os.environ.update(CUDA_VISIBLE_DEVICES="0", PYTHONHASHSEED="42", PYTHONUNBUFFERED="1", USE_TF="0", SOUP_TELEMETRY="0", WANDB_MODE="disabled")

from instruments import LOG, ROOT, atomic_json, gpu_monitor, gpu_snapshot, setup_logging
from train_stage import environment, record_failure, run

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
RESULTS = ROOT / "results"
STATUS = {}


def status(name, outcome, attempts, details=None):
    STATUS[name] = {"exit_code": outcome, "attempts": attempts, "details": details}
    atomic_json(RESULTS / "eval_stage.json", STATUS)


def step(name, args, *, max_attempts=2, success=(0,), limit=None):
    for attempt in range(1, max_attempts + 1):
        label = f"{name}" if attempt == 1 else f"{name}_attempt{attempt}"
        code = run(args, label, limit=limit, fatal=False)
        if code in success:
            status(name, code, attempt)
            return True
        record_failure(label, ROOT / f"logs/{label}.log", f"{name} exited {code}",
                       "Retry unchanged." if attempt < max_attempts else "Continue to the next independent step.")
    status(name, code, attempt)
    return False


def read_json(name):
    path = RESULTS / name
    return json.loads(path.read_bytes()) if path.exists() else {}


def merge_all(python, soup_python, base):
    script = ROOT / "artifacts_stage.py"
    soup = soup_python.with_name("soup") if soup_python else None
    if soup:
        step("soup_merge_help", [soup, "merge", "--help"])
    temp = Path("/kaggle/temp/soup-controls")
    temp.mkdir(exist_ok=True)
    paths = {
        "merged_fp32": (ROOT / "artifacts/merged_fp32", "real", "float32"),
        "merged_fp16": (ROOT / "artifacts/merged_fp16", "real", "float16"),
        "merged_bf16": (ROOT / "artifacts/merged_bf16", "real", "bfloat16"),
        "placebo_fp16": (temp / "placebo_fp16", "placebo", "float16"),
        "shuffled_fp16": (temp / "shuffled_fp16", "shuffled", "float16"),
    }
    outputs = {}
    for key, (dest, kind, dtype) in paths.items():
        adapter = ROOT / f"runs/{kind}/adapter"
        if not (adapter / "adapter_model.safetensors").is_file():
            status(f"merge_{key}", None, 0, "Missing adapter.")
            continue
        soup_done = soup and step(f"soup_merge_{key}", [soup_python, script, "soup_merge", soup, f"merge_{key}", adapter, base, dest, dtype, "fp16"], max_attempts=1)
        if soup_done or step(f"peft_merge_{key}", [python, script, "peft_merge", f"merge_{key}", adapter, base, dest, dtype], max_attempts=1 if soup else 2):
            outputs[key] = dest
    quant = {}
    for key, kind, phase_name in (("merged_4bit", "real", "quantize_merged"), ("base_4bit", "placebo", "quantize_base"), ("shuffled_4bit", "shuffled", "quantize_shuffled")):
        adapter = ROOT / f"runs/{kind}/adapter"
        dest = ROOT / "artifacts" / key
        if not (adapter / "adapter_model.safetensors").is_file():
            status(phase_name, None, 0, "Missing adapter.")
            continue
        # A zero-delta placebo adapter makes the base go through exactly the same Soup pipeline.
        if kind == "placebo":
            identity_check = [
                python, "-c", "import sys; from safetensors import safe_open; "
                "f=safe_open(sys.argv[1],framework='pt'); k=[n for n in f.keys() if '.lora_B.' in n]; "
                "assert len(k)==196 and all(not f.get_tensor(n).count_nonzero().item() for n in k); "
                "print('All 196 placebo B matrices are exactly zero')", adapter / "adapter_model.safetensors",
            ]
            if not step("base_quant_placebo_identity", identity_check, max_attempts=1):
                continue
        soup_done = soup and step(f"soup_{phase_name}", [soup_python, script, "soup_merge", soup, phase_name, adapter, base, dest, "float16", "4bit"], max_attempts=1)
        if soup_done:
            quant[key] = "Soup merge --save-format 4bit"
            outputs[key] = dest
    if len(quant) != 3:
        # Use one quantization procedure for all three arms, even if a Soup artifact was produced.
        for key, source_key, phase_name in (
            ("merged_4bit", "merged_fp16", "quantize_merged"), ("base_4bit", None, "quantize_base"),
            ("shuffled_4bit", "shuffled_fp16", "quantize_shuffled"),
        ):
            if key in outputs:
                outputs[key].rename(outputs[key].with_name(key + "_soup_attempt"))
                del outputs[key]
            source = base if source_key is None else outputs.get(source_key)
            dest = ROOT / "artifacts" / key
            if source and step(f"bnb_{phase_name}", [python, script, "bnb_quantize", phase_name, source, dest], max_attempts=1 if soup else 2):
                quant[key] = "Transformers BitsAndBytesConfig NF4 double quant"
                outputs[key] = dest
    atomic_json(RESULTS / "artifact_procedures.json", {"quantization": quant, "outputs": {key: str(path) for key, path in outputs.items()}})
    if "merged_4bit" in outputs:
        step("standalone", [python, script, "standalone", outputs["merged_4bit"], base])
    return outputs


def verify_all(python, base, paths):
    for case, adapter, merged, quant in (
        ("real", "real", "merged_fp16", "merged_4bit"),
        ("wrong", "real", "merged_fp16", "base_4bit"),
        ("placebo", "placebo", "placebo_fp16", "base_4bit"),
        ("shuffled", "shuffled", "shuffled_fp16", "shuffled_4bit"),
    ):
        if not {merged, quant, "base_4bit"} <= paths.keys():
            status(f"verify_{case}", None, 0, "Missing required artifact.")
            continue
        args = [python, ROOT / "verify.py", "--base", base, "--adapter", ROOT / f"runs/{adapter}/adapter",
                "--merged", paths[merged], "--base4", paths["base_4bit"], "--merged4", paths[quant],
                "--eval", ROOT / "data/eval.jsonl", "--out", RESULTS / f"verify_{case}.json", "--case", case]
        if case == "real":
            for arg, key in (("--merged-fp32", "merged_fp32"), ("--merged-bf16", "merged_bf16")):
                if key in paths:
                    args += [arg, paths[key]]
        for arg, key in (("--shuffled", "shuffled_fp16"), ("--shuffled4", "shuffled_4bit")):
            if key in paths:
                args += [arg, paths[key]]
        step(f"verify_{case}", args, limit=1800)


def parse_json_log(name):
    path = ROOT / f"logs/{name}.log"
    lines = [line.split(" INFO ", 1)[1] for line in path.read_text(encoding="utf-8").splitlines() if " INFO " in line]
    for start, line in enumerate(lines):
        if line.startswith("{"):
            try:
                return json.JSONDecoder().raw_decode("\n".join(lines[start:]))[0]
            except json.JSONDecodeError:
                pass
    return None


def soup_gates(soup_python, base, paths):
    if not soup_python:
        status("soup_gates", None, 0, "Soup installation failed.")
        return
    soup = soup_python.with_name("soup")
    for kind in ("real", "placebo", "shuffled"):
        step(f"soup_check_{kind}", [soup, "adapters", "check", ROOT / f"runs/{kind}/adapter", "--json"], success=(0, 2))
    step("soup_audit_real", [soup, "adapters", "audit", ROOT / "runs/real/adapter", "--config", ROOT / "configs/soup.yaml", "--json"], success=(0, 2))
    step("soup_render_source", [soup_python, ROOT / "soup_evidence.py"], max_attempts=1)
    for name, flag, target in (
        ("a", "--adapter", ROOT / "runs/real/adapter"),
        ("b", "--tuned", paths.get("merged_4bit")),
        ("c", "--adapter", ROOT / "runs/placebo/adapter"),
        ("d", "--tuned", paths.get("base_4bit")),
        ("shuffled", "--adapter", ROOT / "runs/shuffled/adapter"),
    ):
        if target is None or not target.is_dir():
            status(f"ship_{name}", None, 0, "Missing target artifact.")
            continue
        evidence = RESULTS / f"ship_{name}_evidence.json"
        args = [soup, "ship", "--base", base, flag, target, "--task-eval", ROOT / "data/ship_task.jsonl",
                "--device", "cuda", "--emit-evidence", evidence, "--output", RESULTS / f"ship_{name}.json"]
        if (ROOT / "configs/soup.yaml").is_file():
            args += ["--config", ROOT / "configs/soup.yaml"]
        baseline = RESULTS / "soup_baseline.json"
        if name != "a" and baseline.exists():
            args += ["--baseline", baseline]
        step(f"ship_{name}", args, success=(0, 2), limit=1200)
        if evidence.exists():
            LOG.info("Soup ship %s provenance %s", name, read_json(evidence.name).get("provenance"))
            if name == "a":
                step("soup_baseline", [soup_python, ROOT / "soup_evidence.py", evidence], max_attempts=1)
    for name, key in (("real", "merged_4bit"), ("wrong", "base_4bit")):
        if {key, "merged_fp16"} <= paths.keys():
            label = f"quant_check_{name}"
            step(label, [soup, "eval", "quant-check", "--before", paths["merged_fp16"], "--after", paths[key],
                         "--tasks", ROOT / "data/ship_task.jsonl", "--format", "json"], max_attempts=1, success=(0,2), limit=900)
            if value := parse_json_log(label):
                atomic_json(RESULTS / f"{label}.json", value)


def probes(python, soup_python):
    script = ROOT / "probes/run_probes.py"
    step("mask_probes", [python, script, "mask"])
    if soup_python:
        soup = soup_python.with_name("soup")
        stats = read_json("data_stats.json")
        step("probe_P1_doctor", [soup, "data", "doctor", "data/train.jsonl", "--model", MODEL, "--max-length", stats["p96_full_train"], "--train-on-eot"], max_attempts=2, success=(0,2))
        step("probe_P2_doctor", [soup, "data", "doctor", "data/train.jsonl", "--model", ROOT / "artifacts/pad_equals_eos_tokenizer", "--max-length", stats["max_length"], "--train-on-eot"], max_attempts=1, success=(0,2))
        for name in ("P1", "P2"):
            evidence = read_json(f"probe_{name}.json")
            if evidence:
                evidence["soup_gate"] = STATUS.get(f"probe_{name}_doctor")
                atomic_json(RESULTS / f"probe_{name}.json", evidence)
    if (RESULTS / "verify_real.json").is_file():
        step("decoding_probes", [python, script, "decode"], limit=1200)
        step("probe_P11", [python, ROOT / "probes/execution_match.py"], max_attempts=1)
    step("probe_P9", [python, script, "precision"], max_attempts=1)
    for ident, verification, gates in (
        ("P6", "wrong", ["ship_a", "ship_b", "ship_d", "quant_check_wrong"]),
        ("P7", "placebo", ["soup_check_placebo", "ship_c"]),
        ("P8", "shuffled", ["soup_check_shuffled", "ship_shuffled"]),
    ):
        measured = read_json(f"verify_{verification}.json")
        atomic_json(RESULTS / f"probe_{ident}.json", {
            "verification_file": f"verify_{verification}.json", "verdict": measured.get("verdict"),
            "fingerprint_share": measured.get("distances", {}).get("fingerprint_share"),
            "bootstrap": measured.get("bootstrap"),
            "soup_gates": {name: {"status": STATUS.get(name), "output": read_json(name + ".json"),
                                  "log": f"logs/{name}.log"} for name in gates},
        })


def grid(python, soup_python):
    step("memory_grid", [python, ROOT / "mem_grid.py"], max_attempts=1, limit=1800)
    measured = read_json("memory_grid.json")
    if not measured or not soup_python:
        return
    soup = soup_python.with_name("soup")
    step("soup_profile_help", [soup, "profile", "--help"], max_attempts=1)
    for point in measured["points"]:
        mode, b, length = (point[key] for key in ("checkpointing", "batch", "length"))
        label = f"profile_grid_{mode}_{b}_{length}"
        step(label, [soup, "profile", "--model", MODEL, "--seq", length, "--batch", b, "--method", "lora",
                     "--lora-r", "16", "--gradient-checkpointing", "full" if mode == "on" else "off", "--json"], max_attempts=1, limit=120)
        point["soup_profile"] = parse_json_log(label)
    atomic_json(RESULTS / "memory_grid.json", measured)


def training_evidence():
    soup = parse_json_log("soup_profile") if (ROOT / "logs/soup_profile.log").is_file() else None
    if soup is not None:
        atomic_json(RESULTS / "soup_training_profile.json", soup)
    phases = read_json("memory_phases.json")
    real_phases = [row for row in phases if row["name"] == "real_training" and row["state"] == "complete"]
    if real_phases:
        atomic_json(RESULTS / "training_memory_comparison.json", {
            "soup_profile": soup,
            "predicted": read_json("memory_prediction.json")["training_peak_gb"]["on"],
            "measured": real_phases[-1],
        })


def main():
    setup_logging()
    training_evidence()
    with gpu_monitor():
        gpu_snapshot("eval_stage_start")
        python = environment("takehome-verify", ["transformers==4.57.1", "peft==0.18.0", "accelerate==1.11.0", "bitsandbytes==0.48.2", "sqlglot==28.0.0"])
        try:
            soup_python = environment("takehome-soup-eval", ["soup-cli[train]==0.77.2", "bitsandbytes==0.48.2"])
        except Exception as exc:
            LOG.exception("Soup evaluation environment failed")
            soup_python = None
            status("soup_environment", 1, 1, str(exc))
        from subprocess import check_output
        base = Path(check_output([python, "-c", f"from verify import pinned_base; print(pinned_base({MODEL!r}))"], cwd=ROOT, text=True).splitlines()[-1])
        paths = merge_all(python, soup_python, base)
        verify_all(python, base, paths)
        soup_gates(soup_python, base, paths)
        probes(python, soup_python)
        grid(python, soup_python)
        gpu_snapshot("eval_stage_end")


if __name__ == "__main__":
    try:
        main()
    finally:
        # Assemble measured evidence even when one of the phases could not run.
        from summarize import summarize
        summarize()
