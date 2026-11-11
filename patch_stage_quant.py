"""Use the same Soup NF4 save path for the real, placebo and shuffled adapters."""
from pathlib import Path
import shutil

from instruments import ROOT, gpu_monitor
from patch_common import PEFT, SOUP, assert_amendment, base_path, install, mounted, stage_result, step, use

MODELS = (("merged_4bit", "real", "merged_fp16"), ("base_4bit", "placebo", "placebo_fp16"), ("shuffled_4bit", "shuffled", "shuffled_fp16"))


def main():
    amendment = assert_amendment()
    for path in ("data/eval_amended.jsonl", "results/data_stats.json"):
        use(path)
    peft = install("quant-peft", PEFT)
    soup_python = install("quant-soup", SOUP)
    soup = soup_python.with_name("soup")
    base = base_path(peft)
    script = ROOT / "artifacts_stage.py"
    if step([soup, "merge", "--help"], "soup_quant_help") != 0:
        raise RuntimeError("Could not verify Soup's 4-bit merge flags")
    help_text = (ROOT / "logs/soup_quant_help.log").read_text()
    if "--save-format" not in help_text or "BNB-4bit-quantized" not in help_text:
        raise RuntimeError("Soup merge did not advertise a BNB 4-bit format")
    placebo = mounted("runs/placebo/adapter/adapter_model.safetensors")
    check = [
        peft, "-c", "import sys; from safetensors import safe_open; f=safe_open(sys.argv[1],framework='pt'); "
        "b=[n for n in f.keys() if '.lora_B.' in n]; assert len(b)==196; "
        "assert all(not f.get_tensor(k).count_nonzero().item() for k in b); print('196 LoRA B tensors are zero')", placebo,
    ]
    if step(check, "placebo_zero_adapter") != 0:
        raise RuntimeError("Placebo did not establish base identity")
    made = {}
    with gpu_monitor():
        for name, kind, _ in MODELS:
            target = ROOT / f"artifacts/{name}"
            adapter = mounted(f"runs/{kind}/adapter/adapter_model.safetensors").parent
            if step([soup_python, script, "soup_merge", soup, f"quant_{name}", adapter, base, target, "float16", "4bit"], f"soup_{name}", limit=1200) == 0:
                made[name] = target
        procedure = "Soup merge --dtype float16 --save-format 4bit with default double quantization"
        if len(made) != len(MODELS):
            quarantine = Path("/kaggle/temp/incomplete-soup-nf4")
            quarantine.mkdir(exist_ok=True)
            for name, _, _ in MODELS:
                path = ROOT / f"artifacts/{name}"
                if path.exists():
                    shutil.move(path, quarantine / name)
            made = {}
            procedure = "Transformers BitsAndBytesConfig NF4 double quant, applied identically to all three fp16 merges"
            for name, _, merged in MODELS:
                checkpoint = mounted(f"artifacts/{merged}/config.json").parent
                target = ROOT / f"artifacts/{name}"
                if step([peft, script, "bnb_quantize", f"quant_{name}", checkpoint, target], f"bnb_{name}", limit=1200) == 0:
                    made[name] = target
        standalone = ("merged_4bit" in made and step(
            [peft, script, "standalone", made["merged_4bit"], base], "standalone_quantized", attempts=2, limit=900,
        ) == 0)
    hashed = step([peft, ROOT / "patch_stage_merge.py", "inspect", "3", *made.values()], "quant_file_hashes") == 0
    stage_result("3", "complete" if len(made) == 3 and standalone and hashed else "partial",
                 amendment=amendment, procedure=procedure, artifacts=list(made), standalone=standalone, hashes_verified=hashed)


if __name__ == "__main__":
    main()
