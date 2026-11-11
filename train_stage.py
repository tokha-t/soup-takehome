"""Kaggle entry point for data, immutable preregistration, Soup, and controls."""
import json
import os
from pathlib import Path
import shutil
import sys
import time

os.environ.update(
    CUDA_VISIBLE_DEVICES="0", PYTHONHASHSEED="42", PYTHONUNBUFFERED="1",
    TOKENIZERS_PARALLELISM="false", WANDB_MODE="disabled", SOUP_TELEMETRY="0", USE_TF="0",
)

from instruments import ROOT, LOG, atomic_json, command, gpu_monitor, gpu_snapshot, setup_logging

LOGS = ROOT / "logs"
MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
DEADLINE_EPOCH = 1794373200  # November 11, 10:00 in Asia/Almaty.


def record_failure(name, raw_log, cause, next_step):
    path = LOGS / "failed" / name
    path.mkdir(parents=True, exist_ok=True)
    if Path(raw_log).exists():
        shutil.copy2(raw_log, path / "raw.log")
    (path / "note.txt").write_text(f"{cause}\n{next_step}\n", encoding="utf-8")


def run(args, name, *, limit=None, fatal=True):
    available = DEADLINE_EPOCH - time.time()
    if available <= 0:
        raise TimeoutError("Reached the November 11, 10:00 Astana deadline")
    path = LOGS / f"{name}.log"
    status = command(args, path, timeout=min(limit or available, available))
    if status and fatal:
        raise RuntimeError(f"{name} exited {status}. See {path}")
    return status


def environment(name, packages):
    path = Path("/kaggle/temp") / name
    uv = shutil.which("uv")
    run([uv, "venv", "--seed", "--system-site-packages", "--python", sys.executable, path], f"{name}_venv")
    python = path / "bin/python"
    run([python, "-m", "pip", "--disable-pip-version-check", "install", *packages], f"{name}_install")
    return python


def main():
    setup_logging()
    result = json.loads((ROOT / "results/rescore.json").read_bytes())
    if result["gate"] != "confirmed":
        raise RuntimeError("Step 1 rescore gate has not confirmed SQL.")
    harder = result.get("alternative", {}).get("type") == "harder_slice"
    with gpu_monitor():
        gpu_snapshot("session_start")
        python = environment("takehome-peft", [
            "transformers==4.57.1", "peft==0.18.0", "accelerate==1.11.0",
            "bitsandbytes==0.48.2", "sqlglot==28.0.0",
        ])
        run([python, ROOT / "data/build_data.py", *(["--harder"] if harder else [])], "build_data")
        run([python, ROOT / "preregister.py", ROOT / "pipeline_spec.md", ROOT / "configs/base_config.json"], "preregister_command")
        run([python, "-c", "from instruments import setup_logging, record_environment; setup_logging(); record_environment()"], "environment")
        config_path = ROOT / "configs/soup.yaml"
        steps = {"soup_status": None, "runs": {}}
        try:
            soup_python = environment("takehome-soup", ["soup-cli[train]==0.77.2"])
            soup = soup_python.with_name("soup")
            run([soup, "--version"], "soup_version")
            # Init in a clean directory so the template cannot overwrite the registered config.
            staging = ROOT / "configs/soup_init"
            staging.mkdir(exist_ok=True)
            status = command([soup, "init", "--template", "chat"], LOGS / "soup_init.log", cwd=staging, timeout=120)
            if status:
                raise RuntimeError(f"soup init exited {status}")
            shutil.copy2(staging / "soup.yaml", ROOT / "configs/soup_generated.yaml")
            run([soup_python, ROOT / "configs/configure_soup.py"], "soup_configure")
            stats = json.loads((ROOT / "results/data_stats.json").read_bytes())
            run([soup, "data", "validate", "data/train.jsonl", "--format", "chatml", "--min-valid-fraction", "1"], "soup_validate", fatal=False)
            run([soup, "doctor", "--config", config_path], "soup_doctor", fatal=False)
            run([soup, "data", "doctor", "data/train.jsonl", "--model", MODEL, "--max-length", stats["max_length"], "--show-mask", "2", "--train-on-eot"], "soup_data_doctor", fatal=False)
            run([soup, "profile", "--config", config_path, "--json"], "soup_profile", fatal=False)
            # Invoke the upstream soup train CLI with a read-only HF Trainer observer attached.
            for kind in ("real", "placebo", "shuffled"):
                cfg = config_path if kind == "real" else ROOT / "configs" / f"soup_{kind}.yaml"
                status = run([soup_python, ROOT / "configs/soup_observe.py", kind, soup, "train", "--config", cfg, "--yes"], f"soup_train_{kind}", limit=1800, fatal=False)
                if status:
                    steps["soup_status"] = f"{kind} exited {status}; using PEFT for all runs"
                    record_failure(f"soup_train_{kind}", LOGS / f"soup_train_{kind}.log", steps["soup_status"], "Run the same preregistered settings with Transformers and PEFT.")
                    break
            else:
                steps["soup_status"] = "all runs complete"
        except Exception as exc:
            LOG.exception("Soup setup failed. Using the preregistered fallback.")
            steps["soup_status"] = f"{type(exc).__name__}: {exc}"
            record_failure("soup_setup", LOGS / "soup_configure.log", steps["soup_status"], "Run the same preregistered settings with Transformers and PEFT.")
        atomic_json(ROOT / "results/training_stage.json", steps)
        if steps["soup_status"] != "all runs complete":
            for kind in ("real", "placebo", "shuffled"):
                # Preserve any partial Soup artifacts and ensure the comparison uses one backend.
                old = ROOT / f"runs/{kind}/adapter"
                if old.exists():
                    old.rename(old.with_name("soup_adapter_attempt"))
                for attempt in (1, 2):
                    name = f"peft_{kind}_attempt{attempt}"
                    status = run([python, ROOT / "configs/train_peft.py", kind], name, fatal=False)
                    if not status:
                        break
                    record_failure(name, LOGS / f"{name}.log", f"PEFT {kind} exited {status}", "Retry unchanged." if attempt == 1 else "Stop this run after the second failure.")
                steps["runs"][kind] = {"backend": "Transformers + PEFT", "exit_code": status, "attempts": attempt}
                atomic_json(ROOT / "results/training_stage.json", steps)
        else:
            for kind in ("real", "placebo", "shuffled"):
                steps["runs"][kind] = {"backend": "Soup", "exit_code": 0, "attempts": 1}
            atomic_json(ROOT / "results/training_stage.json", steps)
        gpu_snapshot("training_stage_end")


if __name__ == "__main__":
    main()
