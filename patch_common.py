"""Paths, command logs and hashes for the amended Kaggle stages."""
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
from zoneinfo import ZoneInfo

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from instruments import ROOT, atomic_json, command

INPUT = Path("/kaggle/input")
TZ = ZoneInfo("Asia/Almaty")
DEADLINE = datetime(2026, 11, 11, 10, tzinfo=TZ).timestamp()
CUTOFF = datetime(2026, 11, 11, 8, 30, tzinfo=TZ).timestamp()
MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
PEFT = ("transformers==4.57.1", "peft==0.18.0", "accelerate==1.11.0", "bitsandbytes==0.48.2", "sqlglot==28.0.0")
SOUP = ("soup-cli[train]==0.77.2", "bitsandbytes==0.48.2", "torchao==0.17.0")


def stamp():
    return datetime.now(TZ).isoformat(timespec="seconds")


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def mounted(suffix):
    suffix = Path(suffix)
    matches = sorted(path for path in INPUT.rglob(suffix.name) if path.parts[-len(suffix.parts):] == suffix.parts)
    if len(matches) > 1 and all(path.stat().st_size <= 2_000_000 for path in matches):
        if len({sha(path) for path in matches}) == 1:
            return matches[0]
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected exactly one mounted {suffix}; found {matches}")
    return matches[0]


def use(suffix):
    target = ROOT / suffix
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(mounted(suffix), target)
    return target


def step(args, name, *, success=(0,), optional=False, attempts=1, limit=None, cwd=ROOT):
    end = CUTOFF if optional else DEADLINE
    for attempt in range(1, attempts + 1):
        remaining = end - time.time()
        if remaining <= 0:
            return None
        label = name if attempt == 1 else f"{name}_attempt{attempt}"
        log = ROOT / f"logs/{label}.log"
        status = command(args, log, cwd=cwd, timeout=min(limit or remaining, remaining))
        if status in success:
            return status
        failed = ROOT / f"logs/failed/{label}"
        failed.mkdir(parents=True, exist_ok=True)
        shutil.copy2(log, failed / "raw.log")
        (failed / "note.txt").write_text(f"{stamp()} Exit {status} from {name}.\n"
                                        "Retain the raw log and follow the stage fallback or next check.\n", encoding="utf-8")
    return status


def install(name, packages, *, optional=False):
    env = Path(f"/kaggle/temp/soup-patch-{name}")
    uv = shutil.which("uv")
    assert uv and step([uv, "venv", "--seed", "--system-site-packages", "--python", sys.executable, env],
                       f"{name}_venv", optional=optional) == 0
    python = env / "bin/python"
    assert step([python, "-m", "pip", "-q", "--disable-pip-version-check", "install", *packages],
                f"{name}_install", optional=optional) == 0
    return python


def base_path(python):
    return Path(__import__("subprocess").check_output([
        python, "-c",
        f"from huggingface_hub import snapshot_download; print(snapshot_download({MODEL!r},revision={REVISION!r},"
        "allow_patterns=['*.safetensors','*.safetensors.index.json','config.json','generation_config.json',"
        "'tokenizer*','vocab.json','merges.txt','chat_template.jinja']))",
    ], cwd=ROOT, text=True).splitlines()[-1])


def assert_amendment():
    amendment = use("prereg_amendment_1.md")
    expected = use("prereg_amendment_1.sha256").read_text().split()[0]
    if sha(amendment) != expected:
        raise RuntimeError("Amendment hash mismatch")
    return expected


def stage_result(name, state, **details):
    atomic_json(ROOT / f"results/stage_{name}.json", {"status": state, "finished": stamp(), **details})
    print(f"{stamp()} INFO Stage {name} {state}", flush=True)


def read_json(path):
    return json.loads(Path(path).read_bytes())
