"""Run Soup's deployed, adapter, and placebo gates in the patched order."""
import shutil
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

from instruments import ROOT, atomic_json, gpu_monitor
from patch_common import SOUP, assert_amendment, base_path, install, mounted, read_json, sha, stamp, stage_result, step, use
from patch_soup_config import make
from patch_stage_core import checkpoint


def ship(soup, base, config, task, label, flag, target, *, extra=(), optional=False):
    evidence = ROOT / f"results/ship_{label}_evidence.json"
    verdict = ROOT / f"results/ship_{label}.json"
    result = step([soup, "ship", "--base", base, flag, target, "--task-eval", task, "--device", "cuda",
                   "--config", config, "--emit-evidence", evidence, "--output", verdict, *extra],
                  f"ship_{label}", success=(0, 2), attempts=2, limit=1800, optional=optional)
    payload = read_json(evidence) if evidence.is_file() else {}
    record = {
        "exit_code": result, "finished": stamp(), "verdict_file": verdict.name if verdict.is_file() else None,
        "evidence_file": evidence.name if evidence.is_file() else None, "provenance": payload.get("provenance"),
        "target_safetensors_sha256": {p.name: sha(p) for p in target.glob("*.safetensors")},
        "task_sha256": sha(task),
    }
    print(f"Soup ship {label}: {record}", flush=True)
    return record


def main():
    amendment = assert_amendment()
    task = use("data/ship_task_amended.jsonl")
    use("data/train_soup.jsonl")
    use("results/data_stats.json")
    python = install("ship", SOUP)
    soup = python.with_name("soup")
    base = base_path(python)
    real, _placebo_bench = make(soup)
    outcomes = {}
    with TemporaryDirectory(prefix="ship-local-", dir=ROOT) as location, gpu_monitor():
        location = Path(location)
        adapters = {}
        for label in ("real", "placebo"):
            source = mounted(f"runs/{label}/adapter/adapter_model.safetensors").parent
            adapters[label] = Path(shutil.copytree(source, location / label))
        target = checkpoint("merged_4bit")
        outcomes["b"] = ship(soup, base, real, task, "b", "--tuned", target)
        outcomes["a"] = ship(soup, base, real, task, "a", "--adapter", adapters["real"])
        source_args = [python, ROOT / "soup_evidence.py"]
        evidence_a = ROOT / "results/ship_a_evidence.json"
        if evidence_a.is_file():
            source_args += [evidence_a]
        step(source_args, "soup_eval_source")
        baseline = ROOT / "results/soup_baseline.json"
        extra = ["--baseline", baseline] if baseline.is_file() else []
        outcomes["c"] = ship(soup, base, real, task, "c", "--adapter", adapters["placebo"], extra=extra)
    atomic_json(ROOT / "results/ship_checkpoints.json", outcomes)
    stage_result("6", "complete" if all(item["verdict_file"] for item in outcomes.values()) else "partial",
                 amendment=amendment, outcomes=outcomes)


if __name__ == "__main__":
    main()
