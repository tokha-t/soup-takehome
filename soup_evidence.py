"""Record installed Soup's rendering, suite denominators and baseline stamp."""
import inspect
import json
from pathlib import Path
import sys

from soup_cli.eval import gate_suites
from soup_cli.eval.custom import load_eval_tasks
from soup_cli.eval.gate import write_baseline_file
from soup_cli.utils.live_eval import _render_prompt_template

from instruments import ROOT, LOG, atomic_json, setup_logging

setup_logging()
tasks = load_eval_tasks(ROOT / "data/ship_task_amended.jsonl")
assert len(tasks) == 100 and all(task.scoring == "exact" for task in tasks)
source = inspect.getsource(_render_prompt_template)
for line in source.splitlines():
    LOG.info("Soup prompt source %s", line)
suite_counts = {
    name: len(gate_suites.MINI_BENCHMARKS[name]) if name in gate_suites.MINI_BENCHMARKS else len(gate_suites.load_suite_items(name))
    for name in gate_suites.DEFAULT_GENERAL_SUITE
}
atomic_json(ROOT / "results/soup_eval_source.json", {
    "system_turn_supplied": False, "template_injects_qwen_default": True,
    "source": source, "task_rows": len(tasks), "suite_counts": suite_counts,
})
if len(sys.argv) > 1:
    evidence = json.loads(Path(sys.argv[1]).read_bytes())
    LOG.info("Soup emitted provenance %s", json.dumps(evidence.get("provenance")))
    write_baseline_file(str(ROOT / "results/soup_baseline.json"), {
        name: pair["base"] for name, pair in evidence["benchmarks"].items()
    })
