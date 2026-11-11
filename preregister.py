"""Write the immutable main-run preregistration after the data scan."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent


def preregister(spec, model_config):
    prereg = ROOT / "prereg.md"
    if prereg.exists():
        raise FileExistsError("An existing preregistration is immutable.")
    data = json.loads((ROOT / "results/data_stats.json").read_bytes())
    text = Path(spec).read_text(encoding="utf-8")
    predictions = text.split("## 11. Probe predictions", 1)[1].split("## 12.", 1)[0].split("\n\n", 1)[1].strip()
    criteria = text.split("## 13. SHIP criteria", 1)[1].split("## 14.", 1)[0].split("\n\n", 1)[1].strip()
    memory = subprocess.check_output([
        sys.executable, ROOT / "predict_memory.py", "--config", model_config,
        "--historical", "--length", str(data["max_length"]), "--json-out", ROOT / "results/memory_prediction.json",
    ], text=True)
    parts = [
        "# Main SQL pipeline pre-registration",
        "## Memory prediction",
        f"Batch 8, rank 16, maximum length {data['max_length']}. Decimal GB. Device estimates exclude the separately listed CUDA context and use full-vocabulary logits as specified.",
        memory.strip(),
        "The optimized PEFT fallback projects only a suffix covering answer positions. Its measured logits cost may be lower than the full-vocabulary estimate.",
        "## Probe predictions",
        predictions,
        "## SHIP criteria",
        criteria,
        "## Analysis choices fixed before training",
        "Use paired row bootstrap with 10,000 resamples (seed 42). Compute token-weighted NLL within each sample and percentile 95% intervals. Treat KL at or below 1e-6 nats per answer token as approximately zero. Freeze source and model revisions, split hashes, training hyperparameters, generation cap, and sampling seeds 42, 43 and 44.",
        "Valid SQL means SQLite accepts EXPLAIN QUERY PLAN for the extracted SQL. Also report raw-output validity to distinguish format from schema errors.",
        "## Split hashes",
        *[f"- {name} {data[name]['sha256']}" for name in ("train", "train_shuffled", "eval", "eval_leaky")],
    ]
    with prereg.open("x", encoding="utf-8") as handle:
        handle.write("\n\n".join(parts) + "\n")
    digest = hashlib.sha256(prereg.read_bytes()).hexdigest()
    (ROOT / "prereg.sha256").write_text(f"{digest}  prereg.md\n", encoding="utf-8")
    now = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
    with (ROOT / "logs/prereg.log").open("a", encoding="utf-8") as handle:
        print(f"{now} INFO sha256sum prereg.md = {digest}", file=handle)
    print(f"{now} INFO sha256sum prereg.md = {digest}")


if __name__ == "__main__":
    preregister(*sys.argv[1:])
