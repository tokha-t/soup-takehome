"""Validate the amendment and clean evaluation cohort against the attached training commit."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
from zoneinfo import ZoneInfo

from apply_amendment_split import ROOT, audit

source, = Path("/kaggle/input").rglob("prereg.md")
original = source.parent
for name in ("prereg.md", "data/train.jsonl", "data/eval.jsonl", "data/eval_leaky.jsonl", "results/data_stats.json"):
    shutil.copy2(original / name, ROOT / name)
for name in ("prereg", "prereg_amendment_1"):
    actual = hashlib.sha256((ROOT / f"{name}.md").read_bytes()).hexdigest()
    expected = (ROOT / f"{name}.sha256").read_text().split()[0]
    if actual != expected:
        raise RuntimeError(f"{name} hash mismatch")
data = audit()
assert data["shared_statement_count"] == len(data["excluded_eval_rows"]) == 8
assert data["remaining_eval_rows"] == 192
train = [json.loads(line) for line in (ROOT / "data/train.jsonl").read_text().splitlines()]
with (ROOT / "data/train_soup.jsonl").open("w", encoding="utf-8") as output:
    for row in train:
        output.write(json.dumps({"messages": row["messages"]}, ensure_ascii=False) + "\n")
failed = ROOT / "logs/failed/soup_format"
failed.mkdir(parents=True, exist_ok=True)
shutil.copy2(original / "logs/soup_train_real.log", failed / "raw.log")
(failed / "note.txt").write_text(
    "Soup rejected all 1,500 rows because pre_tokenized expected input_ids and the source rows had messages.\n"
    "The three adapters trained with PEFT. After training, converted those same rows to ChatML for strict validation and the separate Soup bench.\n",
    encoding="utf-8",
)
finished = datetime.now(ZoneInfo("Asia/Almaty")).isoformat(timespec="seconds")
(ROOT / "results/stage_1.json").write_text(json.dumps({"status": "complete", "finished": finished,
    "amendment_sha256": hashlib.sha256((ROOT / "prereg_amendment_1.md").read_bytes()).hexdigest(),
    "clean_eval_sha256": data["source_sha256"]["eval_amended"],
    "soup_error_log_sha256": hashlib.sha256((failed / "raw.log").read_bytes()).hexdigest(),
}, indent=2) + "\n")
print(f"{finished} INFO Amendment and statement-level split checks complete")
