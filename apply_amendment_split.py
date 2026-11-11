"""Audit CREATE TABLE overlap in the frozen training and evaluation files."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import sqlite3
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent
CREATE_TABLE = re.compile(r"^create\s+table\s", re.IGNORECASE)


def table_statements(context):
    start = 0
    parts = []
    for end in (match.end() for match in re.finditer(";", context)):
        if sqlite3.complete_statement(context[start:end]):
            parts.append(context[start:end])
            start = end
    parts.append(context[start:])
    result = set()
    for part in parts:
        key = " ".join(part.strip().removesuffix(";").lower().split())
        if key:
            if not CREATE_TABLE.match(key):
                raise ValueError(f"Expected CREATE TABLE statement: {part!r}")
            result.add(key)
    if not result:
        raise ValueError("Schema has no CREATE TABLE statement")
    return result


def load(name):
    return [json.loads(line) for line in (ROOT / f"data/{name}.jsonl").read_text().splitlines()]


def write_rows(path, rows):
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def audit():
    train = load("train")
    eval_rows = load("eval")
    leaky = load("eval_leaky")
    train_keys = set().union(*(table_statements(row["context"]) for row in train))
    eval_keys = set().union(*(table_statements(row["context"]) for row in eval_rows))
    removed = [
        {"source_index": row["source_index"], "category": row["category"], "shared": sorted(table_statements(row["context"]) & train_keys)}
        for row in eval_rows if table_statements(row["context"]) & train_keys
    ]
    drop = {row["source_index"] for row in removed}
    kept = [row for row in eval_rows if row["source_index"] not in drop]
    leaky_no_share = [row["source_index"] for row in leaky if not table_statements(row["context"]) & train_keys]
    assert not any(table_statements(row["context"]) & train_keys for row in kept)
    if leaky_no_share:
        raise ValueError(f"Expected eval_leaky rows to share training statements: {leaky_no_share}")
    write_rows(ROOT / "data/eval_amended.jsonl", kept)
    write_rows(ROOT / "data/ship_task_amended.jsonl", [
        {"prompt": row["messages"][1]["content"], "expected": row["answer"], "scoring": "exact", "category": row["category"]}
        for row in kept[:100]
    ])
    evidence = {
        "train_rows": len(train), "original_eval_rows": len(eval_rows), "shared_statement_count": len(train_keys & eval_keys),
        "shared_statements": sorted(train_keys & eval_keys), "excluded_eval_rows": removed, "remaining_eval_rows": len(kept),
        "eval_leaky_rows": len(leaky), "eval_leaky_without_shared_statement": leaky_no_share,
        "source_sha256": {name: hashlib.sha256((ROOT / f"data/{name}.jsonl").read_bytes()).hexdigest()
                          for name in ("train", "eval", "eval_leaky", "eval_amended", "ship_task_amended")},
    }
    (ROOT / "results/amendment_split.json").write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    print(f"{datetime.now(ZoneInfo('Asia/Almaty')).isoformat(timespec='milliseconds')} INFO "
          f"{evidence['shared_statement_count']} shared statements; excluded {len(removed)}/{len(eval_rows)}; "
          f"kept {len(kept)}; leaky rows with shared statements {len(leaky)}")
    return evidence


if __name__ == "__main__":
    audit()
