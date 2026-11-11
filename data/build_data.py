"""Build deterministic schema-disjoint SQL data after the rescore decision."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import logging
import math
from pathlib import Path
import random
import re

from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
SEED = 42
MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
DATASET = "b-mc2/sql-create-context"
DATA_REVISION = "9d80a6a118b838d9defc3798d659a54a2ac2ff37"
SYSTEM = "You translate questions into SQL. Use the given table schema. Reply with one SQL query only."
CATEGORIES = ("complex", "multi_cond", "agg", "simple")
LITERALS = re.compile(r"""'(?:''|[^'])*'|"(?:""|[^"])*"|`[^`]*`""")
HARDER = re.compile(r"\bJOIN\b|\bGROUP\s+BY\b|\bORDER\s+BY\b|\bHAVING\b|\(\s*SELECT\b", re.I)
COMPLEX = re.compile(HARDER.pattern + r"|\b(?:UNION|INTERSECT|EXCEPT)\b", re.I)
AGG = re.compile(r"\b(?:COUNT|SUM|AVG|MIN|MAX)\s*\(", re.I)
WHERE = re.compile(r"\bWHERE\b(.*)", re.I | re.S)
CONDITIONS = re.compile(r"\b(?:AND|OR)\b", re.I)


def norm(text):
    return " ".join(text.lower().split())


def category(sql):
    sql = LITERALS.sub("''", sql)
    if COMPLEX.search(sql):
        return "complex"
    where = WHERE.search(sql)
    if where and CONDITIONS.search(where[1]):
        return "multi_cond"
    return "agg" if AGG.search(sql) else "simple"


def record(row):
    return {
        **row,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"Schema:\n{row['context']}\n\nQuestion: {row['question']}"},
            {"role": "assistant", "content": row["answer"]},
        ],
    }


def ranked(values, percent):
    ordered = sorted(values)
    return ordered[math.ceil(len(ordered) * percent / 100) - 1]


def stats(values):
    return {**{f"p{p}": ranked(values, p) for p in (50, 95, 99)}, "max": max(values)}


def take_by_category(rows, quota, rng):
    selected = []
    for name in CATEGORIES:
        candidates = [row for row in rows if row["category"] == name]
        selected.extend(rng.sample(candidates, min(len(candidates), quota)))
        if len(candidates) < quota:
            logging.warning("%s supplies %s/%s requested rows", name, len(candidates), quota)
    rng.shuffle(selected)
    return selected


def split(raw, harder=False):
    unique = {}
    for source_index, row in enumerate(raw):
        key = (norm(row["context"]), norm(row["question"]))
        unique.setdefault(key, {
            "source_index": source_index,
            "context": row["context"].strip(), "question": row["question"].strip(),
            "answer": row["answer"].strip(), "category": category(row["answer"]),
        })
    groups = defaultdict(list)
    for (schema, _), row in unique.items():
        if not harder or HARDER.search(LITERALS.sub("''", row["answer"])):
            groups[schema].append(row)
    keys = list(groups)
    rng = random.Random(SEED)
    rng.shuffle(keys)
    eval_schemas = set()
    eval_pool = []
    available = Counter(row["category"] for rows in groups.values() for row in rows)
    counts = Counter()
    for schema in keys:
        if all(counts[name] >= min(50, available[name]) for name in CATEGORIES):
            break
        rows = groups[schema]
        if not any(counts[row["category"]] < min(50, available[row["category"]]) for row in rows):
            continue
        eval_schemas.add(schema)
        eval_pool.extend(rows)
        counts.update(row["category"] for row in rows)
    eval_rows = take_by_category(eval_pool, 50, rng)
    train_rows = take_by_category(
        [row for schema in keys if schema not in eval_schemas for row in groups[schema]], 375, rng,
    )
    train_schemas = {norm(row["context"]) for row in train_rows}
    train_questions = {norm(row["question"]) for row in train_rows}
    leaky_pool = [
        row for schema in sorted(train_schemas) for row in groups[schema]
        if norm(row["question"]) not in train_questions
    ]
    if len(leaky_pool) < 100:
        raise ValueError(f"Only {len(leaky_pool)} leaky probe rows")
    leaky_rows = rng.sample(leaky_pool, 100)
    assert not train_schemas & {norm(row["context"]) for row in eval_rows}
    shuffled_answers = [row["answer"] for row in train_rows]
    random.Random(SEED).shuffle(shuffled_answers)
    shuffled = [{**row, "answer": answer} for row, answer in zip(train_rows, shuffled_answers, strict=True)]
    return {
        "train": train_rows, "eval": eval_rows, "eval_leaky": leaky_rows, "train_shuffled": shuffled,
    }, {"source_rows": len(raw), "dedup_dropped": len(raw) - len(unique), "scope_rows": sum(map(len, groups.values())), "scope_categories": dict(available), "harder": harder}


def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path)
    parser.add_argument("--tokenizer", default=MODEL)
    parser.add_argument("--harder", action="store_true")
    args = parser.parse_args()
    if args.source is None:
        from huggingface_hub import hf_hub_download
        args.source = hf_hub_download(DATASET, "sql_create_context_v4.json", repo_type="dataset", revision=DATA_REVISION)
    raw = json.loads(Path(args.source).read_bytes())
    splits, result = split(raw, args.harder)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, revision=REVISION)
    lengths = []
    full_by_split = {}
    result["dataset_revision"] = DATA_REVISION
    for name, original in splits.items():
        rows = [record(row) for row in original]
        full_text = tokenizer.apply_chat_template([row["messages"] for row in rows], tokenize=False)
        full_counts = list(map(len, tokenizer(full_text, add_special_tokens=False)["input_ids"]))
        answer_counts = list(map(len, tokenizer([row["answer"] for row in rows], add_special_tokens=False)["input_ids"]))
        if name in ("train", "eval"):
            lengths.extend(full_counts)
        full_by_split[name] = full_counts
        counts = Counter(row["category"] for row in rows)
        result[name] = {
            "rows": len(rows), "counts": dict(counts),
            "shares": {key: counts[key] / len(rows) for key in CATEGORIES},
            "full_tokens": stats(full_counts), "answer_tokens": stats(answer_counts),
            "sha256": None,
        }
        path = ROOT / f"data/{name}.jsonl"
        write_jsonl(path, rows)
        result[name]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    result["max_length"] = math.ceil(max(lengths) / 64) * 64
    result["p96_full_train"] = ranked(full_by_split["train"], 96)
    result["train_eval_schema_overlap"] = 0
    result["leaky_schema_overlap"] = 1.0
    result["leaky_question_overlap"] = 0.0
    result["shuffled_over_max_length"] = sum(n > result["max_length"] for n in full_by_split["train_shuffled"])
    result["shuffled_answers_unchanged"] = sum(a["answer"] == b["answer"] for a,b in zip(splits["train"], splits["train_shuffled"], strict=True))
    result["max_new_tokens"] = 2 * result["eval"]["answer_tokens"]["p95"]
    train_example = record(splits["train"][0])
    eval_example = record(splits["eval"][0])
    logging.info("Rendered training %r", tokenizer.apply_chat_template(train_example["messages"], tokenize=False))
    logging.info("Rendered eval %r", tokenizer.apply_chat_template(eval_example["messages"][:-1], tokenize=False, add_generation_prompt=True))
    write_jsonl(ROOT / "data/ship_task.jsonl", [
        {"prompt": row["messages"][1]["content"], "expected": row["answer"], "scoring": "exact", "category": row["category"]}
        for row in map(record, splits["eval"][:100])
    ])
    (ROOT / "results/data_stats.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    logging.info("Data statistics %s", json.dumps(result))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s.%(msecs)03d %(levelname)s %(message)s", datefmt="%Y-%m-%dT%H:%M:%S")
    main()
