"""P11. Compare SQL results on small literal-populated synthetic fixtures."""
from collections import Counter
import json
from pathlib import Path
import re
import sqlite3
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import sqlglot
from sqlglot import exp
from instruments import ROOT, LOG, atomic_json, setup_logging
from sql_metrics import extract_sql


def fixture(context, gold, prediction):
    parsed = sqlglot.parse_one(gold)
    literals = []
    for node in parsed.find_all(exp.Literal):
        value = node.this
        if not node.is_string:
            try:
                value = int(value)
            except ValueError:
                try:
                    value = float(value)
                except ValueError:
                    continue
        if value not in literals:
            literals.append(value)
    with sqlite3.connect(":memory:") as database:
        allowed = {sqlite3.SQLITE_CREATE_TABLE, sqlite3.SQLITE_CREATE_INDEX, sqlite3.SQLITE_INSERT, sqlite3.SQLITE_UPDATE, sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ}
        database.set_authorizer(lambda action, *_: sqlite3.SQLITE_OK if action in allowed else sqlite3.SQLITE_DENY)
        database.executescript(context)
        database.set_authorizer(None)
        quote = lambda value: '"' + value.replace('"', '""') + '"'
        tables = database.execute("SELECT name FROM sqlite_schema WHERE type='table' AND name NOT LIKE 'sqlite_%'").fetchall()
        for (table,) in tables:
            columns = database.execute(f"PRAGMA table_info({quote(table)})").fetchall()
            numeric = (0, 1, 7, 42, 123)
            values = [*literals, *numeric, "synthetic", None]
            insert = f"INSERT OR IGNORE INTO {quote(table)} VALUES ({','.join('?' for _ in columns)})"
            uniform = [[value]*len(columns) for value in values]
            mixed = [[values[(i+j)%len(values)] for j in range(len(columns))] for i in range(len(values))]
            database.executemany(insert, uniform+mixed)
        # Finish constructing the fixture while writes and transaction
        # control are allowed. Only read-only operations follow.
        database.commit()
        database.execute("PRAGMA query_only = ON")
        allowed = {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_FUNCTION, sqlite3.SQLITE_RECURSIVE}
        database.set_authorizer(lambda action, *_: sqlite3.SQLITE_OK if action in allowed else sqlite3.SQLITE_DENY)
        calls = [0]
        def interrupt():
            calls[0] += 1
            return calls[0] > 1000
        database.set_progress_handler(interrupt, 1000)
        gold_rows = database.execute(gold).fetchall()
        try:
            outputs = database.execute(prediction).fetchall()
        except sqlite3.Error as exc:
            return {"matched": False, "error": str(exc), "gold_rows": len(gold_rows)}
        ordered = bool(re.search(r"\bORDER\s+BY\b", gold, re.I))
        return {
            "matched": gold_rows == outputs if ordered else Counter(gold_rows) == Counter(outputs),
            "gold_rows": len(gold_rows), "output_rows": len(outputs),
            "both_empty": not gold_rows and not outputs,
        }


def main():
    setup_logging()
    verify = json.loads((ROOT / "results/verify_real.json").read_bytes())
    rows = [json.loads(line) for line in (ROOT / "data/eval_amended.jsonl").read_text().splitlines()]
    results = {}
    for name in ("base", "merged", "merged4"):
        items = []
        for row, generated in zip(rows, verify["raw_outputs"][name], strict=True):
            assert row["source_index"] == generated["id"]
            try:
                evidence = fixture(row["context"], row["answer"], extract_sql(generated["raw"]))
            except (sqlite3.Error, sqlglot.errors.SqlglotError) as exc:
                evidence = {"matched": None, "gold_error": str(exc)}
            items.append({"source_index": row["source_index"], "lenient": generated["lenient"], **evidence})
        usable = [x for x in items if x["matched"] is not None]
        results[name] = {
            "total": len(items), "usable": len(usable),
            "execution_match": sum(item["matched"] for item in usable) / len(usable) if usable else None,
            "lenient_on_same_rows": sum(item["lenient"] for item in usable) / len(usable) if usable else None,
            "both_empty": sum(item.get("both_empty", False) for item in usable), "rows": items,
        }
    atomic_json(ROOT / "results/probe_P11.json", {"models": results, "soup_gate": {
        "url": "https://trysoup.dev/docs/soup-ship",
        "coverage": "Soup's metric leg uses the configured task scorer; these task rows use exact matching and do not execute SQL against the synthetic fixtures."},
        "limitation": "A finite synthetic fixture can give false execution matches, especially when both results are empty."})
    LOG.info("Probe P11 saved.")


if __name__ == "__main__":
    main()
