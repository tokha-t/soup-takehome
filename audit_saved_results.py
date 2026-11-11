"""Read-only CPU replay of selected saved evidence; no model or GPU required."""
import hashlib
import json
from pathlib import Path
import sqlite3

from s5_followup import evaluation, flags

ROOT = Path(__file__).resolve().parent


def main():
    hashes = {}
    for name in ("prereg", "prereg_amendment_1"):
        actual = hashlib.sha256((ROOT / f"{name}.md").read_bytes()).hexdigest()
        expected = (ROOT / f"{name}.sha256").read_text().split()[0]
        if actual != expected:
            raise ValueError(f"{name} hash mismatch")
        hashes[name] = actual
    verify = json.loads((ROOT / "results/verify_real.json").read_text())
    rows = verify["per_row"]
    tokens = sum(row["tokens"] for row in rows)
    losses = {name: sum(row["nll"][name] for row in rows) / tokens
              for name in ("base", "merged", "base4", "merged4")}
    records, schemas, ids = evaluation()
    seeds = {}
    for seed in (42, 43, 44):
        outputs = json.loads((ROOT / f"results/P4_sampling_seed_{seed}.json").read_text())
        changed = []
        accepted = flags(outputs, records, schemas, changed)
        seeds[str(seed)] = {
            "valid": sum(accepted[row_id] for row_id in ids),
            "stopped": sum(row["stop"] for row in outputs if row["id"] in ids),
            "total": len(ids), "original_flags_different_on_this_sqlite": changed,
        }
    valid = sum(row["valid"] for row in seeds.values())
    total = sum(row["total"] for row in seeds.values())
    print(json.dumps({
        "read_only": True, "sqlite_version": sqlite3.sqlite_version,
        "registration_hashes_match": hashes, "official_rows": len(rows),
        "recomputed_official_answer_nll": losses,
        "compatibility_gold": {"valid": len(ids), "total": len(records)},
        "posthoc_seeds": seeds,
        "posthoc_valid_sql": {"valid": valid, "total": total, "rate": valid / total},
        "posthoc_stopped": sum(row["stopped"] for row in seeds.values()),
        "scope": "Replays saved outputs and stated compatibility rules; does not rerun models or establish training provenance.",
    }, indent=2))


if __name__ == "__main__":
    main()
