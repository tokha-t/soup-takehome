"""Complete P12 with rows that violate the ChatML message type."""
import json
from instruments import ROOT, atomic_json
from patch_common import assert_amendment, install, sha, stage_result, stamp, step, use


def main():
    amendment = assert_amendment()
    source = use("data/train_soup.jsonl")
    lines = source.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1500
    broken = ROOT / "data/train_soup_type_broken.jsonl"
    invalid = [json.dumps({"messages": "BROKEN", "corrupt_row": index}) for index in range(750, 1500)]
    broken.write_text("\n".join(lines[:750] + invalid) + "\n", encoding="utf-8")
    assert all(isinstance(json.loads(row)["messages"], list) for row in lines[:750])
    assert all(isinstance(json.loads(row)["messages"], str) for row in invalid)
    python = install("validate-followup", ("soup-cli==0.77.2",))
    soup = python.with_name("soup")
    codes = {}
    for label, path, extra in (
        ("clean_strict", source, ["--min-valid-fraction", "1.0"]),
        ("half_default", broken, []),
        ("half_strict", broken, ["--min-valid-fraction", "1.0"]),
    ):
        codes[label] = step([soup, "data", "validate", path, *extra], f"validate_type_{label}",
                            success=(0, 2), attempts=1, limit=120)
    record = {
        "total_rows": 1500, "corrupt_rows": 750, "corruption": "messages is a string rather than a list",
        "clean_sha256": sha(source), "corrupt_sha256": sha(broken), "exit_codes": codes,
        "gate_logs": {name: f"logs/stages/7p12/validate_type_{name}.log" for name in codes},
        "initial_fixture": "results/probe_P12.json",
        "reason": "The initial empty-message fixture remained structurally valid to Soup. Preserve that result and test the threshold with type-invalid rows.",
        "finished": stamp(),
    }
    atomic_json(ROOT / "results/probe_P12_type_corruption.json", record)
    stage_result("7p12", "complete" if all(code is not None for code in codes.values()) else "partial",
                 amendment=amendment, exit_codes=codes)


if __name__ == "__main__":
    main()
