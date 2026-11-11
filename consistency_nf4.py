"""POST-HOC reproduction of the Soup NF4 checkpoint."""
from datetime import datetime
from hashlib import sha256
import json
from pathlib import Path
import shlex
import sys
import time

from instruments import LOG, ROOT, atomic_json, gpu_monitor, phase, setup_logging
import patch_common
from patch_common import PEFT, SOUP, base_path, install, mounted, read_json, sha, step, use
from s5_followup import EVAL, LABEL, RESULTS, evaluation, scores

SCRATCH = Path("/kaggle/temp/posthoc-s5-consistency")
TARGET = SCRATCH / "merged_4bit_repeat"
HASHES = RESULTS / "posthoc_consistency_tensor_hashes.json"
GENERATIONS = RESULTS / "posthoc_consistency_nf4_generations.json"
STAGE = RESULTS / "stage_posthoc_consistency_nf4.json"
REFERENCE_SHA = "483fd9ab9ae745da787a627e7c8e3ad23e7343b978a32a3d59976d1874090525"
ADAPTER_SHA = "3ffb665f1c6358f06d6b77268872463af4de734a07b48733bdb6a75ffd495b27"


def tensor_hashes(path):
    """Hash each named safetensors payload and retain its dtype and shape."""
    size = path.stat().st_size
    hashes = {}
    with path.open("rb") as stream:
        header_size = int.from_bytes(stream.read(8), "little")
        assert 0 < header_size <= min(size - 8, 100_000_000)
        header = json.loads(stream.read(header_size))
        data_start = 8 + header_size
        file_digest = sha256()
        stream.seek(0)
        file_digest.update(stream.read(data_start))
        position = 0
        entries = sorted(
            ((key, item) for key, item in header.items() if key != "__metadata__"),
            key=lambda pair: pair[1]["data_offsets"],
        )
        for key, item in entries:
            start, end = item["data_offsets"]
            assert position == start <= end <= size - data_start, key
            digest = sha256()
            while position < end:
                chunk = stream.read(min(end - position, 8 * 1024 * 1024))
                assert chunk, key
                position += len(chunk)
                digest.update(chunk)
                file_digest.update(chunk)
            hashes[key] = {"sha256": digest.hexdigest(), "dtype": item["dtype"], "shape": item["shape"]}
        assert position == size - data_start
    return file_digest.hexdigest(), hashes


def reference_path():
    return mounted("artifacts/merged_4bit_portable/config.json").parent


def compare():
    reference_digest, reference = tensor_hashes(reference_path() / "model.safetensors")
    assert reference_digest == REFERENCE_SHA
    repeated_digest, repeated = tensor_hashes(TARGET / "model.safetensors")
    keys = sorted(reference.keys() | repeated.keys())
    differing = [key for key in keys if reference.get(key) != repeated.get(key)]
    atomic_json(HASHES, {
        "label": LABEL,
        "format_reference": "https://github.com/safetensors/safetensors#format",
        "method": "SHA-256 of each raw safetensors payload. Comparison includes dtype, shape, and key identity.",
        "reference": str(reference_path() / "model.safetensors"),
        "reference_file_sha256": reference_digest, "repeat_file_sha256": repeated_digest,
        "byte_identical_file": reference_digest == repeated_digest,
        "tensor_count_reference": len(reference), "tensor_count_repeat": len(repeated),
        "identical_tensors": len(keys) - len(differing), "differing_tensors": differing,
        "per_tensor": {key: {"reference": reference.get(key), "repeat": repeated.get(key)} for key in keys},
    })
    LOG.info("%s tensor comparison: %s of %s identical", LABEL, len(keys) - len(differing), len(keys))


def greedy():
    import bitsandbytes
    import torch
    from transformers import AutoTokenizer
    from verify import generate, load, read_eval

    assert torch.cuda.device_count() == 1
    records, schemas, ids = evaluation()
    tokenizer = AutoTokenizer.from_pretrained(reference_path(), local_files_only=True)
    rows = read_eval(EVAL, tokenizer)
    verify = read_json(RESULTS / "verify_real.json")
    cap = verify["generation_cap"]
    audit = []
    with phase("posthoc_consistency_nf4_greedy"):
        model = load(TARGET)
        count = sum(isinstance(module, bitsandbytes.nn.Linear4bit) for module in model.modules())
        assert count == 196
        generated = generate(model, tokenizer, rows, cap, rep=1.0, seed=42, audit=audit)
    metrics = scores(generated, records, schemas, ids)
    original = verify["raw_outputs"]["merged4"]
    changes = [a["id"] for a, b in zip(generated, original, strict=True) if a["raw"] != b["raw"]]
    assert [row["id"] for row in generated] == [row["id"] for row in original]
    atomic_json(GENERATIONS, {
        "label": LABEL, "model_sha256": read_json(HASHES)["repeat_file_sha256"],
        "greedy_191": metrics, "archived_merged_4bit_greedy_191": scores(original, records, schemas, ids),
        "raw_generation_differences_all_192": changes, "batching": audit, "max_new_tokens": cap,
        "linear_4bit_modules": count, "outputs": generated, "sqlite_version": __import__("sqlite3").sqlite_version,
    })
    LOG.info("%s repeated NF4 greedy %s", LABEL, metrics)


def remote(deadline):
    patch_common.DEADLINE = datetime.fromisoformat(deadline).timestamp()
    status = {"label": LABEL, "deadline": deadline, "base_revision": patch_common.REVISION}
    try:
        if time.time() >= patch_common.DEADLINE:
            status["status"] = "deadline_before_start"
            return
        for path in ("data/eval_amended.jsonl", "results/verify_real.json"):
            use(path)
        SCRATCH.mkdir(parents=True, exist_ok=True)
        peft = install("consistency-peft", PEFT)
        soup = install("consistency-soup", SOUP).with_name("soup")
        script = Path(__file__).resolve()
        base_record = RESULTS / "posthoc_consistency_base.json"
        status["base_exit"] = step([peft, script, "base", base_record], "posthoc_consistency_base")
        if status["base_exit"] != 0:
            return
        adapter = mounted("runs/real/adapter/adapter_model.safetensors")
        assert sha(adapter) == ADAPTER_SHA
        status["adapter_sha256"] = ADAPTER_SHA
        command = [soup, "merge", "--adapter", adapter.parent, "--base", read_json(base_record)["base"],
                   "--output", TARGET, "--dtype", "float16", "--save-format", "4bit"]
        status["command"] = shlex.join(map(str, command))
        with gpu_monitor():
            status["merge_exit"] = step(command, "posthoc_consistency_merge", cwd=SCRATCH)
            if status["merge_exit"] != 0:
                return
            status["hashes_exit"] = step([peft, script, "compare"], "posthoc_consistency_compare")
            if status["hashes_exit"] != 0:
                return
            status["greedy_exit"] = step([peft, script, "greedy"], "posthoc_consistency_greedy")
    except Exception as error:
        status["error"] = f"{type(error).__name__}: {error}"
        LOG.exception("%s optional NF4 setup failed", LABEL)
    finally:
        status["finished"] = patch_common.stamp()
        status.setdefault("status", "complete" if status.get("greedy_exit") == 0 else "partial")
        atomic_json(STAGE, status)
        print(f"{LABEL} optional NF4 result: {status}", flush=True)


if __name__ == "__main__":
    setup_logging()
    mode = sys.argv[1]
    if mode == "compare":
        compare()
    elif mode == "greedy":
        greedy()
    elif mode == "base":
        atomic_json(Path(sys.argv[2]), {"label": LABEL, "base": str(base_path(Path(sys.executable)))})
    else:
        remote(sys.argv[2])
