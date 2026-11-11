"""Commit the read-only synthetic SQL execution probe with frozen inputs."""
import time

from patch_common import CUTOFF, ROOT, assert_amendment, install, mounted, read_json, sha, stage_result, step, use


def main():
    if time.time() >= CUTOFF:
        stage_result("8p11", "skipped", reason="08:30 cutoff")
        return
    amendment = assert_amendment()
    eval_path = use("data/eval_amended.jsonl")
    if sha(eval_path) != read_json(mounted("results/stage_1.json"))["clean_eval_sha256"]:
        raise RuntimeError("Clean evaluation set hash mismatch")
    verify = use("results/verify_real.json")
    python = install("execution-probe", ("sqlglot==28.0.0",), optional=True)
    status = step([python, ROOT / "probes/execution_match.py"], "probe_P11_execution_retry", optional=True, limit=180)
    result = ROOT / "results/probe_P11.json"
    measured = status == 0 and result.is_file() and all(row["usable"] for row in read_json(result)["models"].values())
    stage_result("8p11", "complete" if measured else "partial", amendment=amendment,
                 exit_code=status, eval_sha256=sha(eval_path), verifier_sha256=sha(verify),
                 raw_log="logs/stages/8p11/probe_P11_execution_retry.log")


if __name__ == "__main__":
    main()
