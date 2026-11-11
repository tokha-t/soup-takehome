"""Measure prompt removal and the three preregistered sampling seeds on the clean split."""
import sys

from instruments import ROOT, atomic_json, gpu_monitor, phase, setup_logging
from patch_common import PEFT, assert_amendment, install, mounted, read_json, sha, stage_result, step, use
from patch_stage_core import checkpoint, eval_path

NEAREST = {
    "url": "https://trysoup.dev/docs/soup-ship",
    "coverage": "Soup ship's metric leg compares base and tuned task accuracy. The prompt ablation, SQL scoring rules, and sampling settings are measured separately here.",
}


def probe():
    from transformers import AutoTokenizer
    from sql_metrics import score
    from verify import generate, generation_summary, load, read_eval

    setup_logging()
    path = checkpoint("merged_4bit")
    data = ROOT / "data/eval_amended.jsonl"
    tokenizer = AutoTokenizer.from_pretrained(path)
    rows = read_eval(data, tokenizer)
    no_system = read_eval(data, tokenizer, with_system=False)
    cap = read_json(ROOT / "results/data_stats.json")["max_new_tokens"]
    greedy = read_json(mounted("results/verify_real.json"))["generations"]["merged4"]
    shipped = read_json(path / "generation_config.json")
    if not shipped.get("do_sample"):
        raise RuntimeError("Saved generation_config does not enable registered sampling")
    actual = {key: shipped[key] for key in ("temperature", "top_p", "top_k", "repetition_penalty") if key in shipped}
    gold = [score(row.context, row.answer, row.answer) for row in rows]
    with phase("probe_decoding_4bit"):
        model = load(path)
        outputs = {}
        for seed in (42, 43, 44):
            samples = generate(model, tokenizer, rows, cap, sampled_config=actual, seed=seed)
            outputs[str(seed)] = samples
            atomic_json(ROOT / f"results/P4_sampling_seed_{seed}.json", samples)
        pooled = generation_summary([row for samples in outputs.values() for row in samples])
        p4 = {
            "shipped_generation_config": shipped, "effective_sampling": actual, "max_new_tokens": cap,
            "sampling_by_seed": {seed: generation_summary(samples) for seed, samples in outputs.items()},
            "sampling_pooled": pooled, "greedy_1_0": greedy,
            "shipped_match_cost": greedy["all"]["lenient"] - pooled["all"]["lenient"],
            "gold_valid_sql": sum(row["valid_sql"] for row in gold) / len(gold), "gold_rows": len(gold),
            "soup_gate": NEAREST,
        }
        atomic_json(ROOT / "results/probe_P4.json", p4)
        audit = []
        without = generate(model, tokenizer, no_system, cap, audit=audit)
        atomic_json(ROOT / "results/probe_P3.json", {
            "with_system": greedy, "without_system": generation_summary(without),
            "default_render": tokenizer.decode(no_system[0].prompt_ids), "batching": audit,
            "soup_gate": NEAREST, "outputs_without_system": without,
        })
        p4["greedy_1_1"] = generation_summary(generate(model, tokenizer, rows, cap, rep=1.1, audit=audit))
        p4["batching"] = audit[1:]
        atomic_json(ROOT / "results/probe_P4.json", p4)


def main():
    amendment = assert_amendment()
    data = eval_path()
    use("results/data_stats.json")
    python = install("decode", PEFT)
    with gpu_monitor():
        status = step([python, ROOT / "patch_stage_decode.py", "probe"], "decode", attempts=2, limit=2700)
    stage_result("5", "complete" if status == 0 else "partial",
                 amendment=amendment, eval_sha256=sha(data), exit_code=status)


if __name__ == "__main__":
    (probe if sys.argv[-1] == "probe" else main)()
