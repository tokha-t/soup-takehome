"""Build the documented Soup ChatML configuration; the adapters were trained with PEFT."""
from pathlib import Path
import sys

from instruments import ROOT
from patch_common import MODEL, REVISION, read_json, step


def make(soup, *, base=MODEL):
    import yaml

    template = ROOT / "configs/patch_soup_generated.yaml"
    if step([soup, "init", "--template", "chat", "--output", template, "--force"], "soup_init_chat") != 0:
        raise RuntimeError("Soup's chat template did not initialize")
    stats = read_json(ROOT / "results/data_stats.json")
    config = yaml.safe_load(template.read_bytes())
    config.update(base=str(base), backend="transformers", output="runs/bench_real")
    config["data"].update(
        train="data/train_soup.jsonl", format="chatml", val_split=0.0,
        max_length=stats["max_length"], train_on_responses_only=True,
    )
    config["training"].update(
        epochs=2, lr=2e-4, batch_size=8, gradient_accumulation_steps=1, quantization="none",
        optimizer="adamw_torch", scheduler="cosine", warmup_ratio=9.999 / 376,
        gradient_checkpointing="full", train_on_eot=True, seed=42, data_seed=42,
        weight_decay=0.0, max_grad_norm=1.0, logging_steps=1, rewind_log=False,
    )
    # The 20-step benchmark ends before R1's first evaluation at step 25.
    # Soup's ChatML schema disallows eval_steps when the file has no validation split.
    config["training"].pop("eval_steps", None)
    config["training"]["lora"].update(
        r=16, alpha=32, dropout=0.0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    paths = []
    for label, lr in (("real", 2e-4), ("placebo", 0.0)):
        config["training"]["lr"] = lr
        config["output"] = f"runs/bench_{label}"
        path = ROOT / f"configs/patch_soup_{label}.yaml"
        path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        paths.append(path)
    return paths


def bench_config(path, steps):
    import yaml

    config = yaml.safe_load(path.read_bytes())
    # Soup converts this fraction against the step budget overridden by the bench command.
    config["training"]["warmup_ratio"] = 9.999 / steps
    target = path.with_name(path.stem + "_bench.yaml")
    target.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return target


if __name__ == "__main__":
    make(Path(sys.argv[1]), base=sys.argv[2] if len(sys.argv) > 2 else MODEL)
