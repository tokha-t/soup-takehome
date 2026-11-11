"""Predict device memory from the pinned Qwen configuration and planned batch."""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
GB = 1_000_000_000
GIB = 2**30
MIB = 2**20


def amended_predictions():
    """Patch 1 §1 values, independent of the original preregistration estimate."""
    return {
        "units": "GiB and MiB", "weights_fp16_gib": 2.875, "adapter_gradient_adamw_gib": .275,
        "logits_mib_per_training_token": [1.45, 2.03], "activation_mib_per_training_token": {"on": .17, "off": 2.6},
        "merged_file_gib": {"fp16": 2.875, "bf16": 2.875, "fp32": 5.75}, "nf4_file_gib": 1.06,
        "load_fp16_cuda_gib": 2.9, "training_fixed_cuda_gib": 3.2, "training_padded_token_mib": [1.6, 2.2],
        "eval_fixed_cuda_gib": 2.9, "eval_padded_token_mib": [.9, 1.5],
        "merge_fp16_bf16_cuda_gib": 3.0, "merge_fp16_bf16_cpu_rss_gib": [2.9, 5.8],
        "merge_fp32_cuda_gib": 5.8, "merge_fp32_cpu_rss_gib": [5.8, 11.5],
        "quantize_cuda_gib": [1.1, 1.3], "quantize_cpu_rss_gib": 2.9, "infer_nf4_cuda_gib": [1.2, 1.5],
    }


def historical_predict(config, batch, length, rank):
    h = config["hidden_size"]
    ff = config["intermediate_size"]
    layers = config["num_hidden_layers"]
    heads = config["num_attention_heads"]
    head_dim = config.get("head_dim", h // heads)
    kv = head_dim * config["num_key_value_heads"]
    vocab = config["vocab_size"]
    shapes = [(h, h), (h, kv), (h, kv), (h, h), (h, ff), (h, ff), (ff, h)]
    linear = layers * sum(in_features * out_features for in_features, out_features in shapes)
    biases_and_norms = layers * (h + 2 * kv + 2 * h) + h
    embedding = vocab * h
    other = embedding * (1 if config["tie_word_embeddings"] else 2) + biases_and_norms
    parameters = linear + other
    lora = rank * layers * sum(in_features + out_features for in_features, out_features in shapes)
    weights, adapter_state = 2 * parameters, 16 * lora
    logits = [vocab * n for n in (10, 14)]
    # The spec rounds one layer's saved intermediates to roughly 97 KB.
    recompute = 2 * (h + 2 * kv + 5 * ff + heads * head_dim)
    activations = {"on": layers * h * 2 + recompute, "off": layers * recompute}
    quantized = 0.516 * linear + 2 * other
    largest_matrix = max(in_size * out_size for in_size, out_size in shapes)
    train_tokens = batch * length
    train = {
        mode: [(weights + adapter_state + train_tokens * (logit + per_token)) / GB for logit in logits]
        for mode, per_token in activations.items()
    }
    phases = {
        "model_load_fp16": weights,
        "adapter_save": weights + 4 * lora,
        "merge_fp32": 4 * parameters + 4 * lora + 8 * largest_matrix,
        "merge_fp16": weights + 4 * lora + 6 * largest_matrix,
        "merge_bf16": weights + 4 * lora + 6 * largest_matrix,
        "quantize_merged": quantized + 2 * largest_matrix,
        "quantize_base": quantized + 2 * largest_matrix,
        "inference_4bit": quantized + train_tokens * 2 * layers * kv * 2 + batch * vocab * 4,
    }
    return {
        "units": "decimal GB unless labelled bytes", "batch": batch, "max_length": length,
        "rank": rank, "parameters": parameters, "linear_parameters": linear, "lora_parameters": lora,
        "weights_fp16_gb": weights / GB, "adapter_gradient_adamw_gb": adapter_state / GB,
        "logits_bytes_per_token_range": logits, "activation_bytes_per_token": activations,
        "cuda_context_gb": [0.3, 0.6], "merged_checkpoint_gb": {"fp16": weights / GB, "fp32": 4 * parameters / GB},
        "nf4_gb": quantized / GB, "training_peak_gb": train,
        "other_phase_peak_gb": {name: amount / GB for name, amount in phases.items()},
        "training_512_sanity_gb": {
            mode: [(weights + adapter_state + 8 * 512 * (logit + per_token)) / GB for logit in logits]
            for mode, per_token in activations.items()
        },
    }


def historical_markdown(values):
    rows = [
        ("Parameters", f"{values['parameters']:,}"),
        ("Base weights fp16", f"{values['weights_fp16_gb']:.3f} GB"),
        ("LoRA parameters", f"{values['lora_parameters']:,}"),
        ("Adapter, gradients and AdamW fp32", f"{values['adapter_gradient_adamw_gb']:.3f} GB"),
        ("Logits per token", "–".join(f"{x/1e6:.3f}" for x in values["logits_bytes_per_token_range"]) + " MB"),
        *[(f"Activations per token, checkpointing {mode}", f"{value/1e6:.3f} MB")
          for mode, value in values["activation_bytes_per_token"].items()],
        ("CUDA context visible to NVIDIA sampler", "–".join(map(str, values["cuda_context_gb"])) + " GB"),
        ("NF4 checkpoint", f"{values['nf4_gb']:.3f} GB"),
        *[(f"Training peak, checkpointing {mode}", "–".join(f"{x:.3f}" for x in interval) + " GB")
          for mode, interval in values["training_peak_gb"].items()],
        *[(name.replace("_", " "), f"{value:.3f} GB") for name, value in values["other_phase_peak_gb"].items()],
    ]
    lines = ["| Term or phase | Prediction |", "| --- | --- |"]
    lines.extend(f"| {key} | {value} |" for key, value in rows)
    return "\n".join(lines)


def predict(config, batch, length, rank):
    old = historical_predict(config, batch, length, rank)
    values = amended_predictions()
    tokens = batch * length
    values.update({k: old[k] for k in ("batch", "max_length", "rank", "parameters", "linear_parameters", "lora_parameters")})
    values["padded_tokens_per_batch"] = tokens
    values["training_checkpoint_on_cuda_gib"] = [values["training_fixed_cuda_gib"] + tokens * slope / 1024
                                                 for slope in values["training_padded_token_mib"]]
    values["eval_cuda_gib"] = [values["eval_fixed_cuda_gib"] + tokens * slope / 1024
                              for slope in values["eval_padded_token_mib"]]
    values["formula_recomputed_from_config"] = {
        "weights_fp16_gib": 2 * old["parameters"] / GIB,
        "adapter_gradient_adamw_gib": 16 * old["lora_parameters"] / GIB,
        "logits_mib_per_token": [n / MIB for n in old["logits_bytes_per_token_range"]],
        "activation_mib_per_token": {mode: n / MIB for mode, n in old["activation_bytes_per_token"].items()},
    }
    return values


def markdown(values):
    lines = ["| Term or phase | Prediction |", "| --- | --- |"]
    for key, val in values.items():
        if "_gib" not in key and "_mib" not in key:
            continue
        unit = "GiB" if "_gib" in key else "MiB/token"
        if isinstance(val, dict):
            reading = ", ".join(f"{k}: {x:.3f}" for k, x in val.items())
        elif isinstance(val, list):
            reading = "–".join(f"{x:.3f}" for x in val)
        else:
            reading = f"{val:.3f}"
        lines.append(f"| {key.replace('_gib', '').replace('_mib', '').replace('_', ' ')} | {reading} {unit} |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--length", type=int, required=True)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--historical", action="store_true", help="Reproduce the original pre-amendment estimate.")
    args = parser.parse_args()
    predictor = historical_predict if args.historical else predict
    result = predictor(json.loads(args.config.read_bytes()), args.batch, args.length, args.rank)
    print((historical_markdown if args.historical else markdown)(result))
    if args.json_out:
        args.json_out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
