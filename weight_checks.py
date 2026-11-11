"""Tensor-level merge and NF4 checks, with a rank-sized SVD."""
from contextlib import ExitStack
import json
from pathlib import Path

import torch
from safetensors import safe_open


class Checkpoint:
    def __init__(self, directory, stack):
        directory = Path(directory)
        files = sorted(directory.glob("model*.safetensors"))
        if not files:
            raise FileNotFoundError(f"No model safetensors in {directory}")
        self.tensors = {}
        for path in files:
            handle = stack.enter_context(safe_open(path, framework="pt", device="cpu"))
            for name in handle.keys():
                if name in self.tensors:
                    raise ValueError(f"Duplicate tensor {name}")
                self.tensors[name] = handle

    def get(self, name):
        return self.tensors[name].get_tensor(name)


def adapter_weights(directory, stack):
    directory = Path(directory)
    config = json.loads((directory / "adapter_config.json").read_bytes())
    weights = stack.enter_context(safe_open(directory / "adapter_model.safetensors", framework="pt", device="cpu"))
    modules = {}
    for key in weights.keys():
        if not key.endswith(".lora_A.weight"):
            continue
        name = key.removeprefix("base_model.model.").removesuffix(".lora_A.weight")
        modules[name] = (key, key.replace(".lora_A.", ".lora_B."))
    scale = config["lora_alpha"] / config["r"]
    return weights, modules, scale


def merge_identity(base, merged, *, count_bf16_cast=False):
    """Compare weights in their saved merge dtype without loading a full model."""
    with ExitStack() as stack:
        before, after = Checkpoint(base, stack), Checkpoint(merged, stack)
        changed, missing = [], []
        cast_changed = cast_total = 0
        for name in sorted(before.tensors.keys() | after.tensors.keys()):
            if name not in before.tensors or name not in after.tensors:
                missing.append(name)
                continue
            source = before.tensors[name].get_slice(name)
            target = after.tensors[name].get_slice(name)
            if source.get_shape() != target.get_shape():
                missing.append(name)
                continue
            shape = source.get_shape()
            chunks = [(slice(start, start + 128),) for start in range(0, shape[0], 128)] if shape else [()]
            differs = False
            for index in chunks:
                raw = source[index] if index else before.get(name)
                tuned = target[index] if index else after.get(name)
                expected = raw.to(tuned.dtype)
                differs |= not torch.equal(tuned, expected)
                if count_bf16_cast and raw.dtype == torch.bfloat16:
                    cast_changed += torch.count_nonzero(raw.float() != raw.half().float()).item()
                    cast_total += raw.numel()
            if differs:
                changed.append(name)
        return {
            "changed_count": len(changed), "changed_tensors": changed, "missing_or_shape_changed": missing,
            "base_bf16_to_fp16_changed_elements": cast_changed if count_bf16_cast else None,
            "base_bf16_elements_compared": cast_total if count_bf16_cast else None,
        }


@torch.inference_mode()
def merge_checks(base, merged, adapter):
    with ExitStack() as stack:
        before, after = Checkpoint(base, stack), Checkpoint(merged, stack)
        weights, modules, scale = adapter_weights(adapter, stack)
        details, unexpected, changed = {}, [], []
        for name in after.tensors.keys() | before.tensors.keys():
            if name not in before.tensors or name not in after.tensors:
                unexpected.append(name)
                continue
            target = after.get(name)
            # Compare against the exact base precision used for this merge.
            source = before.get(name).to(target.dtype)
            differs = not torch.equal(target, source)
            if differs:
                changed.append(name)
            module = name.removesuffix(".weight")
            if module not in modules or not name.endswith(".weight"):
                if differs:
                    unexpected.append(name)
                continue
            a, b = (weights.get_tensor(key).to("cuda", torch.float32) for key in modules[module])
            expected = (b @ a) * scale
            actual = target.to("cuda", torch.float32) - source.to("cuda", torch.float32)
            norm = torch.linalg.vector_norm(expected).item()
            edited = expected != 0
            details[module] = {
                "expected_delta_norm": norm, "actual_delta_norm": torch.linalg.vector_norm(actual).item(),
                "relative_error": torch.linalg.vector_norm(actual - expected).item() / norm if norm else None,
                "expected_nonzero_elements": edited.sum().item(),
                "rounded_to_zero_elements": (edited & (actual == 0)).sum().item(),
                "changed": differs,
            }
        denominator = sum(item["expected_nonzero_elements"] for item in details.values())
        numerator = sum(item["rounded_to_zero_elements"] for item in details.values())
        return {
            "changed_count": len(changed), "changed_tensors": sorted(changed),
            "unexpected_changed_or_missing": sorted(unexpected), "modules": details,
            "expected_module_count": len(modules),
            "expected_delta_norm": sum(item["expected_delta_norm"] ** 2 for item in details.values()) ** .5,
            "rounded_to_zero_share": numerator / denominator if denominator else None,
        }


@torch.inference_mode()
def nf4_checks(base4, merged4, adapter):
    from bitsandbytes.functional import dequantize_4bit

    baseline = dict(base4.named_modules())
    tuned = dict(merged4.named_modules())
    details = {}
    with ExitStack() as stack:
        weights, modules, scale = adapter_weights(adapter, stack)
        for name, keys in modules.items():
            weight_base = baseline[name].weight
            weight_tuned = tuned[name].weight
            left = dequantize_4bit(weight_base.data, weight_base.quant_state).float()
            right = dequantize_4bit(weight_tuned.data, weight_tuned.quant_state).float()
            diff = right - left
            xor = torch.bitwise_xor(weight_base.data.flatten().to(torch.uint8), weight_tuned.data.flatten().to(torch.uint8))
            flips = ((xor & 15) != 0).sum().item() + ((xor >> 4) != 0).sum().item()
            a, b = (weights.get_tensor(key).to("cuda", torch.float32) for key in keys)
            expected = scale * (b @ a)
            deq_norm = torch.linalg.vector_norm(diff).item()
            expected_norm = torch.linalg.vector_norm(expected).item()
            if expected_norm and deq_norm:
                cosine = torch.dot(diff.flatten(), expected.flatten()).item() / (deq_norm * expected_norm)
            else:
                cosine = None
            if expected_norm:
                qb, rb = torch.linalg.qr(b, mode="reduced")
                qa, ra = torch.linalg.qr(a.T, mode="reduced")
                # QR reduces the full SVD to an equivalent r × r SVD.
                u, singular, vh = torch.linalg.svd(scale * (rb @ ra.T), full_matrices=False)
                u, v = qb @ u, qa @ vh.T
                projected = torch.linalg.vector_norm(u.T @ diff @ v).item()
                subspace_retention = projected / torch.linalg.vector_norm(singular).item()
                outside = max(0.0, deq_norm ** 2 - projected ** 2)
            else:
                subspace_retention, outside = None, deq_norm ** 2
            details[name] = {
                "code_flip_rate": flips / diff.numel(), "code_flips": flips, "code_count": diff.numel(),
                "cosine_delta": cosine, "subspace_retention": subspace_retention,
                "outside_subspace_energy": outside, "dequantized_delta_energy": deq_norm ** 2,
            }
    return {"modules": details, "total_code_flip_rate": sum(x["code_flips"] for x in details.values()) / sum(x["code_count"] for x in details.values())}
