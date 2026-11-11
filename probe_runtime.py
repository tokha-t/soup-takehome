"""One-device fp16 inference and the answer-only PEFT probe."""

import math
import random
import time
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, GenerationConfig


SEED = 42
TRAIN_BATCH_SIZE = 8
MAX_TRAIN_LENGTH = 512
GENERATION_BATCH_SIZE = 16


def seed_everything():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


def load_model(model_id, revision=None):
    if torch.cuda.device_count() != 1 or "T4" not in torch.cuda.get_device_name(0):
        raise RuntimeError("The probe requires exactly one visible NVIDIA T4.")
    seed_everything()
    return AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=revision,
        dtype=torch.float16,
        attn_implementation="sdpa",
        device_map={"": 0},
    ).eval()


def _left_pad(sequences, pad_id, device):
    width = max(map(len, sequences))
    padding = [width - len(sequence) for sequence in sequences]
    input_ids = [[pad_id] * extra + list(sequence) for extra, sequence in zip(padding, sequences)]
    attention = [[0] * extra + [1] * len(sequence) for extra, sequence in zip(padding, sequences)]
    return (
        torch.tensor(input_ids, device=device),
        torch.tensor(attention, device=device),
    )


def _answer_sequence(row, max_length=None, *, exclude_stop_id=None):
    # No position after the last supervised target can affect the causal loss.
    labels = row.labels[:max_length]
    last_target = next((index for index, label in enumerate(reversed(labels)) if label != -100), None)
    if last_target is None or last_target == len(labels) - 1:
        raise ValueError(f"An example has no predictable answer tokens at max_length={max_length}.")
    end = len(labels) - last_target
    if exclude_stop_id is not None and labels[end - 1] == exclude_stop_id:
        end -= 1
        if not any(label != -100 for label in labels[1:end]):
            raise ValueError("An example has no answer content before its stop token.")
    return row.input_ids[:end], [-100, *labels[1:end]]


def _answer_nll(model, tokenizer, sequences):
    ids, labels = zip(*sequences)
    input_ids, attention = _left_pad(ids, tokenizer.pad_token_id, model.device)
    width = input_ids.shape[1]
    padded_labels = [[-100] * (width - len(row)) + row for row in labels]
    answer_width = max(len(row) - next(i for i, target in enumerate(row) if target != -100) for row in labels)
    targets = torch.tensor([row[-answer_width:] for row in padded_labels], device=model.device)
    # Feeding through the penultimate token aligns the last answer_width logits
    # with the last answer_width labels. Qwen can project only those positions.
    position_ids = (attention.cumsum(dim=1) - 1).clamp_min(0)
    logits = model(
        input_ids=input_ids[:, :-1],
        attention_mask=attention[:, :-1],
        position_ids=position_ids[:, :-1],
        use_cache=False,
        logits_to_keep=answer_width,
    ).logits
    nll = F.cross_entropy(
        logits.flatten(0, 1).float(),
        targets.flatten(),
        reduction="sum",
        ignore_index=-100,
    )
    token_count = sum(target != -100 for row in labels for target in row)
    return nll, token_count


@torch.inference_mode()
def evaluate(model, tokenizer, rows, score, max_new_tokens, log, check_deadline=None):
    started = time.monotonic()
    model.eval()
    eos = tokenizer.eos_token_id
    generation = GenerationConfig(
        do_sample=False,
        repetition_penalty=1.0,
        max_new_tokens=max_new_tokens,
        eos_token_id=eos,
        pad_token_id=tokenizer.pad_token_id,
        use_cache=True,
    )
    judged = [None] * len(rows)
    stopped = 0
    order = sorted(range(len(rows)), key=lambda index: len(rows[index].prompt_ids))
    for offset in range(0, len(order), GENERATION_BATCH_SIZE):
        if check_deadline is not None:
            check_deadline()
        indices = order[offset:offset + GENERATION_BATCH_SIZE]
        input_ids, attention = _left_pad(
            [rows[index].prompt_ids for index in indices], tokenizer.pad_token_id, model.device
        )
        generated = model.generate(input_ids=input_ids, attention_mask=attention, generation_config=generation,
                                   use_model_defaults=False, do_sample=False, num_beams=1)
        new_tokens = generated[:, input_ids.shape[1]:].tolist()
        for index, tokens in zip(indices, new_tokens):
            if eos in tokens:
                stop_index = tokens.index(eos)
                stopped += stop_index + 1 < max_new_tokens
                tokens = tokens[:stop_index]
            output = tokenizer.decode(tokens, skip_special_tokens=False, clean_up_tokenization_spaces=False)
            judged[index] = score(output, rows[index].answer)
        log(f"Greedy generation completed {offset + len(indices)}/{len(rows)} held-out rows.")
    generation_seconds = time.monotonic() - started

    total_nll = 0.0
    answer_tokens = 0
    sequences = sorted(
        (_answer_sequence(row, exclude_stop_id=eos) for row in rows),
        key=lambda sequence: len(sequence[0]),
    )
    for offset in range(0, len(sequences), TRAIN_BATCH_SIZE):
        if check_deadline is not None:
            check_deadline()
        nll, count = _answer_nll(model, tokenizer, sequences[offset:offset + TRAIN_BATCH_SIZE])
        total_nll += nll.item()
        answer_tokens += count
    if not math.isfinite(total_nll):
        raise FloatingPointError("Nonfinite teacher-forced answer loss.")

    counts = Counter()
    for judgment in judged:
        counts.update(key for key, value in judgment.items() if value)
    wall_seconds = time.monotonic() - started
    return {
        "score": counts["correct"] / len(rows),
        "stop_rate": stopped / len(rows),
        "answer_loss": total_nll / answer_tokens,
        "answer_tokens": answer_tokens,
        "wall_seconds": wall_seconds,
        "generation_seconds": generation_seconds,
        "teacher_forced_seconds": wall_seconds - generation_seconds,
        "scorer_rates": {
            key: counts[key] / len(rows)
            for key in sorted(judged[0]) if key != "correct"
        },
        "correct": [judgment["correct"] for judgment in judged],
    }


def paired_flips(base, adapted):
    pairs = list(zip(base["correct"], adapted["correct"], strict=True))
    return {
        "wrong_to_right": sum(not before and after for before, after in pairs),
        "right_to_wrong": sum(before and not after for before, after in pairs),
    }


def train_peft(model, tokenizer, rows, log, check_deadline=None):
    started = time.monotonic()
    seed_everything()
    # Match Soup's right cutoff while keeping the sampled training split fixed.
    uncropped = [_answer_sequence(row) for row in rows]
    rendered_overlength = sum(len(row.input_ids) > MAX_TRAIN_LENGTH for row in rows)
    truncated_rows = sum(len(ids) > MAX_TRAIN_LENGTH for ids, _ in uncropped)
    without_targets = sum(
        not any(label != -100 for label in row.labels[1:MAX_TRAIN_LENGTH])
        for row in rows
    )
    if without_targets:
        raise ValueError(
            f"Right truncation to 512 removes all supervised targets from {without_targets} of "
            f"{len(rows)} sampled training rows ({rendered_overlength} rendered rows over 512)."
        )
    cropped = [_answer_sequence(row, MAX_TRAIN_LENGTH) for row in rows]
    supervised_tokens_dropped = sum(
        sum(label != -100 for label in before_labels) - sum(label != -100 for label in after_labels)
        for (_, before_labels), (_, after_labels) in zip(uncropped, cropped, strict=True)
    )
    log(
        f"PEFT training uses {len(rows)} rows. Rendered rows over 512: {rendered_overlength}. "
        f"Rows cropped before the last target: {truncated_rows}. Supervised tokens lost: {supervised_tokens_dropped}."
    )
    adapted = get_peft_model(
        model,
        LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            target_modules="all-linear",
            r=16,
            lora_alpha=32,
            lora_dropout=0.0,
            bias="none",
        ),
    ).train()
    adapted.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    log("Enabled non-reentrant gradient checkpointing for the fixed batch size of 8.")
    try:
        params = [param for param in adapted.parameters() if param.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=2e-4, weight_decay=0.0)
        scaler = torch.amp.GradScaler("cuda", init_scale=256.0)
        order = list(range(len(rows)))
        random.Random(SEED).shuffle(order)
        skipped_steps = 0
        optimizer_steps = 0
        for offset in range(0, len(order), TRAIN_BATCH_SIZE):
            indices = order[offset:offset + TRAIN_BATCH_SIZE]
            batch = [cropped[index] for index in indices]
            for _ in range(8):
                if check_deadline is not None:
                    check_deadline()
                optimizer.zero_grad(set_to_none=True)
                nll, tokens = _answer_nll(adapted, tokenizer, batch)
                loss = nll / tokens
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
                previous_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                if scaler.get_scale() >= previous_scale:
                    break
                skipped_steps += 1
                log(f"PEFT retried optimizer batch {optimizer_steps + 1} after fp16 gradient overflow.")
            else:
                raise FloatingPointError("Repeated fp16 gradient overflow in the same optimizer batch.")
            optimizer_steps += 1
            if optimizer_steps % 10 == 0 or offset + len(indices) == len(rows):
                log(f"PEFT trained {offset + len(indices)}/{len(rows)} rows. Batch answer loss {loss.item():.4f}.")
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        adapted.gradient_checkpointing_disable()
        return adapted.eval(), {
            "wall_seconds": time.monotonic() - started,
            "optimizer_steps": optimizer_steps,
            "overflow_retries": skipped_steps,
            "rendered_rows_over_max_length": rendered_overlength,
            "truncated_rows": truncated_rows,
            "supervised_tokens_dropped": supervised_tokens_dropped,
            "trainable_parameters": sum(param.numel() for param in params),
        }
    except Exception:
        adapted.unload()
        raise
