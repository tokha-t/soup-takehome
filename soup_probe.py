"""Run the Soup LoRA probe from the generated chat template."""

from collections import deque
from datetime import datetime, timezone
from importlib import metadata
import json
import os
from pathlib import Path
import queue
import re
import shlex
import shutil
import signal
import subprocess
import sys
from threading import Thread
from time import monotonic, time


MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
# Same text as Qwen's shipped template for explicit system/user/assistant
# messages without tool_calls. The generation block defines the loss mask.
MASKED_TEMPLATE = (
    "{% for message in messages %}"
    "{{ '<|im_start|>' + message.role + '\\n' }}"
    "{% if message.role == 'assistant' %}"
    "{% generation %}{{ message.content + '<|im_end|>' }}{% endgeneration %}"
    "{% else %}{{ message.content + '<|im_end|>' }}{% endif %}"
    "{{ '\\n' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}"
)


class SoupProbeError(RuntimeError):
    """Soup did not produce a valid one-epoch adapter."""


class SoupProbeTimeout(SoupProbeError):
    """A Soup stage ran out of time."""


def _run(command, *, cwd, log, deadline, mirror=None, startup_deadline=None):
    command = [str(part) for part in command]
    display = shlex.join(command)
    log(f"Running {display}")
    if monotonic() >= deadline:
        raise SoupProbeTimeout(f"No time left to run {display}")

    env = os.environ.copy()
    env.update(
        PYTHONUNBUFFERED="1",
        PYTHONHASHSEED="42",
        CUDA_VISIBLE_DEVICES="0",
        NO_COLOR="1",
        HF_HUB_DISABLE_TELEMETRY="1",
        SOUP_TELEMETRY="0",
    )
    with subprocess.Popen(
        command,
        cwd=cwd,
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        start_new_session=True,
    ) as process:
        output = queue.Queue()

        def read_output():
            try:
                for line in process.stdout:
                    output.put((monotonic(), line.removesuffix("\n")))
            finally:
                output.put(None)

        reader = Thread(target=read_output, daemon=True)
        reader.start()
        training_seen = False
        trainer_ready = False
        finished = False
        tail = deque(maxlen=12)

        def write_line(line):
            log(line)
            if mirror is not None:
                timestamp = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
                mirror.write(f"{timestamp} {line}\n")
                mirror.flush()
            if line.strip():
                tail.append(line)

        try:
            while True:
                stop_at = deadline
                if startup_deadline is not None and not training_seen:
                    stop_at = min(stop_at, startup_deadline)
                try:
                    if monotonic() >= stop_at and output.empty():
                        raise queue.Empty
                    item = output.get(timeout=max(0, stop_at - monotonic()))
                except queue.Empty:
                    if startup_deadline is not None and not training_seen and stop_at == startup_deadline:
                        raise SoupProbeTimeout(
                            f"Soup did not reach the optimizer loop within the startup budget: {display}"
                        ) from None
                    raise SoupProbeTimeout(f"Deadline expired running {display}") from None
                if item is None:
                    break
                received_at, line = item
                write_line(line)
                if received_at > stop_at:
                    if stop_at == startup_deadline and not training_seen:
                        raise SoupProbeTimeout(
                            f"Soup did not reach the optimizer loop within the startup budget: {display}"
                        )
                    raise SoupProbeTimeout(f"Deadline expired running {display}")
                if "Training started!" in line:
                    trainer_ready = True
                if (
                    startup_deadline is not None
                    and not training_seen
                    and trainer_ready
                    and re.search(r"\b\d+/63\s*\[", line)
                ):
                    # tqdm first appears after HF Trainer enters the loop.
                    training_seen = True
                    log("Soup entered the optimizer loop (63 steps expected).")
                if "Training Complete!" in line:
                    finished = True

            status = process.wait(timeout=max(0, deadline - monotonic()))
            if status:
                recent_output = "\n".join(tail)
                raise SoupProbeError(f"{display} exited {status}. Last output:\n{recent_output}")
            return training_seen, finished
        finally:
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
            # Keep late process output, including lines written before a timeout.
            reader.join(timeout=2)
            while not output.empty():
                item = output.get_nowait()
                if item is not None:
                    write_line(item[1])


def setup_soup(output_dir, log, *, timeout_seconds):
    """Install Soup into an isolated env reusing Kaggle's installed torch."""
    deadline = monotonic() + timeout_seconds
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    env_dir = output_dir / ".soup_env"
    # Keep the environment out of /kaggle/working and the notebook output archive.
    if Path("/kaggle/working").is_dir():
        env_dir = Path("/tmp/dataset_bakeoff_soup_env")
    uv = shutil.which("uv")
    create_environment = (
        [uv, "venv", "--seed", "--system-site-packages", "--python", sys.executable, env_dir]
        if uv else [sys.executable, "-m", "venv", "--system-site-packages", env_dir]
    )
    _run(
        create_environment,
        cwd=output_dir,
        log=log,
        deadline=deadline,
    )
    python = env_dir / "bin" / "python"
    soup = env_dir / "bin" / "soup"
    constraints = output_dir / "soup_constraints.txt"
    # Reuse the existing CUDA build or fail fast on Soup's incompatible torch floor.
    constraints.write_text(f"torch=={metadata.version('torch')}\n", encoding="utf-8")
    _run(
        [python, "-m", "pip", "install", "-c", constraints, "soup-cli[train]==0.77.2"],
        cwd=output_dir,
        log=log,
        deadline=deadline,
    )
    _run([soup, "--version"], cwd=output_dir, log=log, deadline=deadline)
    _run(
        [python, "-c", (
            "import sys, torch, transformers, peft, trl;"
            "print(sys.version);"
            "print('torch', torch.__version__, 'transformers', transformers.__version__,"
            "'peft', peft.__version__, 'trl', trl.__version__)"
        )],
        cwd=output_dir,
        log=log,
        deadline=deadline,
    )
    return soup


def _verify_mask(prepared, tokenizer, log):
    """Check Qwen rendering and the full answer/EOT mask before Soup launches."""
    rendered_overlength = 0
    truncated_rows = 0
    supervised_tokens_dropped = 0
    for index, example in enumerate(prepared.train, start=1):
        messages = list(example.messages)
        original = tokenizer.apply_chat_template(messages, tokenize=False)
        masked = tokenizer.apply_chat_template(
            messages, chat_template=MASKED_TEMPLATE, tokenize=False
        )
        if masked != original:
            raise SoupProbeError(f"Row {index}: masked Qwen template changed the rendered text.")
        encoded = tokenizer.apply_chat_template(
            messages,
            chat_template=MASKED_TEMPLATE,
            tokenize=True,
            return_dict=True,
            return_assistant_tokens_mask=True,
        )
        target_ids = [
            token for token, keep in zip(encoded["input_ids"], encoded["assistant_masks"])
            if keep
        ]
        if tokenizer.decode(
            target_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
        ) != messages[-1]["content"] + "<|im_end|>":
            raise SoupProbeError(f"Row {index}: Soup's answer/EOT loss mask did not match.")
        labels = [
            token if keep else -100
            for token, keep in zip(encoded["input_ids"], encoded["assistant_masks"])
        ]
        if hasattr(example, "labels") and list(example.labels) != labels:
            raise SoupProbeError(f"Row {index}: Soup's mask differs from the evaluation loss mask.")
        if not any(target != -100 for target in labels[1:512]):
            raise SoupProbeError(
                f"Row {index}: Soup's right truncation at 512 tokens would remove all answer/EOT targets."
            )
        rendered_overlength += len(labels) > 512
        dropped = sum(target != -100 for target in labels[512:])
        truncated_rows += dropped > 0
        supervised_tokens_dropped += dropped
    log(f"Verified equivalent Qwen rendering and answer/EOT masks for {len(prepared.train)} rows.")
    log(
        f"Soup training uses {len(prepared.train)} rows. Rendered rows over 512: "
        f"{rendered_overlength}. Rows cropped before the last target: {truncated_rows}. "
        f"Supervised tokens lost: {supervised_tokens_dropped}."
    )
    return {
        "rendered_rows_over_max_length": rendered_overlength,
        "truncated_rows": truncated_rows,
        "supervised_tokens_dropped": supervised_tokens_dropped,
    }


def train_soup(
    prepared, output_dir, log, *, soup, tokenizer, timeout_seconds, startup_timeout_seconds=None,
    expected_revision=None,
):
    """Train one candidate and return the directory containing its PEFT adapter."""
    import yaml

    start = monotonic()
    deadline = start + timeout_seconds
    startup_deadline = None if startup_timeout_seconds is None else start + startup_timeout_seconds
    setup_deadline = deadline if startup_deadline is None else min(deadline, startup_deadline)
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "soup_train.jsonl"
    with train_path.open("w", encoding="utf-8") as handle:
        for example in prepared.train:
            handle.write(json.dumps({"messages": list(example.messages)}, ensure_ascii=False) + "\n")

    _run(
        [soup, "init", "--template", "chat"],
        cwd=output_dir,
        log=log,
        deadline=setup_deadline,
    )
    config_path = output_dir / "soup.yaml"
    generated = config_path.read_text(encoding="utf-8")
    (output_dir / "soup_template.yaml").write_text(generated, encoding="utf-8")
    config = yaml.safe_load(generated)
    config["base"] = MODEL
    config["backend"] = "transformers"
    config["data"].update(
        train=str(train_path),
        format="chatml",
        val_split=0.0,
        max_length=512,
        train_on_responses_only=True,
        chat_template=MASKED_TEMPLATE,
    )
    config["training"].update(
        epochs=1,
        lr=2e-4,
        batch_size=8,
        gradient_accumulation_steps=1,
        quantization="none",
        train_on_eot=True,
        seed=42,
        data_seed=42,
        optimizer="adamw_torch",
        scheduler="constant",
        warmup_ratio=0.0,
        weight_decay=0.0,
        max_grad_norm=1.0,
        rewind_log=False,
    )
    config["training"]["lora"].update(r=16, alpha=32, target_modules="all-linear", dropout=0.0)
    adapter = output_dir / "soup_adapter"
    config["output"] = str(adapter)
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    with (output_dir / "soup_data_doctor.log").open("w", encoding="utf-8") as doctor_log:
        try:
            _run(
                [
                    soup, "data", "doctor", train_path, "--model", MODEL,
                    "--max-length", "512", "--train-on-eot",
                ],
                cwd=output_dir,
                log=log,
                deadline=setup_deadline,
                mirror=doctor_log,
            )
        except SoupProbeTimeout:
            raise
        except SoupProbeError as error:
            log(f"Soup data doctor returned a failure: {error}. Checking the configured loss mask.")
    truncation = _verify_mask(prepared, tokenizer, log)
    if monotonic() >= setup_deadline:
        raise SoupProbeTimeout("Soup startup deadline expired before launching soup train --yes.")

    start_epoch = time()
    command_start = monotonic()
    try:
        training_seen, finished = _run(
            [soup, "train", "--yes"],
            cwd=output_dir,
            log=log,
            deadline=deadline,
            startup_deadline=startup_deadline,
        )
    finally:
        window = {
            "start_epoch": start_epoch,
            "end_epoch": time(),
            "wall_seconds": monotonic() - command_start,
            **truncation,
        }
        (output_dir / "soup_training_window.json").write_text(
            json.dumps(window, indent=2) + "\n", encoding="utf-8"
        )
    if not finished:
        raise SoupProbeError("Soup exited without confirming the training epoch completed.")
    if startup_deadline is not None and not training_seen:
        raise SoupProbeError("Soup did not show optimizer-loop startup within the startup budget.")
    if not (adapter / "adapter_config.json").is_file() or not any(
        (adapter / filename).is_file() for filename in ("adapter_model.safetensors", "adapter_model.bin")
    ):
        raise SoupProbeError(f"Soup did not save PEFT adapter weights at {adapter}.")
    if expected_revision is not None:
        provenance_path = adapter / "soup_run.json"
        if not provenance_path.is_file():
            raise SoupProbeError("Soup did not emit the base-model revision in soup_run.json.")
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        actual = provenance.get("base_revision")
        if actual != expected_revision:
            raise SoupProbeError(
                f"Soup trained on base revision {actual!r}, expected {expected_revision!r}."
            )
        log(f"Soup used the same pinned Qwen revision as evaluation: {actual}.")
    return adapter
