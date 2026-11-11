"""Raw command logs, phase-local memory peaks, and the session GPU sampler."""
from contextlib import contextmanager
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
import json
import logging
import os
from pathlib import Path
import platform
import shlex
import subprocess
import threading
import time

ROOT = Path(__file__).resolve().parent
LOG = logging.getLogger("pipeline")
LOG_FORMAT = "%(asctime)s.%(msecs)03d %(levelname)s %(message)s"
DATE_FORMAT = "%Y-%m-%dT%H:%M:%S"


def setup_logging():
    logging.Formatter.converter = time.gmtime
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, datefmt=DATE_FORMAT, force=True)


def now():
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".tmp")
    pending.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    pending.replace(path)


def command(args, path, *, cwd=ROOT, timeout=None, env=None):
    args = [str(arg) for arg in args]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", buffering=1, encoding="utf-8") as log:
        def emit(line):
            print(f"{now()} INFO {line}", file=log, flush=True)
            print(f"{now()} INFO {line}", flush=True)
        emit("$ " + shlex.join(args))
        with subprocess.Popen(
            args, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", start_new_session=True,
        ) as child:
            if timeout:
                import signal
                def stop():
                    if child.poll() is None:
                        os.killpg(child.pid, signal.SIGTERM)
                alarm = threading.Timer(timeout, stop)
                alarm.start()
            try:
                for line in child.stdout:
                    emit(line.rstrip("\n"))
                status = child.wait()
            finally:
                if timeout:
                    alarm.cancel()
            emit(f"Exit status {status}")
    return status


def gpu_snapshot(name):
    path = ROOT / "logs/nvidia_smi_raw" / f"{name}.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(["nvidia-smi"], capture_output=True, check=False)
    # Keep command output byte-for-byte, including its own timestamp header.
    with path.open("ab") as stream:
        stream.write(result.stdout)
        stream.write(result.stderr)


@contextmanager
def gpu_monitor():
    path = ROOT / "logs/nvidia_smi.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path = path.with_name(f"nvidia_smi_{int(time.time())}.csv")
    with path.open("xb", buffering=0) as log:
        process = subprocess.Popen([
            "nvidia-smi", "-i", "0", "--query-gpu=timestamp,memory.used,memory.total,utilization.gpu,power.draw",
            "--format=csv", "-lms", "200",
        ], stdout=log)
        try:
            yield path
        finally:
            process.terminate()
            process.wait(timeout=10)


def sampled_peak(start, end):
    values = []
    for path in (ROOT / "logs").glob("nvidia_smi*.csv"):
        for line in path.read_text(encoding="utf-8").splitlines()[1:]:
            try:
                stamp, used, *_ = line.split(",")
                ts = datetime.strptime(stamp.strip(), "%Y/%m/%d %H:%M:%S.%f").timestamp()
                if start <= ts <= end:
                    values.append(int(used.split()[0]))
            except (ValueError, IndexError):
                continue
    return max(values, default=None)


@contextmanager
def phase(name, device=None):
    import psutil
    import torch

    cuda = torch.cuda.is_available()
    device = device or ("CUDA" if cuda else "CPU")
    if cuda:
        gpu_snapshot(name + "_start")
        torch.cuda.reset_peak_memory_stats()
    process = psutil.Process()
    done = threading.Event()
    peak_rss = [process.memory_info().rss]
    def poll():
        while not done.wait(0.2):
            peak_rss[0] = max(peak_rss[0], process.memory_info().rss)
    monitor = threading.Thread(target=poll, daemon=True)
    monitor.start()
    start = time.time()
    status = "complete"
    try:
        yield
        if cuda:
            torch.cuda.synchronize()
    except BaseException:
        status = "failed"
        raise
    finally:
        end = time.time()
        done.set()
        monitor.join()
        if cuda:
            gpu_snapshot(name + "_end")
        allocated = torch.cuda.max_memory_allocated() if cuda else None
        reserved = torch.cuda.max_memory_reserved() if cuda else None
        rss = max(peak_rss[0], process.memory_info().rss)
        nvidia = sampled_peak(start, end) if cuda else None
        gib = 2 ** 30
        record = {
            "name": name, "device": device, "state": status, "start_epoch": start, "end_epoch": end,
            "wall_seconds": end - start, "torch_peak_allocated_gib": allocated / gib if cuda else None,
            "torch_peak_reserved_gib": reserved / gib if cuda else None,
            "peak_process_rss_gib": rss / gib,
            "total_system_ram_gib": psutil.virtual_memory().total / gib,
            "nvidia_smi_peak_mib": nvidia,
            "nvidia_smi_peak_gib": nvidia / 1024 if nvidia is not None else None,
        }
        path = ROOT / "results/memory_phases.json"
        phases = json.loads(path.read_bytes()) if path.exists() else []
        phases.append(record)
        atomic_json(path, phases)
        LOG.info("Memory phase %s", json.dumps(record))


def record_environment():
    import torch
    data = {
        "Python": platform.python_version(), "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(0), "visible_devices": torch.cuda.device_count(),
        "capability": torch.cuda.get_device_capability(),
        "bf16_native": torch.cuda.is_bf16_supported(including_emulation=False),
        "driver": subprocess.check_output(["nvidia-smi", "-i", "0", "--query-gpu=driver_version", "--format=csv,noheader"], text=True).strip(),
    }
    for package in ("torch", "transformers", "peft", "bitsandbytes", "accelerate"):
        try:
            data[package] = version(package)
        except PackageNotFoundError:
            data[package] = None
    LOG.info("Environment %s", json.dumps(data))
    if data["visible_devices"] != 1 or "T4" not in data["device"] or data["bf16_native"]:
        raise RuntimeError("Expected one visible T4 with no native bf16 support.")
    atomic_json(ROOT / "results/environment.json", data)
