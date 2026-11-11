"""Run an archived stage with a fresh, bounded reproduction time budget.

This changes orchestration deadlines only. Run it in a disposable copy because
the original stages write result files and logs relative to their source tree.
"""
import argparse
from contextlib import nullcontext
from datetime import datetime, timezone
import math
import os
from pathlib import Path
import subprocess
import sys
import time


def deadline_from_environment(name, original):
    """Keep the historical deadline unless the replay launcher overrides it."""
    value = os.environ.get(name)
    if value is None:
        return original
    try:
        epoch = float(value)
    except ValueError as error:
        raise ValueError(f"{name} must be a finite positive Unix timestamp") from error
    if not math.isfinite(epoch) or epoch <= 0:
        raise ValueError(f"{name} must be a finite positive Unix timestamp")
    return epoch


def replay_environment(minutes, optional_minutes=None, *, now=None):
    """Return deadlines shared by the stage and every child process."""
    optional_minutes = minutes if optional_minutes is None else optional_minutes
    if not all(math.isfinite(n) and n > 0 for n in (minutes, optional_minutes)):
        raise ValueError("Time budgets must be finite and greater than zero")
    if optional_minutes > minutes:
        raise ValueError("The optional budget cannot exceed the overall budget")
    started = time.time() if now is None else now
    deadline = started + minutes * 60
    environment = os.environ.copy()
    environment["SOUP_REPLAY_DEADLINE_EPOCH"] = str(deadline)
    environment["SOUP_REPLAY_OPTIONAL_EPOCH"] = str(started + optional_minutes * 60)
    return environment, deadline


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--minutes", type=float, default=120,
                        help="Overall stage budget from now (default: 120 minutes)")
    parser.add_argument("--optional-minutes", type=float,
                        help="Optional-work cutoff from now (default: overall budget)")
    parser.add_argument("--gpu-monitor", action="store_true",
                        help="Sample nvidia-smi for direct training or verification runs")
    parser.add_argument("script", help="Stage script in the disposable working copy")
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    try:
        environment, deadline = replay_environment(args.minutes, args.optional_minutes)
    except ValueError as error:
        parser.error(str(error))
    script = Path(args.script).resolve()
    if not script.is_file():
        parser.error(f"Stage script does not exist: {script}")
    iso = datetime.fromtimestamp(deadline, timezone.utc).isoformat()
    arguments = [iso if arg == "@deadline" else arg for arg in args.arguments]
    print(f"REPLAY: {script.name}; overall deadline {iso}; write outputs only in a disposable copy", flush=True)
    if args.gpu_monitor:
        from instruments import gpu_monitor
        monitor = gpu_monitor()
    else:
        monitor = nullcontext()
    try:
        with monitor:
            result = subprocess.run([sys.executable, str(script), *arguments], env=environment,
                                    timeout=max(0.001, deadline - time.time()) + 15)
    except subprocess.TimeoutExpired:
        print("REPLAY: overall time budget exhausted", file=sys.stderr)
        return 124
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
