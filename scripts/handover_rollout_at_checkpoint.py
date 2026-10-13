"""Switch an existing rollout job only after its atomic epoch checkpoint.

The exact old torchrun PID, workspace and output directory are validated before
any signal is sent. Optimizer/scheduler progress resumes from the new checkpoint;
TensorBoard stays in the same run directory. Intended for Linux training hosts.
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-pid", type=int, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    args = parser.parse_args()
    workspace = Path.cwd().resolve()
    process = Path(f"/proc/{args.old_pid}")
    command = (process / "cmdline").read_bytes().split(b"\0")
    decoded = [part.decode() for part in command if part]
    if (
        not any(part.endswith("torchrun") for part in decoded)
        or "scripts/train_absolute_hidden_rollout.py" not in decoded
        or "--output" not in decoded
        or Path(decoded[decoded.index("--output") + 1]).resolve() != args.output.resolve()
        or (process / "cwd").resolve() != workspace
    ):
        raise RuntimeError("old PID does not match the specified rollout job")
    start_time = (process / "stat").read_text().rsplit(")", 1)[1].split()[19]
    print(f"Waiting for {args.checkpoint}; preserving PID {args.old_pid}", flush=True)
    while not args.checkpoint.exists():
        if not process.exists():
            raise RuntimeError("old training exited before writing the requested checkpoint")
        time.sleep(0.1)
    if (process / "stat").read_text().rsplit(")", 1)[1].split()[19] != start_time:
        raise RuntimeError("old PID was reused; refusing to signal another process")
    # The trainer publishes this file with os.replace only after torch.save
    # completes, while the other rank waits at the end-of-epoch barrier.
    print(f"Checkpoint published: {args.checkpoint}; stopping old torchrun", flush=True)
    os.kill(args.old_pid, signal.SIGTERM)
    deadline = time.monotonic() + 60
    while process.exists():
        if time.monotonic() >= deadline:
            raise RuntimeError("old torchrun did not exit; refusing to launch a second job")
        time.sleep(0.1)
    log = args.output / "train.log"
    archive = args.output / f"train-before-compile-{time.time_ns()}.log"
    if log.exists():
        log.rename(archive)
    env = dict(os.environ)
    env["TORCHINDUCTOR_COMPILE_THREADS"] = "8"
    with log.open("wb") as stream:
        child = subprocess.Popen(
            [
                str(workspace / ".venv/bin/torchrun"), "--standalone", "--nproc-per-node=2",
                "scripts/train_absolute_hidden_rollout.py",
                "--data", str(args.data),
                "--checkpoint", str(args.source_checkpoint),
                "--resume", str(args.checkpoint),
                "--output", str(args.output),
                "--compile-rollout", "on",
            ],
            cwd=workspace, env=env, stdin=subprocess.DEVNULL,
            stdout=stream, stderr=subprocess.STDOUT, start_new_session=True,
        )
    print(f"Optimized torchrun PID={child.pid}; log={log}; previous_log={archive}", flush=True)


if __name__ == "__main__":
    main()
