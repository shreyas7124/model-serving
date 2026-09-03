"""Passwordless SSH helpers for multi-node model deploy."""
from __future__ import annotations

import logging
import os
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)


def is_local_target(target: str) -> bool:
    t = (target or "").strip().lower()
    if t in ("", "local", "localhost", "127.0.0.1", "::1"):
        return True
    if "@" in t and t.split("@", 1)[1] in ("localhost", "127.0.0.1", "local"):
        return True
    return False


def normalize_ssh_target(raw: str) -> str:
    raw = (raw or "").strip()
    if not raw or is_local_target(raw):
        return "local"
    if "@" in raw:
        return raw
    user = os.getenv("DEPLOY_SSH_USER", "").strip()
    if user:
        return "%s@%s" % (user, raw)
    return raw


def advertise_host(target: str) -> str:
    t = normalize_ssh_target(target)
    if is_local_target(t):
        return os.getenv("DEPLOY_LOCAL_ADVERTISE_HOST", "127.0.0.1").strip() or "127.0.0.1"
    host = t.split("@", 1)[1] if "@" in t else t
    raw_map = os.getenv("DEPLOY_NODE_URL_HOSTS", "")
    for part in raw_map.split(","):
        part = part.strip()
        if not part or ":" not in part:
            continue
        k, v = part.split(":", 1)
        if k.strip() == host or k.strip() == t:
            return v.strip()
    return host


def _ssh_base(target: str) -> List[str]:
    opts = os.getenv(
        "DEPLOY_SSH_OPTS",
        "-o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15",
    ).split()
    cmd = ["ssh"] + opts
    key = os.getenv("DEPLOY_SSH_KEY", "").strip()
    if key:
        cmd.extend(["-i", key])
    cmd.append(target)
    return cmd


def ssh_run(target: str, remote_cmd: str, *, timeout: float = 120, check: bool = True):
    target = normalize_ssh_target(target)
    if is_local_target(target):
        argv = ["bash", "-lc", remote_cmd]
    else:
        argv = _ssh_base(target) + [remote_cmd]
    logger.info("run on %s: %s", target, remote_cmd[:180])
    try:
        proc = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("Command timed out on %s" % target) from exc
    if check and proc.returncode != 0:
        parts = [
            "Command failed on %s (rc=%s): %s" % (target, proc.returncode, remote_cmd[:160]),
            "stdout: %s" % ((proc.stdout or "")[-1500:]),
            "stderr: %s" % ((proc.stderr or "")[-1500:]),
        ]
        raise RuntimeError(chr(10).join(parts))
    return proc


def scp_put(local_path: str, target: str, remote_path: str, timeout: float = 180) -> None:
    target = normalize_ssh_target(target)
    if is_local_target(target):
        Path(remote_path).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(local_path, remote_path)
        return
    opts = os.getenv(
        "DEPLOY_SSH_OPTS",
        "-o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15",
    ).split()
    remote_dir = remote_path.rsplit("/", 1)[0]
    ssh_run(target, "mkdir -p %s" % shlex.quote(remote_dir), timeout=60)
    cmd = ["scp"] + opts
    key = os.getenv("DEPLOY_SSH_KEY", "").strip()
    if key:
        cmd.extend(["-i", key])
    cmd.extend([local_path, "%s:%s" % (target, remote_path)])
    logger.info("scp %s -> %s:%s", local_path, target, remote_path)
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if proc.returncode != 0:
        raise RuntimeError("scp failed to %s:%s: %s" % (target, remote_path, (proc.stderr or "")[-1000:]))

