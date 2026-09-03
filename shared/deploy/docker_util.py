"""Docker / Compose helpers."""
from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path
from typing import List, Optional, Sequence

logger = logging.getLogger(__name__)


def docker_available() -> bool:
    return shutil.which("docker") is not None


def compose_cmd() -> List[str]:
    """Return base argv for docker compose (plugin or legacy)."""
    if not docker_available():
        raise RuntimeError(
            "Docker is not installed or not on PATH. Install Docker or set "
            "AUTO_DEPLOY_MODEL=false and provide BACKEND_URLS / NIM_API_URL / VLLM_API_URL."
        )
    try:
        r = subprocess.run(
            ["docker", "compose", "version"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if r.returncode == 0:
            return ["docker", "compose"]
    except Exception:
        pass
    if shutil.which("docker-compose"):
        return ["docker-compose"]
    raise RuntimeError(
        "Docker Compose is not available (tried `docker compose` and docker-compose)."
    )


def run_cmd(
    argv: Sequence[str],
    *,
    timeout: Optional[float] = None,
    check: bool = True,
    cwd: Optional[str] = None,
) -> subprocess.CompletedProcess:
    logger.info("Running: %s", " ".join(argv))
    try:
        proc = subprocess.run(
            list(argv),
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=cwd,
        )
    except subprocess.TimeoutExpired as exc:
        msg = "Command timed out: " + " ".join(argv)
        raise RuntimeError(msg) from exc
    if check and proc.returncode != 0:
        parts = [
            "Command failed (%s): %s" % (proc.returncode, " ".join(argv)),
            "stdout: %s" % ((proc.stdout or "")[-2000:]),
            "stderr: %s" % ((proc.stderr or "")[-2000:]),
        ]
        raise RuntimeError(chr(10).join(parts))
    return proc


def compose_up(project: str, compose_file: Path, timeout: float = 600) -> None:
    cmd = compose_cmd() + [
        "-p",
        project,
        "-f",
        str(compose_file),
        "up",
        "-d",
        "--remove-orphans",
    ]
    run_cmd(cmd, timeout=timeout, check=True)


def compose_down(
    project: str,
    compose_file: Path,
    *,
    timeout: float = 60,
    remove_volumes: bool = False,
) -> None:
    cmd = compose_cmd() + [
        "-p",
        project,
        "-f",
        str(compose_file),
        "down",
        "--remove-orphans",
    ]
    if remove_volumes:
        cmd.append("-v")
    try:
        run_cmd(cmd, timeout=timeout, check=False)
    except Exception as exc:
        logger.warning("compose down error: %s", exc)


def compose_ps(project: str, compose_file: Path) -> str:
    cmd = compose_cmd() + ["-p", project, "-f", str(compose_file), "ps"]
    proc = run_cmd(cmd, timeout=60, check=False)
    return (proc.stdout or "") + (proc.stderr or "")

