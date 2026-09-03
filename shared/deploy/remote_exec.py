"""Remote compose up/down over SSH."""
from __future__ import annotations

import logging
import os
import shlex
from pathlib import Path

from .config import DeployConfig
from .scheduler import NodePlan
from .ssh_util import scp_put, ssh_run

logger = logging.getLogger(__name__)


def remote_dir(cfg: DeployConfig) -> str:
    return os.getenv("DEPLOY_REMOTE_DIR", "/tmp/model-serving-runtime").rstrip("/")


def _compose_bin_check() -> str:
    return (
        "if docker compose version >/dev/null 2>&1; then C=\"docker compose\"; "
        "elif command -v docker-compose >/dev/null 2>&1; then C=\"docker-compose\"; "
        "else echo NO_COMPOSE; exit 1; fi"
    )


def deploy_node_plan(cfg: DeployConfig, plan: NodePlan, local_compose: Path) -> str:
    rdir = remote_dir(cfg)
    safe = plan.target.replace("@", "_").replace(":", "_")
    remote_path = "%s/%s-%s.yml" % (rdir, cfg.resolved_project(), safe)
    scp_put(str(local_compose), plan.target, remote_path)
    env_lines = []
    if cfg.ngc_api_key:
        env_lines.append("NGC_API_KEY=%s" % cfg.ngc_api_key)
    if cfg.hf_token:
        env_lines.append("HUGGING_FACE_HUB_TOKEN=%s" % cfg.hf_token)
    if env_lines:
        env_local = local_compose.with_suffix(".env")
        env_local.write_text(chr(10).join(env_lines) + chr(10))
        env_remote = remote_path + ".env"
        scp_put(str(env_local), plan.target, env_remote)
        ssh_run(plan.target, "chmod 600 %s" % shlex.quote(env_remote), check=False)
    project = cfg.resolved_project()
    cmd = (
        "%s; cd %s; $C -p %s -f %s up -d --remove-orphans"
        % (
            _compose_bin_check(),
            shlex.quote(rdir),
            shlex.quote(project),
            shlex.quote(remote_path),
        )
    )
    timeout = float(cfg.wait_seconds) + 180
    ssh_run(plan.target, cmd, timeout=timeout)
    return remote_path


def teardown_node_plan(cfg: DeployConfig, plan: NodePlan, remote_compose_path: str) -> None:
    project = cfg.resolved_project()
    extra = " -v" if cfg.remove_volumes else ""
    down = (
        "%s; $C -p %s -f %s down --remove-orphans%s"
        % (
            _compose_bin_check(),
            shlex.quote(project),
            shlex.quote(remote_compose_path),
            extra,
        )
    )
    try:
        ssh_run(plan.target, down, timeout=float(cfg.teardown_timeout), check=False)
    except Exception as exc:
        logger.warning("teardown on %s failed: %s", plan.target, exc)

