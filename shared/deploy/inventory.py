"""GPU / Docker inventory on local and SSH nodes."""
from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import List, Optional

from .ssh_util import advertise_host, normalize_ssh_target, ssh_run

logger = logging.getLogger(__name__)


@dataclass
class NodeInventory:
    target: str
    advertise: str
    gpu_ids: List[str] = field(default_factory=list)
    docker_ok: bool = False
    error: str = ""

    @property
    def gpu_count(self) -> int:
        return len(self.gpu_ids)


def parse_deploy_nodes(raw: Optional[str] = None) -> List[str]:
    raw = raw if raw is not None else os.getenv("DEPLOY_NODES", "")
    parts = [p.strip() for p in (raw or "").split(",") if p.strip()]
    if not parts:
        return []
    return [normalize_ssh_target(p) for p in parts]


def _probe_one(target: str) -> NodeInventory:
    target = normalize_ssh_target(target)
    inv = NodeInventory(target=target, advertise=advertise_host(target))
    try:
        d = ssh_run(
            target,
            "docker info >/dev/null 2>&1 && echo DOCKER_OK",
            check=False,
            timeout=30,
        )
        inv.docker_ok = "DOCKER_OK" in (d.stdout or "")
        if not inv.docker_ok:
            inv.error = "docker not available: %s" % ((d.stderr or d.stdout or "")[:300])
            return inv

        cmd = os.getenv(
            "DEPLOY_GPU_INVENTORY_CMD",
            "nvidia-smi --query-gpu=index --format=csv,noheader",
        )
        g = ssh_run(target, cmd, check=False, timeout=60)
        ids = []
        for line in (g.stdout or "").splitlines():
            line = line.strip()
            if not line:
                continue
            token = line.split(",")[0].strip()
            if token:
                ids.append(token)
        if not ids and g.returncode != 0:
            inv.error = "GPU inventory failed: %s" % ((g.stderr or g.stdout or "")[:300])
        inv.gpu_ids = ids
        return inv
    except Exception as exc:
        inv.error = str(exc)
        return inv


def inventory_nodes(targets: Optional[List[str]] = None) -> List[NodeInventory]:
    targets = targets if targets is not None else parse_deploy_nodes()
    if not targets:
        targets = ["local"]
    results: List[NodeInventory] = []
    with ThreadPoolExecutor(max_workers=max(1, len(targets))) as pool:
        futs = {pool.submit(_probe_one, t): t for t in targets}
        for fut in as_completed(futs):
            results.append(fut.result())
    order = {t: i for i, t in enumerate(targets)}
    results.sort(key=lambda r: order.get(r.target, 999))
    return results
