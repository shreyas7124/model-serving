"""Ensure NIM/vLLM containers are running; tear down on exit when owned."""
from __future__ import annotations

import atexit
import logging
import os
import signal
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from . import refcount
from .compose_gen import generate_compose, generate_compose_for_node_plan
from .config import DeployConfig, load_deploy_config
from .docker_util import compose_down, compose_up, docker_available
from .health import any_healthy, wait_until_healthy
from .inventory import inventory_nodes, parse_deploy_nodes
from .remote_exec import deploy_node_plan, teardown_node_plan
from .scheduler import NodePlan, schedule_replicas
from .ssh_util import is_local_target

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_registered_handles: List["RuntimeHandle"] = []
_signals_hooked = False


@dataclass
class NodeRuntime:
    target: str
    advertise: str
    compose_file_local: str
    compose_file_remote: str
    urls: List[str]


@dataclass
class RuntimeHandle:
    engine: str
    project: str
    urls: List[str]
    owned: bool
    teardown_on_exit: bool
    compose_file: str = ""
    shared: bool = False
    pid: int = field(default_factory=os.getpid)
    removed: bool = False
    multi_node: bool = False
    node_runtimes: List[NodeRuntime] = field(default_factory=list)
    # keep cfg bits needed for remote teardown
    remove_volumes: bool = False
    teardown_timeout: int = 60
    wait_seconds: int = 300
    ngc_api_key: str = ""
    hf_token: str = ""
    remote_dir: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "engine": self.engine,
            "project": self.project,
            "urls": list(self.urls),
            "owned": self.owned,
            "teardown_on_exit": self.teardown_on_exit,
            "compose_file": self.compose_file,
            "shared": self.shared,
            "pid": self.pid,
            "multi_node": self.multi_node,
            "nodes": [
                {
                    "target": n.target,
                    "advertise": n.advertise,
                    "urls": n.urls,
                    "compose_remote": n.compose_file_remote,
                }
                for n in self.node_runtimes
            ],
        }


def _normalize_existing(urls: Optional[Sequence[str]]) -> List[str]:
    out = []
    for u in urls or []:
        u = (u or "").strip().rstrip("/")
        if not u:
            continue
        if not u.endswith("/chat/completions"):
            if u.endswith("/v1"):
                u = u + "/chat/completions"
            elif "/v1/" not in u:
                u = u + "/v1/chat/completions"
        out.append(u)
    return out


def teardown_runtime(handle: Optional[RuntimeHandle]) -> None:
    if handle is None or handle.removed:
        return
    with _lock:
        if handle.removed:
            return
        handle.removed = True
    if not handle.owned or not handle.teardown_on_exit:
        logger.info(
            "Skip teardown (owned=%s teardown_on_exit=%s) project=%s",
            handle.owned,
            handle.teardown_on_exit,
            handle.project,
        )
        return

    do_down = True
    if handle.shared:
        remaining = refcount.release(handle.project, handle.pid)
        logger.info("Shared project %s refcount after release=%s", handle.project, remaining)
        do_down = remaining <= 0

    if not do_down:
        logger.info("Leaving shared stack %s running (other holders)", handle.project)
        return

    timeout = float(handle.teardown_timeout or os.getenv("AUTO_DEPLOY_TEARDOWN_TIMEOUT_SECONDS", "60"))
    remove_vols = handle.remove_volumes or os.getenv("AUTO_DEPLOY_REMOVE_VOLUMES", "").lower() in (
        "1",
        "true",
        "yes",
        "on",
    )

    if handle.multi_node and handle.node_runtimes:
        logger.info("Tearing down multi-node runtime project=%s nodes=%s", handle.project, len(handle.node_runtimes))
        # Build a minimal cfg-like object for teardown_node_plan
        cfg = load_deploy_config(handle.engine, app_name="teardown")
        cfg.project = handle.project
        cfg.remove_volumes = remove_vols
        cfg.teardown_timeout = int(timeout)
        cfg.ngc_api_key = handle.ngc_api_key
        cfg.hf_token = handle.hf_token

        def _one(nr: NodeRuntime) -> None:
            plan = NodePlan(target=nr.target, advertise=nr.advertise)
            teardown_node_plan(cfg, plan, nr.compose_file_remote)

        with ThreadPoolExecutor(max_workers=max(1, len(handle.node_runtimes))) as pool:
            futs = [pool.submit(_one, nr) for nr in handle.node_runtimes]
            for f in as_completed(futs):
                try:
                    f.result()
                except Exception as exc:
                    logger.warning("node teardown error: %s", exc)
        return

    if not handle.compose_file or not Path(handle.compose_file).exists():
        logger.warning("No compose file for teardown of %s", handle.project)
        return

    logger.info("Tearing down model runtime project=%s", handle.project)
    try:
        compose_down(
            handle.project,
            Path(handle.compose_file),
            timeout=timeout,
            remove_volumes=remove_vols,
        )
    except Exception as exc:
        logger.warning("Teardown failed for %s: %s", handle.project, exc)


def _atexit_teardown() -> None:
    with _lock:
        handles = list(_registered_handles)
    for h in handles:
        try:
            teardown_runtime(h)
        except Exception as exc:
            logger.warning("atexit teardown error: %s", exc)


def _signal_handler(signum, frame) -> None:  # noqa: ARG001
    _atexit_teardown()
    try:
        signal.signal(signum, signal.SIG_DFL)
    except Exception:
        pass
    os.kill(os.getpid(), signum)


def register_runtime_teardown(handle: RuntimeHandle) -> None:
    global _signals_hooked
    with _lock:
        if handle not in _registered_handles:
            _registered_handles.append(handle)
        if not _signals_hooked:
            atexit.register(_atexit_teardown)
            for sig in (signal.SIGINT, signal.SIGTERM):
                try:
                    signal.signal(sig, _signal_handler)
                except Exception:
                    pass
            _signals_hooked = True


def _deploy_local(cfg: DeployConfig, api_key: str) -> RuntimeHandle:
    if not docker_available():
        raise RuntimeError(
            "AUTO_DEPLOY_MODEL requires Docker on the control node for local deploy."
        )
    if cfg.engine == "nim" and not cfg.ngc_api_key:
        raise RuntimeError("NGC_API_KEY is required to auto-deploy NIM containers.")

    compose_file, urls = generate_compose(cfg)
    project = cfg.resolved_project()
    logger.info(
        "Auto-deploying %s project=%s file=%s urls=%s",
        cfg.engine,
        project,
        compose_file,
        urls,
    )
    try:
        compose_up(project, compose_file, timeout=float(cfg.wait_seconds) + 120)
        wait_until_healthy(urls, wait_seconds=cfg.wait_seconds, api_key=api_key)
    except Exception:
        try:
            compose_down(project, compose_file, timeout=cfg.teardown_timeout)
        except Exception:
            pass
        raise

    if cfg.shared:
        refcount.acquire(project, os.getpid(), str(compose_file))

    return RuntimeHandle(
        engine=cfg.engine,
        project=project,
        urls=urls,
        owned=True,
        teardown_on_exit=cfg.teardown_on_exit,
        compose_file=str(compose_file),
        shared=cfg.shared,
        multi_node=False,
        remove_volumes=cfg.remove_volumes,
        teardown_timeout=cfg.teardown_timeout,
        wait_seconds=cfg.wait_seconds,
        ngc_api_key=cfg.ngc_api_key,
        hf_token=cfg.hf_token,
    )


def _deploy_multi_node(cfg: DeployConfig, api_key: str) -> RuntimeHandle:
    if cfg.engine == "nim" and not cfg.ngc_api_key:
        raise RuntimeError("NGC_API_KEY is required to auto-deploy NIM containers.")

    targets = cfg.deploy_nodes or parse_deploy_nodes()
    logger.info("Multi-node deploy targets: %s", targets)
    nodes = inventory_nodes(targets)
    for n in nodes:
        logger.info(
            "Inventory %s advertise=%s gpus=%s docker_ok=%s err=%s",
            n.target,
            n.advertise,
            n.gpu_ids,
            n.docker_ok,
            n.error or "",
        )

    plans = schedule_replicas(cfg, nodes)
    project = cfg.resolved_project()
    node_runtimes: List[NodeRuntime] = []
    all_urls: List[str] = []

    try:
        for plan in plans:
            local_compose, urls = generate_compose_for_node_plan(cfg, plan)
            remote_path = deploy_node_plan(cfg, plan, local_compose)
            nr = NodeRuntime(
                target=plan.target,
                advertise=plan.advertise,
                compose_file_local=str(local_compose),
                compose_file_remote=remote_path,
                urls=urls,
            )
            node_runtimes.append(nr)
            all_urls.extend(urls)

        wait_until_healthy(all_urls, wait_seconds=cfg.wait_seconds, api_key=api_key)
    except Exception:
        # best-effort cleanup of what we started
        for nr in node_runtimes:
            try:
                plan = NodePlan(target=nr.target, advertise=nr.advertise)
                teardown_node_plan(cfg, plan, nr.compose_file_remote)
            except Exception:
                pass
        raise

    if cfg.shared:
        refcount.acquire(project, os.getpid(), ",".join(n.compose_file_remote for n in node_runtimes))

    return RuntimeHandle(
        engine=cfg.engine,
        project=project,
        urls=all_urls,
        owned=True,
        teardown_on_exit=cfg.teardown_on_exit,
        compose_file=node_runtimes[0].compose_file_local if node_runtimes else "",
        shared=cfg.shared,
        multi_node=True,
        node_runtimes=node_runtimes,
        remove_volumes=cfg.remove_volumes,
        teardown_timeout=cfg.teardown_timeout,
        wait_seconds=cfg.wait_seconds,
        ngc_api_key=cfg.ngc_api_key,
        hf_token=cfg.hf_token,
        remote_dir=cfg.remote_dir,
    )


def ensure_model_runtime(
    engine: str,
    *,
    app_name: str = "app",
    model_id: str = "",
    deploy_mode: Optional[str] = None,
    replica_count: Optional[int] = None,
    tensor_parallel_size: Optional[int] = None,
    pipeline_parallel_size: Optional[int] = None,
    gpu_devices: Optional[Sequence[str]] = None,
    gpu_per_replica: Optional[int] = None,
    port_base: Optional[int] = None,
    image: Optional[str] = None,
    project: Optional[str] = None,
    existing_urls: Optional[Sequence[str]] = None,
    register_teardown: bool = True,
    api_key: str = "",
) -> RuntimeHandle:
    """
    Ensure model containers exist when AUTO_DEPLOY_MODEL is on.

    Single-host: local docker compose.
    Multi-host: set DEPLOY_NODES=local,gpu1,user@gpu2 (passwordless SSH).
    Packs replicas onto free GPUs; tears down owned stacks on process exit.
    """
    cfg = load_deploy_config(
        engine,
        app_name=app_name,
        model_id=model_id,
        deploy_mode=deploy_mode,
        replica_count=replica_count,
        tensor_parallel_size=tensor_parallel_size,
        pipeline_parallel_size=pipeline_parallel_size,
        gpu_devices=list(gpu_devices) if gpu_devices is not None else None,
        gpu_per_replica=gpu_per_replica,
        port_base=port_base,
        image=image,
        project=project,
    )
    existing = _normalize_existing(existing_urls)

    if existing and any_healthy(existing, api_key=api_key):
        logger.info("Using existing healthy backends: %s", existing)
        return RuntimeHandle(
            engine=cfg.engine,
            project=cfg.resolved_project(),
            urls=existing,
            owned=False,
            teardown_on_exit=False,
            compose_file="",
            shared=cfg.shared,
        )

    if not cfg.auto_deploy:
        if existing:
            return RuntimeHandle(
                engine=cfg.engine,
                project=cfg.resolved_project(),
                urls=existing,
                owned=False,
                teardown_on_exit=False,
            )
        raise RuntimeError(
            "AUTO_DEPLOY_MODEL=false and no healthy BACKEND_URLS/NIM_API_URL/VLLM_API_URL. "
            "Start model servers or enable AUTO_DEPLOY_MODEL."
        )

    # Multi-node when DEPLOY_NODES is set
    if cfg.is_multi_node():
        handle = _deploy_multi_node(cfg, api_key=api_key)
    else:
        # local-only path still needs docker
        if not docker_available():
            if existing:
                return RuntimeHandle(
                    engine=cfg.engine,
                    project=cfg.resolved_project(),
                    urls=existing,
                    owned=False,
                    teardown_on_exit=False,
                )
            raise RuntimeError(
                "AUTO_DEPLOY_MODEL requires Docker. Install Docker, set DEPLOY_NODES for "
                "remote GPU hosts, or provide BACKEND_URLS."
            )
        handle = _deploy_local(cfg, api_key=api_key)

    if register_teardown and cfg.teardown_on_exit:
        register_runtime_teardown(handle)
    return handle


def _model_spec_deploy_key(spec: Any) -> str:
    """Dedupe key for multi-model deploy (shared backends deploy once)."""
    dep = getattr(spec, "deployment", None)
    if dep is None:
        return f"id:{getattr(spec, 'id', '')}"
    urls = tuple(sorted(u.rstrip("/") for u in (getattr(dep, "backend_urls", None) or []) if u))
    if urls:
        return "urls:" + ",".join(urls)
    cvd = (getattr(dep, "cuda_visible_devices", "") or "").strip()
    image = (getattr(dep, "nim_image", "") or "").strip()
    extra = getattr(dep, "extra", None) or {}
    if not image and isinstance(extra, dict):
        image = str(extra.get("image", "") or "").strip()
    tp = int(getattr(dep, "tensor_parallel_size", 1) or 1)
    mode = (getattr(dep, "deploy_mode", "replica") or "replica").strip().lower()
    mid = getattr(spec, "id", "") or ""
    return f"id:{mid}|img:{image}|cvd:{cvd}|tp:{tp}|mode:{mode}"


def _split_gpu_csv(raw: str) -> List[str]:
    return [p.strip() for p in (raw or "").split(",") if p.strip()]


def _gpus_needed(spec: Any) -> int:
    dep = getattr(spec, "deployment", None)
    if dep is None:
        return 1
    gc = int(getattr(dep, "gpu_count", 0) or 0)
    tp = int(getattr(dep, "tensor_parallel_size", 1) or 1)
    pp = int(getattr(dep, "pipeline_parallel_size", 1) or 1)
    cvd = _split_gpu_csv(getattr(dep, "cuda_visible_devices", "") or "")
    if cvd:
        return len(cvd)
    return max(gc, tp * pp, 1)


def ensure_models_runtime(
    engine: str,
    models: Sequence[Any],
    *,
    app_name: str = "app",
    api_key: str = "",
    default_backend_urls: Optional[Sequence[str]] = None,
    register_teardown: bool = True,
    port_base: Optional[int] = None,
) -> List[RuntimeHandle]:
    """
    Deploy multiple distinct models in parallel.

    - Deduplicates by backend URLs / (model id + image + GPUs + TP).
    - Writes resolved URLs back onto each ModelSpec.deployment.backend_urls.
    - Packs free GPUs from GPU_DEVICES when cuda_visible_devices is empty.
    - Assigns non-overlapping host ports (port_base + offset per unique stack).
    """
    specs = [m for m in (models or []) if m is not None]
    if not specs:
        raise RuntimeError("ensure_models_runtime: no models provided")

    default_urls = _normalize_existing(default_backend_urls)

    # Group specs that share a deploy key
    groups: Dict[str, List[Any]] = {}
    order: List[str] = []
    for spec in specs:
        # Seed empty backend from defaults only when single-model catalog
        dep = spec.deployment
        if (
            not dep.backend_urls
            and not dep.coordinator_url
            and default_urls
            and len(specs) == 1
        ):
            dep.backend_urls = list(default_urls)
        key = _model_spec_deploy_key(spec)
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(spec)

    # Global GPU pool for packing when CVD not set
    pool = _split_gpu_csv(os.getenv("GPU_DEVICES", os.getenv("CUDA_VISIBLE_DEVICES", "")))
    used: set = set()

    # Base port from engine defaults
    if port_base is None:
        if (engine or "").lower() == "nim":
            base = int(os.getenv("NIM_PORT", "8000") or "8000")
        else:
            base = int(os.getenv("VLLM_PORT", "8000") or "8000")
    else:
        base = int(port_base)

    # Pre-assign ports and GPUs for groups that need deploy
    plan: List[Dict[str, Any]] = []
    port_offset = 0
    for key in order:
        group = groups[key]
        lead = group[0]
        dep = lead.deployment
        existing = _normalize_existing(list(dep.backend_urls or []))
        if dep.coordinator_url:
            existing = _normalize_existing([dep.coordinator_url]) + existing

        gpus = _split_gpu_csv(getattr(dep, "cuda_visible_devices", "") or "")
        need = _gpus_needed(lead)
        if not gpus and pool:
            free = [g for g in pool if g not in used]
            if len(free) < need:
                raise RuntimeError(
                    f"Not enough GPU_DEVICES for model {getattr(lead, 'id', key)}: "
                    f"need {need}, free {len(free)} (pool={pool})"
                )
            gpus = free[:need]
            used.update(gpus)
        elif gpus:
            used.update(gpus)

        tp = int(getattr(dep, "tensor_parallel_size", 1) or 1)
        pp = int(getattr(dep, "pipeline_parallel_size", 1) or 1)
        mode = (getattr(dep, "deploy_mode", "") or None) or None
        image = (getattr(dep, "nim_image", "") or "").strip()
        extra = getattr(dep, "extra", None) or {}
        if not image and isinstance(extra, dict):
            image = str(extra.get("image", "") or "").strip()
        gpr = int(getattr(dep, "gpu_count", 0) or 0) or max(tp * pp, len(gpus) or 1)

        slug_id = re_slug(getattr(lead, "name", None) or getattr(lead, "id", "model"))
        model_port = base + port_offset
        # Reserve ports for potential multi-replica of this stack
        replicas = 1
        try:
            replicas = max(1, int(os.getenv("VLLM_REPLICA_COUNT", os.getenv("NIM_REPLICA_COUNT", "1")) or "1"))
        except ValueError:
            replicas = 1
        if mode == "sharded":
            replicas = 1
        port_offset += max(replicas, 1)

        plan.append(
            {
                "key": key,
                "group": group,
                "lead": lead,
                "existing": existing,
                "gpus": gpus,
                "tp": tp,
                "pp": pp,
                "mode": mode,
                "image": image or None,
                "gpr": gpr,
                "port": model_port,
                "project_suffix": slug_id,
                "model_id": getattr(lead, "id", "") or "",
            }
        )

    def _one(entry: Dict[str, Any]) -> RuntimeHandle:
        handle = ensure_model_runtime(
            engine,
            app_name=f"{app_name}-{entry['project_suffix']}",
            model_id=entry["model_id"],
            deploy_mode=entry["mode"],
            replica_count=1 if entry["mode"] == "sharded" else None,
            tensor_parallel_size=entry["tp"],
            pipeline_parallel_size=entry["pp"],
            gpu_devices=entry["gpus"] or None,
            gpu_per_replica=entry["gpr"],
            port_base=entry["port"],
            image=entry["image"],
            project=f"ms-{engine}-{app_name}-{entry['project_suffix']}",
            existing_urls=entry["existing"] or None,
            register_teardown=register_teardown,
            api_key=api_key,
        )
        # Write URLs back to all specs in the group
        urls = list(handle.urls or entry["existing"] or [])
        for spec in entry["group"]:
            if urls:
                spec.deployment.backend_urls = list(urls)
            if entry["gpus"] and not (spec.deployment.cuda_visible_devices or "").strip():
                spec.deployment.cuda_visible_devices = ",".join(entry["gpus"])
        return handle

    handles: List[RuntimeHandle] = []
    if len(plan) == 1:
        handles.append(_one(plan[0]))
        return handles

    # Parallel deploy distinct stacks
    errors: List[BaseException] = []
    with ThreadPoolExecutor(max_workers=max(1, len(plan))) as pool_ex:
        futs = {pool_ex.submit(_one, entry): entry for entry in plan}
        for fut in as_completed(futs):
            entry = futs[fut]
            try:
                handles.append(fut.result())
            except BaseException as exc:
                logger.exception(
                    "Multi-model deploy failed for %s: %s",
                    entry.get("model_id"), 
                    exc,
                )
                errors.append(exc)
    if errors and not handles:
        raise errors[0]
    if errors:
        logger.warning(
            "Some model stacks failed to deploy (%d ok, %d failed)",
            len(handles),
            len(errors),
        )
    return handles


def re_slug(s: str) -> str:
    import re as _re

    s = _re.sub(r"[^a-zA-Z0-9_.-]+", "-", (s or "model").strip().lower())
    return (s.strip("-") or "model")[:48]

