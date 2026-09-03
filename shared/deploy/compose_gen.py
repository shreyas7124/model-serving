"""Generate compose YAML for NIM or vLLM model stacks."""
from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from .config import DeployConfig

try:
    from .scheduler import NodePlan, ReplicaPlacement
except Exception:  # pragma: no cover
    NodePlan = None  # type: ignore
    ReplicaPlacement = None  # type: ignore


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def compose_output_path(cfg: DeployConfig, suffix: str = "") -> Path:
    if cfg.compose_file and not suffix:
        return Path(cfg.compose_file)
    deploy_dir = _repo_root() / "deploy"
    deploy_dir.mkdir(parents=True, exist_ok=True)
    name = f"docker-compose.{cfg.engine}.{cfg.resolved_project()}"
    if suffix:
        name += f".{suffix}"
    return deploy_dir / f"{name}.generated.yml"


def _gpu_slices(cfg: DeployConfig) -> List[List[str]]:
    replicas = cfg.resolved_replicas()
    gpr = cfg.gpus_per_replica
    needed = replicas * gpr
    gpus = list(cfg.gpu_devices)
    if not gpus:
        gpus = [str(i) for i in range(needed)]
    if len(gpus) < needed:
        raise RuntimeError(
            f"Not enough GPU_DEVICES ({len(gpus)}) for {replicas} x {gpr} GPUs (need {needed})"
        )
    slices: List[List[str]] = []
    idx = 0
    for _ in range(replicas):
        slices.append(gpus[idx : idx + gpr])
        idx += gpr
    return slices


def _join(lines: List[str]) -> str:
    return "\n".join(lines) + "\n"


def _vllm_service_lines(cfg: DeployConfig, name: str, devices: Sequence[str], host_port: int) -> List[str]:
    cvd = ",".join(devices)
    ids = ", ".join('"%s"' % d for d in devices)
    hf_cache = os.path.expanduser(os.getenv("HF_CACHE", "~/.cache/huggingface"))
    lines = [
        f"  {name}:",
        f"    image: {cfg.image}",
        "    command:",
        '      - "--model"',
        '      - "%s"' % cfg.model_id,
        '      - "--tensor-parallel-size"',
        '      - "%s"' % cfg.tensor_parallel_size,
        '      - "--max-model-len"',
        '      - "%s"' % cfg.vllm_max_model_len,
        '      - "--host"',
        '      - "0.0.0.0"',
        '      - "--port"',
        '      - "8000"',
    ]
    if cfg.enable_prefix_caching:
        lines.append('      - "--enable-prefix-caching"')
    if cfg.pipeline_parallel_size > 1:
        lines.extend(
            [
                '      - "--pipeline-parallel-size"',
                '      - "%s"' % cfg.pipeline_parallel_size,
            ]
        )
    lines.extend(
        [
            "    environment:",
            '      CUDA_VISIBLE_DEVICES: "%s"' % cvd,
            '      HUGGING_FACE_HUB_TOKEN: "%s"' % cfg.hf_token,
            "    ports:",
            '      - "%s:8000"' % host_port,
            "    volumes:",
            "      - %s:/root/.cache/huggingface" % hf_cache,
            "    ipc: host",
            "    deploy:",
            "      resources:",
            "        reservations:",
            "          devices:",
            "            - driver: nvidia",
            "              capabilities: [gpu]",
            "              device_ids: [%s]" % ids,
        ]
    )
    return lines


def _nim_service_lines(cfg: DeployConfig, name: str, devices: Sequence[str], host_port: int) -> List[str]:
    cvd = ",".join(devices)
    ids = ", ".join('"%s"' % d for d in devices)
    key = cfg.ngc_api_key or "${NGC_API_KEY}"
    return [
        f"  {name}:",
        f"    image: {cfg.image}",
        "    runtime: nvidia",
        "    environment:",
        '      NGC_API_KEY: "%s"' % key,
        '      CUDA_VISIBLE_DEVICES: "%s"' % cvd,
        '      NVIDIA_VISIBLE_DEVICES: "%s"' % cvd,
        "    ports:",
        '      - "%s:8000"' % host_port,
        '    shm_size: "16gb"',
        "    deploy:",
        "      resources:",
        "        reservations:",
        "          devices:",
        "            - driver: nvidia",
        "              capabilities: [gpu]",
        "              device_ids: [%s]" % ids,
    ]


def generate_vllm_compose(cfg: DeployConfig) -> Tuple[Path, List[str]]:
    slices = _gpu_slices(cfg)
    lines = [
        f"# AUTO-GENERATED model runtime ({cfg.engine}) project={cfg.resolved_project()}",
        f"# mode={cfg.deploy_mode} replicas={len(slices)} tp={cfg.tensor_parallel_size}",
        "services:",
    ]
    urls: List[str] = []
    for i, devices in enumerate(slices):
        name = f"vllm-{i}"
        host_port = cfg.port_base + i
        lines.extend(_vllm_service_lines(cfg, name, devices, host_port))
        urls.append("http://127.0.0.1:%s/v1/chat/completions" % host_port)
    path = compose_output_path(cfg)
    path.write_text(_join(lines))
    return path, urls


def generate_nim_compose(cfg: DeployConfig) -> Tuple[Path, List[str]]:
    slices = _gpu_slices(cfg)
    lines = [
        f"# AUTO-GENERATED model runtime ({cfg.engine}) project={cfg.resolved_project()}",
        f"# mode={cfg.deploy_mode} replicas={len(slices)}",
        "services:",
    ]
    urls: List[str] = []
    for i, devices in enumerate(slices):
        name = f"nim-{i}"
        host_port = cfg.port_base + i
        lines.extend(_nim_service_lines(cfg, name, devices, host_port))
        urls.append("http://127.0.0.1:%s/v1/chat/completions" % host_port)
    path = compose_output_path(cfg)
    path.write_text(_join(lines))
    return path, urls


def generate_compose(cfg: DeployConfig) -> Tuple[Path, List[str]]:
    if cfg.engine == "nim":
        return generate_nim_compose(cfg)
    return generate_vllm_compose(cfg)


def generate_compose_for_node_plan(cfg: DeployConfig, plan: "NodePlan") -> Tuple[Path, List[str]]:
    """Compose file containing only replicas scheduled on one node."""
    safe = plan.target.replace("@", "_").replace(":", "_")
    lines = [
        f"# AUTO-GENERATED {cfg.engine} node={plan.target} project={cfg.resolved_project()}",
        "services:",
    ]
    for rep in plan.replicas:
        if cfg.engine == "nim":
            lines.extend(
                _nim_service_lines(cfg, rep.service_name, rep.device_ids, rep.host_port)
            )
        else:
            lines.extend(
                _vllm_service_lines(cfg, rep.service_name, rep.device_ids, rep.host_port)
            )
    path = compose_output_path(cfg, suffix=safe)
    path.write_text(_join(lines))
    return path, list(plan.urls)
