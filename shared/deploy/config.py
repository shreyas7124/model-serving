"""Environment configuration for model runtime auto-deploy."""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import List, Optional


def env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or str(raw).strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def split_csv(raw: str) -> List[str]:
    return [p.strip() for p in (raw or "").split(",") if p.strip()]


def _slug(s: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9_.-]+", "-", (s or "app").strip().lower())
    return s.strip("-") or "app"


@dataclass
class DeployConfig:
    auto_deploy: bool = True
    teardown_on_exit: bool = True
    shared: bool = False
    project: str = ""
    engine: str = "vllm"  # nim | vllm
    app_name: str = "app"
    model_id: str = ""
    deploy_mode: str = "hybrid"
    replica_count: int = 1
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    gpu_per_replica: int = 1
    gpu_devices: List[str] = None  # type: ignore
    port_base: int = 8000
    wait_seconds: int = 300
    teardown_timeout: int = 60
    remove_volumes: bool = False
    compose_file: str = ""
    image: str = ""
    # credentials
    ngc_api_key: str = ""
    hf_token: str = ""
    vllm_max_model_len: int = 8192
    enable_prefix_caching: bool = True
    # multi-node SSH
    deploy_nodes: List[str] = None  # type: ignore
    remote_dir: str = "/tmp/model-serving-runtime"
    ssh_user: str = ""

    def __post_init__(self) -> None:
        if self.gpu_devices is None:
            self.gpu_devices = []
        if self.deploy_nodes is None:
            self.deploy_nodes = []

    @property
    def gpus_per_replica(self) -> int:
        return max(self.gpu_per_replica, self.tensor_parallel_size * self.pipeline_parallel_size, 1)

    def resolved_project(self) -> str:
        if self.project:
            return _slug(self.project)
        if self.shared:
            return _slug(f"model-serving-{self.engine}-shared")
        return _slug(f"model-serving-{self.engine}-{self.app_name}")

    def resolved_replicas(self) -> int:
        if self.deploy_mode == "sharded":
            return 1
        return max(1, self.replica_count)

    def is_multi_node(self) -> bool:
        """True when DEPLOY_NODES lists hosts (possibly including local)."""
        return bool(self.deploy_nodes)


def load_deploy_config(
    engine: str,
    *,
    app_name: str = "app",
    model_id: str = "",
    deploy_mode: Optional[str] = None,
    replica_count: Optional[int] = None,
    tensor_parallel_size: Optional[int] = None,
    existing_hint_port: int = 8000,
) -> DeployConfig:
    engine = (engine or "vllm").strip().lower()
    if engine not in ("nim", "vllm"):
        raise ValueError(f"engine must be nim|vllm, got {engine}")

    mode = (
        deploy_mode
        or os.getenv("MODEL_DEPLOY_MODE", os.getenv("LLM_DEPLOY_MODE", "hybrid"))
    ).strip().lower()
    if mode not in ("replica", "sharded", "hybrid"):
        mode = "hybrid"

    tp = tensor_parallel_size
    if tp is None:
        tp = env_int("TENSOR_PARALLEL_SIZE", env_int("TP_SIZE", 1))
    pp = env_int("PIPELINE_PARALLEL_SIZE", env_int("PP_SIZE", 1))

    if replica_count is None:
        if engine == "nim":
            replica_count = env_int(
                "NIM_REPLICA_COUNT", env_int("VLLM_REPLICA_COUNT", env_int("REPLICA_COUNT", 1))
            )
        else:
            replica_count = env_int(
                "VLLM_REPLICA_COUNT", env_int("NIM_REPLICA_COUNT", env_int("REPLICA_COUNT", 1))
            )

    if engine == "vllm":
        model = model_id or os.getenv(
            "VLLM_MODEL", os.getenv("HF_MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")
        )
        image = os.getenv("VLLM_IMAGE", "vllm/vllm-openai:latest")
        port_base = env_int("VLLM_PORT", existing_hint_port)
    else:
        model = model_id or os.getenv("NIM_MODEL_NAME", "meta/llama-3.1-8b-instruct")
        image = os.getenv(
            "NIM_IMAGE", "nvcr.io/nim/meta/llama-3.1-8b-instruct:latest"
        )
        port_base = env_int("NIM_PORT", existing_hint_port)

    gpus = split_csv(os.getenv("GPU_DEVICES", os.getenv("CUDA_VISIBLE_DEVICES", "")))
    gpr = env_int("GPU_PER_REPLICA", max(tp * pp, 1))

    # Default auto-deploy on for local convenience; teardown on exit ON as requested
    auto = env_bool("AUTO_DEPLOY_MODEL", default=True)
    teardown = env_bool("AUTO_DEPLOY_TEARDOWN_ON_EXIT", default=True)

    return DeployConfig(
        auto_deploy=auto,
        teardown_on_exit=teardown,
        shared=env_bool("AUTO_DEPLOY_SHARED", default=False),
        project=os.getenv("AUTO_DEPLOY_PROJECT", "").strip(),
        engine=engine,
        app_name=app_name or "app",
        model_id=model,
        deploy_mode=mode,
        replica_count=max(1, int(replica_count)),
        tensor_parallel_size=max(1, int(tp)),
        pipeline_parallel_size=max(1, int(pp)),
        gpu_per_replica=gpr,
        gpu_devices=gpus,
        port_base=port_base,
        wait_seconds=env_int("AUTO_DEPLOY_WAIT_SECONDS", 300),
        teardown_timeout=env_int("AUTO_DEPLOY_TEARDOWN_TIMEOUT_SECONDS", 60),
        remove_volumes=env_bool("AUTO_DEPLOY_REMOVE_VOLUMES", default=False),
        image=image,
        ngc_api_key=os.getenv("NGC_API_KEY", "").strip(),
        hf_token=os.getenv("HUGGING_FACE_HUB_TOKEN", os.getenv("HF_TOKEN", "")).strip(),
        vllm_max_model_len=env_int("VLLM_MAX_MODEL_LEN", 8192),
        enable_prefix_caching=env_bool("VLLM_ENABLE_PREFIX_CACHING", default=True),
        deploy_nodes=split_csv(os.getenv("DEPLOY_NODES", "")),
        remote_dir=os.getenv("DEPLOY_REMOTE_DIR", "/tmp/model-serving-runtime").strip() or "/tmp/model-serving-runtime",
        ssh_user=os.getenv("DEPLOY_SSH_USER", "").strip(),
    )
