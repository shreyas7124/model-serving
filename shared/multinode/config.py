"""
Multi-node configuration loaded from environment variables.

Design goals:
  - Single-node remains the zero-config default
  - Multi-node enables shared state (DB path / Redis), sticky sessions,
    horizontal replicas behind a load balancer, and node identity
  - Works for Flask APIs, Socket.IO voice apps, and Streamlit UIs
"""
from __future__ import annotations

import os
import socket
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _default_hostname() -> str:
    return os.getenv("HOSTNAME") or socket.gethostname() or "node-local"


@dataclass
class MultiNodeConfig:
    """Runtime multi-node settings for one application process."""

    # Cluster topology
    enabled: bool = False
    node_id: str = "node-0"
    node_host: str = "0.0.0.0"
    node_port: int = 0
    public_url: str = ""
    cluster_name: str = "model-serving"
    role: str = "all"  # all | api | worker | prefill | decode
    peers: List[str] = field(default_factory=list)

    # Process model
    workers: int = 1
    threads: int = 8

    # Shared state
    shared_data_dir: str = "shared/database"
    auth_db_path: str = ""
    chat_db_path: str = ""
    redis_url: str = ""  # e.g. redis://redis:6379/0
    use_redis: bool = False

    # Load balancing / sessions
    sticky_sessions: bool = True
    session_secret: str = "change-me-in-production"

    # Upstream model backends (comma-separated for multi-backend)
    backend_urls: List[str] = field(default_factory=list)
    backend_strategy: str = "round_robin"  # round_robin | random | first

    # LLM placement: multi-replica and/or single-model sharding
    # replica  = N full model copies (horizontal)
    # sharded  = 1 logical model split across GPUs/nodes (TP/PP/device_map)
    # hybrid   = sharded replicas (each replica is itself multi-GPU/node)
    model_deploy_mode: str = "hybrid"

    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    model_parallel_size: int = 1
    shard_rank: int = 0
    shard_world_size: int = 1
    shard_group: str = "default"
    cuda_visible_devices: str = ""
    hf_device_map: str = ""
    hf_max_memory: str = ""
    model_shard_backends: List[str] = field(default_factory=list)
    model_coordinator_url: str = ""

    def resolved_auth_db(self) -> str:

        if self.auth_db_path:
            return self.auth_db_path
        return os.path.join(self.shared_data_dir, "users.db")

    def resolved_chat_db(self) -> str:
        if self.chat_db_path:
            return self.chat_db_path
        return os.path.join(self.shared_data_dir, "chat_history.db")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "node_id": self.node_id,
            "node_host": self.node_host,
            "node_port": self.node_port,
            "public_url": self.public_url,
            "cluster_name": self.cluster_name,
            "role": self.role,
            "peers": list(self.peers),
            "workers": self.workers,
            "threads": self.threads,
            "shared_data_dir": self.shared_data_dir,
            "auth_db_path": self.resolved_auth_db(),
            "chat_db_path": self.resolved_chat_db(),
            "redis_url": self.redis_url or None,
            "use_redis": self.use_redis,
            "sticky_sessions": self.sticky_sessions,
            "backend_urls": list(self.backend_urls),
            "backend_strategy": self.backend_strategy,
            "hostname": _default_hostname(),
            "model_placement": {
                "deploy_mode": self.model_deploy_mode,
                "tensor_parallel_size": self.tensor_parallel_size,
                "pipeline_parallel_size": self.pipeline_parallel_size,
                "model_parallel_size": self.model_parallel_size,
                "shard_rank": self.shard_rank,
                "shard_world_size": self.shard_world_size,
                "shard_group": self.shard_group,
                "cuda_visible_devices": self.cuda_visible_devices or None,
                "hf_device_map": self.hf_device_map or None,
                "model_shard_backends": list(self.model_shard_backends),
                "model_coordinator_url": self.model_coordinator_url or None,
            },
        }



_CONFIG: Optional[MultiNodeConfig] = None


def get_multinode_config(
    default_port: int = 0,
    app_name: str = "app",
    reload: bool = False,
) -> MultiNodeConfig:
    """
    Load (and cache) multi-node config from the environment.

    default_port: app-specific fallback when PORT / NODE_PORT unset
    app_name: used only for default node_id prefix
    """
    global _CONFIG
    if _CONFIG is not None and not reload:
        # Allow callers to still override port display if previously 0
        if default_port and not _CONFIG.node_port:
            _CONFIG.node_port = default_port
        return _CONFIG

    peers_raw = os.getenv("CLUSTER_PEERS", os.getenv("MULTI_NODE_PEERS", ""))
    peers = [p.strip() for p in peers_raw.split(",") if p.strip()]

    backends_raw = os.getenv(
        "BACKEND_URLS",
        os.getenv("NIM_API_URLS", os.getenv("MODEL_BACKEND_URLS", "")),
    )
    backend_urls = [b.strip() for b in backends_raw.split(",") if b.strip()]
    # Fall back to single NIM_API_URL if present
    single = os.getenv("NIM_API_URL", "").strip()
    if single and single not in backend_urls:
        backend_urls = backend_urls or [single]

    port = _env_int("PORT", _env_int("NODE_PORT", default_port or 0))
    host = os.getenv("HOST", os.getenv("NODE_HOST", "0.0.0.0"))
    node_id = os.getenv("NODE_ID", f"{app_name}-{_default_hostname()}")

    enabled = _env_bool("MULTI_NODE", default=bool(peers) or _env_bool("CLUSTER_ENABLED"))
    redis_url = os.getenv("REDIS_URL", "").strip()
    shared_dir = os.getenv("SHARED_DATA_DIR", "shared/database")

    tp = max(1, _env_int("TENSOR_PARALLEL_SIZE", _env_int("TP_SIZE", 1)))
    pp = max(1, _env_int("PIPELINE_PARALLEL_SIZE", _env_int("PP_SIZE", 1)))
    mp = max(1, _env_int("MODEL_PARALLEL_SIZE", tp * pp if (tp > 1 or pp > 1) else 1))
    shard_backends = [
        b.strip()
        for b in os.getenv("MODEL_SHARD_BACKENDS", "").split(",")
        if b.strip()
    ]
    # Default hybrid: advanced models rarely fit on one server; combine
    # multi-GPU/node sharding with optional multi-replica backends.
    deploy_mode = os.getenv("MODEL_DEPLOY_MODE", os.getenv("LLM_DEPLOY_MODE", "hybrid")).strip().lower()
    if deploy_mode not in ("replica", "sharded", "hybrid"):
        deploy_mode = "hybrid"

    world = max(1, _env_int("SHARD_WORLD_SIZE", _env_int("WORLD_SIZE", mp if mp > 1 else 1)))

    cfg = MultiNodeConfig(
        enabled=enabled,
        node_id=node_id,
        node_host=host,
        node_port=port,
        public_url=os.getenv("PUBLIC_URL", os.getenv("NODE_PUBLIC_URL", "")).rstrip("/"),
        cluster_name=os.getenv("CLUSTER_NAME", "model-serving"),
        role=os.getenv("NODE_ROLE", "all").lower(),
        peers=peers,
        workers=max(1, _env_int("WEB_CONCURRENCY", _env_int("WORKERS", 1))),
        threads=max(1, _env_int("THREADS", 8)),
        shared_data_dir=shared_dir,
        auth_db_path=os.getenv("AUTH_DB_PATH", ""),
        chat_db_path=os.getenv("CHAT_DB_PATH", ""),
        redis_url=redis_url,
        use_redis=_env_bool("USE_REDIS", default=bool(redis_url)),
        sticky_sessions=_env_bool("STICKY_SESSIONS", default=True),
        session_secret=os.getenv("SECRET_KEY", os.getenv("SESSION_SECRET", "change-me-in-production")),
        backend_urls=backend_urls,
        backend_strategy=os.getenv("BACKEND_STRATEGY", "round_robin").lower(),
        model_deploy_mode=deploy_mode,
        tensor_parallel_size=tp,
        pipeline_parallel_size=pp,
        model_parallel_size=mp,
        shard_rank=max(0, _env_int("SHARD_RANK", _env_int("RANK", 0))),
        shard_world_size=world,
        shard_group=os.getenv("SHARD_GROUP", "default"),
        cuda_visible_devices=os.getenv("CUDA_VISIBLE_DEVICES", os.getenv("GPU_DEVICES", "")),
        hf_device_map=os.getenv("HF_DEVICE_MAP", ""),
        hf_max_memory=os.getenv("HF_MAX_MEMORY", ""),
        model_shard_backends=shard_backends,
        model_coordinator_url=os.getenv(
            "MODEL_COORDINATOR_URL", os.getenv("SHARD_COORDINATOR_URL", "")
        ).strip(),
    )
    _CONFIG = cfg
    return cfg

