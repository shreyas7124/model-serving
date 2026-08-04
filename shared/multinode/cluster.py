"""
Cluster identity, backend load balancing, and health payloads.
"""
from __future__ import annotations

import itertools
import os
import random
import socket
import threading
import time
from typing import Any, Dict, List, Optional

from .config import MultiNodeConfig, get_multinode_config
from .model_placement import (
    ModelPlacement,
    describe_placement_for_logs,
    get_model_placement,
    resolve_inference_url,
)


def node_identity(cfg: Optional[MultiNodeConfig] = None) -> Dict[str, Any]:
    """Compact identity block for /health and logs."""
    cfg = cfg or get_multinode_config()
    return {
        "node_id": cfg.node_id,
        "hostname": os.getenv("HOSTNAME") or socket.gethostname(),
        "role": cfg.role,
        "cluster": cfg.cluster_name,
        "multi_node": cfg.enabled,
        "public_url": cfg.public_url or None,
        "listen": f"{cfg.node_host}:{cfg.node_port}" if cfg.node_port else cfg.node_host,
        "workers": cfg.workers,
        "pid": os.getpid(),
        "model_deploy_mode": cfg.model_deploy_mode,
        "tensor_parallel_size": cfg.tensor_parallel_size,
        "pipeline_parallel_size": cfg.pipeline_parallel_size,
        "shard_rank": cfg.shard_rank,
        "shard_world_size": cfg.shard_world_size,
    }



class BackendPool:
    """
    Simple multi-backend URL pool for NIM / OpenAI-compatible upstreams.

    Strategies: round_robin | random | first
    """

    def __init__(self, urls: List[str], strategy: str = "round_robin"):
        self.urls = [u.rstrip("/") for u in urls if u]
        self.strategy = (strategy or "round_robin").lower()
        self._lock = threading.Lock()
        self._cycle = itertools.cycle(self.urls) if self.urls else None
        self._failures: Dict[str, float] = {}
        self.cooldown_seconds = float(os.getenv("BACKEND_COOLDOWN_SECONDS", "30"))

    def __len__(self) -> int:
        return len(self.urls)

    def mark_failure(self, url: str) -> None:
        with self._lock:
            self._failures[url] = time.time()

    def mark_success(self, url: str) -> None:
        with self._lock:
            self._failures.pop(url, None)

    def _available(self) -> List[str]:
        now = time.time()
        out = []
        for u in self.urls:
            failed_at = self._failures.get(u)
            if failed_at is None or (now - failed_at) >= self.cooldown_seconds:
                out.append(u)
        return out or list(self.urls)

    def next(self) -> Optional[str]:
        if not self.urls:
            return None
        available = self._available()
        if self.strategy == "random":
            return random.choice(available)
        if self.strategy == "first":
            return available[0]
        # round_robin across full list but skip cooling-down when possible
        with self._lock:
            if self._cycle is None:
                return available[0]
            for _ in range(len(self.urls)):
                candidate = next(self._cycle)
                if candidate in available:
                    return candidate
            return available[0]

    def all(self) -> List[str]:
        return list(self.urls)


class ClusterInfo:
    """Aggregates node + backend + model-placement info for health endpoints."""

    def __init__(
        self,
        cfg: Optional[MultiNodeConfig] = None,
        app_name: str = "app",
        extra: Optional[Dict[str, Any]] = None,
    ):
        self.cfg = cfg or get_multinode_config(app_name=app_name)
        self.app_name = app_name
        self.extra = extra or {}
        self.started_at = time.time()
        self.backend_pool = BackendPool(
            self.cfg.backend_urls,
            strategy=self.cfg.backend_strategy,
        )
        # Full placement (replica + sharded) derived from env + backend list
        self.placement: ModelPlacement = get_model_placement(
            replica_backends=self.cfg.backend_urls
        )
        # Prefer coordinator / shard gateway when this node does not hold weights
        if self.placement.coordinator_url and self.placement.coordinator_url not in self.backend_pool.urls:
            # Coordinator becomes the primary upstream for API proxies
            pass

    def next_backend(self) -> Optional[str]:
        """Resolve next inference URL (coordinator → replica pool → shard gateway)."""
        return resolve_inference_url(self.placement, self.backend_pool.next)

    def health(self, **more: Any) -> Dict[str, Any]:
        payload = {
            "status": "healthy",
            "app": self.app_name,
            "uptime_seconds": round(time.time() - self.started_at, 1),
            "node": node_identity(self.cfg),
            "cluster": {
                "name": self.cfg.cluster_name,
                "enabled": self.cfg.enabled,
                "peers": self.cfg.peers,
                "role": self.cfg.role,
                "sticky_sessions": self.cfg.sticky_sessions,
                "shared_state": {
                    "auth_db": self.cfg.resolved_auth_db(),
                    "chat_db": self.cfg.resolved_chat_db(),
                    "redis": self.cfg.redis_url or None,
                    "use_redis": self.cfg.use_redis,
                },
            },
            "backends": {
                "urls": self.backend_pool.all(),
                "strategy": self.cfg.backend_strategy,
                "count": len(self.backend_pool),
                "mode": "multi_replica" if len(self.backend_pool) > 1 else "single",
            },
            "model_placement": self.placement.to_dict(),
        }
        payload.update(self.extra)
        payload.update(more)
        return payload



def apply_flask_multinode(app, cfg: Optional[MultiNodeConfig] = None) -> MultiNodeConfig:
    """
    Apply multi-node friendly Flask settings (secret, proxy headers hint).
    """
    cfg = cfg or get_multinode_config()
    app.config["SECRET_KEY"] = cfg.session_secret
    # Trust X-Forwarded-* when behind a load balancer / ingress
    app.config["PREFERRED_URL_SCHEME"] = os.getenv("PREFERRED_URL_SCHEME", "http")
    try:
        from werkzeug.middleware.proxy_fix import ProxyFix

        app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1)
    except Exception:
        pass
    return cfg


def run_flask_app(app, cfg: MultiNodeConfig, default_port: int, debug: bool = False):
    """
    Run a Flask app with multi-node host/port/worker settings.

    Uses waitress when workers>1 or MULTI_NODE is on (production-ish),
    otherwise falls back to Flask dev server.
    """
    host = cfg.node_host or "0.0.0.0"
    port = cfg.node_port or default_port
    cfg.node_port = port

    print(f"Node: {cfg.node_id}  role={cfg.role}  multi_node={cfg.enabled}")
    print(f"Listen: {host}:{port}  workers={cfg.workers}")
    print(
        f"LLM placement: mode={cfg.model_deploy_mode} "
        f"tp={cfg.tensor_parallel_size} pp={cfg.pipeline_parallel_size} "
        f"shard={cfg.shard_rank}/{cfg.shard_world_size}"
    )
    if cfg.peers:
        print(f"Peers: {', '.join(cfg.peers)}")
    if cfg.backend_urls:
        print(f"Replica backends ({cfg.backend_strategy}): {', '.join(cfg.backend_urls)}")
    if cfg.model_shard_backends:
        print(f"Shard backends (one logical model): {', '.join(cfg.model_shard_backends)}")
    if cfg.model_coordinator_url:
        print(f"Model coordinator: {cfg.model_coordinator_url}")
    if cfg.public_url:
        print(f"Public URL: {cfg.public_url}")


    use_waitress = cfg.enabled or cfg.workers > 1 or os.getenv("USE_WAITRESS", "").lower() in (
        "1",
        "true",
        "yes",
    )
    if use_waitress:
        try:
            from waitress import serve

            # waitress is threaded; map workers*threads roughly
            threads = max(cfg.threads, cfg.workers * 4)
            print(f"Serving with waitress (threads={threads})")
            serve(app, host=host, port=port, threads=threads)
            return
        except ImportError:
            print("waitress not installed; falling back to Flask dev server")

    app.run(host=host, port=port, debug=debug, threaded=True)


def run_socketio_app(socketio, app, cfg: MultiNodeConfig, default_port: int, debug: bool = False):
    """Run Flask-SocketIO with multi-node host/port. Prefer eventlet/gevent message queue when Redis set."""
    host = cfg.node_host or "0.0.0.0"
    port = cfg.node_port or default_port
    cfg.node_port = port

    print(f"Node: {cfg.node_id}  role={cfg.role}  multi_node={cfg.enabled}")
    print(f"Listen: {host}:{port} (Socket.IO)")
    if cfg.use_redis and cfg.redis_url:
        print(f"Socket.IO message queue: {cfg.redis_url}")

    # message_queue enables multi-process / multi-node Socket.IO fan-out
    if cfg.use_redis and cfg.redis_url:
        try:
            socketio.server.manager_driver  # noqa: B018 — existence check
        except Exception:
            pass
        # Re-bind is hard after init; document that apps should pass message_queue at SocketIO() construct time

    socketio.run(app, host=host, port=port, debug=debug, allow_unsafe_werkzeug=True)
