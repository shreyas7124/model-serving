"""
Mooncake-backed cache for Moonshot / Kimi models.

Uses MooncakeDistributedStore (from mooncake-transfer-engine) as an L2
KV store for completion responses. Falls back gracefully when Mooncake
is not installed or the master is unreachable.

Mooncake is Kimi's KVCache-centric serving infrastructure:
https://github.com/kvcache-ai/Mooncake
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from typing import Any, Dict, Optional

from .response_cache import ResponseCache

logger = logging.getLogger(__name__)


def is_moonshot_model(model_name: str) -> bool:
    """Return True when the model id looks like a Moonshot / Kimi model."""
    if not model_name:
        return False
    name = model_name.lower().strip()
    return (
        name.startswith("moonshotai/")
        or name.startswith("moonshot/")
        or "kimi" in name
        or name.startswith("moonshotai")
    )


class MooncakeResponseCache(ResponseCache):
    """
    Response cache with optional MooncakeDistributedStore L2 backend.

    Lookup order: local LRU (L1) → Mooncake store (L2) → miss.
    Writes go to both L1 and Mooncake when the store is available.
    """

    def __init__(
        self,
        max_size: int = 256,
        ttl_seconds: int = 3600,
        enabled: bool = True,
        key_prefix: str = "hf-ide",
        master_server_addr: Optional[str] = None,
        local_hostname: Optional[str] = None,
        metadata_server: Optional[str] = None,
        protocol: Optional[str] = None,
        global_segment_size: Optional[int] = None,
        local_buffer_size: Optional[int] = None,
        rdma_devices: Optional[str] = None,
    ):
        super().__init__(max_size=max_size, ttl_seconds=ttl_seconds, enabled=enabled)
        self.key_prefix = key_prefix.rstrip(":")
        self._mooncake = None
        self._mooncake_lock = threading.RLock()
        self._mooncake_enabled = False
        self._mooncake_error: Optional[str] = None
        self.mooncake_hits = 0
        self.mooncake_misses = 0
        self.mooncake_writes = 0

        if not enabled:
            return

        self._init_mooncake(
            master_server_addr=master_server_addr
            or os.getenv("MOONCAKE_MASTER", "127.0.0.1:50051"),
            local_hostname=local_hostname
            or os.getenv("MOONCAKE_LOCAL_HOSTNAME", "localhost"),
            metadata_server=metadata_server
            or os.getenv("MOONCAKE_METADATA_SERVER", "P2PHANDSHAKE"),
            protocol=protocol or os.getenv("MOONCAKE_PROTOCOL", "tcp"),
            global_segment_size=global_segment_size
            or int(os.getenv("MOONCAKE_GLOBAL_SEGMENT_SIZE", str(512 * 1024 * 1024))),
            local_buffer_size=local_buffer_size
            or int(os.getenv("MOONCAKE_LOCAL_BUFFER_SIZE", str(128 * 1024 * 1024))),
            rdma_devices=rdma_devices
            if rdma_devices is not None
            else os.getenv("MOONCAKE_RDMA_DEVICES", ""),
        )

    def _init_mooncake(
        self,
        master_server_addr: str,
        local_hostname: str,
        metadata_server: str,
        protocol: str,
        global_segment_size: int,
        local_buffer_size: int,
        rdma_devices: str,
    ) -> None:
        try:
            from mooncake.store import MooncakeDistributedStore  # type: ignore
        except ImportError:
            self._mooncake_error = (
                "mooncake-transfer-engine not installed "
                "(pip install mooncake-transfer-engine)"
            )
            logger.info("Mooncake unavailable: %s", self._mooncake_error)
            return

        try:
            store = MooncakeDistributedStore()
            # setup signature from Mooncake quick start
            store.setup(
                local_hostname=local_hostname,
                metadata_server=metadata_server,
                global_segment_size=int(global_segment_size),
                local_buffer_size=int(local_buffer_size),
                protocol=protocol,
                rdma_devices=rdma_devices or "",
                master_server_addr=master_server_addr,
            )
            self._mooncake = store
            self._mooncake_enabled = True
            logger.info(
                "Mooncake store connected (master=%s, protocol=%s)",
                master_server_addr,
                protocol,
            )
        except Exception as exc:  # noqa: BLE001 — soft-fail to L1 only
            self._mooncake = None
            self._mooncake_enabled = False
            self._mooncake_error = str(exc)
            logger.warning("Mooncake setup failed, using L1 only: %s", exc)

    def _full_key(self, key: str) -> str:
        return f"{self.key_prefix}:{key}"

    def _mooncake_get(self, key: str) -> Optional[Any]:
        if not self._mooncake_enabled or self._mooncake is None:
            return None
        full = self._full_key(key)
        try:
            with self._mooncake_lock:
                raw = self._mooncake.get(full)
            if raw is None:
                self.mooncake_misses += 1
                return None
            if isinstance(raw, str):
                raw_bytes = raw.encode("utf-8")
            elif isinstance(raw, memoryview):
                raw_bytes = raw.tobytes()
            else:
                raw_bytes = bytes(raw)
            payload = json.loads(raw_bytes.decode("utf-8"))
            # Optional embedded TTL
            ts = payload.get("_ts")
            if ts is not None and time.time() - float(ts) > self.ttl_seconds:
                self.mooncake_misses += 1
                try:
                    with self._mooncake_lock:
                        if hasattr(self._mooncake, "remove"):
                            self._mooncake.remove(full)
                        elif hasattr(self._mooncake, "delete"):
                            self._mooncake.delete(full)
                except Exception:
                    pass
                return None
            self.mooncake_hits += 1
            return payload.get("value")
        except Exception as exc:  # noqa: BLE001
            self.mooncake_misses += 1
            logger.debug("Mooncake get failed for %s: %s", full, exc)
            return None

    def _mooncake_set(self, key: str, value: Any) -> None:
        if not self._mooncake_enabled or self._mooncake is None:
            return
        full = self._full_key(key)
        try:
            blob = json.dumps(
                {"value": value, "_ts": time.time()},
                ensure_ascii=True,
                default=str,
            ).encode("utf-8")
            with self._mooncake_lock:
                self._mooncake.put(full, blob)
            self.mooncake_writes += 1
        except Exception as exc:  # noqa: BLE001
            logger.debug("Mooncake put failed for %s: %s", full, exc)

    def get(self, key: str) -> Optional[Any]:
        if not self.enabled:
            return None

        # L1 local LRU
        hit = super().get(key)
        if hit is not None:
            return hit

        # Undo the miss counted by super().get so L2 can still be a hit
        with self._lock:
            self.misses = max(0, self.misses - 1)

        # L2 Mooncake
        value = self._mooncake_get(key)
        if value is not None:
            # Promote to L1
            super().set(key, value)
            with self._lock:
                self.hits += 1
            return value

        with self._lock:
            self.misses += 1
        return None

    def set(self, key: str, value: Any) -> None:
        if not self.enabled:
            return
        super().set(key, value)
        self._mooncake_set(key, value)

    def clear(self) -> None:
        super().clear()
        self.mooncake_hits = 0
        self.mooncake_misses = 0
        self.mooncake_writes = 0
        # Mooncake store has no bulk clear in the simple API; leave remote entries
        # to TTL expiry. Local L1 is wiped above.

    def close(self) -> None:
        if self._mooncake is not None:
            try:
                with self._mooncake_lock:
                    if hasattr(self._mooncake, "close"):
                        self._mooncake.close()
            except Exception as exc:  # noqa: BLE001
                logger.debug("Mooncake close error: %s", exc)
            finally:
                self._mooncake = None
                self._mooncake_enabled = False

    def stats(self) -> Dict[str, Any]:
        base = super().stats()
        base.update(
            {
                "backend": "mooncake+l1" if self._mooncake_enabled else "l1",
                "mooncake_enabled": self._mooncake_enabled,
                "mooncake_error": self._mooncake_error,
                "mooncake_hits": self.mooncake_hits,
                "mooncake_misses": self.mooncake_misses,
                "mooncake_writes": self.mooncake_writes,
                "key_prefix": self.key_prefix,
            }
        )
        return base


def build_response_cache_for_model(
    model_name: str,
    max_size: int = 256,
    ttl_seconds: int = 3600,
    enabled: bool = True,
    force_mooncake: Optional[bool] = None,
) -> ResponseCache:
    """
    Factory: Mooncake cache for Moonshot/Kimi models, plain LRU otherwise.

    force_mooncake:
      None  → auto (Moonshot models only)
      True  → always try Mooncake
      False → never use Mooncake
    """
    use_mooncake = (
        force_mooncake
        if force_mooncake is not None
        else is_moonshot_model(model_name)
    )
    # Allow explicit env override
    env_flag = os.getenv("MOONCAKE_CACHE", "").lower()
    if env_flag in ("1", "true", "yes", "on"):
        use_mooncake = True
    elif env_flag in ("0", "false", "no", "off"):
        use_mooncake = False

    if use_mooncake:
        cache = MooncakeResponseCache(
            max_size=max_size,
            ttl_seconds=ttl_seconds,
            enabled=enabled,
            key_prefix=os.getenv("MOONCAKE_KEY_PREFIX", f"hf-ide:{model_name}"),
        )
        return cache

    return ResponseCache(
        max_size=max_size,
        ttl_seconds=ttl_seconds,
        enabled=enabled,
    )
