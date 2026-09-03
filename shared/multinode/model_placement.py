"""
LLM placement strategies for multi-node / multi-GPU serving.

Two complementary modes (can be combined at the cluster level):

1. **replica** — multiple full copies of the model (horizontal scale-out).
   Each replica is a complete model on one or more local GPUs.
   Traffic is load-balanced via BackendPool (BACKEND_URLS).

2. **sharded** — one logical model split across GPUs and/or nodes because
   it does not fit on a single GPU/node (tensor / pipeline / device_map).
   - *Intra-node*: HF `device_map` + `max_memory` across local GPUs
   - *Inter-node*: ranks join a process group (torchrun / accelerate) or
     each node hosts a pipeline stage / TP shard exposed as a backend

Environment variables (see MultiNodeConfig + README):
  MODEL_DEPLOY_MODE=hybrid|sharded|replica  (default: hybrid)

  MODEL_PARALLEL_SIZE / TENSOR_PARALLEL_SIZE / PIPELINE_PARALLEL_SIZE
  CUDA_VISIBLE_DEVICES, HF_DEVICE_MAP, HF_MAX_MEMORY
  SHARD_RANK, SHARD_WORLD_SIZE, SHARD_GROUP
  MODEL_SHARD_BACKENDS  (ordered pipeline/TP endpoints for one logical model)
"""
from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _env_int(name: str, default: int = 0) -> int:
    raw = os.getenv(name)
    if raw is None or str(raw).strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _split_csv(raw: str) -> List[str]:
    return [p.strip() for p in (raw or "").split(",") if p.strip()]


@dataclass
class ModelPlacement:
    """
    Describes how the LLM weights are placed for this process / cluster.
    """

    # hybrid | sharded | replica  (default hybrid for large models)
    deploy_mode: str = "hybrid"


    # Parallelism degrees (1 = disabled)
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    model_parallel_size: int = 1  # convenience: max(tp, pp) or tp*pp

    # This process's shard identity (for sharded / hybrid)
    shard_rank: int = 0
    shard_world_size: int = 1
    shard_group: str = "default"  # logical model id shared by all ranks

    # Local GPU visibility
    cuda_visible_devices: str = ""  # e.g. "0,1,2,3"
    local_gpu_count: int = 0

    # HuggingFace / accelerate placement
    device_map: Optional[str] = None  # auto | sequential | balanced | none
    max_memory: Dict[str, str] = field(default_factory=dict)  # {0: "20GiB", "cpu": "64GiB"}
    low_cpu_mem_usage: bool = True
    offload_folder: str = ""
    torch_dtype: str = ""

    # Ordered backends that together form ONE sharded logical model
    # (pipeline stages or a coordinator URL). Empty when this process loads weights.
    shard_backends: List[str] = field(default_factory=list)

    # Independent full-model replica URLs (horizontal). Distinct from shard_backends.
    replica_backends: List[str] = field(default_factory=list)

    # Optional coordinator that fans out to shards (e.g. vLLM router, custom gateway)
    coordinator_url: str = ""

    @property
    def is_sharded(self) -> bool:
        return self.deploy_mode in ("sharded", "hybrid") and (
            self.tensor_parallel_size > 1
            or self.pipeline_parallel_size > 1
            or self.shard_world_size > 1
            or bool(self.shard_backends)
            or (self.device_map and self.device_map not in ("none", ""))
        )

    @property
    def is_multi_replica(self) -> bool:
        return len(self.replica_backends) > 1 or self.deploy_mode in ("replica", "hybrid")

    @property
    def loads_weights_locally(self) -> bool:
        """False when this process only proxies to remote shards/replicas."""
        if self.coordinator_url:
            return False
        if self.shard_backends and self.deploy_mode == "sharded":
            # API node in front of remote pipeline stages
            if os.getenv("LOAD_MODEL_WEIGHTS", "").lower() in ("0", "false", "no"):
                return False
        return _env_bool("LOAD_MODEL_WEIGHTS", default=True)

    def hf_from_pretrained_kwargs(self) -> Dict[str, Any]:
        """
        Keyword args for AutoModelForCausalLM.from_pretrained under this placement.
        """
        kwargs: Dict[str, Any] = {
            "low_cpu_mem_usage": self.low_cpu_mem_usage,
        }
        if self.device_map and self.device_map.lower() not in ("none", "null"):
            kwargs["device_map"] = self.device_map
        if self.max_memory:
            # accelerate expects int device keys where possible
            mm: Dict[Any, str] = {}
            for k, v in self.max_memory.items():
                try:
                    mm[int(k)] = v
                except (TypeError, ValueError):
                    mm[k] = v
            kwargs["max_memory"] = mm
        if self.offload_folder:
            kwargs["offload_folder"] = self.offload_folder
        return kwargs

    def apply_cuda_visible_devices(self) -> None:
        """Set CUDA_VISIBLE_DEVICES early if configured and not already set."""
        if self.cuda_visible_devices and "CUDA_VISIBLE_DEVICES" not in os.environ:
            os.environ["CUDA_VISIBLE_DEVICES"] = self.cuda_visible_devices
            logger.info("CUDA_VISIBLE_DEVICES=%s", self.cuda_visible_devices)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "deploy_mode": self.deploy_mode,
            "tensor_parallel_size": self.tensor_parallel_size,
            "pipeline_parallel_size": self.pipeline_parallel_size,
            "model_parallel_size": self.model_parallel_size,
            "shard_rank": self.shard_rank,
            "shard_world_size": self.shard_world_size,
            "shard_group": self.shard_group,
            "cuda_visible_devices": self.cuda_visible_devices or None,
            "local_gpu_count": self.local_gpu_count,
            "device_map": self.device_map,
            "max_memory": self.max_memory or None,
            "low_cpu_mem_usage": self.low_cpu_mem_usage,
            "offload_folder": self.offload_folder or None,
            "shard_backends": list(self.shard_backends),
            "replica_backends": list(self.replica_backends),
            "coordinator_url": self.coordinator_url or None,
            "is_sharded": self.is_sharded,
            "is_multi_replica": self.is_multi_replica,
            "loads_weights_locally": self.loads_weights_locally,
        }


def _detect_local_gpu_count() -> int:
    cvd = os.getenv("CUDA_VISIBLE_DEVICES", "").strip()
    if cvd and cvd.lower() not in ("", "none"):
        # count entries (supports "0,1" or UUID lists)
        return len([x for x in cvd.split(",") if x.strip() != ""])
    try:
        import torch

        if torch.cuda.is_available():
            return int(torch.cuda.device_count())
    except Exception:
        pass
    return 0


def _parse_max_memory(raw: str) -> Dict[str, str]:
    """
    Parse HF_MAX_MEMORY.

    Formats:
      - JSON object: {"0":"20GiB","1":"20GiB","cpu":"64GiB"}
      - CSV pairs: 0=20GiB,1=20GiB,cpu=64GiB
    """
    raw = (raw or "").strip()
    if not raw:
        return {}
    if raw.startswith("{"):
        try:
            data = json.loads(raw)
            return {str(k): str(v) for k, v in data.items()}
        except json.JSONDecodeError:
            logger.warning("Invalid HF_MAX_MEMORY JSON: %s", raw)
            return {}
    out: Dict[str, str] = {}
    for part in raw.split(","):
        part = part.strip()
        if not part or "=" not in part:
            continue
        k, v = part.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def get_model_placement(
    replica_backends: Optional[List[str]] = None,
) -> ModelPlacement:
    """
    Build ModelPlacement from environment.

    replica_backends: optional list already resolved by MultiNodeConfig.backend_urls
    """
    # Default hybrid: most frontier models need multi-GPU/node placement
    # and benefit from multiple sharded replicas when available.
    mode = os.getenv("MODEL_DEPLOY_MODE", os.getenv("LLM_DEPLOY_MODE", "hybrid")).strip().lower()
    if mode not in ("replica", "sharded", "hybrid"):
        mode = "hybrid"


    tp = max(1, _env_int("TENSOR_PARALLEL_SIZE", _env_int("TP_SIZE", 1)))
    pp = max(1, _env_int("PIPELINE_PARALLEL_SIZE", _env_int("PP_SIZE", 1)))
    mp = max(1, _env_int("MODEL_PARALLEL_SIZE", tp * pp if (tp > 1 or pp > 1) else 1))

    world = max(1, _env_int("SHARD_WORLD_SIZE", _env_int("WORLD_SIZE", mp if mp > 1 else 1)))
    rank = max(0, _env_int("SHARD_RANK", _env_int("RANK", 0)))
    if rank >= world:
        rank = rank % world

    cvd = os.getenv("CUDA_VISIBLE_DEVICES", os.getenv("GPU_DEVICES", "")).strip()
    device_map = os.getenv("HF_DEVICE_MAP", "").strip() or None
    # Default device_map for sharded local load
    if mode in ("sharded", "hybrid") and device_map is None and _detect_local_gpu_count() > 1:
        device_map = os.getenv("HF_DEVICE_MAP_DEFAULT", "auto")
    if device_map and device_map.lower() == "none":
        device_map = None

    max_memory = _parse_max_memory(os.getenv("HF_MAX_MEMORY", ""))
    # Optional per-GPU budget helper: HF_MAX_MEMORY_PER_GPU=20GiB
    per_gpu = os.getenv("HF_MAX_MEMORY_PER_GPU", "").strip()
    if per_gpu and not max_memory:
        n = _detect_local_gpu_count() or tp
        max_memory = {str(i): per_gpu for i in range(max(n, 1))}
        cpu_mem = os.getenv("HF_MAX_MEMORY_CPU", "").strip()
        if cpu_mem:
            max_memory["cpu"] = cpu_mem

    shard_backends = _split_csv(os.getenv("MODEL_SHARD_BACKENDS", ""))
    replicas = list(replica_backends or [])
    if not replicas:
        replicas = _split_csv(
            os.getenv("BACKEND_URLS", os.getenv("NIM_API_URLS", os.getenv("MODEL_REPLICA_URLS", "")))
        )
        single = os.getenv("NIM_API_URL", "").strip()
        if single and single not in replicas:
            replicas = replicas or [single]
        for _vk in ("VLLM_API_URLS", "VLLM_API_URL"):
            for part in _split_csv(os.getenv(_vk, "")):
                if part and part not in replicas:
                    replicas.append(part)

    coordinator = os.getenv("MODEL_COORDINATOR_URL", os.getenv("SHARD_COORDINATOR_URL", "")).strip()

    placement = ModelPlacement(
        deploy_mode=mode,
        tensor_parallel_size=tp,
        pipeline_parallel_size=pp,
        model_parallel_size=mp,
        shard_rank=rank,
        shard_world_size=world,
        shard_group=os.getenv("SHARD_GROUP", os.getenv("MODEL_SHARD_GROUP", "default")),
        cuda_visible_devices=cvd,
        local_gpu_count=_detect_local_gpu_count(),
        device_map=device_map,
        max_memory=max_memory,
        low_cpu_mem_usage=_env_bool("HF_LOW_CPU_MEM_USAGE", default=True),
        offload_folder=os.getenv("HF_OFFLOAD_FOLDER", "").strip(),
        torch_dtype=os.getenv("HF_TORCH_DTYPE", "").strip(),
        shard_backends=shard_backends,
        replica_backends=replicas,
        coordinator_url=coordinator,
    )
    return placement


def resolve_inference_url(placement: ModelPlacement, replica_pool_next) -> Optional[str]:
    """
    Choose the upstream URL for a chat/completion request.

    Priority:
      1. coordinator (single entry to a sharded mesh)
      2. replica pool (multi-instance full models)
      3. first shard backend (only if this node does not load weights)
    """
    if placement.coordinator_url:
        return placement.coordinator_url.rstrip("/")
    if callable(replica_pool_next):
        url = replica_pool_next()
        if url:
            return url
    if placement.shard_backends and not placement.loads_weights_locally:
        # Pipeline: clients hit stage-0 or a listed gateway
        return placement.shard_backends[0].rstrip("/")
    return None


def describe_placement_for_logs(placement: ModelPlacement) -> str:
    parts = [
        f"mode={placement.deploy_mode}",
        f"tp={placement.tensor_parallel_size}",
        f"pp={placement.pipeline_parallel_size}",
        f"rank={placement.shard_rank}/{placement.shard_world_size}",
        f"gpus_local={placement.local_gpu_count}",
        f"device_map={placement.device_map or 'none'}",
        f"replicas={len(placement.replica_backends)}",
        f"shard_backends={len(placement.shard_backends)}",
    ]
    if placement.coordinator_url:
        parts.append(f"coordinator={placement.coordinator_url}")
    return " ".join(parts)
