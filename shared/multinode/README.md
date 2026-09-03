# Multi-Node Deployment (Shared)

All applications support multi-node deployment through `shared.multinode`.

This covers **two LLM scaling patterns** (they can be combined):

| Pattern | When to use | How |
|--------|-------------|-----|
| **Hybrid (default)** | Frontier models that need multi-GPU/node **and** scale-out | Sharded placement + optional multi-replica `BACKEND_URLS` |
| **Single-model sharded** | One logical model only (no extra full copies) | TP/PP/`device_map` across GPUs and/or nodes |
| **Multi-replica** | Throughput / HA when each full copy fits one node | N full copies; load-balance with `BACKEND_URLS` |


## Concepts

| Concept | Purpose |
|--------|---------|
| **Node** | One running process (or container) of an app |
| **Cluster** | Several nodes behind a load balancer |
| **Replica backends** | Full model instances (`BACKEND_URLS`) — horizontal scale |
| **Shard backends** | Pieces of **one** model (`MODEL_SHARD_BACKENDS`) — capacity scale |
| **Coordinator** | Optional gateway in front of a sharded mesh (`MODEL_COORDINATOR_URL`) |
| **Shared state** | Auth DB + chat history on a shared volume or Redis |
| **Sticky sessions** | Keep a client on the same app node (WebRTC / Streamlit) |

## 1) Multiple full LLM instances (replicas)

```bash
MULTI_NODE=true
MODEL_DEPLOY_MODE=replica   # override default hybrid when each copy fits one node
# Each URL is a complete model (NIM container, vLLM, etc.)
BACKEND_URLS=http://nim-a:8000/v1/chat/completions,http://nim-b:8000/v1/chat/completions
BACKEND_STRATEGY=round_robin   # round_robin | random | first
```


```
Clients → LB → [app nodes] → BackendPool → NIM-A
                                         → NIM-B
```

## 2) One LLM spread across GPUs / nodes (sharded)

### Intra-node (multi-GPU, one machine)

Model too large for one GPU but fits on several GPUs of the same host:

```bash
MODEL_DEPLOY_MODE=sharded
CUDA_VISIBLE_DEVICES=0,1,2,3
TENSOR_PARALLEL_SIZE=4          # or rely on device_map alone
HF_DEVICE_MAP=auto              # accelerate splits layers across GPUs
HF_MAX_MEMORY_PER_GPU=20GiB
# HF_MAX_MEMORY={"0":"20GiB","1":"20GiB","cpu":"64GiB"}
HF_TORCH_DTYPE=bfloat16
```

HF apps are **vLLM-only** clients: set these on the **vLLM containers**, not the Streamlit/Flask processes.
Apps use `BACKEND_URLS` / `VLLM_API_URL` and never load Transformers weights.

### Inter-node (pipeline / multi-host TP)

Model does not fit on one node — each host runs a shard or stage:

```bash
MODEL_DEPLOY_MODE=sharded
SHARD_GROUP=kimi-k3-prod
SHARD_WORLD_SIZE=4
SHARD_RANK=0                    # 0..3 on each rank process
# Ordered endpoints that together form ONE logical model
MODEL_SHARD_BACKENDS=http://gpu-node0:8000/v1/chat/completions,http://gpu-node1:8000/v1/chat/completions
# Or a single coordinator / router:
MODEL_COORDINATOR_URL=http://vllm-router:8000/v1/chat/completions
LOAD_MODEL_WEIGHTS=false        # API nodes only proxy; workers hold weights
```

Typical layout:

```
                    ┌──────────────┐
 Clients ──────────►│ App / API    │  LOAD_MODEL_WEIGHTS=false
                    │ (coordinator)│
                    └──────┬───────┘
           ┌───────────────┼───────────────┐
           ▼               ▼               ▼
      ┌─────────┐     ┌─────────┐     ┌─────────┐
      │ Shard 0 │     │ Shard 1 │     │ Shard 2 │  one logical model
      │ TP/PP   │◄───►│ TP/PP   │◄───►│ TP/PP   │
      └─────────┘     └─────────┘     └─────────┘
```

Workers that **do** hold weights (e.g. torchrun ranks, NIM multi-node, vLLM TP):

```bash
MODEL_DEPLOY_MODE=sharded
SHARD_RANK=1
SHARD_WORLD_SIZE=4
CUDA_VISIBLE_DEVICES=0,1
TENSOR_PARALLEL_SIZE=2
PIPELINE_PARALLEL_SIZE=2
LOAD_MODEL_WEIGHTS=true
HF_DEVICE_MAP=auto
```

## 3) Hybrid (default — sharded replicas)

Default for this repo: assume advanced models need multi-GPU/node sharding,
and allow multiple such meshes for throughput.

```bash
# MODEL_DEPLOY_MODE=hybrid   # default if unset
# Two independent full copies; each copy is multi-GPU internally
BACKEND_URLS=http://replica-a-router:8000/v1/chat/completions,http://replica-b-router:8000/v1/chat/completions
TENSOR_PARALLEL_SIZE=8
HF_DEVICE_MAP=auto
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
```


## Common environment variables

```bash
# Cluster / app nodes
MULTI_NODE=true
CLUSTER_NAME=model-serving
CLUSTER_PEERS=http://node1:8080,http://node2:8080
NODE_ID=api-1
NODE_ROLE=all          # all | api | worker | prefill | decode
HOST=0.0.0.0
PORT=8080
PUBLIC_URL=https://models.example.com
WORKERS=2
THREADS=8

# Shared state
SHARED_DATA_DIR=/data/model-serving
REDIS_URL=redis://redis:6379/0
USE_REDIS=true
SECRET_KEY=replace-with-long-random-string
STICKY_SESSIONS=true

# --- LLM placement ---
MODEL_DEPLOY_MODE=hybrid       # hybrid (default) | sharded | replica


# Multi-replica (full instances)
BACKEND_URLS=http://nim-a:8000/v1/chat/completions,http://nim-b:8000/v1/chat/completions
BACKEND_STRATEGY=round_robin

# Single-model sharding
TENSOR_PARALLEL_SIZE=1
PIPELINE_PARALLEL_SIZE=1
MODEL_PARALLEL_SIZE=1
SHARD_RANK=0
SHARD_WORLD_SIZE=1
SHARD_GROUP=default
CUDA_VISIBLE_DEVICES=0,1
HF_DEVICE_MAP=auto
HF_MAX_MEMORY_PER_GPU=20GiB
HF_MAX_MEMORY_CPU=64GiB
MODEL_SHARD_BACKENDS=
MODEL_COORDINATOR_URL=
LOAD_MODEL_WEIGHTS=true
```

## Architecture (replicas + optional shards)

```
                   ┌─────────────────┐
  Clients ────────►│  Load Balancer  │  sticky for Streamlit/WebRTC
                   └────────┬────────┘
            ┌───────────────┼───────────────┐
            ▼               ▼               ▼
        ┌───────┐       ┌───────┐       ┌───────┐
        │ App 1 │       │ App 2 │       │ App 3 │
        └───┬───┘       └───┬───┘       └───┬───┘
            └───────────────┼───────────────┘
                            ▼
              ┌──────────────────────────┐
              │ Shared volume / Redis    │
              └────────────┬─────────────┘
           ┌───────────────┴───────────────┐
           ▼                               ▼
    ┌─────────────┐                 ┌─────────────┐
    │ Replica A   │                 │ Replica B   │  full models
    │ (maybe TP)  │                 │ (maybe TP)  │
    └─────────────┘                 └─────────────┘
```

## Docker Compose

```bash
docker compose -f deploy/docker-compose.multinode.yml up -d
```

See also `deploy/env.multinode.example`.

## Kubernetes (sketch)

- **Replicas**: `Deployment` with `replicas: N`; Service + Ingress; session affinity for Streamlit/WebRTC
- **Sharded model**: one `StatefulSet` (or Job with torchrun) per logical model; ranks via `SHARD_RANK` / pod ordinal; optional separate API Deployment with `LOAD_MODEL_WEIGHTS=false` and `MODEL_COORDINATOR_URL`
- Mount PVC at `SHARED_DATA_DIR` or use Redis
- Set `NODE_ID` from pod name

## Health

`GET /health` includes:

```json
{
  "node": {
    "node_id": "api-1",
    "multi_node": true,
    "model_deploy_mode": "sharded",
    "tensor_parallel_size": 4,
    "shard_rank": 0,
    "shard_world_size": 4
  },
  "backends": { "urls": ["..."], "mode": "multi_replica" },
  "model_placement": {
    "deploy_mode": "sharded",
    "is_sharded": true,
    "is_multi_replica": false,
    "loads_weights_locally": true,
    "device_map": "auto"
  }
}
```

## Code entry points

- `get_multinode_config()` — cluster + placement-related env
- `get_model_placement()` — replica vs sharded details + HF `from_pretrained` kwargs
- `ClusterInfo.next_backend()` — coordinator → replica pool → shard gateway
- `ClusterInfo.health()` — full topology for ops


## HF apps (vLLM-only)

- Require `BACKEND_URLS`, `VLLM_API_URL`, or `MODEL_COORDINATOR_URL`
- `LOAD_MODEL_WEIGHTS=false` (defaulted in app code)
- Multi-replica: `python deploy/scripts/generate_vllm_compose.py`
- Switchyard (hf-ide-assistant): each tier points at its own vLLM `backend_urls`
