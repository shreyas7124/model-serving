# Model Serving Applications

This repository contains 8 different applications for locally hosted open-weight model serving with various interfaces and capabilities.

## Applications Overview

1. **nim-streamlit-chat** - NIM model with Streamlit chat interface
2. **hf-streamlit-chat** - Streamlit chat UI over vLLM (HF model weights served by vLLM)
3. **nim-webrtc-voice** - NIM model with WebRTC voice interface
4. **hf-webrtc-voice** - WebRTC voice UI over vLLM
5. **nim-streamlit-video** - NIM model with Streamlit video interface (live/pre-recorded)
6. **hf-streamlit-video** - HuggingFace model with Streamlit video interface (live/pre-recorded)
7. **nim-ide-assistant** - NIM model for Cline/Cursor coding assistance
8. **hf-ide-assistant** - OpenAI-compatible IDE assistant over vLLM (+ Switchyard multi-vLLM)

## Features

All applications include:
- Optional user authentication
- Chat/session history tracking
- Local model serving (NIM containers or vLLM for HF apps)
- Easy setup and deployment

## Prerequisites

- Python 3.8+
- Docker (for NIM and vLLM model servers)
- NVIDIA GPU (recommended for better performance)
- Node.js (for WebRTC applications)

## Quick Start

Each application has its own directory with specific setup instructions. Navigate to the respective directory and follow the README.md file.

## Directory Structure

```
model-serving/
├── nim-streamlit-chat/
├── hf-streamlit-chat/
├── nim-webrtc-voice/
├── hf-webrtc-voice/
├── nim-streamlit-video/
├── hf-streamlit-video/
├── nim-ide-assistant/
├── hf-ide-assistant/
└── shared/
    ├── auth/
    └── database/
```



## Multi-Node Deployment

All applications support multi-node / multi-replica deployment through `shared/multinode`.

- **Shared helpers**: `shared/multinode/` (`MultiNodeConfig`, `ClusterInfo`, backend pool, Flask/Socket.IO runners)
- **Shared state**: `SHARED_DATA_DIR` / `AUTH_DB_PATH` / `CHAT_DB_PATH` (auth + chat history)
- **Redis**: optional `REDIS_URL` for Socket.IO message queues across voice nodes
- **Backends**: `BACKEND_URLS` for multi-NIM round-robin
- **Reference stack**: `deploy/docker-compose.multinode.yml` + `deploy/nginx.multinode.conf`
- **Env template**: `deploy/env.multinode.example`
- **Docs**: [`shared/multinode/README.md`](shared/multinode/README.md)
- **HF / vLLM stack**: `deploy/docker-compose.vllm.yml`, `deploy/env.vllm.example`,
  `deploy/scripts/generate_vllm_compose.py` (replica/hybrid multi-instance)

## Auto-deploy model containers

NIM and HF apps can **start their model servers via Docker Compose** on startup:

- `AUTO_DEPLOY_MODEL=true` (default) — deploy if backends are not already healthy
- `AUTO_DEPLOY_TEARDOWN_ON_EXIT=true` (default) — stop **owned** containers when the app exits
- Engine: NIM apps → NIM images; HF apps → vLLM
- Placement: `MODEL_DEPLOY_MODE`, `NIM_REPLICA_COUNT` / `VLLM_REPLICA_COUNT`, `TENSOR_PARALLEL_SIZE`, `GPU_DEVICES`
- CLI: `python -m shared.deploy up|down|status|generate --engine nim|vllm`
- Pre-existing healthy `BACKEND_URLS` are used as-is and are **not** torn down on exit
- Set `AUTO_DEPLOY_MODEL=false` in production/K8s when an orchestrator owns GPUs
- **Multi physical GPU nodes**: set `DEPLOY_NODES=local,user@gpu1,user@gpu2` with passwordless SSH;
  control node inventories GPUs, packs replicas, `scp` + `docker compose up` on each host.
  See `deploy/env.multinode-ssh.example`. Cross-node TP is not auto-deployed (use coordinator URL).

## HF apps and vLLM

All **hf-*** applications are **vLLM-only** clients (no in-process Transformers/torch weights).
Set `BACKEND_URLS` or `VLLM_API_URL` to one or more OpenAI-compatible vLLM servers.
`MODEL_DEPLOY_MODE=replica|hybrid|sharded` plus `VLLM_REPLICA_COUNT` / `TENSOR_PARALLEL_SIZE`
drive how many vLLM containers the generator creates.

GPU **working-set** KV stays inside vLLM (prefix cache). Optional **Mooncake** on the vLLM
stack is hierarchical overflow / cross-replica share — not a replacement for GPU KV.
The IDE may still use Mooncake as an **app-level response** L2 cache for Moonshot/Kimi ids.

## NeMo Switchyard (IDE assistants)

**nim-ide-assistant** and **hf-ide-assistant** support [NeMo Switchyard](https://github.com/NVIDIA-NeMo/Switchyard)-style multi-model routing:

- Select several models; clients use route id `switchyard/agent` or a specific model id
- **Escalation router**: weak → judge → strong session latch
- **Independent deployment parameters** per model (backends, GPUs, TP, dtype, timeouts, …)
- Optional export to official `switchyard-server` `routes.toml`

Docs: [`shared/switchyard/README.md`](shared/switchyard/README.md)  
Examples: `nim-ide-assistant/switchyard-models.example.json`, `hf-ide-assistant/switchyard-models.example.json`

Each app README has a **Multi-Node Deployment** section with app-specific ports and notes.

## License

MIT
# model-serving
