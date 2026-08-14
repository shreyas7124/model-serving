# Model Serving Applications

This repository contains 8 different applications for locally hosted open-weight model serving with various interfaces and capabilities.

## Applications Overview

1. **nim-streamlit-chat** - NIM model with Streamlit chat interface
2. **hf-streamlit-chat** - HuggingFace model with Streamlit chat interface
3. **nim-webrtc-voice** - NIM model with WebRTC voice interface
4. **hf-webrtc-voice** - HuggingFace model with WebRTC voice interface
5. **nim-streamlit-video** - NIM model with Streamlit video interface (live/pre-recorded)
6. **hf-streamlit-video** - HuggingFace model with Streamlit video interface (live/pre-recorded)
7. **nim-ide-assistant** - NIM model for Cline/Cursor coding assistance
8. **hf-ide-assistant** - HuggingFace model for Cline/Cursor coding assistance

## Features

All applications include:
- Optional user authentication
- Chat/session history tracking
- Local model serving
- Easy setup and deployment

## Prerequisites

- Python 3.8+
- Docker (for NIM models)
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
