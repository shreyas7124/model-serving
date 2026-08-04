# HuggingFace WebRTC Voice Interface

A voice-enabled chat interface powered by HuggingFace Transformers with WebRTC, speech recognition, and text-to-speech capabilities.

## Features

- 🎤 Voice input using Web Speech API
- 🔊 Text-to-speech responses
- 💬 Text chat fallback
- 🔐 Optional user authentication
- 📝 Conversation history (for logged-in users)
- 🌐 Real-time communication with Socket.IO
- 🤗 Local HuggingFace model inference

## Prerequisites

- Python 3.8+
- Modern web browser with Web Speech API support (Chrome, Edge recommended)
- NVIDIA GPU (optional, but recommended for better performance)
- CUDA toolkit (if using GPU)

## Setup

### 1. Install Python Dependencies

```bash
cd hf-webrtc-voice
pip install -r requirements.txt
```

**Note**: The first run will download the model, which may take some time.

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env with your preferred model
```

### 3. Run the Server

```bash
python server.py
```

The server will start on port 5001 (different from NIM version to avoid conflicts).

### 4. Open in Browser

Navigate to `http://localhost:5001` in your web browser.

## Usage

### Login Options

1. **Login**: Use existing credentials
2. **Register**: Create a new account
3. **Guest Mode**: Use without authentication (no history saved)

### Voice Chat

1. Click "Start Recording" to begin voice input
2. Speak your message
3. Click "Stop Recording" when done
4. The AI will respond with both text and voice

### Text Chat

You can also type messages in the text input field as an alternative to voice.

## Browser Compatibility

### Fully Supported
- Google Chrome (Desktop & Mobile)
- Microsoft Edge
- Safari (iOS 14.5+)

### Limited Support
- Firefox (no Web Speech API support)
- Opera

## Configuration

Edit `.env` file:

- `HF_MODEL_NAME`: HuggingFace model to use (default: microsoft/DialoGPT-medium)
- `SECRET_KEY`: Flask secret key for sessions

### Recommended Models

**Small Models (CPU-friendly):**
- `microsoft/DialoGPT-small` - 117M parameters
- `microsoft/DialoGPT-medium` - 345M parameters (default)
- `facebook/blenderbot-400M-distill` - 400M parameters

**Larger Models (GPU recommended):**
- `microsoft/DialoGPT-large` - 762M parameters
- `meta-llama/Llama-2-7b-chat-hf` - 7B parameters (requires HF token)
- `mistralai/Mistral-7B-Instruct-v0.1` - 7B parameters

## Architecture

### Backend (Flask + Socket.IO + HuggingFace)
- Handles authentication
- Manages chat history
- Local model inference with Transformers
- Real-time message handling

### Frontend (HTML/CSS/JavaScript)
- Web Speech API for voice recognition
- Speech Synthesis API for text-to-speech
- Socket.IO for real-time communication
- Responsive UI

## Troubleshooting

### Voice Recognition Not Working

1. Ensure you're using a supported browser (Chrome/Edge recommended)
2. Grant microphone permissions when prompted
3. Use HTTPS (required for production)
4. Check browser console for errors

### Model Loading Issues

If the model fails to load:
```bash
# Pre-download the model
python -c "from transformers import AutoModelForCausalLM; AutoModelForCausalLM.from_pretrained('microsoft/DialoGPT-medium')"
```

### Out of Memory Errors

1. Use a smaller model (e.g., DialoGPT-small)
2. Ensure you have enough RAM/VRAM
3. Close other applications

### Slow Response Times

- Use GPU if available
- Try a smaller model
- Reduce conversation history length

## Performance Tips

- **GPU Acceleration**: The app automatically uses GPU if available
- **Model Size**: Start with smaller models and scale up based on your hardware
- **Memory Management**: Monitor memory usage, especially with larger models

## Security Notes

- Change the `SECRET_KEY` in production
- Use HTTPS in production environments
- Implement rate limiting for production use
- Consider adding CSRF protection

## Port Configuration

This application runs on port **5001** by default to avoid conflicts with the NIM voice interface (port 5000). You can change this in `server.py` if needed.




## Internet access tool

When a user message contains `http://` or `https://` URLs, the app:

1. Fetches each page
2. Converts HTML to readable text for the model (may truncate for the prompt)
3. **Always logs the URL and the full fetched body** under `WEB_ACCESS_LOG_DIR` (default `shared/database/web_access_logs/`)

### Logs (full content, not just the model snippet)

| Artifact | Location |
|----------|----------|
| Fixed index + full body stream (no rotation) | `WEB_ACCESS_LOG_DIR/<app>-web-access.log` |
| Per-URL full content files | `WEB_ACCESS_LOG_DIR/pages/<app>/*.txt` |

Each per-URL file includes the URL, status, title, SHA-256, and the **complete** extracted text.

**Log full ⇒ web access terminates** (no rotating backups). Clear logs or raise `WEB_ACCESS_LOG_MAX_SIZE` (e.g. `2TB`) to resume after restart.

### Config

```bash
WEB_ACCESS_ENABLED=true
WEB_ACCESS_LOG_DIR=shared/database/web_access_logs
WEB_ACCESS_LOG_MAX_SIZE=1TB              # e.g. 50GB, 1TB, 2GiB; access stops when full
WEB_ACCESS_TIMEOUT=20
WEB_ACCESS_MAX_BYTES=2000000
WEB_ACCESS_MAX_PROMPT_CHARS=12000
WEB_ACCESS_MAX_URLS=5
```

Disable with `WEB_ACCESS_ENABLED=false`.

## Multi-Node Deployment

This app supports horizontal multi-node deployment via `shared.multinode`.

### Quick enable

```bash
# In .env (see also deploy/env.multinode.example)
MULTI_NODE=true
CLUSTER_NAME=model-serving
NODE_ID=hf-voice-1
HOST=0.0.0.0
PORT=5001
PUBLIC_URL=https://models.example.com
SHARED_DATA_DIR=/data/model-serving
REDIS_URL=redis://redis:6379/0
USE_REDIS=true
SECRET_KEY=replace-with-long-random-string
# Optional: multiple model backends
BACKEND_URLS=http://nim-a:8000/v1/chat/completions,http://nim-b:8000/v1/chat/completions
BACKEND_STRATEGY=round_robin
```

### Architecture notes

| Concern | Approach |
|--------|----------|
| Shared auth / chat history | Mount the same `SHARED_DATA_DIR` on every node (NFS/EFS) or set `AUTH_DB_PATH` / `CHAT_DB_PATH` |
| Load balancing | Put nginx/ALB/ingress in front; enable **sticky sessions** for Streamlit & WebRTC |
| Socket.IO fan-out | Set `REDIS_URL` + `USE_REDIS=true` so all voice nodes share a message queue |
| Model backends | `BACKEND_URLS` round-robins NIM / OpenAI-compatible upstreams |
| Health | `GET /health` (APIs) or sidebar "Cluster / node" (Streamlit) reports `node_id` and peers |

### Docker Compose reference

```bash
# From repo root — example scales NIM IDE behind nginx + Redis
docker compose -f deploy/docker-compose.multinode.yml up -d
```

Full variable reference: [`shared/multinode/README.md`](../shared/multinode/README.md) and [`deploy/env.multinode.example`](../deploy/env.multinode.example).
