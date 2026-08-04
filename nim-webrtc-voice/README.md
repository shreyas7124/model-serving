# NIM WebRTC Voice Interface

A voice-enabled chat interface powered by NVIDIA NIM with WebRTC, speech recognition, and text-to-speech capabilities.

## Features

- 🎤 Voice input using Web Speech API
- 🔊 Text-to-speech responses
- 💬 Text chat fallback
- 🔐 Optional user authentication
- 📝 Conversation history (for logged-in users)
- 🌐 Real-time communication with Socket.IO

## Prerequisites

- Python 3.8+
- Docker (for running NIM container)
- Modern web browser with Web Speech API support (Chrome, Edge recommended)
- NVIDIA GPU with Docker support (recommended)

## Setup

### 1. Start NIM Container

```bash
docker run -d \
  --gpus all \
  --name nim-llama \
  -p 8000:8000 \
  -e NGC_API_KEY=your_ngc_api_key \
  nvcr.io/nim/meta/llama-3.1-8b-instruct:latest
```

### 2. Install Python Dependencies

```bash
cd nim-webrtc-voice
pip install -r requirements.txt
```

### 3. Configure Environment

```bash
cp .env.example .env
# Edit .env with your configuration
```

### 4. Run the Server

```bash
python server.py
```

### 5. Open in Browser

Navigate to `http://localhost:5000` in your web browser.

**Note**: For voice features to work, you need to use HTTPS in production or localhost in development.

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

- `NIM_API_URL`: URL of your NIM API endpoint
- `NIM_MODEL_NAME`: Model name to use
- `SECRET_KEY`: Flask secret key for sessions

## Architecture

### Backend (Flask + Socket.IO)
- Handles authentication
- Manages chat history
- Communicates with NIM API
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

### No Audio Output

1. Check browser audio settings
2. Ensure volume is not muted
3. Try a different browser

### Connection Issues

1. Ensure the Flask server is running
2. Check that port 5000 is not blocked
3. Verify NIM container is running

### CORS Errors

The server is configured to allow all origins. If you encounter CORS issues:
1. Check browser console for specific errors
2. Ensure credentials are being sent with requests

## Security Notes

- Change the `SECRET_KEY` in production
- Use HTTPS in production environments
- Implement rate limiting for production use
- Consider adding CSRF protection

## Performance Tips

- Use a GPU for better NIM performance
- Adjust `max_tokens` in server.py for response length
- Consider implementing response streaming for longer responses




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
NODE_ID=nim-voice-1
HOST=0.0.0.0
PORT=5000
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
