# NIM IDE Coding Assistant

An OpenAI-compatible API server that connects NIM models to IDEs like Cline and Cursor for coding assistance.

## Features

- 🔌 OpenAI-compatible API endpoints
- 🤖 Powered by NVIDIA NIM models
- 💻 Works with Cline, Cursor, and other OpenAI-compatible tools
- 🔑 API key authentication
- 📝 Optimized for coding tasks

## Prerequisites

- Python 3.8+
- Docker (for running NIM container)
- NVIDIA GPU with Docker support (recommended)
- Cline or Cursor IDE extension

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
cd nim-ide-assistant
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

The server will start on port 8080 and display configuration details.

## IDE Configuration

### For Cline (VS Code Extension)

1. Open VS Code Settings
2. Search for "Cline"
3. Configure:
   - **API Provider**: OpenAI Compatible
   - **Base URL**: `http://localhost:8080/v1`
   - **API Key**: `nim-coding-assistant-key` (or your custom key from .env)
   - **Model**: `meta/llama-3.1-8b-instruct` (or your configured model)

### For Cursor

1. Open Cursor Settings (Cmd/Ctrl + ,)
2. Go to "Models" section
3. Add Custom Model:
   - **Provider**: OpenAI Compatible
   - **Base URL**: `http://localhost:8080/v1`
   - **API Key**: `nim-coding-assistant-key`
   - **Model ID**: `meta/llama-3.1-8b-instruct`

### For Continue (VS Code Extension)

Edit `~/.continue/config.json`:

```json
{
  "models": [
    {
      "title": "NIM Llama",
      "provider": "openai",
      "model": "meta/llama-3.1-8b-instruct",
      "apiBase": "http://localhost:8080/v1",
      "apiKey": "nim-coding-assistant-key"
    }
  ]
}
```

## API Endpoints

### List Models
```bash
GET /v1/models
Authorization: Bearer nim-coding-assistant-key
```

### Chat Completions
```bash
POST /v1/chat/completions
Authorization: Bearer nim-coding-assistant-key
Content-Type: application/json

{
  "model": "meta/llama-3.1-8b-instruct",
  "messages": [
    {"role": "user", "content": "Write a Python function to sort a list"}
  ]
}
```

### Text Completions
```bash
POST /v1/completions
Authorization: Bearer nim-coding-assistant-key
Content-Type: application/json

{
  "model": "meta/llama-3.1-8b-instruct",
  "prompt": "def fibonacci(n):",
  "max_tokens": 100
}
```

### Health Check
```bash
GET /health
```

## Configuration

Edit `.env` file:

- `NIM_API_URL`: URL of your NIM API endpoint
- `NIM_MODEL_NAME`: Model name to use
- `API_KEY`: API key for authentication (change in production!)
- `MAX_TOKENS`: Maximum tokens per response (default: 4096)
- `TEMPERATURE`: Response randomness 0.0-1.0 (default: 0.7)
- `CONTEXT_WINDOW`: Maximum context window size (default: 8192)

### Understanding Context Window vs Max Tokens

**CONTEXT_WINDOW**: The maximum total size (in tokens) of the conversation history that the model can "see" at once. This includes:
- System messages
- Previous user messages
- Previous assistant responses
- Current user message

**MAX_TOKENS**: The maximum number of tokens the model can generate in a single response. This is per-request and controls the length of each individual answer.

**Example**:
- `CONTEXT_WINDOW=8192` - Model can see up to 8,192 tokens of conversation history
- `MAX_TOKENS=4096` - Each response can be up to 4,096 tokens long

### Adjusting Context and Token Limits

Configure these in the `.env` file:

```bash
# For longer code generation and more conversation history
MAX_TOKENS=8192          # Longer individual responses
CONTEXT_WINDOW=16384     # More conversation history

# For faster, shorter responses (less memory usage)
MAX_TOKENS=2048          # Shorter individual responses
CONTEXT_WINDOW=4096      # Less conversation history
```

**Important Notes**:
- MAX_TOKENS should always be less than CONTEXT_WINDOW
- These are server-side defaults; IDEs can override them per-request
- Larger values require more GPU memory
- The model's native context limit may be lower than your setting

## Recommended NIM Models for Coding

- **meta/llama-3.1-8b-instruct** - Good balance of speed and quality
- **meta/llama-3.1-70b-instruct** - Higher quality, requires more resources
- **codellama/CodeLlama-34b-Instruct-hf** - Specialized for coding tasks

## Usage Examples

### Code Generation
Ask Cline/Cursor: "Create a REST API with Flask that has CRUD endpoints for a todo list"

### Code Explanation
Select code and ask: "Explain what this function does"

### Debugging
Paste error and ask: "Help me fix this error"

### Refactoring
Select code and ask: "Refactor this to be more efficient"

## Troubleshooting

### Connection Refused

1. Ensure the server is running: `python server.py`
2. Check the port is not in use: `lsof -i :8080`
3. Verify NIM container is running: `docker ps | grep nim`

### Authentication Errors

1. Check API key matches in `.env` and IDE configuration
2. Ensure Authorization header is being sent

### Slow Responses

1. Use a GPU for better performance
2. Try a smaller model
3. Reduce `max_tokens` in requests

### Model Not Found

1. Verify NIM container is running with the correct model
2. Check `NIM_MODEL_NAME` in `.env` matches the container model

## Security Notes

- **Change the API_KEY** in production
- Use HTTPS in production environments
- Consider implementing rate limiting
- Restrict access to localhost or trusted networks

## Performance Tips

- Use GPU for significantly faster inference
- Adjust `max_tokens` based on your needs
- Consider using a smaller model for faster responses
- Cache common requests if possible

## Port Configuration

This application runs on port **8080** by default. You can change this in `server.py` if needed.
