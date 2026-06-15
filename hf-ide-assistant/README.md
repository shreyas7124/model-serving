# HuggingFace IDE Coding Assistant

An OpenAI-compatible API server that connects HuggingFace models to IDEs like Cline and Cursor for coding assistance.

## Features

- 🔌 OpenAI-compatible API endpoints
- 🤗 Powered by HuggingFace Transformers
- 💻 Works with Cline, Cursor, and other OpenAI-compatible tools
- 🔑 API key authentication
- 📝 Local model inference
- 🚀 GPU acceleration support

## Prerequisites

- Python 3.8+
- NVIDIA GPU (optional, but recommended for better performance)
- CUDA toolkit (if using GPU)
- Cline or Cursor IDE extension

## Setup

### 1. Install Python Dependencies

```bash
cd hf-ide-assistant
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

The server will start on port 8081 and display configuration details.

## IDE Configuration

### For Cline (VS Code Extension)

1. Open VS Code Settings
2. Search for "Cline"
3. Configure:
   - **API Provider**: OpenAI Compatible
   - **Base URL**: `http://localhost:8081/v1`
   - **API Key**: `hf-coding-assistant-key` (or your custom key from .env)
   - **Model**: `microsoft/DialoGPT-medium` (or your configured model)

### For Cursor

1. Open Cursor Settings (Cmd/Ctrl + ,)
2. Go to "Models" section
3. Add Custom Model:
   - **Provider**: OpenAI Compatible
   - **Base URL**: `http://localhost:8081/v1`
   - **API Key**: `hf-coding-assistant-key`
   - **Model ID**: `microsoft/DialoGPT-medium`

### For Continue (VS Code Extension)

Edit `~/.continue/config.json`:

```json
{
  "models": [
    {
      "title": "HuggingFace Local",
      "provider": "openai",
      "model": "microsoft/DialoGPT-medium",
      "apiBase": "http://localhost:8081/v1",
      "apiKey": "hf-coding-assistant-key"
    }
  ]
}
```

## API Endpoints

### List Models
```bash
GET /v1/models
Authorization: Bearer hf-coding-assistant-key
```

### Chat Completions
```bash
POST /v1/chat/completions
Authorization: Bearer hf-coding-assistant-key
Content-Type: application/json

{
  "model": "microsoft/DialoGPT-medium",
  "messages": [
    {"role": "user", "content": "Write a Python function to sort a list"}
  ]
}
```

### Text Completions
```bash
POST /v1/completions
Authorization: Bearer hf-coding-assistant-key
Content-Type: application/json

{
  "model": "microsoft/DialoGPT-medium",
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

- `HF_MODEL_NAME`: HuggingFace model to use (default: microsoft/DialoGPT-medium)
- `API_KEY`: API key for authentication (change in production!)
- `MAX_TOKENS`: Maximum tokens per response (default: 2048)
- `TEMPERATURE`: Response randomness 0.0-1.0 (default: 0.7)
- `CONTEXT_WINDOW`: Maximum context window size (default: 4096)

### Understanding Context Window vs Max Tokens

**CONTEXT_WINDOW**: The maximum total size (in tokens) of the conversation history that the model can "see" at once. This includes:
- System messages
- Previous user messages
- Previous assistant responses
- Current user message

**MAX_TOKENS**: The maximum number of tokens the model can generate in a single response. This is per-request and controls the length of each individual answer.

**Example**:
- `CONTEXT_WINDOW=4096` - Model can see up to 4,096 tokens of conversation history
- `MAX_TOKENS=2048` - Each response can be up to 2,048 tokens long

### Adjusting Context and Token Limits

Configure these in the `.env` file:

```bash
# For longer code generation (requires more memory)
MAX_TOKENS=4096          # Longer individual responses
CONTEXT_WINDOW=8192      # More conversation history

# For faster, shorter responses (recommended for CPU)
MAX_TOKENS=1024          # Shorter individual responses
CONTEXT_WINDOW=2048      # Less conversation history
```

**Important Notes**:
- MAX_TOKENS should always be less than CONTEXT_WINDOW
- These are server-side defaults; IDEs can override them per-request
- Larger values require more memory; adjust based on your hardware
- HuggingFace models typically have smaller native context windows than NIM models
- DialoGPT models work best with smaller context windows (2048-4096)

## Recommended Models for Coding

**Conversational Models:**
- `microsoft/DialoGPT-medium` - 345M parameters (default, good for general chat)
- `microsoft/DialoGPT-large` - 762M parameters

**Code-Specific Models:**
- `Salesforce/codegen-350M-mono` - Specialized for code generation
- `Salesforce/codegen-2B-mono` - Larger code generation model
- `bigcode/starcoder` - 15B parameters (requires significant resources)

**Instruction-Following Models:**
- `meta-llama/Llama-2-7b-chat-hf` - 7B parameters (requires HF token)
- `mistralai/Mistral-7B-Instruct-v0.1` - 7B parameters

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
2. Check the port is not in use: `lsof -i :8081`

### Authentication Errors

1. Check API key matches in `.env` and IDE configuration
2. Ensure Authorization header is being sent

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

### Slow Responses

- Use GPU if available
- Try a smaller model
- Reduce conversation history length

## Performance Tips

- **GPU Acceleration**: The app automatically uses GPU if available
- **Model Size**: Start with smaller models and scale up based on your hardware
- **Memory Management**: Monitor memory usage, especially with larger models
- **Conversation History**: Clear conversation history periodically for better performance

## Security Notes

- **Change the API_KEY** in production
- Use HTTPS in production environments
- Consider implementing rate limiting
- Restrict access to localhost or trusted networks

## Port Configuration

This application runs on port **8081** by default (different from NIM version on 8080). You can change this in `server.py` if needed.

## Comparison with NIM Version

**HuggingFace Version:**
- ✅ Fully local, no external dependencies
- ✅ Free and open source
- ✅ Wide model selection
- ⚠️ Requires more local resources
- ⚠️ May be slower without GPU

**NIM Version:**
- ✅ Optimized inference performance
- ✅ Better GPU utilization
- ✅ Enterprise-grade models
- ⚠️ Requires NVIDIA GPU
- ⚠️ Requires NGC account
