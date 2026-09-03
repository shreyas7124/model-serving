# Model Serving Applications - Complete Setup Guide

This guide will help you set up and run all 8 model serving applications.

## Quick Overview

| # | Application | Port | Description |
|---|-------------|------|-------------|
| 1 | nim-streamlit-chat | 8501 | NIM + Streamlit chat interface |
| 2 | hf-streamlit-chat | 8501 | HuggingFace + Streamlit chat interface |
| 3 | nim-webrtc-voice | 5000 | NIM + WebRTC voice interface |
| 4 | hf-webrtc-voice | 5001 | HuggingFace + WebRTC voice interface |
| 5 | nim-streamlit-video | 8502 | NIM + Streamlit video interface (Coming Soon) |
| 6 | hf-streamlit-video | 8503 | HuggingFace + Streamlit video interface (Coming Soon) |
| 7 | nim-ide-assistant | 8080 | NIM IDE coding assistant |
| 8 | hf-ide-assistant | 8081 | HuggingFace IDE coding assistant |

## Prerequisites

### Common Requirements
- Python 3.8 or higher
- pip (Python package manager)
- Git

### For NIM Applications (1, 3, 5, 7)
- Docker
- NVIDIA GPU with Docker support
- NVIDIA Container Toolkit
- NGC API Key (get from https://catalog.ngc.nvidia.com/)

### For HuggingFace Applications (2, 4, 6, 8)
- **vLLM** server(s) (required; see deploy/docker-compose.vllm.yml)
- NVIDIA GPU for vLLM hosts
- HF apps themselves are CPU-friendly API/UI clients

### For Voice Applications (3, 4)
- Modern web browser (Chrome or Edge recommended)
- Microphone access

## Installation Steps

### 1. Clone the Repository

```bash
git clone https://github.com/7sg-ai/model-serving.git
cd model-serving
```

### 2. Set Up NIM Container (for NIM applications)

```bash
# Pull and run NIM container
docker run -d \
  --gpus all \
  --name nim-llama \
  -p 8000:8000 \
  -e NGC_API_KEY=your_ngc_api_key_here \
  nvcr.io/nim/meta/llama-3.1-8b-instruct:latest

# Verify it's running
docker ps | grep nim

# Check logs
docker logs nim-llama
```

### 3. Install Python Dependencies

Each application has its own requirements. You can install them individually or create a virtual environment for each.

#### Option A: Individual Installation

```bash
# For nim-streamlit-chat
cd nim-streamlit-chat
pip install -r requirements.txt
cd ..

# For hf-streamlit-chat
cd hf-streamlit-chat
pip install -r requirements.txt
cd ..

# Repeat for other applications...
```

#### Option B: Using Virtual Environments (Recommended)

```bash
# For nim-streamlit-chat
cd nim-streamlit-chat
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
deactivate
cd ..

# Repeat for other applications...
```

## Running the Applications

### Application 1: NIM Streamlit Chat

```bash
cd nim-streamlit-chat
cp .env.example .env
# Edit .env if needed
streamlit run app.py
```

Access at: http://localhost:8501

### Application 2: HuggingFace Streamlit Chat

```bash
cd hf-streamlit-chat
cp .env.example .env
# Edit .env if needed
streamlit run app.py
```

Access at: http://localhost:8501

### Application 3: NIM WebRTC Voice

```bash
cd nim-webrtc-voice
cp .env.example .env
# Edit .env if needed
python server.py
```

Access at: http://localhost:5000

### Application 4: HuggingFace WebRTC Voice

```bash
cd hf-webrtc-voice
cp .env.example .env
# Edit .env if needed
python server.py
```

Access at: http://localhost:5001

### Application 7: NIM IDE Assistant

```bash
cd nim-ide-assistant
cp .env.example .env
# Edit .env if needed
python server.py
```

Configure your IDE with:
- Base URL: http://localhost:8080/v1
- API Key: nim-coding-assistant-key

### Application 8: HuggingFace IDE Assistant

```bash
cd hf-ide-assistant
cp .env.example .env
# Edit .env if needed
python server.py
```

Configure your IDE with:
- Base URL: http://localhost:8081/v1
- API Key: hf-coding-assistant-key

## Configuration

### Environment Variables

Each application has a `.env.example` file. Copy it to `.env` and customize:

```bash
cp .env.example .env
```

Common variables:
- `NIM_API_URL`: URL of NIM API (default: http://localhost:8000/v1/chat/completions)
- `NIM_MODEL_NAME`: NIM model name (default: meta/llama-3.1-8b-instruct)
- `HF_MODEL_NAME / BACKEND_URLS (vLLM)`: HuggingFace model name (default: microsoft/DialoGPT-medium)
- `API_KEY`: API key for IDE assistants
- `SECRET_KEY`: Flask secret key for voice applications

### Shared Database

All applications share the same authentication and chat history databases located in:
- `shared/database/users.db` - User authentication
- `shared/database/chat_history.db` - Conversation history

These are created automatically on first run.

## Troubleshooting

### NIM Container Issues

```bash
# Check if container is running
docker ps | grep nim

# View logs
docker logs nim-llama

# Restart container
docker restart nim-llama

# Stop and remove container
docker stop nim-llama
docker rm nim-llama
```

### Port Already in Use

```bash
# Find process using port (e.g., 8501)
lsof -i :8501

# Kill the process
kill -9 <PID>
```

### Model Download Issues

HuggingFace models are downloaded on first run. If download fails:

```bash
# Pre-download model
python -c "from transformers import AutoModelForCausalLM; AutoModelForCausalLM.from_pretrained('microsoft/DialoGPT-medium')"
```

### GPU Not Detected

```bash
# Check NVIDIA driver
nvidia-smi

# Check CUDA availability in Python
python -c "import torch; print(torch.cuda.is_available())"
```

### Permission Issues

```bash
# Fix permissions for shared directory
chmod -R 755 shared/
```

## Security Considerations

### For Development
- Default API keys are fine for local development
- Databases are stored locally

### For Production
1. Change all API keys in `.env` files
2. Use strong SECRET_KEY values
3. Enable HTTPS
4. Implement rate limiting
5. Use proper authentication
6. Restrict network access
7. Regular security updates

## Performance Optimization

### For Better Performance
1. Use NVIDIA GPU when available
2. Adjust model sizes based on your hardware
3. Use smaller models for faster responses
4. Monitor memory usage
5. Close unnecessary applications

### Recommended Hardware
- **Minimum**: 16GB RAM, 4-core CPU
- **Recommended**: 32GB RAM, 8-core CPU, NVIDIA GPU with 8GB+ VRAM
- **Optimal**: 64GB RAM, 16-core CPU, NVIDIA GPU with 24GB+ VRAM

## Next Steps

1. **Explore Each Application**: Try out different interfaces
2. **Customize Models**: Experiment with different models
3. **Integrate with IDEs**: Set up Cline or Cursor
4. **Build Your Own**: Use these as templates for your projects

## Getting Help

- Check individual README files in each application directory
- Review error logs in terminal
- Ensure all prerequisites are installed
- Verify environment variables are set correctly

## License

MIT License - See LICENSE file for details
