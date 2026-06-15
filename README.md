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

## License

MIT
# model-serving
