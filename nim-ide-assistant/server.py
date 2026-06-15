"""
NIM Model IDE Coding Assistant - OpenAI Compatible API Server
Can be connected to Cline or Cursor for coding assistance
"""
from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import os
import sys
from datetime import datetime

# Add parent directory to path for shared modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.auth import AuthManager
from shared.database import ChatHistory

app = Flask(__name__)
CORS(app)

# Configuration
NIM_API_URL = os.getenv("NIM_API_URL", "http://localhost:8000/v1/chat/completions")
MODEL_NAME = os.getenv("NIM_MODEL_NAME", "meta/llama-3.1-8b-instruct")
API_KEY = os.getenv("API_KEY", "nim-coding-assistant-key")

# Model Parameters
DEFAULT_MAX_TOKENS = int(os.getenv("MAX_TOKENS", "4096"))
DEFAULT_TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))
CONTEXT_WINDOW = int(os.getenv("CONTEXT_WINDOW", "8192"))

# Initialize managers
auth_manager = AuthManager()
chat_history = ChatHistory()

def verify_api_key(request):
    """Verify API key from request"""
    auth_header = request.headers.get('Authorization', '')
    if auth_header.startswith('Bearer '):
        token = auth_header[7:]
        return token == API_KEY
    return False

@app.route('/v1/models', methods=['GET'])
def list_models():
    """List available models (OpenAI compatible)"""
    if not verify_api_key(request):
        return jsonify({'error': 'Invalid API key'}), 401
    
    return jsonify({
        'object': 'list',
        'data': [
            {
                'id': MODEL_NAME,
                'object': 'model',
                'created': int(datetime.now().timestamp()),
                'owned_by': 'nim'
            }
        ]
    })

@app.route('/v1/chat/completions', methods=['POST'])
def chat_completions():
    """Handle chat completions (OpenAI compatible)"""
    if not verify_api_key(request):
        return jsonify({'error': 'Invalid API key'}), 401
    
    data = request.json
    messages = data.get('messages', [])
    temperature = data.get('temperature', DEFAULT_TEMPERATURE)
    max_tokens = data.get('max_tokens', DEFAULT_MAX_TOKENS)
    stream = data.get('stream', False)
    
    # Add system message for coding assistance if not present
    if not messages or messages[0].get('role') != 'system':
        system_message = {
            'role': 'system',
            'content': 'You are an expert coding assistant. Provide clear, concise, and accurate code solutions. Include explanations when helpful.'
        }
        messages = [system_message] + messages
    
    try:
        # Forward request to NIM API
        nim_payload = {
            'model': MODEL_NAME,
            'messages': messages,
            'temperature': temperature,
            'max_tokens': max_tokens,
            'stream': stream
        }
        
        response = requests.post(NIM_API_URL, json=nim_payload, timeout=60)
        response.raise_for_status()
        
        # Return NIM response
        return jsonify(response.json())
    
    except Exception as e:
        return jsonify({
            'error': {
                'message': str(e),
                'type': 'server_error',
                'code': 'nim_error'
            }
        }), 500

@app.route('/v1/completions', methods=['POST'])
def completions():
    """Handle text completions (OpenAI compatible)"""
    if not verify_api_key(request):
        return jsonify({'error': 'Invalid API key'}), 401
    
    data = request.json
    prompt = data.get('prompt', '')
    temperature = data.get('temperature', DEFAULT_TEMPERATURE)
    max_tokens = data.get('max_tokens', DEFAULT_MAX_TOKENS)
    
    # Convert to chat format
    messages = [
        {
            'role': 'system',
            'content': 'You are an expert coding assistant.'
        },
        {
            'role': 'user',
            'content': prompt
        }
    ]
    
    try:
        nim_payload = {
            'model': MODEL_NAME,
            'messages': messages,
            'temperature': temperature,
            'max_tokens': max_tokens
        }
        
        response = requests.post(NIM_API_URL, json=nim_payload, timeout=60)
        response.raise_for_status()
        
        nim_response = response.json()
        
        # Convert chat response to completion format
        completion_response = {
            'id': nim_response.get('id', 'cmpl-' + str(int(datetime.now().timestamp()))),
            'object': 'text_completion',
            'created': nim_response.get('created', int(datetime.now().timestamp())),
            'model': MODEL_NAME,
            'choices': [
                {
                    'text': nim_response['choices'][0]['message']['content'],
                    'index': 0,
                    'finish_reason': nim_response['choices'][0].get('finish_reason', 'stop')
                }
            ],
            'usage': nim_response.get('usage', {})
        }
        
        return jsonify(completion_response)
    
    except Exception as e:
        return jsonify({
            'error': {
                'message': str(e),
                'type': 'server_error',
                'code': 'nim_error'
            }
        }), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'model': MODEL_NAME})

if __name__ == '__main__':
    print(f"Starting NIM IDE Assistant on port 8080")
    print(f"Model: {MODEL_NAME}")
    print(f"API Key: {API_KEY}")
    print(f"Max Tokens: {DEFAULT_MAX_TOKENS}")
    print(f"Temperature: {DEFAULT_TEMPERATURE}")
    print(f"Context Window: {CONTEXT_WINDOW}")
    print(f"\nConfigure your IDE with:")
    print(f"  Base URL: http://localhost:8080/v1")
    print(f"  API Key: {API_KEY}")
    print(f"  Model: {MODEL_NAME}")
    app.run(host='0.0.0.0', port=8080, debug=True)
