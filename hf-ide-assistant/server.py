"""
HuggingFace Model IDE Coding Assistant - OpenAI Compatible API Server
Can be connected to Cline or Cursor for coding assistance
"""
from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
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
MODEL_NAME = os.getenv("HF_MODEL_NAME", "microsoft/DialoGPT-medium")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
API_KEY = os.getenv("API_KEY", "hf-coding-assistant-key")

# Model Parameters
DEFAULT_MAX_TOKENS = int(os.getenv("MAX_TOKENS", "2048"))
DEFAULT_TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))
CONTEXT_WINDOW = int(os.getenv("CONTEXT_WINDOW", "4096"))

# Initialize managers
auth_manager = AuthManager()
chat_history = ChatHistory()

# Load model
print(f"Loading model {MODEL_NAME} on {DEVICE}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
model.to(DEVICE)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Model loaded successfully!")

# Store conversation histories
conversation_histories = {}

def verify_api_key(request):
    """Verify API key from request"""
    auth_header = request.headers.get('Authorization', '')
    if auth_header.startswith('Bearer '):
        token = auth_header[7:]
        return token == API_KEY
    return False

def generate_response(prompt, conversation_id=None):
    """Generate response using HuggingFace model"""
    try:
        # Get or create conversation history
        if conversation_id and conversation_id in conversation_histories:
            chat_history_ids = conversation_histories[conversation_id]
        else:
            chat_history_ids = None
        
        new_input_ids = tokenizer.encode(
            prompt + tokenizer.eos_token, 
            return_tensors='pt'
        ).to(DEVICE)
        
        bot_input_ids = torch.cat(
            [chat_history_ids, new_input_ids], 
            dim=-1
        ) if chat_history_ids is not None else new_input_ids
        
        new_chat_history_ids = model.generate(
            bot_input_ids,
            max_length=2000,
            pad_token_id=tokenizer.eos_token_id,
            no_repeat_ngram_size=3,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.7
        )
        
        response = tokenizer.decode(
            new_chat_history_ids[:, bot_input_ids.shape[-1]:][0],
            skip_special_tokens=True
        )
        
        # Store conversation history
        if conversation_id:
            conversation_histories[conversation_id] = new_chat_history_ids
        
        return response
    except Exception as e:
        return f"Error: {str(e)}"

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
                'owned_by': 'huggingface'
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
    
    # Extract the last user message
    user_message = ""
    for msg in reversed(messages):
        if msg.get('role') == 'user':
            user_message = msg.get('content', '')
            break
    
    if not user_message:
        return jsonify({'error': 'No user message found'}), 400
    
    # Generate conversation ID from request
    conversation_id = request.headers.get('X-Conversation-ID', 'default')
    
    # Generate response
    response_text = generate_response(user_message, conversation_id)
    
    # Return OpenAI-compatible response
    return jsonify({
        'id': 'chatcmpl-' + str(int(datetime.now().timestamp())),
        'object': 'chat.completion',
        'created': int(datetime.now().timestamp()),
        'model': MODEL_NAME,
        'choices': [
            {
                'index': 0,
                'message': {
                    'role': 'assistant',
                    'content': response_text
                },
                'finish_reason': 'stop'
            }
        ],
        'usage': {
            'prompt_tokens': 0,
            'completion_tokens': 0,
            'total_tokens': 0
        }
    })

@app.route('/v1/completions', methods=['POST'])
def completions():
    """Handle text completions (OpenAI compatible)"""
    if not verify_api_key(request):
        return jsonify({'error': 'Invalid API key'}), 401
    
    data = request.json
    prompt = data.get('prompt', '')
    
    if not prompt:
        return jsonify({'error': 'No prompt provided'}), 400
    
    # Generate response
    response_text = generate_response(prompt)
    
    # Return OpenAI-compatible response
    return jsonify({
        'id': 'cmpl-' + str(int(datetime.now().timestamp())),
        'object': 'text_completion',
        'created': int(datetime.now().timestamp()),
        'model': MODEL_NAME,
        'choices': [
            {
                'text': response_text,
                'index': 0,
                'finish_reason': 'stop'
            }
        ],
        'usage': {
            'prompt_tokens': 0,
            'completion_tokens': 0,
            'total_tokens': 0
        }
    })

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model': MODEL_NAME,
        'device': DEVICE
    })

if __name__ == '__main__':
    print(f"Starting HuggingFace IDE Assistant on port 8081")
    print(f"Model: {MODEL_NAME}")
    print(f"Device: {DEVICE}")
    print(f"API Key: {API_KEY}")
    print(f"Max Tokens: {DEFAULT_MAX_TOKENS}")
    print(f"Temperature: {DEFAULT_TEMPERATURE}")
    print(f"Context Window: {CONTEXT_WINDOW}")
    print(f"\nConfigure your IDE with:")
    print(f"  Base URL: http://localhost:8081/v1")
    print(f"  API Key: {API_KEY}")
    print(f"  Model: {MODEL_NAME}")
    app.run(host='0.0.0.0', port=8081, debug=True)
