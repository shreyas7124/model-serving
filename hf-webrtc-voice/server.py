"""
HuggingFace Model with WebRTC Voice Interface - Backend Server
Supports optional login and chat history
"""
from flask import Flask, request, jsonify, session
from flask_cors import CORS
from flask_socketio import SocketIO, emit
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
import sys
from datetime import timedelta

# Add parent directory to path for shared modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.auth import AuthManager
from shared.database import ChatHistory
from shared.multinode import (
    ClusterInfo,
    apply_flask_multinode,
    get_multinode_config,
    run_socketio_app,
)
from shared.tools import get_web_access_tool


app = Flask(__name__)
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=7)
CORS(app, supports_credentials=True)

# Multi-node / cluster configuration
MN_CFG = get_multinode_config(default_port=5001, app_name="hf-webrtc-voice")
apply_flask_multinode(app, MN_CFG)
CLUSTER = ClusterInfo(MN_CFG, app_name="hf-webrtc-voice")

_socketio_kwargs = {"cors_allowed_origins": "*"}
if MN_CFG.use_redis and MN_CFG.redis_url:
    _socketio_kwargs["message_queue"] = MN_CFG.redis_url
socketio = SocketIO(app, **_socketio_kwargs)

# Configuration
MODEL_NAME = os.getenv("HF_MODEL_NAME", "microsoft/DialoGPT-medium")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize managers
auth_manager = AuthManager()
chat_history = ChatHistory()
WEB = get_web_access_tool("hf-webrtc-voice")



# Load model
print(f"Loading model {MODEL_NAME} on {DEVICE}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
model.to(DEVICE)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Model loaded successfully!")

# Store active conversations in memory
conversations = {}

@app.route('/api/auth/register', methods=['POST'])
def register():
    """Register a new user"""
    data = request.json
    username = data.get('username')
    password = data.get('password')
    
    if not username or not password:
        return jsonify({'error': 'Username and password required'}), 400
    
    if auth_manager.register_user(username, password):
        return jsonify({'message': 'Registration successful'}), 201
    else:
        return jsonify({'error': 'Username already exists'}), 409

@app.route('/api/auth/login', methods=['POST'])
def login():
    """Login user"""
    data = request.json
    username = data.get('username')
    password = data.get('password')
    
    session_token = auth_manager.login(username, password)
    if session_token:
        user_id = auth_manager.validate_session(session_token)
        session['session_token'] = session_token
        session['user_id'] = user_id
        session['username'] = username
        session.permanent = True
        return jsonify({
            'message': 'Login successful',
            'session_token': session_token,
            'username': username
        }), 200
    else:
        return jsonify({'error': 'Invalid credentials'}), 401

@app.route('/api/auth/logout', methods=['POST'])
def logout():
    """Logout user"""
    session_token = session.get('session_token')
    if session_token:
        auth_manager.logout(session_token)
    session.clear()
    return jsonify({'message': 'Logout successful'}), 200

@app.route('/api/auth/guest', methods=['POST'])
def guest_login():
    """Continue as guest"""
    session['user_id'] = None
    session['username'] = 'Guest'
    session.permanent = False
    return jsonify({'message': 'Guest session created', 'username': 'Guest'}), 200

@app.route('/api/conversations', methods=['GET'])
def get_conversations():
    """Get user's conversation history"""
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({'error': 'Not authenticated'}), 401
    
    convs = chat_history.get_user_conversations(user_id, "hf-webrtc-voice")
    return jsonify({'conversations': convs}), 200

@app.route('/api/conversations/<int:conv_id>', methods=['GET'])
def get_conversation(conv_id):
    """Get specific conversation history"""
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({'error': 'Not authenticated'}), 401
    
    messages = chat_history.get_conversation_history(conv_id)
    return jsonify({'messages': messages}), 200

@app.route('/health', methods=['GET'])
def health():
    """Health check with multi-node identity"""
    return jsonify(
        CLUSTER.health(model=MODEL_NAME, device=DEVICE, web_access=WEB.stats())
    ), 200


def generate_response(prompt, chat_history_ids, web_context: str = ""):
    """Generate response using HuggingFace model"""

    try:
        model_prompt = prompt
        if web_context:
            model_prompt = (
                f"{web_context}\n\n"
                f"User question (may reference the URL(s) above):\n{prompt}"
            )
        new_input_ids = tokenizer.encode(
            model_prompt + tokenizer.eos_token,
            return_tensors='pt'
        ).to(DEVICE)

        
        bot_input_ids = torch.cat(
            [chat_history_ids, new_input_ids], 
            dim=-1
        ) if chat_history_ids is not None else new_input_ids
        
        new_chat_history_ids = model.generate(
            bot_input_ids,
            max_length=1000,
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
        
        return response, new_chat_history_ids
    except Exception as e:
        return f"Error: {str(e)}", chat_history_ids

@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    print('Client connected')
    emit('connected', {'message': 'Connected to server'})

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    print('Client disconnected')
    if request.sid in conversations:
        del conversations[request.sid]

@socketio.on('start_conversation')
def handle_start_conversation(data):
    """Start a new conversation"""
    user_id = session.get('user_id')
    
    # Create conversation in database if user is logged in
    if user_id:
        conv_id = chat_history.create_conversation(user_id, "hf-webrtc-voice")
    else:
        conv_id = None
    
    # Store conversation in memory
    conversations[request.sid] = {
        'id': conv_id,
        'messages': [],
        'chat_history_ids': None
    }
    
    emit('conversation_started', {'conversation_id': conv_id})

@socketio.on('voice_message')
def handle_voice_message(data):
    """Handle incoming voice message (transcribed text)"""
    text = data.get('text', '')
    
    if not text:
        emit('error', {'message': 'No text provided'})
        return
    
    # Get or create conversation
    if request.sid not in conversations:
        handle_start_conversation({})
    
    conv = conversations[request.sid]
    
    # Add user message
    conv['messages'].append({"role": "user", "content": text})
    
    # Save to database if logged in
    if conv['id']:
        chat_history.add_message(conv['id'], "user", text)

    # Internet access: fetch URLs; log full page content (not only model context)
    session_key = str(conv.get("id") or request.sid)
    web_results, web_context = WEB.process_user_text(text, session_id=session_key)
    if web_results:
        emit(
            "web_fetch",
            {
                "urls": [
                    {
                        "url": wr.url,
                        "ok": wr.ok,
                        "status_code": wr.status_code,
                        "title": wr.title,
                        "log_path": wr.log_path,
                        "content_sha256": wr.content_sha256,
                        "raw_length": wr.raw_length,
                        "error": wr.error,
                    }
                    for wr in web_results
                ],
                "log_file": WEB.log_file_path,
            },
        )
    
    # Get AI response
    response, new_history_ids = generate_response(
        text, conv['chat_history_ids'], web_context=web_context
    )
    conv['chat_history_ids'] = new_history_ids
    
    # Add assistant message
    conv['messages'].append({"role": "assistant", "content": response})
    
    # Save to database if logged in
    if conv['id']:
        chat_history.add_message(conv['id'], "assistant", response)
    
    # Send response back to client
    emit('ai_response', {'text': response})


@socketio.on('text_message')
def handle_text_message(data):
    """Handle text message (for testing without voice)"""
    handle_voice_message(data)

if __name__ == '__main__':
    run_socketio_app(socketio, app, MN_CFG, default_port=5001, debug=not MN_CFG.enabled)

