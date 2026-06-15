"""
NIM Model with WebRTC Voice Interface - Backend Server
Supports optional login and chat history
"""
from flask import Flask, request, jsonify, session
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import requests
import os
import sys
from datetime import timedelta

# Add parent directory to path for shared modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.auth import AuthManager
from shared.database import ChatHistory

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'your-secret-key-change-this')
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=7)
CORS(app, supports_credentials=True)
socketio = SocketIO(app, cors_allowed_origins="*")

# Configuration
NIM_API_URL = os.getenv("NIM_API_URL", "http://localhost:8000/v1/chat/completions")
MODEL_NAME = os.getenv("NIM_MODEL_NAME", "meta/llama-3.1-8b-instruct")

# Initialize managers
auth_manager = AuthManager()
chat_history = ChatHistory()

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
    
    convs = chat_history.get_user_conversations(user_id, "nim-webrtc-voice")
    return jsonify({'conversations': convs}), 200

@app.route('/api/conversations/<int:conv_id>', methods=['GET'])
def get_conversation(conv_id):
    """Get specific conversation history"""
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({'error': 'Not authenticated'}), 401
    
    messages = chat_history.get_conversation_history(conv_id)
    return jsonify({'messages': messages}), 200

def query_nim_model(messages):
    """Query the NIM model API"""
    try:
        payload = {
            "model": MODEL_NAME,
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 512
        }
        
        response = requests.post(NIM_API_URL, json=payload, timeout=30)
        response.raise_for_status()
        
        result = response.json()
        return result['choices'][0]['message']['content']
    except Exception as e:
        return f"Error: {str(e)}"

@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    print('Client connected')
    emit('connected', {'message': 'Connected to server'})

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    print('Client disconnected')

@socketio.on('start_conversation')
def handle_start_conversation(data):
    """Start a new conversation"""
    user_id = session.get('user_id')
    
    # Create conversation in database if user is logged in
    if user_id:
        conv_id = chat_history.create_conversation(user_id, "nim-webrtc-voice")
    else:
        conv_id = None
    
    # Store conversation in memory
    conversations[request.sid] = {
        'id': conv_id,
        'messages': []
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
    
    # Get AI response
    response = query_nim_model(conv['messages'])
    
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
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)
