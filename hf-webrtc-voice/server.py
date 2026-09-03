"""
HuggingFace Model with WebRTC Voice Interface - Backend Server (vLLM backend)
Supports optional login and chat history.
"""
from flask import Flask, request, jsonify, session
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import os
import sys
from datetime import timedelta

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.auth import AuthManager
from shared.database import ChatHistory
from shared.backends import OpenAIChatClient, require_backend_urls
from shared.deploy import ensure_model_runtime
from shared.multinode import (
    BackendPool,
    ClusterInfo,
    apply_flask_multinode,
    get_multinode_config,
    run_socketio_app,
)
from shared.tools import get_web_access_tool

os.environ.setdefault("LOAD_MODEL_WEIGHTS", "false")

app = Flask(__name__)
app.config["PERMANENT_SESSION_LIFETIME"] = timedelta(days=7)
CORS(app, supports_credentials=True)

MN_CFG = get_multinode_config(default_port=5001, app_name="hf-webrtc-voice")
apply_flask_multinode(app, MN_CFG)
CLUSTER = ClusterInfo(MN_CFG, app_name="hf-webrtc-voice")

_socketio_kwargs = {"cors_allowed_origins": "*"}
if MN_CFG.use_redis and MN_CFG.redis_url:
    _socketio_kwargs["message_queue"] = MN_CFG.redis_url
socketio = SocketIO(app, **_socketio_kwargs)

MODEL_NAME = os.getenv(
    "HF_MODEL_NAME", os.getenv("VLLM_MODEL", "meta-llama/Llama-3.1-8B-Instruct")
)
DEFAULT_TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))
DEFAULT_MAX_TOKENS = int(os.getenv("MAX_TOKENS", "1024"))
VLLM_API_KEY = os.getenv("VLLM_API_KEY", os.getenv("OPENAI_API_KEY", ""))
REQUEST_TIMEOUT = float(os.getenv("VLLM_TIMEOUT", "300"))

_RUNTIME = ensure_model_runtime(
    "vllm",
    app_name="hf-webrtc-voice",
    model_id=MODEL_NAME,
    deploy_mode=MN_CFG.model_deploy_mode,
    replica_count=int(os.getenv("VLLM_REPLICA_COUNT", "1") or "1"),
    tensor_parallel_size=MN_CFG.tensor_parallel_size,
    existing_urls=MN_CFG.backend_urls or CLUSTER.backend_pool.urls,
    api_key=VLLM_API_KEY,
)
backends = require_backend_urls(_RUNTIME.urls or MN_CFG.backend_urls or CLUSTER.backend_pool.urls)
CLUSTER.backend_pool = BackendPool(backends, strategy=MN_CFG.backend_strategy)
MN_CFG.backend_urls = backends

VLLM_CLIENT = OpenAIChatClient(
    next_url=CLUSTER.next_backend,
    model=MODEL_NAME,
    api_key=VLLM_API_KEY,
    timeout=REQUEST_TIMEOUT,
    mark_success=CLUSTER.backend_pool.mark_success,
    mark_failure=CLUSTER.backend_pool.mark_failure,
    default_temperature=DEFAULT_TEMPERATURE,
    default_max_tokens=DEFAULT_MAX_TOKENS,
)

auth_manager = AuthManager()
chat_history = ChatHistory()
WEB = get_web_access_tool("hf-webrtc-voice")

print(f"vLLM voice backends: {CLUSTER.backend_pool.all()}")
print("Model loaded via remote vLLM (no local weights).")

conversations = {}


@app.route("/api/auth/register", methods=["POST"])
def register():
    data = request.json or {}
    username = data.get("username")
    password = data.get("password")
    if not username or not password:
        return jsonify({"error": "Username and password required"}), 400
    if auth_manager.register_user(username, password):
        return jsonify({"message": "Registration successful"}), 201
    return jsonify({"error": "Username already exists"}), 409


@app.route("/api/auth/login", methods=["POST"])
def login():
    data = request.json or {}
    username = data.get("username")
    password = data.get("password")
    session_token = auth_manager.login(username, password)
    if session_token:
        user_id = auth_manager.validate_session(session_token)
        session["session_token"] = session_token
        session["user_id"] = user_id
        session["username"] = username
        session.permanent = True
        return jsonify(
            {
                "message": "Login successful",
                "session_token": session_token,
                "username": username,
            }
        ), 200
    return jsonify({"error": "Invalid credentials"}), 401


@app.route("/api/auth/logout", methods=["POST"])
def logout():
    session_token = session.get("session_token")
    if session_token:
        auth_manager.logout(session_token)
    session.clear()
    return jsonify({"message": "Logout successful"}), 200


@app.route("/api/auth/guest", methods=["POST"])
def guest_login():
    session["user_id"] = None
    session["username"] = "Guest"
    session.permanent = False
    return jsonify({"message": "Guest session created", "username": "Guest"}), 200


@app.route("/api/conversations", methods=["GET"])
def get_conversations():
    user_id = session.get("user_id")
    if not user_id:
        return jsonify({"error": "Not authenticated"}), 401
    convs = chat_history.get_user_conversations(user_id, "hf-webrtc-voice")
    return jsonify({"conversations": convs}), 200


@app.route("/api/conversations/<int:conv_id>", methods=["GET"])
def get_conversation(conv_id):
    user_id = session.get("user_id")
    if not user_id:
        return jsonify({"error": "Not authenticated"}), 401
    messages = chat_history.get_conversation_history(conv_id)
    return jsonify({"messages": messages}), 200


@app.route("/health", methods=["GET"])
def health():
    return jsonify(
        CLUSTER.health(
            model=MODEL_NAME,
            inference_backend="vllm",
            loads_weights_locally=False,
            web_access=WEB.stats(),
        )
    ), 200


def generate_response(prompt, prior_messages, web_context: str = ""):
    messages = []
    for msg in prior_messages or []:
        role = msg.get("role")
        content = msg.get("content", "")
        if role in ("user", "assistant", "system") and content:
            messages.append({"role": role, "content": content})
    model_prompt = prompt
    if web_context:
        model_prompt = (
            f"{web_context}\n\n"
            f"User question (may reference the URL(s) above):\n{prompt}"
        )
    messages.append({"role": "user", "content": model_prompt})
    try:
        _data, text = VLLM_CLIENT.chat(messages)
        return text or "(empty response)"
    except Exception as e:
        return f"Error: {e}"


@socketio.on("connect")
def handle_connect():
    print("Client connected")
    emit("connected", {"message": "Connected to server"})


@socketio.on("disconnect")
def handle_disconnect():
    print("Client disconnected")
    if request.sid in conversations:
        del conversations[request.sid]


@socketio.on("start_conversation")
def handle_start_conversation(data):
    user_id = session.get("user_id")
    if user_id:
        conv_id = chat_history.create_conversation(user_id, "hf-webrtc-voice")
    else:
        conv_id = None
    conversations[request.sid] = {"id": conv_id, "messages": []}
    emit("conversation_started", {"conversation_id": conv_id})


@socketio.on("voice_message")
def handle_voice_message(data):
    text = (data or {}).get("text", "")
    if not text:
        emit("error", {"message": "No text provided"})
        return

    if request.sid not in conversations:
        handle_start_conversation({})

    conv = conversations[request.sid]
    prior = list(conv["messages"])
    conv["messages"].append({"role": "user", "content": text})

    if conv["id"]:
        chat_history.add_message(conv["id"], "user", text)

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

    response = generate_response(text, prior, web_context=web_context)
    conv["messages"].append({"role": "assistant", "content": response})

    if conv["id"]:
        chat_history.add_message(conv["id"], "assistant", response)

    emit("ai_response", {"text": response})


@socketio.on("text_message")
def handle_text_message(data):
    handle_voice_message(data)


if __name__ == "__main__":
    run_socketio_app(socketio, app, MN_CFG, default_port=5001, debug=not MN_CFG.enabled)
