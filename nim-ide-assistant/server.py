"""
NIM Model IDE Coding Assistant - OpenAI Compatible API Server
Can be connected to Cline or Cursor for coding assistance

Tuned for complicated coding workloads:
  - 1M context window
  - High max-token responses for multi-file generation
  - Response caching + conversation memory for faster follow-ups
"""
from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import os
import sys
import copy
from datetime import datetime

# Add parent directory to path for shared modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.auth import AuthManager
from shared.database import ChatHistory
from shared.cache import ResponseCache, ConversationMemory
from shared.multinode import (
    ClusterInfo,
    apply_flask_multinode,
    get_multinode_config,
    run_flask_app,
)

app = Flask(__name__)
CORS(app)

# Multi-node / cluster configuration
MN_CFG = get_multinode_config(default_port=8080, app_name="nim-ide-assistant")
apply_flask_multinode(app, MN_CFG)
CLUSTER = ClusterInfo(MN_CFG, app_name="nim-ide-assistant")

# Configuration
NIM_API_URL = os.getenv("NIM_API_URL", "http://localhost:8000/v1/chat/completions")
MODEL_NAME = os.getenv("NIM_MODEL_NAME", "meta/llama-3.1-8b-instruct")
API_KEY = os.getenv("API_KEY", "nim-coding-assistant-key")
# Seed backend pool with default single URL when BACKEND_URLS unset
if not CLUSTER.backend_pool.urls:
    CLUSTER.backend_pool = type(CLUSTER.backend_pool)(
        [NIM_API_URL], strategy=MN_CFG.backend_strategy
    )


# Model Parameters — defaults tuned for complicated coding use cases
# CONTEXT_WINDOW: total prompt budget the assistant will retain/trim to (1M)
# MAX_TOKENS: per-response generation limit (long multi-file code answers)
DEFAULT_MAX_TOKENS = int(os.getenv("MAX_TOKENS", "32768"))
DEFAULT_TEMPERATURE = float(os.getenv("TEMPERATURE", "0.3"))
CONTEXT_WINDOW = int(os.getenv("CONTEXT_WINDOW", "1048576"))

# Cache / memory configuration
CACHE_ENABLED = os.getenv("CACHE_ENABLED", "true").lower() in ("1", "true", "yes")
CACHE_MAX_SIZE = int(os.getenv("CACHE_MAX_SIZE", "256"))
CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", "3600"))
MEMORY_MAX_CONVERSATIONS = int(os.getenv("MEMORY_MAX_CONVERSATIONS", "128"))
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "300"))

CODING_SYSTEM_PROMPT = (
    "You are an expert coding assistant optimized for complex, multi-file software "
    "engineering tasks. Provide clear, correct, production-quality code. Prefer "
    "complete solutions with necessary imports, error handling, and brief explanations. "
    "When editing existing code, preserve style and only change what is required."
)

# Initialize managers
auth_manager = AuthManager()
chat_history = ChatHistory()
response_cache = ResponseCache(
    max_size=CACHE_MAX_SIZE,
    ttl_seconds=CACHE_TTL_SECONDS,
    enabled=CACHE_ENABLED,
)
conversation_memory = ConversationMemory(
    context_window=CONTEXT_WINDOW,
    max_conversations=MEMORY_MAX_CONVERSATIONS,
)


def verify_api_key(req):
    """Verify API key from request"""
    auth_header = req.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        token = auth_header[7:]
        return token == API_KEY
    return False


def ensure_system_message(messages):
    """Prepend coding system prompt when the client did not supply one."""
    if not messages or messages[0].get("role") != "system":
        return [{"role": "system", "content": CODING_SYSTEM_PROMPT}] + list(messages)
    return list(messages)


def merge_with_memory(conversation_id, messages, max_tokens):
    """
    Merge client messages with server-side conversation memory and trim
    to CONTEXT_WINDOW, reserving room for the completion.
    """
    messages = ensure_system_message(messages)

    if conversation_id:
        stored = conversation_memory.get(conversation_id)
        if stored:
            # Prefer full client history when it is longer; otherwise extend memory
            if len(messages) <= len(stored):
                # Client may only send the latest turn — append new user msgs
                known = {(m.get("role"), m.get("content")) for m in stored}
                merged = list(stored)
                for msg in messages:
                    key = (msg.get("role"), msg.get("content"))
                    if key not in known and msg.get("role") != "system":
                        merged.append({"role": msg["role"], "content": msg.get("content", "")})
                messages = merged

    trimmed = conversation_memory.trim_to_context(
        messages,
        reserve_for_response=max_tokens,
    )

    if conversation_id:
        conversation_memory.set(conversation_id, trimmed)

    return trimmed


def remember_exchange(conversation_id, messages, assistant_content):
    """Persist the latest user + assistant turn into conversation memory."""
    if not conversation_id or not assistant_content:
        return

    # Start from the trimmed prompt we just used
    history = list(messages)
    history.append({"role": "assistant", "content": assistant_content})
    conversation_memory.set(conversation_id, history)

    # Best-effort durable log (non-fatal if DB fails)
    try:
        # Use a stable numeric id derived from conversation string when possible
        conv_key = abs(hash(conversation_id)) % (10**9)
        chat_history.add_message(conv_key, "assistant", assistant_content)
    except Exception:
        pass


@app.route("/v1/models", methods=["GET"])
def list_models():
    """List available models (OpenAI compatible)"""
    if not verify_api_key(request):
        return jsonify({"error": "Invalid API key"}), 401

    return jsonify(
        {
            "object": "list",
            "data": [
                {
                    "id": MODEL_NAME,
                    "object": "model",
                    "created": int(datetime.now().timestamp()),
                    "owned_by": "nim",
                    "context_window": CONTEXT_WINDOW,
                    "max_tokens": DEFAULT_MAX_TOKENS,
                }
            ],
        }
    )


@app.route("/v1/chat/completions", methods=["POST"])
def chat_completions():
    """Handle chat completions (OpenAI compatible)"""
    if not verify_api_key(request):
        return jsonify({"error": "Invalid API key"}), 401

    data = request.json or {}
    messages = data.get("messages", [])
    temperature = data.get("temperature", DEFAULT_TEMPERATURE)
    max_tokens = int(data.get("max_tokens", DEFAULT_MAX_TOKENS))
    stream = data.get("stream", False)
    conversation_id = (
        request.headers.get("X-Conversation-ID")
        or data.get("user")
        or data.get("conversation_id")
    )

    # Cap max_tokens so prompt + completion fit the context window
    max_tokens = max(1, min(max_tokens, CONTEXT_WINDOW - 512))

    messages = merge_with_memory(conversation_id, messages, max_tokens)

    # Response cache (skip streaming — not cacheable as a single JSON body)
    cache_key = None
    if not stream:
        cache_key = response_cache.make_key(
            MODEL_NAME, messages, temperature, max_tokens
        )
        cached = response_cache.get(cache_key)
        if cached is not None:
            result = copy.deepcopy(cached)
            result["cached"] = True
            return jsonify(result)

    try:
        nim_payload = {
            "model": MODEL_NAME,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream,
        }

        backend = CLUSTER.next_backend() or NIM_API_URL
        try:
            response = requests.post(
                backend, json=nim_payload, timeout=REQUEST_TIMEOUT
            )
            response.raise_for_status()
            CLUSTER.backend_pool.mark_success(backend)
        except Exception:
            CLUSTER.backend_pool.mark_failure(backend)
            raise
        result = response.json()

        # Store assistant reply in memory + cache


        try:
            assistant_content = result["choices"][0]["message"]["content"]
            remember_exchange(conversation_id, messages, assistant_content)
        except (KeyError, IndexError, TypeError):
            pass

        if cache_key is not None:
            response_cache.set(cache_key, result)

        result["cached"] = False
        return jsonify(result)

    except Exception as e:
        return (
            jsonify(
                {
                    "error": {
                        "message": str(e),
                        "type": "server_error",
                        "code": "nim_error",
                    }
                }
            ),
            500,
        )


@app.route("/v1/completions", methods=["POST"])
def completions():
    """Handle text completions (OpenAI compatible)"""
    if not verify_api_key(request):
        return jsonify({"error": "Invalid API key"}), 401

    data = request.json or {}
    prompt = data.get("prompt", "")
    temperature = data.get("temperature", DEFAULT_TEMPERATURE)
    max_tokens = int(data.get("max_tokens", DEFAULT_MAX_TOKENS))
    max_tokens = max(1, min(max_tokens, CONTEXT_WINDOW - 512))

    messages = [
        {"role": "system", "content": CODING_SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    messages = conversation_memory.trim_to_context(
        messages, reserve_for_response=max_tokens
    )

    cache_key = response_cache.make_key(
        MODEL_NAME, messages, temperature, max_tokens, extra={"mode": "completion"}
    )
    cached = response_cache.get(cache_key)
    if cached is not None:
        result = copy.deepcopy(cached)
        result["cached"] = True
        return jsonify(result)

    try:
        nim_payload = {
            "model": MODEL_NAME,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        backend = CLUSTER.next_backend() or NIM_API_URL
        try:
            response = requests.post(
                backend, json=nim_payload, timeout=REQUEST_TIMEOUT
            )
            response.raise_for_status()
            CLUSTER.backend_pool.mark_success(backend)
        except Exception:
            CLUSTER.backend_pool.mark_failure(backend)
            raise

        nim_response = response.json()
        text = nim_response["choices"][0]["message"]["content"]



        completion_response = {
            "id": nim_response.get(
                "id", "cmpl-" + str(int(datetime.now().timestamp()))
            ),
            "object": "text_completion",
            "created": nim_response.get(
                "created", int(datetime.now().timestamp())
            ),
            "model": MODEL_NAME,
            "choices": [
                {
                    "text": text,
                    "index": 0,
                    "finish_reason": nim_response["choices"][0].get(
                        "finish_reason", "stop"
                    ),
                }
            ],
            "usage": nim_response.get("usage", {}),
            "cached": False,
        }

        response_cache.set(cache_key, completion_response)
        return jsonify(completion_response)

    except Exception as e:
        return (
            jsonify(
                {
                    "error": {
                        "message": str(e),
                        "type": "server_error",
                        "code": "nim_error",
                    }
                }
            ),
            500,
        )


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint (includes multi-node identity)"""
    return jsonify(
        CLUSTER.health(
            model=MODEL_NAME,
            context_window=CONTEXT_WINDOW,
            max_tokens=DEFAULT_MAX_TOKENS,
            temperature=DEFAULT_TEMPERATURE,
            cache=response_cache.stats(),
            memory=conversation_memory.stats(),
        )
    )



@app.route("/v1/cache", methods=["DELETE"])
def clear_cache():
    """Clear response cache (and optionally conversation memory)."""
    if not verify_api_key(request):
        return jsonify({"error": "Invalid API key"}), 401

    clear_memory = request.args.get("memory", "false").lower() in (
        "1",
        "true",
        "yes",
    )
    response_cache.clear()
    if clear_memory:
        conversation_memory.clear()

    return jsonify(
        {
            "status": "cleared",
            "memory_cleared": clear_memory,
            "cache": response_cache.stats(),
            "memory": conversation_memory.stats(),
        }
    )


if __name__ == "__main__":
    print("Starting NIM IDE Assistant")
    print(f"Model: {MODEL_NAME}")
    print(f"API Key: {API_KEY}")
    print(f"Max Tokens: {DEFAULT_MAX_TOKENS}")
    print(f"Temperature: {DEFAULT_TEMPERATURE}")
    print(f"Context Window: {CONTEXT_WINDOW}")
    print(f"Cache Enabled: {CACHE_ENABLED} (size={CACHE_MAX_SIZE}, ttl={CACHE_TTL_SECONDS}s)")
    print(f"Conversation Memory: max {MEMORY_MAX_CONVERSATIONS} sessions")
    base = MN_CFG.public_url or f"http://localhost:{MN_CFG.node_port or 8080}"
    print("\nConfigure your IDE with:")
    print(f"  Base URL: {base}/v1")
    print(f"  API Key: {API_KEY}")
    print(f"  Model: {MODEL_NAME}")
    run_flask_app(app, MN_CFG, default_port=8080, debug=not MN_CFG.enabled)

