"""
HuggingFace Model IDE Coding Assistant - OpenAI Compatible API Server
Can be connected to Cline or Cursor for coding assistance

Tuned for complicated coding workloads:
  - 1M context window (trimmed server-side to model capacity)
  - High max-token responses for multi-file generation
  - Response caching + conversation memory for faster follow-ups
"""
from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
import sys
import copy
from datetime import datetime

# Add parent directory to path for shared modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.auth import AuthManager
from shared.database import ChatHistory
from shared.cache import (
    ConversationMemory,
    build_response_cache_for_model,
    is_moonshot_model,
)
from shared.multinode import (
    ClusterInfo,
    apply_flask_multinode,
    describe_placement_for_logs,
    get_model_placement,
    get_multinode_config,
    run_flask_app,
)



app = Flask(__name__)
CORS(app)

# Multi-node / cluster configuration
MN_CFG = get_multinode_config(default_port=8081, app_name="hf-ide-assistant")
apply_flask_multinode(app, MN_CFG)
CLUSTER = ClusterInfo(MN_CFG, app_name="hf-ide-assistant")


# Configuration
MODEL_NAME = os.getenv("HF_MODEL_NAME", "moonshotai/Kimi-K3")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
API_KEY = os.getenv("API_KEY", "hf-coding-assistant-key")

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

CODING_SYSTEM_PROMPT = (
    "You are an expert coding assistant optimized for complex, multi-file software "
    "engineering tasks. Provide clear, correct, production-quality code. Prefer "
    "complete solutions with necessary imports, error handling, and brief explanations. "
    "When editing existing code, preserve style and only change what is required."
)

# Initialize managers
auth_manager = AuthManager()
chat_history = ChatHistory()
# Moonshot / Kimi models → Mooncake L2 KV store (+ local L1); others → L1 only
response_cache = build_response_cache_for_model(
    MODEL_NAME,
    max_size=CACHE_MAX_SIZE,
    ttl_seconds=CACHE_TTL_SECONDS,
    enabled=CACHE_ENABLED,
)
conversation_memory = ConversationMemory(
    context_window=CONTEXT_WINDOW,
    max_conversations=MEMORY_MAX_CONVERSATIONS,
)
USING_MOONSHOT = is_moonshot_model(MODEL_NAME)


# Model placement: multi-replica and/or single-model multi-GPU/node sharding
PLACEMENT = get_model_placement(replica_backends=MN_CFG.backend_urls)
PLACEMENT.apply_cuda_visible_devices()
print(f"LLM placement: {describe_placement_for_logs(PLACEMENT)}")

# Load model (Kimi-K3 and other large models need device_map / dtype / max_memory)
print(f"Loading model {MODEL_NAME} on {DEVICE}...")
_trust_remote = os.getenv("HF_TRUST_REMOTE_CODE", "true").lower() in ("1", "true", "yes")
_torch_dtype_name = (
    PLACEMENT.torch_dtype
    or os.getenv("HF_TORCH_DTYPE", "bfloat16" if DEVICE == "cuda" else "float32")
)
_dtype_map = {
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float16": torch.float16,
    "fp16": torch.float16,
    "float32": torch.float32,
    "fp32": torch.float32,
    "auto": "auto",
}
_torch_dtype = _dtype_map.get(
    str(_torch_dtype_name).lower(),
    torch.bfloat16 if DEVICE == "cuda" else torch.float32,
)
# Prefer placement device_map (sharded multi-GPU); fall back to env / cuda auto
_device_map = PLACEMENT.device_map or os.getenv(
    "HF_DEVICE_MAP", "auto" if DEVICE == "cuda" else None
)
if _device_map and str(_device_map).lower() in ("none", "null", ""):
    _device_map = None

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=_trust_remote,
)
_load_kwargs = {
    "trust_remote_code": _trust_remote,
    "torch_dtype": _torch_dtype,
}
_load_kwargs.update(PLACEMENT.hf_from_pretrained_kwargs())
# Ensure device_map from placement/env wins if set
if _device_map:
    _load_kwargs["device_map"] = _device_map
elif "device_map" not in _load_kwargs and DEVICE == "cuda":
    _load_kwargs["device_map"] = "auto"

model = None
if PLACEMENT.loads_weights_locally:
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, **_load_kwargs)
    if _load_kwargs.get("device_map") is None:
        model.to(DEVICE)
    model.eval()
else:
    print(
        "LOAD_MODEL_WEIGHTS=false or coordinator set — "
        "this node proxies to remote shards/replicas (no local weights)."
    )


if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


# Effective context is the minimum of configured window and model capacity
_model_max_length = getattr(tokenizer, "model_max_length", CONTEXT_WINDOW) or CONTEXT_WINDOW
# Some tokenizers report an absurdly large model_max_length; clamp sensibly
if _model_max_length > 10_000_000:
    _model_max_length = CONTEXT_WINDOW
EFFECTIVE_CONTEXT_WINDOW = max(512, min(CONTEXT_WINDOW, int(_model_max_length)))

_cache_stats = response_cache.stats()
print(
    f"Model loaded successfully! "
    f"(configured context={CONTEXT_WINDOW}, effective={EFFECTIVE_CONTEXT_WINDOW})"
)
print(
    f"Cache backend: {_cache_stats.get('backend', 'l1')}"
    + (
        f" (Mooncake for Moonshot/Kimi)"
        if USING_MOONSHOT
        else ""
    )
)
if _cache_stats.get("mooncake_error"):
    print(f"Mooncake note: {_cache_stats['mooncake_error']}")



def verify_api_key(req):
    """Verify API key from request"""
    auth_header = req.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        token = auth_header[7:]
        return token == API_KEY
    return False


def messages_to_prompt(messages):
    """Flatten chat messages into a single prompt string for causal LMs."""
    parts = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "system":
            parts.append(f"System: {content}")
        elif role == "assistant":
            parts.append(f"Assistant: {content}")
        else:
            parts.append(f"User: {content}")
    parts.append("Assistant:")
    return "\n".join(parts)


def ensure_system_message(messages):
    if not messages or messages[0].get("role") != "system":
        return [{"role": "system", "content": CODING_SYSTEM_PROMPT}] + list(messages)
    return list(messages)


def merge_with_memory(conversation_id, messages, max_tokens):
    """Merge client messages with memory and trim to effective context."""
    messages = ensure_system_message(messages)

    # Use effective (model-aware) window for trimming
    original_window = conversation_memory.context_window
    conversation_memory.context_window = EFFECTIVE_CONTEXT_WINDOW
    try:
        if conversation_id:
            stored = conversation_memory.get(conversation_id)
            if stored and len(messages) <= len(stored):
                known = {(m.get("role"), m.get("content")) for m in stored}
                merged = list(stored)
                for msg in messages:
                    key = (msg.get("role"), msg.get("content"))
                    if key not in known and msg.get("role") != "system":
                        merged.append(
                            {"role": msg["role"], "content": msg.get("content", "")}
                        )
                messages = merged

        trimmed = conversation_memory.trim_to_context(
            messages, reserve_for_response=max_tokens
        )
        if conversation_id:
            conversation_memory.set(conversation_id, trimmed)
        return trimmed
    finally:
        conversation_memory.context_window = original_window


def _model_input_device():
    """Resolve the device for input tensors (handles device_map='auto')."""
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device(DEVICE)


def generate_response(messages, max_tokens, temperature):
    """Generate a completion from chat messages using the local HF model."""
    if model is None:
        raise RuntimeError(
            "No local model weights loaded. Set LOAD_MODEL_WEIGHTS=true "
            "or point BACKEND_URLS / MODEL_COORDINATOR_URL at a sharded serving mesh."
        )
    prompt = messages_to_prompt(messages)


    # Token budget: leave room for generation within effective context
    max_new = max(1, min(int(max_tokens), EFFECTIVE_CONTEXT_WINDOW - 64))
    max_input_len = max(64, EFFECTIVE_CONTEXT_WINDOW - max_new)

    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_input_len,
    )
    target_device = _model_input_device()
    input_ids = encoded["input_ids"].to(target_device)
    attention_mask = encoded.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(target_device)


    prompt_tokens = int(input_ids.shape[-1])

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            no_repeat_ngram_size=3,
            do_sample=temperature > 0,
            top_k=50,
            top_p=0.95,
            temperature=max(temperature, 1e-5) if temperature > 0 else 1.0,
            use_cache=True,  # KV cache for faster decoding
        )

    generated = output_ids[:, prompt_tokens:]
    response = tokenizer.decode(generated[0], skip_special_tokens=True).strip()
    completion_tokens = int(generated.shape[-1])

    return response, prompt_tokens, completion_tokens


def remember_exchange(conversation_id, messages, assistant_content):
    if not conversation_id or not assistant_content:
        return
    history = list(messages)
    history.append({"role": "assistant", "content": assistant_content})

    original_window = conversation_memory.context_window
    conversation_memory.context_window = EFFECTIVE_CONTEXT_WINDOW
    try:
        conversation_memory.set(conversation_id, history)
    finally:
        conversation_memory.context_window = original_window

    try:
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
                    "owned_by": "huggingface",
                    "context_window": CONTEXT_WINDOW,
                    "effective_context_window": EFFECTIVE_CONTEXT_WINDOW,
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
    temperature = float(data.get("temperature", DEFAULT_TEMPERATURE))
    max_tokens = int(data.get("max_tokens", DEFAULT_MAX_TOKENS))
    max_tokens = max(1, min(max_tokens, EFFECTIVE_CONTEXT_WINDOW - 64))

    conversation_id = (
        request.headers.get("X-Conversation-ID")
        or data.get("user")
        or data.get("conversation_id")
        or "default"
    )

    if not messages:
        return jsonify({"error": "No messages provided"}), 400

    messages = merge_with_memory(conversation_id, messages, max_tokens)

    cache_key = response_cache.make_key(
        MODEL_NAME, messages, temperature, max_tokens
    )
    cached = response_cache.get(cache_key)
    if cached is not None:
        result = copy.deepcopy(cached)
        result["cached"] = True
        return jsonify(result)

    try:
        response_text, prompt_tokens, completion_tokens = generate_response(
            messages, max_tokens, temperature
        )
    except Exception as e:
        return (
            jsonify(
                {
                    "error": {
                        "message": str(e),
                        "type": "server_error",
                        "code": "hf_error",
                    }
                }
            ),
            500,
        )

    remember_exchange(conversation_id, messages, response_text)

    result = {
        "id": "chatcmpl-" + str(int(datetime.now().timestamp())),
        "object": "chat.completion",
        "created": int(datetime.now().timestamp()),
        "model": MODEL_NAME,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": response_text},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
        "cached": False,
    }

    response_cache.set(cache_key, result)
    return jsonify(result)


@app.route("/v1/completions", methods=["POST"])
def completions():
    """Handle text completions (OpenAI compatible)"""
    if not verify_api_key(request):
        return jsonify({"error": "Invalid API key"}), 401

    data = request.json or {}
    prompt = data.get("prompt", "")
    temperature = float(data.get("temperature", DEFAULT_TEMPERATURE))
    max_tokens = int(data.get("max_tokens", DEFAULT_MAX_TOKENS))
    max_tokens = max(1, min(max_tokens, EFFECTIVE_CONTEXT_WINDOW - 64))

    if not prompt:
        return jsonify({"error": "No prompt provided"}), 400

    messages = [
        {"role": "system", "content": CODING_SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    original_window = conversation_memory.context_window
    conversation_memory.context_window = EFFECTIVE_CONTEXT_WINDOW
    try:
        messages = conversation_memory.trim_to_context(
            messages, reserve_for_response=max_tokens
        )
    finally:
        conversation_memory.context_window = original_window

    cache_key = response_cache.make_key(
        MODEL_NAME, messages, temperature, max_tokens, extra={"mode": "completion"}
    )
    cached = response_cache.get(cache_key)
    if cached is not None:
        result = copy.deepcopy(cached)
        result["cached"] = True
        return jsonify(result)

    try:
        response_text, prompt_tokens, completion_tokens = generate_response(
            messages, max_tokens, temperature
        )
    except Exception as e:
        return (
            jsonify(
                {
                    "error": {
                        "message": str(e),
                        "type": "server_error",
                        "code": "hf_error",
                    }
                }
            ),
            500,
        )

    result = {
        "id": "cmpl-" + str(int(datetime.now().timestamp())),
        "object": "text_completion",
        "created": int(datetime.now().timestamp()),
        "model": MODEL_NAME,
        "choices": [
            {
                "text": response_text,
                "index": 0,
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
        "cached": False,
    }

    response_cache.set(cache_key, result)
    return jsonify(result)


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint (includes multi-node identity)"""
    return jsonify(
        CLUSTER.health(
            model=MODEL_NAME,
            device=DEVICE,
            context_window=CONTEXT_WINDOW,
            effective_context_window=EFFECTIVE_CONTEXT_WINDOW,
            max_tokens=DEFAULT_MAX_TOKENS,
            temperature=DEFAULT_TEMPERATURE,
            cache=response_cache.stats(),
            memory=conversation_memory.stats(),
            moonshot=USING_MOONSHOT,
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
    print("Starting HuggingFace IDE Assistant")
    print(f"Model: {MODEL_NAME}")
    print(f"Device: {DEVICE}")
    print(f"API Key: {API_KEY}")
    print(f"Max Tokens: {DEFAULT_MAX_TOKENS}")
    print(f"Temperature: {DEFAULT_TEMPERATURE}")
    print(f"Context Window: {CONTEXT_WINDOW} (effective: {EFFECTIVE_CONTEXT_WINDOW})")
    print(f"Cache Enabled: {CACHE_ENABLED} (size={CACHE_MAX_SIZE}, ttl={CACHE_TTL_SECONDS}s)")
    print(f"Cache backend: {response_cache.stats().get('backend', 'l1')}")
    if USING_MOONSHOT:
        print("Moonshot model detected → Mooncake cache preferred (L1 + L2)")
    print(f"Conversation Memory: max {MEMORY_MAX_CONVERSATIONS} sessions")
    base = MN_CFG.public_url or f"http://localhost:{MN_CFG.node_port or 8081}"
    print("\nConfigure your IDE with:")
    print(f"  Base URL: {base}/v1")
    print(f"  API Key: {API_KEY}")
    print(f"  Model: {MODEL_NAME}")
    try:
        run_flask_app(app, MN_CFG, default_port=8081, debug=not MN_CFG.enabled)
    finally:
        close = getattr(response_cache, "close", None)
        if callable(close):
            close()


