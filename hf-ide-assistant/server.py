"""
HuggingFace Model IDE Coding Assistant - OpenAI Compatible API Server
Can be connected to Cline or Cursor for coding assistance

Tuned for complicated coding workloads:
  - 1M context window (trimmed server-side to model capacity)
  - High max-token responses for multi-file generation
  - Response caching + conversation memory for faster follow-ups
  - NeMo Switchyard multi-model selection + escalation router
"""
from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
import sys
import copy
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

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
from shared.switchyard import (
    ModelSpec,
    get_switchyard_config,
    get_switchyard_router,
    write_routes_toml,
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

# --- Local HF model registry (primary + optional Switchyard local tiers) ---
# Maps model id -> (tokenizer, model, effective_context)
_LOCAL_MODELS: Dict[str, Dict[str, Any]] = {}


def _dtype_from_name(name: str):
    _dtype_map = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
        "auto": "auto",
    }
    return _dtype_map.get(
        str(name or "").lower(),
        torch.bfloat16 if DEVICE == "cuda" else torch.float32,
    )


def _load_hf_model(
    model_id: str,
    *,
    torch_dtype_name: str = "",
    device_map: Optional[str] = None,
    trust_remote: Optional[bool] = None,
    max_memory: Optional[Dict[str, str]] = None,
    placement_kwargs: Optional[Dict[str, Any]] = None,
):
    """Load one HF causal LM; returns (tokenizer, model, effective_context)."""
    _trust = (
        trust_remote
        if trust_remote is not None
        else os.getenv("HF_TRUST_REMOTE_CODE", "true").lower() in ("1", "true", "yes")
    )
    _torch_dtype_name = (
        torch_dtype_name
        or PLACEMENT.torch_dtype
        or os.getenv("HF_TORCH_DTYPE", "bfloat16" if DEVICE == "cuda" else "float32")
    )
    _torch_dtype = _dtype_from_name(_torch_dtype_name)
    _device_map = device_map or PLACEMENT.device_map or os.getenv(
        "HF_DEVICE_MAP", "auto" if DEVICE == "cuda" else None
    )
    if _device_map and str(_device_map).lower() in ("none", "null", ""):
        _device_map = None

    print(f"Loading HF model {model_id} on {DEVICE}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=_trust)
    _load_kwargs: Dict[str, Any] = {
        "trust_remote_code": _trust,
        "torch_dtype": _torch_dtype,
    }
    if placement_kwargs:
        _load_kwargs.update(placement_kwargs)
    else:
        _load_kwargs.update(PLACEMENT.hf_from_pretrained_kwargs())
    if _device_map:
        _load_kwargs["device_map"] = _device_map
    elif "device_map" not in _load_kwargs and DEVICE == "cuda":
        _load_kwargs["device_map"] = "auto"
    if max_memory:
        mm: Dict[Any, str] = {}
        for k, v in max_memory.items():
            try:
                mm[int(k)] = v
            except (TypeError, ValueError):
                mm[k] = v
        _load_kwargs["max_memory"] = mm

    model = AutoModelForCausalLM.from_pretrained(model_id, **_load_kwargs)
    if _load_kwargs.get("device_map") is None:
        model.to(DEVICE)
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    _model_max_length = getattr(tokenizer, "model_max_length", CONTEXT_WINDOW) or CONTEXT_WINDOW
    if _model_max_length > 10_000_000:
        _model_max_length = CONTEXT_WINDOW
    effective = max(512, min(CONTEXT_WINDOW, int(_model_max_length)))
    print(f"Loaded {model_id} (effective_context={effective})")
    return tokenizer, model, effective


# Load primary model (legacy single-model path / default local tier)
tokenizer = None
model = None
EFFECTIVE_CONTEXT_WINDOW = CONTEXT_WINDOW

if PLACEMENT.loads_weights_locally:
    tokenizer, model, EFFECTIVE_CONTEXT_WINDOW = _load_hf_model(MODEL_NAME)
    _LOCAL_MODELS[MODEL_NAME] = {
        "tokenizer": tokenizer,
        "model": model,
        "effective_context": EFFECTIVE_CONTEXT_WINDOW,
    }
else:
    print(
        "LOAD_MODEL_WEIGHTS=false or coordinator set — "
        "this node proxies to remote shards/replicas (no local weights)."
    )

_cache_stats = response_cache.stats()
print(
    f"Primary model ready "
    f"(configured context={CONTEXT_WINDOW}, effective={EFFECTIVE_CONTEXT_WINDOW})"
)
print(
    f"Cache backend: {_cache_stats.get('backend', 'l1')}"
    + (f" (Mooncake for Moonshot/Kimi)" if USING_MOONSHOT else "")
)
if _cache_stats.get("mooncake_error"):
    print(f"Mooncake note: {_cache_stats['mooncake_error']}")


def _parse_max_memory_str(raw: str) -> Dict[str, str]:
    import json as _json

    raw = (raw or "").strip()
    if not raw:
        return {}
    if raw.startswith("{"):
        try:
            return {str(k): str(v) for k, v in _json.loads(raw).items()}
        except Exception:
            return {}
    out = {}
    for part in raw.split(","):
        if "=" in part:
            k, v = part.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def _ensure_local_model(spec: ModelSpec) -> Optional[Dict[str, Any]]:
    """Load a Switchyard local model on demand (independent deploy params)."""
    if spec.id in _LOCAL_MODELS:
        return _LOCAL_MODELS[spec.id]
    if not spec.deployment.load_weights_locally and spec.chat_url():
        return None
    # Apply per-model CUDA visibility only if not already set for process
    cvd = spec.deployment.cuda_visible_devices
    if cvd and "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = cvd
        print(f"CUDA_VISIBLE_DEVICES={cvd} for local model {spec.id}")

    max_mem = _parse_max_memory_str(spec.deployment.max_memory)
    if not max_mem and spec.deployment.max_memory_per_gpu:
        # Best-effort single-entry; full multi-GPU map left to device_map=auto
        max_mem = {"0": spec.deployment.max_memory_per_gpu}
        if spec.deployment.max_memory_cpu:
            max_mem["cpu"] = spec.deployment.max_memory_cpu

    trust = spec.deployment.trust_remote_code
    tok, mdl, eff = _load_hf_model(
        spec.id,
        torch_dtype_name=spec.deployment.torch_dtype,
        device_map=spec.deployment.device_map or None,
        trust_remote=trust,
        max_memory=max_mem or None,
    )
    entry = {"tokenizer": tok, "model": mdl, "effective_context": eff}
    _LOCAL_MODELS[spec.id] = entry
    return entry


# --- Switchyard config (after primary load so local hook can use registry) ---
SY_CFG = get_switchyard_config(
    default_model_id=MODEL_NAME,
    default_backend_url="",
    owned_by="hf-switchyard",
    reload=True,
)

# Auto-mark primary as local weak if Switchyard enabled with local-only models
if SY_CFG.enabled and model is not None:
    for m in SY_CFG.models:
        if m.deployment.load_weights_locally or (
            not m.chat_url() and m.id == MODEL_NAME
        ):
            m.deployment.load_weights_locally = True
            if m.id == MODEL_NAME and MODEL_NAME not in _LOCAL_MODELS:
                _LOCAL_MODELS[MODEL_NAME] = {
                    "tokenizer": tokenizer,
                    "model": model,
                    "effective_context": EFFECTIVE_CONTEXT_WINDOW,
                }


def _local_generate(
    spec: ModelSpec,
    messages: List[Dict[str, Any]],
    temperature: float,
    max_tokens: int,
) -> Tuple[str, Dict[str, Any]]:
    """Switchyard hook: generate with a locally loaded HF model."""
    entry = _ensure_local_model(spec)
    if entry is None:
        # Fall back to primary if ids match
        if spec.id == MODEL_NAME and model is not None:
            entry = _LOCAL_MODELS.get(MODEL_NAME)
        if entry is None:
            raise RuntimeError(
                f"No local weights for '{spec.id}'. Set deployment.load_weights_locally "
                f"or provide backend_url."
            )
    tok = entry["tokenizer"]
    mdl = entry["model"]
    eff_ctx = int(entry.get("effective_context") or EFFECTIVE_CONTEXT_WINDOW)

    prompt = messages_to_prompt(messages)
    max_new = max(1, min(int(max_tokens), eff_ctx - 64))
    max_input_len = max(64, eff_ctx - max_new)
    encoded = tok(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_input_len,
    )
    try:
        target_device = next(mdl.parameters()).device
    except StopIteration:
        target_device = torch.device(DEVICE)
    input_ids = encoded["input_ids"].to(target_device)
    attention_mask = encoded.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(target_device)
    prompt_tokens = int(input_ids.shape[-1])

    with torch.no_grad():
        output_ids = mdl.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new,
            pad_token_id=tok.eos_token_id,
            eos_token_id=tok.eos_token_id,
            no_repeat_ngram_size=3,
            do_sample=temperature > 0,
            top_k=50,
            top_p=0.95,
            temperature=max(temperature, 1e-5) if temperature > 0 else 1.0,
            use_cache=True,
        )
    generated = output_ids[:, prompt_tokens:]
    response = tok.decode(generated[0], skip_special_tokens=True).strip()
    completion_tokens = int(generated.shape[-1])
    return response, {
        "id": "chatcmpl-" + str(int(datetime.now().timestamp())),
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


SY_ROUTER = get_switchyard_router(
    SY_CFG,
    local_generate=_local_generate if (model is not None or SY_CFG.enabled) else None,
    fallback_backend=lambda: CLUSTER.next_backend(),
    fallback_model_id=MODEL_NAME,
    reload=True,
)
if SY_CFG.enabled and SY_CFG.routes_toml_path:
    try:
        path = write_routes_toml(SY_CFG, SY_CFG.routes_toml_path)
        print(f"Switchyard routes.toml written: {path}")
    except Exception as exc:
        print(f"Switchyard routes.toml export failed: {exc}")

# Eager-load additional local Switchyard models
if SY_CFG.enabled:
    for m in SY_CFG.models:
        if m.deployment.load_weights_locally and m.id not in _LOCAL_MODELS:
            try:
                _ensure_local_model(m)
            except Exception as exc:
                print(f"Warning: failed to load local Switchyard model {m.id}: {exc}")


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


def merge_with_memory(conversation_id, messages, max_tokens, context_window=None):
    """Merge client messages with memory and trim to effective context."""
    messages = ensure_system_message(messages)

    # Use effective (model-aware) window for trimming
    window = context_window or EFFECTIVE_CONTEXT_WINDOW
    original_window = conversation_memory.context_window
    conversation_memory.context_window = window
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

    if SY_CFG.enabled:
        payload = SY_ROUTER.list_models_payload()
        for item in payload.get("data", []):
            item.setdefault("created", int(datetime.now().timestamp()))
            if item.get("id") == MODEL_NAME:
                item["effective_context_window"] = EFFECTIVE_CONTEXT_WINDOW
        return jsonify(payload)

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
    requested_model = data.get("model") or MODEL_NAME

    conversation_id = (
        request.headers.get("X-Conversation-ID")
        or data.get("user")
        or data.get("conversation_id")
        or "default"
    )

    if not messages:
        return jsonify({"error": "No messages provided"}), 400

    ctx_window = EFFECTIVE_CONTEXT_WINDOW
    if SY_CFG.enabled:
        spec = SY_ROUTER.resolve_spec(requested_model) or SY_CFG.weak()
        if spec:
            local = _LOCAL_MODELS.get(spec.id)
            if local:
                ctx_window = int(local.get("effective_context") or spec.context_window)
            else:
                ctx_window = min(EFFECTIVE_CONTEXT_WINDOW, spec.context_window or EFFECTIVE_CONTEXT_WINDOW)
            max_tokens = max(1, min(max_tokens, ctx_window - 64))

    messages = merge_with_memory(
        conversation_id, messages, max_tokens, context_window=ctx_window
    )

    cache_model = requested_model or MODEL_NAME
    if SY_CFG.enabled:
        cache_model = f"sy:{SY_CFG.strategy}:{cache_model}"
    cache_key = response_cache.make_key(
        cache_model, messages, temperature, max_tokens
    )
    cached = response_cache.get(cache_key)
    if cached is not None:
        result = copy.deepcopy(cached)
        result["cached"] = True
        return jsonify(result)

    try:
        if SY_CFG.enabled:
            result, response_text, served = SY_ROUTER.chat_completions(
                messages,
                requested_model=requested_model,
                temperature=temperature,
                max_tokens=max_tokens,
                session_id=conversation_id,
            )
            remember_exchange(conversation_id, messages, response_text)
            if isinstance(result, dict):
                result = dict(result)
                result["cached"] = False
            response_cache.set(cache_key, result)
            return jsonify(result)

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
    requested_model = data.get("model") or MODEL_NAME
    conversation_id = (
        request.headers.get("X-Conversation-ID")
        or data.get("user")
        or data.get("conversation_id")
        or "completion"
    )

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

    cache_model = requested_model or MODEL_NAME
    if SY_CFG.enabled:
        cache_model = f"sy:{SY_CFG.strategy}:{cache_model}"
    cache_key = response_cache.make_key(
        cache_model, messages, temperature, max_tokens, extra={"mode": "completion"}
    )
    cached = response_cache.get(cache_key)
    if cached is not None:
        result = copy.deepcopy(cached)
        result["cached"] = True
        return jsonify(result)

    try:
        if SY_CFG.enabled:
            chat_result, response_text, served = SY_ROUTER.chat_completions(
                messages,
                requested_model=requested_model,
                temperature=temperature,
                max_tokens=max_tokens,
                session_id=conversation_id,
            )
            usage = chat_result.get("usage", {}) if isinstance(chat_result, dict) else {}
            model_out = (served.id if served else None) or requested_model or MODEL_NAME
            result = {
                "id": "cmpl-" + str(int(datetime.now().timestamp())),
                "object": "text_completion",
                "created": int(datetime.now().timestamp()),
                "model": model_out,
                "choices": [
                    {
                        "text": response_text,
                        "index": 0,
                        "finish_reason": "stop",
                    }
                ],
                "usage": usage,
                "cached": False,
            }
            if isinstance(chat_result, dict) and chat_result.get("switchyard"):
                result["switchyard"] = chat_result["switchyard"]
            response_cache.set(cache_key, result)
            return jsonify(result)

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
            switchyard=SY_ROUTER.health_block(),
            local_models=list(_LOCAL_MODELS.keys()),
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
        if SY_ROUTER.escalation and request.args.get("session"):
            SY_ROUTER.escalation.reset_session(request.args.get("session"))

    return jsonify(
        {
            "status": "cleared",
            "memory_cleared": clear_memory,
            "cache": response_cache.stats(),
            "memory": conversation_memory.stats(),
            "switchyard": SY_ROUTER.health_block(),
        }
    )


@app.route("/v1/switchyard/session", methods=["DELETE"])
def reset_switchyard_session():
    """Reset escalation latch/streak for a conversation session."""
    if not verify_api_key(request):
        return jsonify({"error": "Invalid API key"}), 401
    if not SY_CFG.enabled or not SY_ROUTER.escalation:
        return jsonify({"error": "Escalation router not active"}), 400
    data = request.json or {}
    session_id = (
        request.headers.get("X-Conversation-ID")
        or data.get("session_id")
        or data.get("conversation_id")
        or request.args.get("session")
        or "default"
    )
    SY_ROUTER.escalation.reset_session(session_id)
    return jsonify({"status": "reset", "session_id": session_id})


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
    if SY_CFG.enabled:
        print(
            f"Switchyard: ON  strategy={SY_CFG.strategy}  route_id={SY_CFG.route_id}  "
            f"models={len(SY_CFG.models)}"
        )
        for m in SY_CFG.models:
            print(
                f"  - [{m.role}] {m.name} id={m.id} "
                f"local={m.deployment.load_weights_locally} "
                f"backend={m.chat_url() or '(local)'} "
                f"deploy={m.deployment.to_dict()}"
            )
    else:
        print("Switchyard: OFF (set SWITCHYARD_ENABLED=true or configure models)")
    base = MN_CFG.public_url or f"http://localhost:{MN_CFG.node_port or 8081}"
    print("\nConfigure your IDE with:")
    print(f"  Base URL: {base}/v1")
    print(f"  API Key: {API_KEY}")
    if SY_CFG.enabled:
        print(f"  Model (route): {SY_CFG.route_id}")
        for m in SY_CFG.selectable_models():
            print(f"  Model (direct): {m.id}")
    else:
        print(f"  Model: {MODEL_NAME}")
    try:
        run_flask_app(app, MN_CFG, default_port=8081, debug=not MN_CFG.enabled)
    finally:
        close = getattr(response_cache, "close", None)
        if callable(close):
            close()
