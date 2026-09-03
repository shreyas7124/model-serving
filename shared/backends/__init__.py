"""OpenAI-compatible HTTP backends (vLLM, NIM, etc.)."""
from .openai_chat import (
    OpenAIChatClient,
    chat_completions,
    extract_assistant_text,
    normalize_chat_url,
    require_backend_urls,
    resolve_vllm_backend_urls,
)

__all__ = [
    "OpenAIChatClient",
    "chat_completions",
    "extract_assistant_text",
    "normalize_chat_url",
    "require_backend_urls",
    "resolve_vllm_backend_urls",
]
