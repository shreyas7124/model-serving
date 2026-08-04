"""
Shared response cache and conversation memory for IDE assistants.

Speeds up repeated / similar coding requests and keeps rolling
conversation context within the configured window.
"""
from __future__ import annotations

import hashlib
import json
import threading
import time
from collections import OrderedDict
from typing import Any, Dict, List, Optional


class ResponseCache:
    """Thread-safe LRU cache with TTL for chat/completion responses."""

    def __init__(
        self,
        max_size: int = 256,
        ttl_seconds: int = 3600,
        enabled: bool = True,
    ):
        self.max_size = max(1, max_size)
        self.ttl_seconds = max(1, ttl_seconds)
        self.enabled = enabled
        self._store: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self._lock = threading.RLock()
        self.hits = 0
        self.misses = 0

    @staticmethod
    def make_key(
        model: str,
        messages: List[Dict[str, Any]],
        temperature: float,
        max_tokens: int,
        extra: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Build a stable cache key from request parameters."""
        payload = {
            "model": model,
            "messages": messages,
            "temperature": round(float(temperature), 4),
            "max_tokens": int(max_tokens),
        }
        if extra:
            payload["extra"] = extra
        raw = json.dumps(payload, sort_keys=True, ensure_ascii=True, default=str)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def get(self, key: str) -> Optional[Any]:
        if not self.enabled:
            return None

        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                self.misses += 1
                return None

            if time.time() - entry["ts"] > self.ttl_seconds:
                del self._store[key]
                self.misses += 1
                return None

            # Move to end (most recently used)
            self._store.move_to_end(key)
            self.hits += 1
            return entry["value"]

    def set(self, key: str, value: Any) -> None:
        if not self.enabled:
            return

        with self._lock:
            if key in self._store:
                self._store.move_to_end(key)
            self._store[key] = {"value": value, "ts": time.time()}

            while len(self._store) > self.max_size:
                self._store.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._store.clear()
            self.hits = 0
            self.misses = 0

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            total = self.hits + self.misses
            hit_rate = (self.hits / total) if total else 0.0
            return {
                "enabled": self.enabled,
                "size": len(self._store),
                "max_size": self.max_size,
                "ttl_seconds": self.ttl_seconds,
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": round(hit_rate, 4),
            }


class ConversationMemory:
    """
    In-memory conversation store with context-window trimming.

    Keeps system messages and the most recent turns that fit within
    the configured token budget (approx chars/4 when no tokenizer).
    """

    def __init__(self, context_window: int = 1_048_576, max_conversations: int = 128):
        self.context_window = max(1024, context_window)
        self.max_conversations = max(1, max_conversations)
        self._conversations: OrderedDict[str, List[Dict[str, str]]] = OrderedDict()
        self._lock = threading.RLock()

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """Rough token estimate without a tokenizer (~4 chars/token)."""
        if not text:
            return 0
        return max(1, len(text) // 4)

    def _message_tokens(self, message: Dict[str, str]) -> int:
        return self.estimate_tokens(message.get("content", "")) + 4  # role overhead

    def get(self, conversation_id: str) -> List[Dict[str, str]]:
        with self._lock:
            messages = self._conversations.get(conversation_id)
            if messages is None:
                return []
            self._conversations.move_to_end(conversation_id)
            return list(messages)

    def set(self, conversation_id: str, messages: List[Dict[str, str]]) -> None:
        with self._lock:
            trimmed = self.trim_to_context(messages)
            if conversation_id in self._conversations:
                self._conversations.move_to_end(conversation_id)
            self._conversations[conversation_id] = trimmed
            while len(self._conversations) > self.max_conversations:
                self._conversations.popitem(last=False)

    def append(
        self,
        conversation_id: str,
        role: str,
        content: str,
    ) -> List[Dict[str, str]]:
        with self._lock:
            history = list(self._conversations.get(conversation_id, []))
            history.append({"role": role, "content": content})
            trimmed = self.trim_to_context(history)
            self._conversations[conversation_id] = trimmed
            self._conversations.move_to_end(conversation_id)
            while len(self._conversations) > self.max_conversations:
                self._conversations.popitem(last=False)
            return list(trimmed)

    def trim_to_context(
        self,
        messages: List[Dict[str, str]],
        reserve_for_response: int = 0,
    ) -> List[Dict[str, str]]:
        """
        Keep system messages + newest turns within the context budget.

        reserve_for_response: tokens reserved for the upcoming completion
        so prompt + completion stay under CONTEXT_WINDOW.
        """
        if not messages:
            return []

        budget = max(512, self.context_window - max(0, reserve_for_response))

        system_msgs = [m for m in messages if m.get("role") == "system"]
        other_msgs = [m for m in messages if m.get("role") != "system"]

        system_tokens = sum(self._message_tokens(m) for m in system_msgs)
        remaining = budget - system_tokens
        if remaining <= 0:
            # Extreme case: only keep the last system message
            return system_msgs[-1:] if system_msgs else []

        kept_reversed: List[Dict[str, str]] = []
        used = 0
        for msg in reversed(other_msgs):
            cost = self._message_tokens(msg)
            if used + cost > remaining:
                break
            kept_reversed.append(msg)
            used += cost

        kept = list(reversed(kept_reversed))
        return system_msgs + kept

    def clear(self, conversation_id: Optional[str] = None) -> None:
        with self._lock:
            if conversation_id is None:
                self._conversations.clear()
            elif conversation_id in self._conversations:
                del self._conversations[conversation_id]

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "conversations": len(self._conversations),
                "max_conversations": self.max_conversations,
                "context_window": self.context_window,
            }
