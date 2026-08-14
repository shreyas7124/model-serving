"""
Unified Switchyard router used by NIM / HF IDE assistants.

Strategies:
  - escalation  — EscalationRouter (weak → judge → strong latch)
  - capability  — one-shot judge picks weak vs strong before answering
  - random      — weighted random among selectable models
  - passthrough — single default / selected model
  - external    — proxy to switchyard-server
"""
from __future__ import annotations

import logging
import random
import time
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

import requests

from .client import ModelClient
from .config import ModelSpec, SwitchyardConfig, get_switchyard_config
from .escalation import EscalationRouter, build_escalation_router

logger = logging.getLogger(__name__)

# local_generate(spec, messages, temperature, max_tokens) -> (text, meta_dict)
LocalGenerateFn = Callable[
    [ModelSpec, List[Dict[str, Any]], float, int],
    Tuple[str, Dict[str, Any]],
]


class SwitchyardRouter:
    def __init__(
        self,
        cfg: SwitchyardConfig,
        *,
        local_generate: Optional[LocalGenerateFn] = None,
        fallback_backend: Optional[Callable[[], Optional[str]]] = None,
        fallback_model_id: str = "",
    ):
        self.cfg = cfg
        self.local_generate = local_generate
        self.fallback_backend = fallback_backend
        self.fallback_model_id = fallback_model_id
        self.escalation: Optional[EscalationRouter] = None
        if cfg.enabled and cfg.strategy == "escalation":
            self.escalation = build_escalation_router(cfg, local_generate=local_generate)
        self._clients: Dict[str, ModelClient] = {}

    def _client(self, spec: ModelSpec) -> ModelClient:
        key = spec.name
        if key not in self._clients:
            self._clients[key] = ModelClient(spec)
        return self._clients[key]

    # ----- listing -----

    def list_models_payload(self) -> Dict[str, Any]:
        created = int(datetime.now().timestamp())
        data: List[Dict[str, Any]] = []

        if self.cfg.enabled:
            # Virtual route id (escalation entrypoint)
            if self.cfg.strategy in ("escalation", "capability", "random"):
                data.append(
                    {
                        "id": self.cfg.route_id,
                        "object": "model",
                        "created": created,
                        "owned_by": self.cfg.owned_by,
                        "root": "switchyard-route",
                        "strategy": self.cfg.strategy,
                        "context_window": (
                            (self.cfg.weak() or self.cfg.default_model() or ModelSpec("x", "x")).context_window
                        ),
                    }
                )
            if self.cfg.expose_models:
                for m in self.cfg.selectable_models():
                    entry = m.to_openai_model_entry(owned_by=self.cfg.owned_by)
                    entry["created"] = created
                    data.append(entry)
        elif self.fallback_model_id:
            data.append(
                {
                    "id": self.fallback_model_id,
                    "object": "model",
                    "created": created,
                    "owned_by": self.cfg.owned_by,
                }
            )

        # Deduplicate by id
        seen = set()
        unique = []
        for item in data:
            if item["id"] in seen:
                continue
            seen.add(item["id"])
            unique.append(item)
        return {"object": "list", "data": unique}

    def health_block(self) -> Dict[str, Any]:
        block: Dict[str, Any] = {
            "enabled": self.cfg.enabled,
            "strategy": self.cfg.strategy if self.cfg.enabled else None,
            "route_id": self.cfg.route_id if self.cfg.enabled else None,
            "models": [m.to_dict() for m in self.cfg.models] if self.cfg.enabled else [],
            "external_url": self.cfg.external_url or None,
        }
        if self.escalation:
            block["escalation"] = self.escalation.stats()
        return block

    # ----- resolution -----

    def resolve_spec(self, requested_model: Optional[str]) -> Optional[ModelSpec]:
        """Pick a ModelSpec for direct (non-route) requests."""
        if not self.cfg.enabled:
            return None
        key = (requested_model or "").strip()
        if not key or key == self.cfg.route_id:
            return None  # means: use strategy route
        if not self.cfg.allow_model_select:
            return None
        return self.cfg.model_by_id_or_name(key)

    def is_route_request(self, requested_model: Optional[str]) -> bool:
        if not self.cfg.enabled:
            return False
        key = (requested_model or "").strip()
        if not key:
            return True  # default to route
        if key == self.cfg.route_id:
            return True
        if key.lower() in ("switchyard", "auto", "escalation", "agent"):
            return True
        # Unknown model with allow_model_select → try direct; else route
        if self.cfg.allow_model_select and self.cfg.model_by_id_or_name(key):
            return False
        return True

    # ----- generation helpers -----

    def _call_spec(
        self,
        spec: ModelSpec,
        messages: List[Dict[str, Any]],
        temperature: float,
        max_tokens: int,
    ) -> Tuple[Dict[str, Any], str]:
        if self.local_generate and (
            spec.deployment.load_weights_locally or not spec.chat_url()
        ):
            text, meta = self.local_generate(spec, messages, temperature, max_tokens)
            result = {
                "id": meta.get("id", f"chatcmpl-{int(time.time())}"),
                "object": "chat.completion",
                "created": int(time.time()),
                "model": spec.id,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": text},
                        "finish_reason": "stop",
                    }
                ],
                "usage": meta.get("usage", {}),
                "switchyard_target": spec.name,
                "switchyard_model_id": spec.id,
            }
            return result, text

        client = self._client(spec)
        if not client.pool.urls:
            # Last resort: fallback_backend URL with this model id
            if self.fallback_backend:
                url = self.fallback_backend()
                if url:
                    # Temporarily use a synthetic client
                    from .config import DeploymentParams

                    tmp = ModelSpec(
                        name=spec.name,
                        id=spec.id,
                        role=spec.role,
                        max_tokens=spec.max_tokens,
                        temperature=spec.temperature,
                        context_window=spec.context_window,
                        request_timeout=spec.request_timeout,
                        extra_body=spec.extra_body,
                        api_key=spec.api_key,
                        api_key_env=spec.api_key_env,
                        deployment=DeploymentParams(backend_urls=[url]),
                    )
                    client = ModelClient(tmp)

        result = client.chat(messages, temperature=temperature, max_tokens=max_tokens)
        return result, client.extract_assistant_text(result)

    def _proxy_external(
        self,
        messages: List[Dict[str, Any]],
        *,
        model: str,
        temperature: float,
        max_tokens: int,
        stream: bool = False,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        base = self.cfg.external_url.rstrip("/")
        url = base
        if not url.endswith("/chat/completions"):
            if url.endswith("/v1"):
                url = url + "/chat/completions"
            else:
                url = url + "/v1/chat/completions"
        payload = {
            "model": model or self.cfg.route_id,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream,
        }
        if extra:
            payload.update(extra)
        headers = {"Content-Type": "application/json"}
        if self.cfg.external_api_key_env:
            import os

            key = os.getenv(self.cfg.external_api_key_env, "")
            if key:
                headers["Authorization"] = f"Bearer {key}"
        resp = requests.post(url, json=payload, headers=headers, timeout=300)
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, dict):
            data.setdefault("switchyard", {"route": "external", "url": base})
        return data

    def _capability_route(
        self,
        messages: List[Dict[str, Any]],
        temperature: float,
        max_tokens: int,
    ) -> Tuple[Dict[str, Any], str, ModelSpec]:
        weak = self.cfg.weak()
        strong = self.cfg.strong()
        judge = self.cfg.judge() or weak
        if not weak:
            raise RuntimeError("No weak/default model configured for capability routing")
        if not strong:
            strong = weak

        # Quick classifier: ask judge if task needs strong
        preview = []

        for m in messages[-6:]:
            c = m.get("content", "")
            if not isinstance(c, str):
                c = str(c)
            preview.append({"role": m.get("role", "user"), "content": c[:500]})
        judge_msgs = [
            {
                "role": "system",
                "content": (
                    "You classify coding tasks. "
                    'Return ONLY JSON: {"verdict":"strong"|"weak","reason":"..."}. '
                    "Use strong for complex multi-file refactors, architecture, hard bugs; "
                    "weak for simple edits, explanations, small snippets."
                ),
            },
            {
                "role": "user",
                "content": "Task messages:\n" + str(preview),
            },
        ]
        pick = strong
        try:
            jclient = self._client(judge)
            jres = jclient.chat(judge_msgs, temperature=0.0, max_tokens=256)
            jtext = jclient.extract_assistant_text(jres).lower()
            if "weak" in jtext and "strong" not in jtext.split("weak")[0][-20:]:
                # crude: if verdict weak
                if '"verdict": "weak"' in jtext or '"verdict":"weak"' in jtext:
                    pick = weak
                elif "verdict" in jtext and "strong" in jtext:
                    pick = strong
                else:
                    pick = weak if "weak" in jtext else strong
            elif "strong" in jtext:
                pick = strong
            else:
                pick = weak
        except Exception as exc:
            logger.warning("Capability judge failed, defaulting to weak: %s", exc)
            pick = weak

        mt = max(1, min(int(max_tokens), pick.context_window - 512))
        resp, text = self._call_spec(pick, messages, temperature, mt)
        resp = dict(resp)
        resp["switchyard"] = {
            "route": "capability",
            "served_by": pick.role,
            "model_id": pick.id,
        }
        return resp, text, pick

    def _random_route(
        self,
        messages: List[Dict[str, Any]],
        temperature: float,
        max_tokens: int,
    ) -> Tuple[Dict[str, Any], str, ModelSpec]:
        models = self.cfg.selectable_models()
        if not models:
            raise RuntimeError("No models configured for random routing")
        pick = random.choice(models)
        mt = max(1, min(int(max_tokens), pick.context_window - 512))
        resp, text = self._call_spec(pick, messages, temperature, mt)
        resp = dict(resp)
        resp["switchyard"] = {
            "route": "random",
            "served_by": pick.name,
            "model_id": pick.id,
        }
        return resp, text, pick

    # ----- main entry -----

    def chat_completions(
        self,
        messages: List[Dict[str, Any]],
        *,
        requested_model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        session_id: str = "default",
        stream: bool = False,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, Any], str, Optional[ModelSpec]]:
        """
        Route a chat completion request.

        Returns (openai_response_dict, assistant_text, serving_ModelSpec|None).
        """
        if not self.cfg.enabled:
            raise RuntimeError("Switchyard is not enabled")

        # External proxy mode
        if self.cfg.strategy == "external" or (
            self.cfg.external_url and self.cfg.strategy == "external"
        ):
            model = requested_model or self.cfg.route_id
            temp = 0.3 if temperature is None else temperature
            mt = 32768 if max_tokens is None else max_tokens
            data = self._proxy_external(
                messages,
                model=model,
                temperature=temp,
                max_tokens=mt,
                stream=stream,
                extra=extra,
            )
            text = ""
            try:
                text = data["choices"][0]["message"]["content"] or ""
            except (KeyError, IndexError, TypeError):
                pass
            return data, text, None

        # Direct model selection
        if not self.is_route_request(requested_model):
            spec = self.resolve_spec(requested_model)
            if spec is None:
                raise RuntimeError(f"Unknown model: {requested_model}")
            temp = spec.temperature if temperature is None else float(temperature)
            mt = spec.max_tokens if max_tokens is None else int(max_tokens)
            mt = max(1, min(mt, spec.context_window - 512))
            resp, text = self._call_spec(spec, messages, temp, mt)
            resp = dict(resp)
            resp["switchyard"] = {
                "route": "direct",
                "served_by": spec.name,
                "model_id": spec.id,
            }
            return resp, text, spec

        # Strategy routes
        strategy = self.cfg.strategy
        if strategy == "escalation":
            if not self.escalation:
                raise RuntimeError("Escalation router not configured (need weak + strong models)")
            # Use weak defaults for unspecified temp/tokens; router applies per-tier
            temp = temperature
            mt = max_tokens
            result = self.escalation.route(
                messages,
                session_id=session_id or "default",
                temperature=temp,
                max_tokens=mt,
            )
            return result.response, result.assistant_text, result.model

        if strategy == "capability":
            temp = float(temperature) if temperature is not None else (
                self.cfg.weak().temperature if self.cfg.weak() else 0.3
            )
            mt = int(max_tokens) if max_tokens is not None else (
                self.cfg.weak().max_tokens if self.cfg.weak() else 32768
            )
            return self._capability_route(messages, temp, mt)

        if strategy == "random":
            temp = float(temperature) if temperature is not None else 0.3
            mt = int(max_tokens) if max_tokens is not None else 32768
            return self._random_route(messages, temp, mt)

        # passthrough
        spec = self.cfg.default_model()
        if not spec:
            raise RuntimeError("No models configured")
        temp = spec.temperature if temperature is None else float(temperature)
        mt = spec.max_tokens if max_tokens is None else int(max_tokens)
        mt = max(1, min(mt, spec.context_window - 512))
        resp, text = self._call_spec(spec, messages, temp, mt)
        resp = dict(resp)
        resp["switchyard"] = {
            "route": "passthrough",
            "served_by": spec.name,
            "model_id": spec.id,
        }
        return resp, text, spec


_ROUTER: Optional[SwitchyardRouter] = None


def get_switchyard_router(
    cfg: Optional[SwitchyardConfig] = None,
    *,
    local_generate: Optional[LocalGenerateFn] = None,
    fallback_backend: Optional[Callable[[], Optional[str]]] = None,
    fallback_model_id: str = "",
    reload: bool = False,
) -> SwitchyardRouter:
    global _ROUTER
    if _ROUTER is not None and not reload and cfg is None:
        return _ROUTER
    cfg = cfg or get_switchyard_config()
    _ROUTER = SwitchyardRouter(
        cfg,
        local_generate=local_generate,
        fallback_backend=fallback_backend,
        fallback_model_id=fallback_model_id,
    )
    return _ROUTER
