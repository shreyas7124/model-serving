"""
Export SwitchyardConfig to a NeMo Switchyard routes.toml for switchyard-server.
"""
from __future__ import annotations

import os
import re
from typing import Any, Dict, List

from .config import ModelSpec, SwitchyardConfig


def _toml_str(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _safe_key(name: str) -> str:
    key = re.sub(r"[^A-Za-z0-9_]", "_", name or "model")
    if key and key[0].isdigit():
        key = "m_" + key
    return key or "model"


def export_routes_toml(cfg: SwitchyardConfig) -> str:
    """
    Build a routes.toml string compatible with switchyard-server.
    """
    lines: List[str] = [
        "# Auto-generated from model-serving Switchyard config",
        "# https://github.com/NVIDIA-NeMo/Switchyard",
        "schema_version = 1",
        "",
    ]

    # Group models by openai base URL → llm_clients
    clients: Dict[str, Dict[str, Any]] = {}
    target_client: Dict[str, str] = {}

    for m in cfg.models:
        base = m.openai_base_url() or "http://127.0.0.1:8000/v1"
        client_key = _safe_key(
            re.sub(r"https?://", "", base).replace("/", "_").replace(".", "_").replace(":", "_")
        )
        if client_key not in clients:
            clients[client_key] = {
                "format": m.format or "openai_chat",
                "base_url": base,
                "api_key_env": m.api_key_env or "",
            }
        target_client[m.name] = client_key

    for ckey, cval in clients.items():
        lines.append(f"[llm_clients.{ckey}]")
        lines.append(f"format = {_toml_str(cval['format'])}")
        lines.append(f"base_url = {_toml_str(cval['base_url'])}")
        if cval.get("api_key_env"):
            lines.append(f"api_key_env = {_toml_str(cval['api_key_env'])}")
        lines.append("max_retries = 2")
        lines.append("")

    for m in cfg.models:
        tkey = _safe_key(m.name)
        lines.append(f"[targets.{tkey}]")
        lines.append(f"id = {_toml_str(m.id)}")
        lines.append(f"llm_client = {_toml_str(target_client[m.name])}")
        if m.extra_body:
            # simple flat extra_body
            parts = []
            for k, v in m.extra_body.items():
                if isinstance(v, bool):
                    parts.append(f"{k} = {'true' if v else 'false'}")
                elif isinstance(v, (int, float)):
                    parts.append(f"{k} = {v}")
                else:
                    parts.append(f"{k} = {_toml_str(str(v))}")
            if parts:
                lines.append("extra_body = { " + ", ".join(parts) + " }")
        lines.append("")

    weak = cfg.weak()
    strong = cfg.strong()
    judge = cfg.judge()
    route_key = _safe_key(cfg.route_id.replace("/", "_"))

    if cfg.strategy == "escalation" and weak and strong:
        lines.append(f"[routes.{route_key}]")
        lines.append(f"id = {_toml_str(cfg.route_id)}")
        lines.append('type = "llm_classifier"')
        lines.append('mode = "escalation"')
        lines.append(f"classifier_target = {_toml_str(_safe_key((judge or weak).name))}")
        lines.append(f"strong_target = {_toml_str(_safe_key(strong.name))}")
        lines.append(f"weak_target = {_toml_str(_safe_key(weak.name))}")
        if cfg.escalation.prompt:
            lines.append(f"prompt = {_toml_str(cfg.escalation.prompt)}")
        lines.append(f"max_output_tokens = {cfg.escalation.max_output_tokens}")
        lines.append(
            "escalation = { "
            f"confirmations = {cfg.escalation.confirmations}, "
            f"recent_turn_window = {cfg.escalation.recent_turn_window}, "
            f"window_message_chars = {cfg.escalation.window_message_chars}"
            " }"
        )
        lines.append("")
    elif cfg.strategy == "capability" and weak and strong:
        lines.append(f"[routes.{route_key}]")
        lines.append(f"id = {_toml_str(cfg.route_id)}")
        lines.append('type = "llm_classifier"')
        lines.append('mode = "capability"')
        lines.append(f"classifier_target = {_toml_str(_safe_key((judge or weak).name))}")
        lines.append(f"strong_target = {_toml_str(_safe_key(strong.name))}")
        lines.append(f"weak_target = {_toml_str(_safe_key(weak.name))}")
        lines.append("base_threshold = 0.5")
        lines.append("session_affinity = true")
        lines.append("message_hash_fallback = true")
        lines.append("")
    elif cfg.strategy == "random" and cfg.models:
        names = [_safe_key(m.name) for m in cfg.selectable_models()]
        lines.append(f"[routes.{route_key}]")
        lines.append(f"id = {_toml_str(cfg.route_id)}")
        lines.append('type = "random"')
        lines.append("targets = [" + ", ".join(_toml_str(n) for n in names) + "]")
        lines.append("weights = [" + ", ".join("1" for _ in names) + "]")
        lines.append("")
    elif cfg.models:
        m = cfg.default_model() or cfg.models[0]
        lines.append(f"[routes.{route_key}]")
        lines.append(f"id = {_toml_str(cfg.route_id)}")
        lines.append('type = "passthrough"')
        lines.append(f"target = {_toml_str(_safe_key(m.name))}")
        lines.append("")

    # Also export direct passthrough routes per selectable model
    for m in cfg.selectable_models():
        pkey = _safe_key(f"direct_{m.name}")
        lines.append(f"[routes.{pkey}]")
        lines.append(f"id = {_toml_str(m.id)}")
        lines.append('type = "passthrough"')
        lines.append(f"target = {_toml_str(_safe_key(m.name))}")
        if m.context_window:
            lines.append(f"context_window = {int(m.context_window)}")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def write_routes_toml(cfg: SwitchyardConfig, path: str = "") -> str:
    """
    Write routes.toml to path (or cfg.routes_toml_path). Returns path written.
    """
    path = path or cfg.routes_toml_path
    if not path:
        path = os.path.join(os.getcwd(), "switchyard-routes.toml")
    path = os.path.expanduser(path)
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    content = export_routes_toml(cfg)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return path
