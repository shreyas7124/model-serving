"""Backend health probes."""
from __future__ import annotations

import logging
import time
from typing import Iterable, List, Optional
from urllib.parse import urlparse

import requests

logger = logging.getLogger(__name__)


def models_url_from_chat_url(chat_url: str) -> str:
    u = (chat_url or "").rstrip("/")
    if u.endswith("/chat/completions"):
        base = u[: -len("/chat/completions")]
        return base + "/models"
    if u.endswith("/v1"):
        return u + "/models"
    if "/v1/" in u:
        # strip last segment
        parts = u.rsplit("/", 1)
        return parts[0] + "/models"
    return u + "/v1/models"


def probe_backend(chat_url: str, timeout: float = 5.0, api_key: str = "") -> bool:
    url = models_url_from_chat_url(chat_url)
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    try:
        r = requests.get(url, headers=headers, timeout=timeout)
        if r.status_code < 500:
            return True
    except Exception as exc:
        logger.debug("probe failed %s: %s", url, exc)
    # fallback: root health
    try:
        parsed = urlparse(chat_url)
        root = f"{parsed.scheme}://{parsed.netloc}/health"
        r = requests.get(root, timeout=timeout)
        return r.status_code < 500
    except Exception:
        return False


def wait_until_healthy(
    urls: Iterable[str],
    *,
    wait_seconds: float = 300,
    interval: float = 3.0,
    api_key: str = "",
) -> List[str]:
    urls = [u for u in urls if u]
    if not urls:
        return []
    deadline = time.time() + max(1.0, wait_seconds)
    pending = set(urls)
    healthy: List[str] = []
    while pending and time.time() < deadline:
        done = []
        for u in list(pending):
            if probe_backend(u, api_key=api_key):
                healthy.append(u)
                done.append(u)
                logger.info("Backend healthy: %s", u)
        for u in done:
            pending.discard(u)
        if pending:
            time.sleep(interval)
    if pending:
        raise RuntimeError(
            f"Timed out waiting for backends to become healthy: {sorted(pending)}"
        )
    return healthy


def any_healthy(urls: Iterable[str], api_key: str = "") -> bool:
    return any(probe_backend(u, api_key=api_key) for u in urls if u)
