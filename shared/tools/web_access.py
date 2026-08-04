"""
Internet access tool for Streamlit and WebRTC chat apps.

- Detects URLs in user messages
- Fetches full page content (HTML → readable text)
- Logs **every** URL and the **full** fetched body to a fixed log file
  (not only the truncated snippet injected into the model prompt)
- When log storage reaches the configured limit, web access is **terminated**
  (no rotation / no further fetches) until logs are cleared
- Returns context blocks suitable for prepending to LLM messages

"""
from __future__ import annotations

import hashlib
import html
import logging
import os
import re
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
from urllib.parse import urlparse


import requests

# URL pattern: http(s) only (avoid javascript:, file:, etc.)
_URL_RE = re.compile(
    r"https?://[^\s<>\"'\)\]\}]+",
    re.IGNORECASE,
)

# Strip common trailing punctuation stuck to URLs in prose
_TRAIL_PUNCT = ".,;:!?)]}'\"…"


def _env_bool(name: str, default: bool = True) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not str(raw).strip():
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def parse_size(value: str, default_bytes: int) -> int:
    """
    Parse a size string into bytes.

    Accepts:
      - plain integers (bytes): "1024"
      - decimal units:  "1KB", "50MB", "2GB", "1TB"  (1000-based)
      - binary units:   "1KiB", "50MiB", "2GiB", "1TiB" (1024-based)
      - bare letter forms: "1K", "50M", "2G", "1T" (treated as binary KiB/MiB/GiB/TiB)
    """
    if value is None:
        return default_bytes
    raw = str(value).strip()
    if not raw:
        return default_bytes
    # plain integer bytes
    try:
        return int(raw)
    except ValueError:
        pass
    m = re.fullmatch(
        r"(?i)\s*([0-9]+(?:\.[0-9]+)?)\s*([kmgt]i?b?)\s*",
        raw,
    )
    if not m:
        return default_bytes
    amount = float(m.group(1))
    unit = m.group(2).lower()
    # Normalize: k/kb/kib, m/mb/mib, ...
    decimal = {
        "k": 1000,
        "kb": 1000,
        "m": 1000**2,
        "mb": 1000**2,
        "g": 1000**3,
        "gb": 1000**3,
        "t": 1000**4,
        "tb": 1000**4,
    }
    binary = {
        "ki": 1024,
        "kib": 1024,
        "mi": 1024**2,
        "mib": 1024**2,
        "gi": 1024**3,
        "gib": 1024**3,
        "ti": 1024**4,
        "tib": 1024**4,
    }
    # Single-letter K/M/G/T → binary (common ops shorthand)
    if unit in ("k", "m", "g", "t"):
        mult = {"k": 1024, "m": 1024**2, "g": 1024**3, "t": 1024**4}[unit]
    elif unit in binary:
        mult = binary[unit]
    elif unit in decimal:
        mult = decimal[unit]
    else:
        return default_bytes
    return int(amount * mult)


def _env_size(name: str, default_bytes: int) -> int:
    """Read a size env var supporting GB/TB (and byte) suffixes."""
    raw = os.getenv(name)
    if raw is None or not str(raw).strip():
        return default_bytes
    return parse_size(raw, default_bytes)


def format_size(num_bytes: int) -> str:
    """Human-readable size for logs/stats (decimal GB/TB preferred)."""
    if num_bytes is None:
        return "n/a"
    n = float(num_bytes)
    for unit, div in (
        ("TB", 1000**4),
        ("GB", 1000**3),
        ("MB", 1000**2),
        ("KB", 1000),
    ):
        if n >= div:
            val = n / div
            if val >= 100:
                return f"{val:.0f}{unit}"
            if val >= 10:
                return f"{val:.1f}{unit}"
            return f"{val:.2f}{unit}"
    return f"{int(n)}B"




def extract_urls(text: str) -> List[str]:
    """Return unique http(s) URLs found in text, order-preserving."""
    if not text:
        return []
    seen = set()
    out: List[str] = []
    for m in _URL_RE.finditer(text):
        url = m.group(0).rstrip(_TRAIL_PUNCT)
        # Drop empty fragments-only junk
        if not url or url in seen:
            continue
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            continue
        seen.add(url)
        out.append(url)
    return out


def html_to_text(raw: str) -> str:
    """Best-effort HTML → plain text without heavy deps."""
    if not raw:
        return ""
    # Remove script/style blocks
    text = re.sub(
        r"(?is)<(script|style|noscript|svg|iframe)[^>]*>.*?</\1>",
        " ",
        raw,
    )
    # Line breaks for block-ish tags
    text = re.sub(r"(?i)<br\s*/?>", "\n", text)
    text = re.sub(r"(?i)</(p|div|h[1-6]|li|tr|section|article)>", "\n", text)
    # Drop remaining tags
    text = re.sub(r"(?s)<[^>]+>", " ", text)
    text = html.unescape(text)
    # Collapse whitespace
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return text.strip()


@dataclass
class FetchResult:
    url: str
    ok: bool
    status_code: Optional[int] = None
    content_type: str = ""
    title: str = ""
    text: str = ""
    raw_length: int = 0
    error: str = ""
    fetched_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    log_path: str = ""
    content_sha256: str = ""

    def prompt_block(self, max_chars: int = 12000) -> str:
        """Truncated block for LLM context (full body is still in the log)."""
        body = self.text if self.ok else f"[fetch failed: {self.error}]"
        truncated = False
        if len(body) > max_chars:
            body = body[:max_chars] + "\n…[truncated for model context; full content in web access log]"
            truncated = True
        title_line = f"Title: {self.title}\n" if self.title else ""
        meta = (
            f"URL: {self.url}\n"
            f"{title_line}"
            f"Status: {self.status_code if self.status_code is not None else 'n/a'}\n"
            f"Content-Type: {self.content_type or 'n/a'}\n"
            f"Fetched-At: {self.fetched_at}\n"
            f"Full-Content-Log: {self.log_path or 'n/a'}\n"
            f"Content-SHA256: {self.content_sha256 or 'n/a'}\n"
            f"Truncated-For-Prompt: {truncated}\n"
        )
        return f"----- BEGIN WEB PAGE -----\n{meta}\n{body}\n----- END WEB PAGE -----"


class WebAccessTool:
    """
    Fetch URLs and always log full content to disk.

    Logging guarantee: for every attempted URL we write a structured record
    including the complete response body (or error), independent of what is
    passed into the model prompt.
    """

    def __init__(
        self,
        enabled: Optional[bool] = None,
        log_dir: Optional[str] = None,
        timeout: Optional[int] = None,
        max_bytes: Optional[int] = None,
        max_prompt_chars: Optional[int] = None,
        user_agent: Optional[str] = None,
        app_name: str = "app",
    ):
        self.enabled = (
            enabled
            if enabled is not None
            else _env_bool("WEB_ACCESS_ENABLED", True)
        )
        self.app_name = app_name
        self.timeout = timeout if timeout is not None else _env_int("WEB_ACCESS_TIMEOUT", 20)
        self.max_bytes = max_bytes if max_bytes is not None else _env_int(
            "WEB_ACCESS_MAX_BYTES", 2_000_000
        )
        self.max_prompt_chars = (
            max_prompt_chars
            if max_prompt_chars is not None
            else _env_int("WEB_ACCESS_MAX_PROMPT_CHARS", 12000)
        )
        self.user_agent = user_agent or os.getenv(
            "WEB_ACCESS_USER_AGENT",
            "ModelServing-WebAccess/1.0 (+https://github.com/7sg-ai/model-serving)",
        )
        base_log = log_dir or os.getenv(
            "WEB_ACCESS_LOG_DIR",
            os.path.join(os.getenv("SHARED_DATA_DIR", "shared/database"), "web_access_logs"),
        )
        self.log_dir = Path(base_log)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        # Fixed-size log budget (no rotation). When exceeded, web access stops.
        # Prefer WEB_ACCESS_LOG_MAX_SIZE (e.g. "1TB", "50GB"); fall back to bytes env.
        # Default 1TB.
        _default_log_max = parse_size("1TB", 1_000_000_000_000)
        size_raw = os.getenv("WEB_ACCESS_LOG_MAX_SIZE", "").strip()
        if size_raw:
            self.log_max_bytes = parse_size(size_raw, _default_log_max)
        else:
            # Backward compatible: WEB_ACCESS_LOG_MAX_BYTES still accepted
            self.log_max_bytes = _env_size("WEB_ACCESS_LOG_MAX_BYTES", _default_log_max)


        self._lock = threading.RLock()
        self._terminated = False
        self._terminate_reason = ""
        self._logger = self._build_logger()
        self.fetches = 0
        self.failures = 0
        # Honor existing full logs from a previous run
        if self.enabled and self._log_usage_bytes() >= self.log_max_bytes:
            self._terminate("log storage already at capacity on startup")

    def _build_logger(self) -> logging.Logger:
        name = f"web_access.{self.app_name}"
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)
        logger.propagate = False
        # Single non-rotating file handler (no rollover backups)
        log_file = self.log_dir / f"{self.app_name}-web-access.log"
        has_file = any(
            isinstance(h, logging.FileHandler)
            and getattr(h, "baseFilename", None) == str(log_file.resolve())
            for h in logger.handlers
        )
        if not has_file:
            handler = logging.FileHandler(log_file, encoding="utf-8")
            handler.setFormatter(
                logging.Formatter("%(asctime)s %(levelname)s %(message)s")
            )
            logger.addHandler(handler)
        return logger

    @property
    def log_file_path(self) -> str:
        return str(self.log_dir / f"{self.app_name}-web-access.log")

    @property
    def terminated(self) -> bool:
        return self._terminated

    def _pages_dir(self) -> Path:
        return self.log_dir / "pages" / self.app_name

    def _log_usage_bytes(self) -> int:
        """Total bytes used by this app's web-access log file + per-URL page files."""
        total = 0
        main = Path(self.log_file_path)
        if main.is_file():
            total += main.stat().st_size
        pages = self._pages_dir()
        if pages.is_dir():
            for p in pages.rglob("*"):
                if p.is_file():
                    try:
                        total += p.stat().st_size
                    except OSError:
                        pass
        return total

    def _terminate(self, reason: str) -> None:
        """Permanently disable further web fetches for this process (logs full)."""
        if self._terminated:
            return
        self._terminated = True
        self._terminate_reason = reason
        self.enabled = False
        msg = (
            f"event=web_access_terminated app={self.app_name} "
            f"reason={reason!r} log_usage_bytes={self._log_usage_bytes()} "
            f"log_max_bytes={self.log_max_bytes}"
        )
        try:
            with self._lock:
                self._logger.error(msg)
        except Exception:
            pass

    def _ensure_log_capacity(self, upcoming_bytes: int = 0) -> bool:
        """
        Return True if a fetch may proceed. If usage would exceed the budget,
        terminate web access and return False.
        """
        if self._terminated or not self.enabled:
            return False
        usage = self._log_usage_bytes()
        if usage + max(0, upcoming_bytes) >= self.log_max_bytes:
            self._terminate(
                f"log storage full ({usage + max(0, upcoming_bytes)} >= {self.log_max_bytes} bytes)"
            )
            return False
        return True


    def _write_full_content_file(self, url: str, body: str, meta: Dict[str, Any]) -> str:
        """
        Persist the complete page body to its own file and return the path.
        This is the authoritative full-content log (not truncated).
        """
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:12]
        safe_host = re.sub(r"[^a-zA-Z0-9.-]+", "_", urlparse(url).netloc)[:80] or "unknown"
        fname = f"{ts}_{safe_host}_{digest}.txt"
        path = self._pages_dir()
        path.mkdir(parents=True, exist_ok=True)
        full_path = path / fname

        header_lines = [
            f"URL: {url}",
            f"Fetched-At: {meta.get('fetched_at', '')}",
            f"Status: {meta.get('status_code', '')}",
            f"Content-Type: {meta.get('content_type', '')}",
            f"Title: {meta.get('title', '')}",
            f"App: {self.app_name}",
            f"OK: {meta.get('ok', '')}",
            f"Error: {meta.get('error', '')}",
            f"Raw-Length: {meta.get('raw_length', len(body))}",
            f"Content-SHA256: {meta.get('content_sha256', '')}",
            "=" * 72,
            "",
        ]
        with open(full_path, "w", encoding="utf-8") as f:
            f.write("\n".join(header_lines))
            f.write(body if body is not None else "")
            if body and not body.endswith("\n"):
                f.write("\n")
        return str(full_path)

    def _log_event(self, result: FetchResult, session_id: str = "") -> None:
        """Index line in the fixed log pointing at the full content file."""
        # Full content already on disk at result.log_path — also embed a
        # complete copy in the main log so operators have one stream.

        record = (
            f"event=web_fetch app={self.app_name} session={session_id or '-'} "
            f"ok={result.ok} status={result.status_code} url={result.url!r} "
            f"content_type={result.content_type!r} title={result.title!r} "
            f"raw_length={result.raw_length} sha256={result.content_sha256} "
            f"full_content_path={result.log_path!r} error={result.error!r}\n"
            f"----- FULL CONTENT START url={result.url!r} -----\n"
            f"{result.text}\n"
            f"----- FULL CONTENT END url={result.url!r} -----\n"
        )
        with self._lock:
            self._logger.info(record)

    def fetch_url(self, url: str, session_id: str = "") -> FetchResult:
        """Fetch a single URL; always logs full content (or error body)."""
        if self._terminated:
            return FetchResult(
                url=url,
                ok=False,
                error=f"web access terminated: {self._terminate_reason or 'log storage full'}",
                log_path=self.log_file_path,
            )
        if not self.enabled:
            return FetchResult(
                url=url,
                ok=False,
                error="web access disabled",
                log_path=self.log_file_path,
            )
        # Refuse before network I/O if logs are already at capacity
        if not self._ensure_log_capacity(upcoming_bytes=0):
            return FetchResult(
                url=url,
                ok=False,
                error=f"web access terminated: {self._terminate_reason or 'log storage full'}",
                log_path=self.log_file_path,
            )

        result = FetchResult(url=url, ok=False, log_path="")

        try:
            headers = {
                "User-Agent": self.user_agent,
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,text/plain;q=0.8,*/*;q=0.5",
            }
            with requests.get(
                url,
                headers=headers,
                timeout=self.timeout,
                stream=True,
                allow_redirects=True,
            ) as resp:
                result.status_code = resp.status_code
                result.content_type = (resp.headers.get("Content-Type") or "").split(";")[0].strip()
                # Cap download size
                chunks: List[bytes] = []
                total = 0
                for chunk in resp.iter_content(chunk_size=65536):
                    if not chunk:
                        continue
                    total += len(chunk)
                    if total > self.max_bytes:
                        chunks.append(chunk[: max(0, self.max_bytes - (total - len(chunk)))])
                        break
                    chunks.append(chunk)
                raw_bytes = b"".join(chunks)
                result.raw_length = len(raw_bytes)
                # Decode
                encoding = resp.encoding or "utf-8"
                try:
                    raw_text = raw_bytes.decode(encoding, errors="replace")
                except LookupError:
                    raw_text = raw_bytes.decode("utf-8", errors="replace")

                ctype = result.content_type.lower()
                if "html" in ctype or raw_text.lstrip()[:1] == "<":
                    # Title
                    tm = re.search(r"(?is)<title[^>]*>(.*?)</title>", raw_text)
                    if tm:
                        result.title = html.unescape(re.sub(r"\s+", " ", tm.group(1))).strip()
                    result.text = html_to_text(raw_text)
                else:
                    result.text = raw_text

                result.content_sha256 = hashlib.sha256(
                    result.text.encode("utf-8", errors="replace")
                ).hexdigest()
                result.ok = 200 <= resp.status_code < 400
                if not result.ok:
                    result.error = f"HTTP {resp.status_code}"
                    self.failures += 1
                else:
                    self.fetches += 1
        except Exception as exc:  # noqa: BLE001 — surface to caller + log
            result.ok = False
            result.error = str(exc)
            result.text = f"[exception while fetching] {exc}"
            result.content_sha256 = hashlib.sha256(
                result.text.encode("utf-8", errors="replace")
            ).hexdigest()
            self.failures += 1

        meta = {
            "fetched_at": result.fetched_at,
            "status_code": result.status_code,
            "content_type": result.content_type,
            "title": result.title,
            "ok": result.ok,
            "error": result.error,
            "raw_length": result.raw_length,
            "content_sha256": result.content_sha256,
        }
        # Estimate bytes this write will add (body + headers + index record)
        upcoming = len((result.text or "").encode("utf-8", errors="replace")) + 2048
        if not self._ensure_log_capacity(upcoming_bytes=upcoming):
            # Do not write more once full; surface termination to caller
            result.ok = False
            result.error = (
                result.error + "; " if result.error else ""
            ) + f"web access terminated: {self._terminate_reason or 'log storage full'}"
            result.log_path = self.log_file_path
            return result

        result.log_path = self._write_full_content_file(url, result.text, meta)
        self._log_event(result, session_id=session_id)
        # Re-check after write; stop future fetches if we crossed the limit
        if self._log_usage_bytes() >= self.log_max_bytes:
            self._terminate(
                f"log storage full after write ({self._log_usage_bytes()} >= {self.log_max_bytes} bytes)"
            )
        return result


    def fetch_urls(
        self,
        urls: Sequence[str],
        session_id: str = "",
        max_urls: Optional[int] = None,
    ) -> List[FetchResult]:
        limit = max_urls if max_urls is not None else _env_int("WEB_ACCESS_MAX_URLS", 5)
        results: List[FetchResult] = []
        for url in list(urls)[: max(0, limit)]:
            results.append(self.fetch_url(url, session_id=session_id))
        return results

    def process_user_text(
        self,
        text: str,
        session_id: str = "",
    ) -> Tuple[List[FetchResult], str]:
        """
        Extract URLs from user text, fetch each, log full content, return
        (results, context_string_for_llm).
        """
        if self._terminated or not self.enabled or not text:
            return [], ""

        urls = extract_urls(text)
        if not urls:
            return [], ""
        results = self.fetch_urls(urls, session_id=session_id)
        blocks = [r.prompt_block(self.max_prompt_chars) for r in results]
        context = (
            "The user referenced web URL(s). Full page contents were fetched and "
            "logged server-side. Use the following extracted content when answering:\n\n"
            + "\n\n".join(blocks)
        )
        return results, context

    def enrich_messages(
        self,
        messages: List[Dict[str, str]],
        user_text: Optional[str] = None,
        session_id: str = "",
    ) -> Tuple[List[Dict[str, str]], List[FetchResult]]:
        """
        If the latest user message (or user_text) contains URLs, fetch them,
        log full content, and inject a system/context message before the user turn.
        """
        text = user_text
        if text is None:
            for msg in reversed(messages or []):
                if msg.get("role") == "user":
                    text = msg.get("content", "")
                    break
        results, context = self.process_user_text(text or "", session_id=session_id)
        if not context:
            return list(messages or []), results
        enriched = list(messages or [])
        # Insert context immediately before the last user message when possible
        insert_at = len(enriched)
        for i in range(len(enriched) - 1, -1, -1):
            if enriched[i].get("role") == "user":
                insert_at = i
                break
        enriched.insert(
            insert_at,
            {
                "role": "system",
                "content": context,
            },
        )
        return enriched, results

    def stats(self) -> Dict[str, Any]:
        usage = self._log_usage_bytes()
        return {
            "enabled": self.enabled and not self._terminated,
            "terminated": self._terminated,
            "terminate_reason": self._terminate_reason or None,
            "app_name": self.app_name,
            "log_dir": str(self.log_dir),
            "log_file": self.log_file_path,
            "log_usage_bytes": usage,
            "log_max_bytes": self.log_max_bytes,
            "log_usage": format_size(usage),
            "log_max": format_size(self.log_max_bytes),
            "log_usage_ratio": round(usage / self.log_max_bytes, 4) if self.log_max_bytes else None,
            "fetches": self.fetches,
            "failures": self.failures,
            "timeout": self.timeout,
            "max_bytes": self.max_bytes,
            "max_prompt_chars": self.max_prompt_chars,
        }




_TOOLS: Dict[str, WebAccessTool] = {}
_TOOLS_LOCK = threading.Lock()


def get_web_access_tool(app_name: str = "app") -> WebAccessTool:
    """Process-wide singleton per app_name."""
    with _TOOLS_LOCK:
        tool = _TOOLS.get(app_name)
        if tool is None:
            tool = WebAccessTool(app_name=app_name)
            _TOOLS[app_name] = tool
        return tool
