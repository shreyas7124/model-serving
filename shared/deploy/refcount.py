"""Simple file-based refcount for shared AUTO_DEPLOY_PROJECT stacks."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict

try:
    import fcntl
except ImportError:  # Windows
    fcntl = None  # type: ignore


def _state_dir() -> Path:
    base = os.getenv("SHARED_DATA_DIR", os.getenv("AUTO_DEPLOY_STATE_DIR", ""))
    if base:
        d = Path(base) / ".model-runtime"
    else:
        d = Path(os.getenv("TMPDIR", "/tmp")) / "model-serving-runtime"
    d.mkdir(parents=True, exist_ok=True)
    return d


def state_path(project: str) -> Path:
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in project)
    return _state_dir() / f"{safe}.json"


def _lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(path.suffix + ".lock")
    fh = open(lock_path, "a+")
    if fcntl is not None:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
    return fh


def _unlock(fh) -> None:
    try:
        if fcntl is not None:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
    finally:
        fh.close()


def read_state(project: str) -> Dict[str, Any]:
    p = state_path(project)
    if not p.exists():
        return {"project": project, "refcount": 0, "owned_pids": [], "compose_file": ""}
    try:
        return json.loads(p.read_text())
    except Exception:
        return {"project": project, "refcount": 0, "owned_pids": [], "compose_file": ""}


def write_state(project: str, data: Dict[str, Any]) -> None:
    p = state_path(project)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True))
    tmp.replace(p)


def acquire(project: str, pid: int, compose_file: str) -> int:
    """Increment refcount; return new count."""
    fh = _lock(state_path(project))
    try:
        st = read_state(project)
        pids = [int(x) for x in st.get("owned_pids") or [] if str(x).isdigit()]
        if pid not in pids:
            pids.append(pid)
        st["owned_pids"] = pids
        st["refcount"] = len(pids)
        st["compose_file"] = compose_file or st.get("compose_file") or ""
        st["project"] = project
        st["updated_at"] = time.time()
        write_state(project, st)
        return int(st["refcount"])
    finally:
        _unlock(fh)


def release(project: str, pid: int) -> int:
    """Decrement refcount; return new count."""
    fh = _lock(state_path(project))
    try:
        st = read_state(project)
        pids = [int(x) for x in st.get("owned_pids") or [] if str(x).isdigit()]
        pids = [p for p in pids if p != pid]
        # drop dead pids
        alive = []
        for p in pids:
            try:
                os.kill(p, 0)
                alive.append(p)
            except OSError:
                pass
        st["owned_pids"] = alive
        st["refcount"] = len(alive)
        st["updated_at"] = time.time()
        write_state(project, st)
        return int(st["refcount"])
    finally:
        _unlock(fh)
