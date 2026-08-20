#!/usr/bin/env python3
"""Generate profile_data/ scenarios for each model-serving application.

Creates 1000 expected Q/A (or multi-turn) scenarios per app as JSONL.
WebRTC apps also get WAV audio recordings of each question.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import struct
import subprocess
import sys
import wave
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
N_SCENARIOS = 1000

APPS = [
    "nim-streamlit-chat",
    "hf-streamlit-chat",
    "nim-webrtc-voice",
    "hf-webrtc-voice",
    "nim-ide-assistant",
    "hf-ide-assistant",
]

# ---------------------------------------------------------------------------
# Shared content banks
# ---------------------------------------------------------------------------

TOPICS = [
    "python",
    "javascript",
    "typescript",
    "rust",
    "go",
    "java",
    "sql",
    "docker",
    "kubernetes",
    "linux",
    "networking",
    "security",
    "machine learning",
    "data structures",
    "algorithms",
    "apis",
    "databases",
    "testing",
    "git",
    "cloud",
    "performance",
    "concurrency",
    "frontend",
    "backend",
    "devops",
]

CHAT_INTENTS = [
    "explain",
    "compare",
    "troubleshoot",
    "summarize",
    "brainstorm",
    "define",
    "give_example",
    "pros_cons",
    "step_by_step",
    "best_practices",
]

VOICE_INTENTS = [
    "quick_fact",
    "how_to",
    "status_check",
    "reminder",
    "translate_simple",
    "math",
    "weather_style",
    "schedule",
    "definition",
    "recommendation",
]

LANGUAGES = [
    "Python",
    "TypeScript",
    "JavaScript",
    "Rust",
    "Go",
    "Java",
    "C++",
    "SQL",
    "Bash",
    "Swift",
]

IDE_TASKS = [
    "implement_feature",
    "fix_bug",
    "refactor",
    "add_tests",
    "write_docs",
    "optimize",
    "migrate_api",
    "add_logging",
    "security_harden",
    "code_review",
]

FRAMEWORKS = [
    "Flask",
    "FastAPI",
    "Express",
    "React",
    "Next.js",
    "Django",
    "Spring Boot",
    "Actix",
    "Gin",
    "Streamlit",
]


def backend_for(app: str) -> str:
    return "nim" if app.startswith("nim-") else "hf"


def app_kind(app: str) -> str:
    if "streamlit-chat" in app:
        return "chat"
    if "webrtc-voice" in app:
        return "voice"
    if "ide-assistant" in app:
        return "ide"
    raise ValueError(app)


# ---------------------------------------------------------------------------
# Scenario generators
# ---------------------------------------------------------------------------

def chat_scenario(i: int, app: str) -> Dict[str, Any]:
    topic = TOPICS[i % len(TOPICS)]
    intent = CHAT_INTENTS[i % len(CHAT_INTENTS)]
    variant = i // len(TOPICS) + 1
    backend = backend_for(app)

    questions = {
        "explain": f"Explain {topic} concept #{variant} in plain language for a mid-level engineer.",
        "compare": f"Compare two common approaches to {topic} (option A vs option B). When should I pick each?",
        "troubleshoot": f"My {topic} setup fails with error code E{1000 + (i % 9000)}. What are the most likely causes and fixes?",
        "summarize": f"Summarize the key ideas behind modern {topic} practices in under 8 bullet points.",
        "brainstorm": f"Brainstorm 5 practical project ideas that use {topic} and could ship in a weekend.",
        "define": f"Define the core terminology used in {topic}, focusing on terms a new teammate must know.",
        "give_example": f"Give a concrete worked example related to {topic}, including inputs and expected outputs.",
        "pros_cons": f"What are the pros and cons of adopting {topic} in a production service?",
        "step_by_step": f"Provide a step-by-step checklist to get started with {topic} locally.",
        "best_practices": f"List best practices for {topic} that reduce outages and on-call load.",
    }
    question = questions[intent]

    answer = (
        f"[{backend.upper()} chat scenario {i:04d}] Intent={intent}, topic={topic}.\n\n"
        f"1) Restate the goal: help the user with '{intent.replace('_', ' ')}' about {topic}.\n"
        f"2) Core answer: provide accurate, structured guidance tailored to variant {variant}.\n"
        f"3) Concrete detail: include one example, one pitfall, and one verification step.\n"
        f"4) Next actions: offer 2 follow-up questions the user can ask to go deeper.\n"
        f"Expected qualities: clear, concise, actionable, no fabricated citations."
    )

    return {
        "id": f"{app}-{i:04d}",
        "app": app,
        "backend": backend,
        "modality": "text_chat",
        "intent": intent,
        "topic": topic,
        "question": question,
        "answer": answer,
        "tags": ["streamlit", "chat", topic.replace(" ", "-"), intent],
        "metadata": {
            "scenario_index": i,
            "expected_format": "markdown_or_plain_text",
            "max_turns": 1,
        },
    }


def voice_scenario(i: int, app: str) -> Dict[str, Any]:
    intent = VOICE_INTENTS[i % len(VOICE_INTENTS)]
    topic = TOPICS[i % len(TOPICS)]
    n = (i % 50) + 1
    m = ((i % 12) + 2)
    variant = (i // len(TOPICS)) + 1
    backend = backend_for(app)
    # Include variant / numbers so each of 1000 prompts is distinct for audio + eval.
    questions = {
        "quick_fact": f"In one or two sentences, what is {topic}? Focus on point number {variant}.",
        "how_to": f"Quickly tell me how to get started with {topic}, tip set {variant}.",
        "status_check": f"Give me a short status-style summary of common {topic} health checks, checklist {variant}.",
        "reminder": f"Remind me of the top three things to verify before deploying a {topic} change, batch {variant}.",
        "translate_simple": f"Explain {topic} like I am five, version {variant}, in under twenty seconds of speech.",
        "math": f"What is {n} times {m}, and how would you use that number as a batch size in {topic}?",
        "weather_style": f"Give a brief go or no-go recommendation for rolling out a {topic} update today, decision {variant}.",
        "schedule": f"Outline a five minute plan to review our {topic} setup, agenda {variant}.",
        "definition": f"Define {topic} in one spoken sentence, definition card {variant}.",
        "recommendation": f"Recommend one tool or practice for better {topic} results, pick number {variant}.",
    }
    question = questions[intent]


    answer = (
        f"Spoken-friendly answer for scenario {i:04d} ({backend}). "
        f"Intent {intent} on {topic}: give a brief direct reply, then one practical tip. "
        f"Keep it under about 40 words so TTS stays snappy."
    )

    audio_rel = f"audio/q_{i:04d}.wav"
    return {
        "id": f"{app}-{i:04d}",
        "app": app,
        "backend": backend,
        "modality": "voice",
        "intent": intent,
        "topic": topic,
        "question": question,
        "answer": answer,
        "question_audio": audio_rel,
        "tags": ["webrtc", "voice", topic.replace(" ", "-"), intent],
        "metadata": {
            "scenario_index": i,
            "expected_format": "short_spoken_text",
            "max_turns": 1,
            "audio_format": "wav",
            "audio_sample_rate_hz": 22050,


        },
    }


def _ide_code_snippet(lang: str, task: str, i: int) -> str:
    if lang == "Python":
        return (
            "```python\n"
            f"# module service_{i % 97}.py\n"
            "from typing import Optional\n\n"
            "def process(items: list[int], limit: Optional[int] = None) -> list[int]:\n"
            "    out = []\n"
            "    for x in items:\n"
            "        if limit is not None and len(out) >= limit:\n"
            "            break\n"
            "        if x % 2 == 0:\n"
            "            out.append(x * 2)\n"
            "    return out\n"
            "```"
        )
    if lang in ("TypeScript", "JavaScript"):
        return (
            "```typescript\n"
            f"// handlers/task_{i % 53}.ts\n"
            "export async function handle(req: { id: string }) {\n"
            "  const data = await db.find(req.id);\n"
            "  if (!data) throw new Error('not found');\n"
            "  return { ok: true, data };\n"
            "}\n"
            "```"
        )
    if lang == "Go":
        return (
            "```go\n"
            f"// package workers // file w_{i % 41}.go\n"
            "func Run(jobs <-chan int) int {\n"
            "    total := 0\n"
            "    for j := range jobs {\n"
            "        total += j\n"
            "    }\n"
            "    return total\n"
            "}\n"
            "```"
        )
    if lang == "Rust":
        return (
            "```rust\n"
            f"// src/lib_{i % 29}.rs\n"
            "pub fn normalize(xs: &[i32]) -> Vec<i32> {\n"
            "    xs.iter().filter(|x| **x > 0).copied().collect()\n"
            "}\n"
            "```"
        )
    if lang == "SQL":
        return (
            "```sql\n"
            f"-- query_{i % 31}.sql\n"
            "SELECT u.id, COUNT(o.id) AS orders\n"
            "FROM users u\n"
            "LEFT JOIN orders o ON o.user_id = u.id\n"
            "GROUP BY u.id\n"
            "HAVING COUNT(o.id) > 0;\n"
            "```"
        )
    return (
        f"```{lang.lower()}\n"
        f"// placeholder snippet for {task} scenario {i}\n"
        "function main() { return 0; }\n"
        "```"
    )


def ide_scenario(i: int, app: str) -> Dict[str, Any]:
    """Multi-turn coding assistant trajectory (typical Cline/Cursor flow)."""
    lang = LANGUAGES[i % len(LANGUAGES)]
    task = IDE_TASKS[i % len(IDE_TASKS)]
    fw = FRAMEWORKS[i % len(FRAMEWORKS)]
    backend = backend_for(app)
    feature = f"feature-{topic_slug(i)}"
    snippet = _ide_code_snippet(lang, task, i)

    turns: List[Dict[str, str]] = []

    # Turn 1: user presents task + code context
    u1 = (
        f"I'm working in a {fw} codebase ({lang}). Task: {task.replace('_', ' ')} for `{feature}`.\n\n"
        f"Here is the current code:\n{snippet}\n\n"
        f"Please inspect it and propose a concrete plan before editing."
    )
    a1 = (
        f"Plan for `{feature}` ({task}, {lang}/{fw}):\n"
        f"1. Clarify current behavior and edge cases in the snippet.\n"
        f"2. Identify the minimal code change set (files + functions).\n"
        f"3. Implement the change with clear names and error handling.\n"
        f"4. Add/adjust tests and a quick manual verification step.\n"
        f"5. Call out risks (concurrency, backward compatibility, perf).\n"
        f"If you want, I can apply the edits next."
    )
    turns.append({"role": "user", "content": u1})
    turns.append({"role": "assistant", "content": a1})

    # Turn 2: user asks to implement
    u2 = (
        f"Looks good. Implement the change for `{feature}`. "
        f"Show the full updated code and briefly explain what changed."
    )
    a2 = (
        f"Implemented `{feature}` for scenario {i:04d}.\n\n"
        f"{_ide_code_snippet(lang, task, i + 17)}\n\n"
        f"Changes:\n"
        f"- Extended the main path to cover the {task.replace('_', ' ')} requirement.\n"
        f"- Added validation/guard rails appropriate for {lang}.\n"
        f"- Kept the public interface stable where possible.\n"
        f"- Left a clear extension point for follow-up work in {fw}."
    )
    turns.append({"role": "user", "content": u2})
    turns.append({"role": "assistant", "content": a2})

    # Turn 3: tests / verification
    u3 = (
        f"Add focused tests for the new behavior and list commands I should run "
        f"to verify in {fw}."
    )
    a3 = (
        f"Tests for `{feature}`:\n\n"
        f"```{lang.lower()}\n"
        f"// test cases: happy path, empty input, invalid input, boundary for scenario {i:04d}\n"
        f"```\n\n"
        f"Verify with:\n"
        f"1. Unit tests for the modified module\n"
        f"2. One integration check through the {fw} entrypoint\n"
        f"3. Lint/typecheck if available\n"
        f"Expected result: all new tests pass; no regression in existing suite."
    )
    turns.append({"role": "user", "content": u3})
    turns.append({"role": "assistant", "content": a3})

    # Turn 4 (every other scenario): follow-up refactor or review
    if i % 2 == 0:
        u4 = "Do a quick code review of your change. Any bugs or cleaner refactors?"
        a4 = (
            f"Review notes for scenario {i:04d}:\n"
            f"- Check error paths return actionable messages.\n"
            f"- Ensure resource cleanup / cancellation is handled.\n"
            f"- Consider extracting a helper if the function grew past ~40 lines.\n"
            f"- Confirm logging does not leak secrets.\n"
            f"No critical issues assumed in the golden answer; flag style nits only."
        )
        turns.append({"role": "user", "content": u4})
        turns.append({"role": "assistant", "content": a4})

    # Convenience single-string fields (first Q / final A) plus full transcript
    return {
        "id": f"{app}-{i:04d}",
        "app": app,
        "backend": backend,
        "modality": "ide_multiturn",
        "intent": task,
        "language": lang,
        "framework": fw,
        "question": u1,
        "answer": turns[-1]["content"],
        "turns": turns,
        "tags": [
            "ide",
            "coding",
            "multi-turn",
            lang.lower(),
            fw.lower().replace(" ", "-"),
            task,
        ],
        "metadata": {
            "scenario_index": i,
            "expected_format": "multi_turn_chat_messages",
            "turn_count": len(turns),
            "user_turns": sum(1 for t in turns if t["role"] == "user"),
            "assistant_turns": sum(1 for t in turns if t["role"] == "assistant"),
            "feature": feature,
        },
    }


def topic_slug(i: int) -> str:
    t = TOPICS[i % len(TOPICS)]
    return f"{t.replace(' ', '-')}-{(i % 200) + 1}"


def generate_scenario(app: str, i: int) -> Dict[str, Any]:
    kind = app_kind(app)
    if kind == "chat":
        return chat_scenario(i, app)
    if kind == "voice":
        return voice_scenario(i, app)
    return ide_scenario(i, app)


# ---------------------------------------------------------------------------
# Audio generation
# ---------------------------------------------------------------------------

def write_tone_wav(path: Path, text: str, sample_rate: int = 16000) -> None:
    """Fast deterministic speech-like WAV derived from text (no TTS dependency)."""
    import array

    path.parent.mkdir(parents=True, exist_ok=True)
    # Short clips: duration scales lightly with text length
    duration = min(4.0, max(0.8, 0.03 * len(text)))
    n_samples = int(sample_rate * duration)
    words = text.split() or ["hello"]

    samples = array.array("h")
    cursor = 0
    gap = max(1, int(0.04 * sample_rate))
    attack = max(1, int(0.008 * sample_rate))
    release = max(1, int(0.015 * sample_rate))

    for wi, word in enumerate(words):
        remaining_words = len(words) - wi
        remaining_samples = n_samples - cursor
        allot = max(gap + 1, remaining_samples // remaining_words)
        base_freq = 140.0 + (sum(ord(c) for c in word) % 120)
        two_pi_f = 2.0 * math.pi * base_freq / sample_rate
        two_pi_2f = 2.0 * two_pi_f
        tone_len = max(1, allot - gap)
        for s in range(allot):
            if s >= tone_len:
                samples.append(0)
            else:
                env = 1.0
                if s < attack:
                    env = s / attack
                elif s > tone_len - release:
                    env = max(0.0, (tone_len - s) / release)
                val = int(env * (10000.0 * math.sin(two_pi_f * s) + 3500.0 * math.sin(two_pi_2f * s)))
                if val > 32767:
                    val = 32767
                elif val < -32767:
                    val = -32767
                samples.append(val)
            cursor += 1
            if cursor >= n_samples:
                break
        if cursor >= n_samples:
            break

    if len(samples) < n_samples:
        samples.extend([0] * (n_samples - len(samples)))

    with wave.open(str(path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(samples.tobytes())



def write_say_wav(path: Path, text: str, voice: str = "Samantha") -> bool:
    """Use macOS `say` to synthesize speech to WAV. Returns True on success."""
    path.parent.mkdir(parents=True, exist_ok=True)
    aiff_path = path.with_suffix(".aiff")
    try:
        subprocess.run(
            ["say", "-v", voice, "-o", str(aiff_path), text],
            check=True,
            capture_output=True,
            timeout=60,
        )
        # Convert AIFF -> WAV via afconvert if present, else keep and rename attempt
        af = subprocess.run(
            ["afconvert", "-f", "WAVE", "-d", "LEI16", str(aiff_path), str(path)],
            capture_output=True,
            timeout=60,
        )
        if aiff_path.exists():
            aiff_path.unlink(missing_ok=True)
        return af.returncode == 0 and path.exists()
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        if aiff_path.exists():
            aiff_path.unlink(missing_ok=True)
        return False


def generate_audio_for_voice_app(
    app_dir: Path,
    scenarios: Sequence[Dict[str, Any]],
    *,
    use_say: bool,
    workers: int,
) -> Tuple[int, int]:
    """Generate question audio files. Returns (ok, failed).

    When use_say is True, synthesize each unique question text once and copy
    the WAV to all scenarios that share that question (much faster).
    """
    import hashlib
    import shutil

    audio_dir = app_dir / "profile_data" / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = app_dir / "profile_data" / ".audio_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Group scenarios by question text
    by_text: Dict[str, List[Dict[str, Any]]] = {}
    for sc in scenarios:
        by_text.setdefault(sc["question"], []).append(sc)

    def synth_one(text: str) -> Path:
        key = hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]
        cached = cache_dir / f"{key}.wav"
        if cached.exists() and cached.stat().st_size > 44:
            return cached
        if use_say:
            if write_say_wav(cached, text):
                return cached
        write_tone_wav(cached, text)
        return cached

    ok = fail = 0
    texts = list(by_text.keys())

    # Synthesize unique questions in parallel
    text_to_cache: Dict[str, Path] = {}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(synth_one, t): t for t in texts}
        for fut in as_completed(futs):
            t = futs[fut]
            try:
                text_to_cache[t] = fut.result()
            except Exception:
                text_to_cache[t] = Path()

    for text, group in by_text.items():
        src = text_to_cache.get(text)
        if not src or not src.exists():
            fail += len(group)
            continue
        for sc in group:
            dest = app_dir / "profile_data" / sc["question_audio"]
            dest.parent.mkdir(parents=True, exist_ok=True)
            try:
                if dest.resolve() != src.resolve():
                    shutil.copyfile(src, dest)
                ok += 1
            except OSError:
                fail += 1
    return ok, fail



# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------

def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=N_SCENARIOS)
    parser.add_argument(
        "--apps",
        nargs="*",
        default=APPS,
        help="Subset of apps to generate",
    )
    parser.add_argument(
        "--say",
        action="store_true",
        help="Use macOS say for WebRTC question audio (slower). "
        "Default is deterministic tone WAV placeholders encoding the question length.",
    )
    parser.add_argument("--audio-workers", type=int, default=8)
    parser.add_argument(
        "--skip-audio",
        action="store_true",
        help="Only write JSONL even for voice apps",
    )
    args = parser.parse_args()

    summary = []
    for app in args.apps:
        app_dir = ROOT / app
        if not app_dir.is_dir():
            print(f"SKIP missing app dir: {app}", file=sys.stderr)
            continue
        out_dir = app_dir / "profile_data"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "scenarios.jsonl"

        scenarios = [generate_scenario(app, i) for i in range(args.count)]
        n = write_jsonl(out_path, scenarios)

        audio_ok = audio_fail = 0
        if app_kind(app) == "voice" and not args.skip_audio:
            audio_ok, audio_fail = generate_audio_for_voice_app(
                app_dir,
                scenarios,
                use_say=args.say,
                workers=args.audio_workers,
            )

        # Small README for the folder
        readme = out_dir / "README.md"
        kind = app_kind(app)
        readme.write_text(
            f"# profile_data — `{app}`\n\n"
            f"- **scenarios**: `{n}` rows in `scenarios.jsonl`\n"
            f"- **kind**: `{kind}`\n"
            f"- **backend**: `{backend_for(app)}`\n"
            + (
                f"- **audio**: `audio/q_XXXX.wav` per scenario (question only)\n"
                if kind == "voice"
                else ""
            )
            + (
                "- **format**: multi-turn `turns[]` with user/assistant roles (IDE coding flow)\n"
                if kind == "ide"
                else "- **format**: single-turn `question` / `answer`\n"
            )
            + "\nEach JSONL line is one evaluation scenario with expected answer content "
            "for profiling / regression checks.\n",
            encoding="utf-8",
        )

        summary.append(
            {
                "app": app,
                "scenarios": n,
                "jsonl": str(out_path.relative_to(ROOT)),
                "audio_ok": audio_ok,
                "audio_fail": audio_fail,
            }
        )
        print(
            f"{app}: wrote {n} scenarios"
            + (f", audio ok={audio_ok} fail={audio_fail}" if kind == "voice" else "")
        )

    print(json.dumps({"summary": summary}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
