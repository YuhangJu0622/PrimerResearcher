"""Per-question artifact writer and crash-tolerant batch runner."""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

from agent.runtime.render import LEVEL_ORDER, render_event


def _utc_timestamp() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _local_clock() -> str:
    return datetime.now().strftime("%H:%M:%S")


class QuestionLogger:
    """Write exactly one question's timeline to two parallel artifacts.

    ``<stem>.log`` is the human-readable timeline in the style of
    ``[HH:MM:SS] [LEVEL] [component] message``; only events with a renderer and a
    level at or above ``level`` appear there.  ``<stem>.events.jsonl`` receives
    every structured event verbatim, one JSON object per line, for machine
    analysis.  Both artifacts are flushed and fsynced per record so a crashed
    batch leaves a resumable, truthful trace.
    """

    def __init__(self, path: Path, *, question_number: int, question_id: str, level: str = "INFO") -> None:
        self.path = path
        self.events_path = path.with_suffix(".events.jsonl")
        self.question_number = question_number
        self.question_id = question_id
        self.attempt = 0
        self.sequence = 0
        self.level = LEVEL_ORDER[level.upper()]
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Both artifacts exist from the start so consumers and resume logic can
        # rely on them even when every event is filtered out of the human log.
        for artifact in (self.path, self.events_path):
            artifact.touch(exist_ok=True)

    def write(self, event: str, *, level: str = "INFO", phase: str = "runtime", **payload: Any) -> None:
        self.sequence += 1
        record = {
            "ts": _utc_timestamp(), "question_number": self.question_number,
            "question_id": self.question_id, "attempt": self.attempt,
            "event_seq": self.sequence, "phase": phase, "event": event,
            "level": level, **payload,
        }
        rendered = render_event(event, payload, question_number=self.question_number, attempt=self.attempt)
        line = None
        if rendered is not None and LEVEL_ORDER[rendered.level] >= self.level:
            line = f"[{_local_clock()}] [{rendered.level}] [{rendered.component}] {rendered.message}"
        self._append(self.events_path, json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
        if line is not None:
            self._append(self.path, line)

    @staticmethod
    def _append(path: Path, line: str) -> None:
        with path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(line + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    def emit_loop_event(self, value: dict[str, Any]) -> None:
        event = dict(value)
        event_name = str(event.pop("type", "unknown"))
        error_code = event.get("error_code")
        level = "DEBUG" if event_name == "model_api_response" else "ERROR" if event_name in {"generation_error", "invalid_tool_json", "tool_error", "web_search_failed"} or error_code in {
            "tool_error", "tool_unavailable"
        } else "WARNING" if event_name in {
            "tool_timeout", "context_limit_exceeded", "tool_calls_rejected", "no_final_answer",
            "forced_final_attempt_failed", "forced_final_exhausted", "forced_final_tool_calls_rejected",
            "model_api_attempt_failed", "model_api_retry_scheduled", "web_search_timeout",
            "tool_execution_timeout", "tool_execution_failed",
            "serper_request_failed", "serper_retry_scheduled", "serper_key_switch_scheduled",
            "serper_key_blacklisted", "web_fetch_fallback", "web_fetch_stage_failed",
            "web_fetch_retry_scheduled", "web_fetch_fallbacks_exhausted",
            "webpage_processing_failed", "evidence_extraction_failed", "evidence_retry_scheduled",
            "evidence_extraction_exhausted",
        } or error_code == "tool_timeout" else "INFO"
        phase = "tool" if event_name.startswith(("tool_arguments", "tool_execution", "web_search", "search_", "serper_", "page_", "web_fetch", "webpage_", "evidence_")) else "agent"
        self.write(event_name, level=level, phase=phase, **event)


def _question_stem(question_number: int) -> str:
    return f"{question_number:06d}"


def _is_completed_log(path: Path) -> bool:
    """Only a valid final state is resumable; failed questions remain eligible for rerun.

    Reads the structured ``.events.jsonl`` sidecar; falls back to the legacy
    key=value log format for run directories created before the split.
    """
    events_path = path.with_suffix(".events.jsonl")
    source = events_path if events_path.exists() else path
    if not source.exists():
        return False
    try:
        for line in source.read_text(encoding="utf-8").splitlines():
            if events_path == source:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if record.get("event") == "question_finished" and record.get("terminal_state") == "valid":
                    return True
            elif "event=question_finished" in line and '"terminal_state":"valid"' in line:
                return True
        return False
    except OSError:
        return False


def _write_trajectory(path: Path, *, messages: list[dict[str, Any]]) -> None:
    """Write only the final attempt's model message sequence, never its retry history."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"messages": messages}
    temporary = path.with_suffix(".json.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _create_loop(loop_factory: Callable[..., Any], event_sink: Callable[[dict[str, Any]], None]) -> Any:
    """Keep legacy no-argument factories usable in direct library callers and older tests."""
    try:
        signature = inspect.signature(loop_factory)
        if "event_sink" in signature.parameters:
            return loop_factory(event_sink=event_sink)
    except (TypeError, ValueError):
        pass
    return loop_factory()


async def run_jsonl(
    input_path: str | Path, logs_dir: str | Path, trajectories_dir: str | Path,
    loop_factory: Callable[..., Any], *, system_prompt: str | Callable[[str], str], max_retries: int = 1,
    concurrency: int = 1, start_index: int | None = None, end_index: int | None = None,
    log_level: str = "INFO",
) -> dict[str, int]:
    """Run a contiguous, inclusive 0-based range of nonblank JSONL records.

    Artifact filenames retain the record's position in the original input.  This
    makes independently run ranges safe to place in the same resumable run.
    """
    if start_index is not None and start_index < 0:
        raise ValueError("start_index must be non-negative")
    if end_index is not None and end_index < 0:
        raise ValueError("end_index must be non-negative")
    if start_index is not None and end_index is not None and start_index > end_index:
        raise ValueError("start_index must be less than or equal to end_index")

    first_index = start_index if start_index is not None else 0
    source, logs, trajectories = Path(input_path), Path(logs_dir), Path(trajectories_dir)
    counts = {"skipped": 0, "succeeded": 0, "failed": 0}
    pending: list[tuple[int, str, str]] = []
    question_number = 0
    for line_number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        question_number += 1
        record_index = question_number - 1
        if record_index < first_index or (end_index is not None and record_index > end_index):
            continue
        row = json.loads(line)
        query_id = str(row.get("id", line_number))
        stem = _question_stem(question_number)
        if _is_completed_log(logs / f"{stem}.log"):
            counts["skipped"] += 1
            continue
        query = row.get("query") or row.get("question")
        if not isinstance(query, str) or not query.strip():
            logger = QuestionLogger(
                logs / f"{stem}.log", question_number=question_number,
                question_id=query_id, level=log_level,
            )
            logger.write("invalid_input", level="ERROR", phase="input", line_number=line_number)
            logger.write("question_finished", level="ERROR", terminal_state="invalid_input")
            counts["failed"] += 1
            continue
        pending.append((question_number, query_id, query))

    semaphore = asyncio.Semaphore(concurrency)
    lock = asyncio.Lock()

    async def execute(number: int, query_id: str, query: str) -> None:
        async with semaphore:
            stem = _question_stem(number)
            logger = QuestionLogger(
                logs / f"{stem}.log", question_number=number,
                question_id=query_id, level=log_level,
            )
            logger.write("question_started", phase="question", query=query,
                         query_sha256=hashlib.sha256(query.encode("utf-8")).hexdigest(),
                         query_chars=len(query), max_attempts=max_retries + 1)
            final = None
            for attempt in range(max_retries + 1):
                logger.attempt = attempt + 1
                logger.write("attempt_started", phase="attempt")
                selected_system_prompt = system_prompt(query) if callable(system_prompt) else system_prompt
                final = await _create_loop(loop_factory, logger.emit_loop_event).run(query, selected_system_prompt)
                logger.write("attempt_finished", phase="attempt", terminal_state=final.terminal_state,
                             elapsed_seconds=final.elapsed_seconds, tool_rounds=final.tool_rounds)
                if final.terminal_state == "valid":
                    break
                if attempt < max_retries:
                    logger.write("retry_scheduled", level="WARNING", phase="recovery",
                                 previous_terminal_state=final.terminal_state, next_attempt=attempt + 2)
            assert final is not None
            _write_trajectory(trajectories / f"{stem}.json", messages=final.messages)
            logger.write("question_finished", level="INFO" if final.terminal_state == "valid" else "ERROR",
                         phase="question", terminal_state=final.terminal_state, attempts=attempt + 1)
            async with lock:
                if final.terminal_state == "valid":
                    counts["succeeded"] += 1
                else:
                    counts["failed"] += 1

    await asyncio.gather(*(execute(number, query_id, query) for number, query_id, query in pending))
    return counts
