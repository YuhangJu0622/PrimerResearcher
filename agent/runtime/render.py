"""Human-readable line rendering for per-question logs.

Every structured event lands verbatim in the machine-readable ``.events.jsonl``
sidecar.  This module decides which events also deserve a line in the human
``.log`` timeline, at what level, and in what wording.  Renderers must tolerate
missing fields: the event vocabulary evolves faster than this table, and an
unknown or partial event must never break a rollout.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

LEVEL_ORDER = {"DEBUG": 10, "INFO": 20, "WARNING": 30, "ERROR": 40}


@dataclass(frozen=True)
class RenderedLine:
    level: str
    component: str
    message: str


def _short(value: Any, limit: int = 60) -> str:
    text = " ".join(str(value if value is not None else "").split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _quote(value: Any, limit: int = 60) -> str:
    return f'"{_short(value, limit)}"'


def _seconds(value: Any) -> str:
    return f"{value:.1f}s" if isinstance(value, (int, float)) else "?"


def _usage(payload: dict[str, Any], key: str) -> str:
    value = (payload.get("usage") or {}).get(key)
    return str(value) if isinstance(value, (int, float)) else "?"


def _arguments_text(value: Any) -> str:
    if isinstance(value, dict) and isinstance(value.get("query"), str):
        return _quote(value["query"], 80)
    return _short(value, 100)


def _network_failure_details(payload: dict[str, Any]) -> str:
    status = payload.get("http_status")
    error_type = payload.get("error_type")
    parts = []
    if isinstance(status, int):
        parts.append(f"HTTP {status}")
    if error_type:
        parts.append(str(error_type))
    return f" [{', '.join(parts)}]" if parts else ""


def _web_fetch_http_response(p: dict[str, Any], _: int, __: int) -> RenderedLine:
    status = p.get("http_status")
    failed = isinstance(status, int) and status >= 400
    return RenderedLine(
        "WARNING" if failed else "DEBUG", "tool",
        f"HTTP {status if status is not None else '?'} via {p.get('source', '?')} "
        f"({p.get('response_chars', '?')} chars)"
        + (f": {_short(p.get('url'), 80)}" if failed else ""),
    )


def _web_fetch_http_error(p: dict[str, Any], _: int, __: int) -> RenderedLine:
    return RenderedLine(
        "WARNING", "tool",
        f"HTTP transport error via {p.get('source', '?')}: "
        f"{p.get('error_type') or p.get('error_code') or '?'}"
        f"{_network_failure_details(p)}: {_short(p.get('url'), 80)}",
    )


# (payload, question_number, attempt) -> RenderedLine, or None for jsonl-only events.
_Renderer = Callable[[dict[str, Any], int, int], "RenderedLine | None"]


def _none(_: dict[str, Any], __: int, ___: int) -> None:
    return None


def _question_started(p: dict[str, Any], question_number: int, _: int) -> RenderedLine:
    return RenderedLine("INFO", "batch",
                        f"Question {question_number:06d} started "
                        f"({p.get('max_attempts', '?')} attempt(s)): {_short(p.get('query'), 120)}")


def _invalid_input(p: dict[str, Any], _: int, __: int) -> RenderedLine:
    return RenderedLine("ERROR", "batch",
                        f"Invalid input record at source line {p.get('line_number', '?')}")


def _question_finished(p: dict[str, Any], _: int, __: int) -> RenderedLine:
    state = str(p.get("terminal_state", "?"))
    level = "INFO" if state == "valid" else "ERROR"
    return RenderedLine(level, "batch",
                        f"Question finished: {state} after {p.get('attempts', '?')} attempt(s)")


_RENDERERS: dict[str, _Renderer] = {
    # ---- batch / lifecycle -------------------------------------------------
    "question_started": _question_started,
    "invalid_input": _invalid_input,
    "attempt_started": lambda _p, _q, attempt: RenderedLine("INFO", "batch", f"Attempt {attempt} started"),
    "attempt_finished": lambda p, _, attempt: RenderedLine(
        "INFO", "batch",
        f"Attempt {attempt} finished: {p.get('terminal_state', '?')} "
        f"(rounds={p.get('tool_rounds', '?')}, {_seconds(p.get('elapsed_seconds'))})"),
    "retry_scheduled": lambda p, _, __: RenderedLine(
        "WARNING", "batch",
        f"Attempt {p.get('next_attempt', '?')} scheduled "
        f"(previous ended: {p.get('previous_terminal_state', '?')})"),
    "question_finished": _question_finished,
    # ---- agent loop --------------------------------------------------------
    "agent_run_started": lambda p, _, __: RenderedLine(
        "INFO", "agent",
        f"Run started (max_rounds={p.get('max_tool_rounds', '?')}, "
        f"tool_timeout={p.get('tool_timeout_seconds', '?')}s, "
        f"concurrency={p.get('tool_concurrency', '?')})"),
    "model_request_started": lambda p, _, __: RenderedLine(
        "DEBUG", "agent",
        f"Model request round {p.get('tool_round', '?')} "
        f"(messages={p.get('message_count', '?')}, prompt~{p.get('prompt_token_count', '?')} tok)"),
    "model_api_attempt_started": lambda p, _, __: RenderedLine(
        "DEBUG", "agent",
        f"Model API attempt {p.get('attempt', '?')}/{p.get('max_attempts', '?')} "
        f"(timeout {p.get('timeout_seconds', '?')}s)"),
    "model_api_attempt_succeeded": lambda p, _, __: RenderedLine(
        "DEBUG", "agent",
        f"Model API attempt {p.get('attempt', '?')} ok ({p.get('elapsed_seconds', '?')}s)"),
    "model_api_attempt_failed": lambda p, _, __: RenderedLine(
        "WARNING", "agent",
        f"Model API attempt {p.get('attempt', '?')}/{p.get('max_attempts', '?')} failed: "
        f"{p.get('error_type', '?')} ({_seconds(p.get('elapsed_seconds'))})"),
    "model_api_retry_scheduled": lambda p, _, __: RenderedLine(
        "DEBUG", "agent", f"Model API retry in {p.get('delay_seconds', '?')}s"),
    "model_request_succeeded": lambda p, _, __: RenderedLine(
        "INFO", "agent",
        f"Round {p.get('tool_round', '?')}: model ok ({_seconds(p.get('elapsed_seconds'))}, "
        f"prompt {_usage(p, 'prompt_tokens')} + completion {_usage(p, 'completion_tokens')} tok)"),
    "model_output": lambda p, _, __: RenderedLine(
        "DEBUG", "agent", f"Output (round {p.get('tool_round', '?')}):\n{p.get('raw', '')}"),
    "model_api_response": lambda p, _, __: RenderedLine(
        "DEBUG", "vllm",
        f"Aggregated API response for round {p.get('tool_round', '?')} "
        f"({p.get('stream_chunk_count', '?')} SSE chunks): {p.get('response')}",
    ),
    "assistant_message_parsed": lambda p, _, __: RenderedLine(
        "DEBUG", "agent",
        f"Assistant parsed (tool_calls={p.get('tool_call_count', 0)}, "
        f"final_content={p.get('has_final_content', False)})"),
    "tool_call_started": lambda p, _, __: RenderedLine(
        "INFO", "agent",
        f"Calling {p.get('tool_name', '?')} [{p.get('call_id', '?')}] "
        f"round {p.get('tool_round', '?')}: {_arguments_text(p.get('arguments'))}"),
    "tool_call_finished": lambda p, _, __: (
        RenderedLine("INFO", "agent",
                     f"Tool [{p.get('call_id', '?')}] ok ({_seconds(p.get('elapsed_seconds'))}, "
                     f"{p.get('search_result_count', 0)} results, {p.get('evidence_count', 0)} evidence)")
        if p.get("status") == "success" else
        RenderedLine("ERROR", "agent",
                     f"Tool [{p.get('call_id', '?')}] failed: {p.get('error_code') or 'error'} "
                     f"({_seconds(p.get('elapsed_seconds'))})")),
    "tool_arguments_validated": _none,
    "tool_execution_timeout": lambda p, _, __: RenderedLine(
        "WARNING", "agent",
        f"Tool [{p.get('call_id', '?')}] execution timed out after {p.get('timeout_seconds', '?')}s"),
    "tool_execution_failed": lambda p, _, __: RenderedLine(
        "ERROR", "agent",
        f"Tool [{p.get('call_id', '?')}] execution failed: {p.get('error_type', '?')}"),
    "generation_error": lambda p, _, __: RenderedLine(
        "ERROR", "agent",
        f"Generation error: {p.get('error', '?')}: {_short(p.get('error_message'), 160)}"),
    "context_limit_exceeded": lambda p, _, __: RenderedLine(
        "ERROR", "agent",
        f"Context limit exceeded ({p.get('token_count', '?')} >= "
        f"{p.get('max_context_tokens', '?')} tok, force_final={p.get('force_final', False)})"),
    "forced_final_entered": lambda p, _, __: RenderedLine(
        "WARNING", "agent",
        f"Forced-final mode entered (reasons: {', '.join(map(str, p.get('trigger_reasons') or []))})"),
    "forced_final_trigger_reason_added": lambda p, _, __: RenderedLine(
        "DEBUG", "agent",
        f"Forced-final reason added: {', '.join(map(str, p.get('trigger_reasons') or []))}"),
    "forced_final_answer_accepted": lambda p, _, __: RenderedLine(
        "INFO", "agent", f"Forced-final answer accepted (attempt {p.get('attempt', '?')})"),
    "forced_final_tool_calls_rejected": lambda p, _, __: RenderedLine(
        "WARNING", "agent",
        f"Forced-final: rejected {p.get('count', '?')} tool call(s) at attempt {p.get('attempt', '?')}"),
    "forced_final_attempt_failed": lambda p, _, __: RenderedLine(
        "WARNING", "agent",
        f"Forced-final attempt {p.get('attempt', '?')}/{p.get('attempt_limit', '?')} rejected: "
        f"{p.get('failure_reason', '?')}"),
    "forced_final_exhausted": lambda p, _, __: RenderedLine(
        "ERROR", "agent",
        f"Forced-final exhausted after {p.get('attempts', '?')} attempt(s) ({p.get('last_failure', '?')})"),
    "runtime_final_answer_required": lambda p, _, __: RenderedLine(
        "DEBUG", "agent",
        f"Final answer flagged required (reasons: {', '.join(map(str, p.get('trigger_reasons') or []))})"),
    "invalid_tool_json": lambda p, _, __: RenderedLine(
        "ERROR", "agent", f"Terminal: invalid tool JSON ({p.get('error', '?')})"),
    "unknown_tool": lambda _p, _q, _a: RenderedLine("ERROR", "agent", "Terminal: unknown tool requested"),
    "no_final_answer": lambda _p, _q, _a: RenderedLine(
        "ERROR", "agent", "Terminal: model returned no final answer"),
    "agent_finished": lambda p, _, __: RenderedLine(
        "INFO", "agent",
        f"Run finished: {p.get('terminal_state', '?')} "
        f"(rounds={p.get('tool_rounds', '?')}, {_seconds(p.get('elapsed_seconds'))})"),
    # ---- tool: search ------------------------------------------------------
    "web_search_started": lambda p, _, __: RenderedLine(
        "DEBUG", "tool", f"web_search start (timeout {p.get('total_timeout_seconds', '?')}s)"),
    "web_search_finished": lambda p, _, __: RenderedLine(
        "DEBUG", "tool", f"web_search ok ({_seconds(p.get('elapsed_seconds'))})"),
    "web_search_timeout": lambda p, _, __: RenderedLine(
        "WARNING", "tool",
        f"web_search total timeout after {p.get('timeout_seconds', '?')}s "
        f"(query {_quote(p.get('query'), 60)})"),
    "web_search_failed": lambda p, _, __: RenderedLine(
        "ERROR", "tool",
        f"web_search failed: {p.get('error_type', '?')}: "
        f"{_short(p.get('error_message'), 120)} ({_seconds(p.get('elapsed_seconds'))})"),
    "search_finished": lambda p, _, __: RenderedLine(
        "WARNING", "tool",
        f"Search ended with status {p.get('status', '?')} (0 results, query {_quote(p.get('query'), 60)})"),
    "search_cache_hit": lambda p, _, __: RenderedLine(
        "DEBUG", "tool", f"Search cache hit ({p.get('result_count', 0)} results)"),
    "search_cache_miss": lambda p, _, __: RenderedLine(
        "DEBUG", "tool", f"Search cache miss ({p.get('cache_mode', '?')})"),
    "search_query_attempt_started": lambda p, _, __: RenderedLine(
        "DEBUG", "tool",
        f"Search variant {p.get('query_variant', '?')}: {_quote(p.get('candidate_query'), 80)}"),
    "search_query_attempt_finished": lambda p, _, __: RenderedLine(
        "INFO", "tool",
        f"Search {_quote(p.get('candidate_query'), 60)} -> {p.get('result_count', 0)} result(s)"),
    "search_replacement_candidate_selected": lambda p, _, __: RenderedLine(
        "DEBUG", "tool",
        f"Replacement candidate rank={p.get('search_rank', '?')} selected "
        f"(evidence={p.get('current_evidence_count', 0)})"),
    "search_results_normalized": lambda p, _, __: RenderedLine(
        "DEBUG", "tool",
        f"Normalized {p.get('raw_result_count', '?')} -> {p.get('normalized_result_count', '?')} results"),
    "web_search_aggregation_finished": lambda p, _, __: RenderedLine(
        "DEBUG", "tool",
        f"Aggregation: {p.get('search_result_count', '?')} results, "
        f"{p.get('evidence_count', '?')} evidence, truncated={p.get('truncated', False)}"),
    # ---- tool: serper ------------------------------------------------------
    "serper_request_started": lambda p, _, __: RenderedLine(
        "DEBUG", "tool",
        f"Serper key {p.get('key_index', '?')} attempt {p.get('attempt', '?')}/{p.get('max_attempts', '?')}"),
    "serper_request_succeeded": lambda p, _, __: RenderedLine(
        "DEBUG", "tool",
        f"Serper key {p.get('key_index', '?')} attempt {p.get('attempt', '?')} ok "
        f"({p.get('result_count', 0)} results, {_seconds(p.get('elapsed_seconds'))})"),
    "serper_request_failed": lambda p, _, __: RenderedLine(
        "WARNING", "tool",
        f"Serper key {p.get('key_index', '?')} attempt {p.get('attempt', '?')} failed: "
        f"{p.get('error_type', '?')}: {_short(p.get('error_message'), 80)} "
        f"({_seconds(p.get('elapsed_seconds'))})"
        + (", will retry" if p.get("retryable") else "")),
    "serper_retry_scheduled": lambda p, _, __: RenderedLine(
        "DEBUG", "tool",
        f"Serper retry in {p.get('delay_seconds', '?')}s (next attempt {p.get('next_attempt', '?')})"),
    "serper_key_switch_scheduled": lambda p, _, __: RenderedLine(
        "WARNING", "tool",
        f"Serper switching key {p.get('key_index', '?')} (HTTP {p.get('http_status', '?')}, "
        f"{p.get('reason', '?')})"),
    "serper_key_blacklisted": lambda p, _, __: RenderedLine(
        "WARNING", "tool",
        f"Serper key {p.get('key_index', '?')} blacklisted "
        f"(HTTP {p.get('http_status', '?')}: {_short(p.get('reason'), 80)})"),
    # ---- tool: page fetch --------------------------------------------------
    "webpage_processing_started": lambda p, _, __: RenderedLine(
        "DEBUG", "tool",
        f"Page rank={p.get('search_rank', '?')}: {_short(p.get('title'), 60)}"),
    "webpage_processing_failed": lambda p, _, __: RenderedLine(
        "WARNING", "tool",
        f"Page rank={p.get('search_rank', '?')} failed at {p.get('stage', '?')} "
        f"({p.get('error_code') or p.get('error_type') or '?'}) {_short(p.get('url'), 70)}"),
    "webpage_processing_succeeded": lambda p, _, __: RenderedLine(
        "DEBUG", "tool",
        f"Page rank={p.get('search_rank', '?')} ok ({p.get('page_chars', '?')} chars, "
        f"evidence {p.get('evidence_chars', '?')} chars)"),
    "page_cache_hit": lambda p, _, __: RenderedLine(
        "DEBUG", "tool", f"Page cache hit ({p.get('page_chars', '?')} chars)"),
    "page_cache_miss": lambda p, _, __: RenderedLine(
        "DEBUG", "tool", f"Page cache miss ({p.get('cache_mode', '?')})"),
    "page_cached": lambda p, _, __: RenderedLine(
        "DEBUG", "tool",
        f"Page cached ({p.get('page_chars', '?')} chars, truncated={p.get('truncated', False)})"),
    "web_fetch_stage_started": lambda p, _, __: RenderedLine(
        "DEBUG", "tool", f"Fetch stage {p.get('source', '?')} (timeout {p.get('stage_timeout_seconds', '?')}s)"),
    "web_fetch_stage_succeeded": lambda p, _, __: RenderedLine(
        "INFO", "tool",
        f"Visit ok via {p.get('source', '?')} ({p.get('content_chars', '?')} chars): "
        f"{_short(p.get('url'), 80)}"),
    "web_fetch_stage_failed": lambda p, _, __: RenderedLine(
        "WARNING", "tool",
        f"Fetch failed via {p.get('source', '?')} ({p.get('error_code', '?')}), "
        f"action={p.get('action', '?')}"
        f"{_network_failure_details(p)}: {_short(p.get('url'), 80)}"),
    "web_fetch_fallback": lambda p, _, __: RenderedLine(
        "DEBUG", "tool", f"Falling back from {p.get('source', '?')} ({p.get('error_code', '?')})"),
    "web_fetch_fallbacks_exhausted": lambda p, _, __: RenderedLine(
        "WARNING", "tool", f"Fetch exhausted all sources: {_short(p.get('url'), 80)}"),
    "web_fetch_attempt_started": lambda p, _, __: RenderedLine(
        "DEBUG", "tool",
        f"Fetch attempt {p.get('attempt', '?')}/{p.get('max_attempts', '?')} via {p.get('source', '?')} "
        f"({p.get('remaining_seconds', '?')}s left)"),
    "web_fetch_attempt_finished": lambda p, _, __: RenderedLine(
        "DEBUG", "tool",
        f"Fetch attempt {p.get('attempt', '?')} via {p.get('source', '?')}: "
        f"{p.get('action', '?')} ({p.get('content_chars', 0)} chars)"),
    "web_fetch_retry_scheduled": lambda p, _, __: RenderedLine(
        "DEBUG", "tool",
        f"Fetch retry in {p.get('delay_seconds', '?')}s (next attempt {p.get('next_attempt', '?')})"),
    "web_fetch_http_response": _web_fetch_http_response,
    "web_fetch_http_error": _web_fetch_http_error,
    "web_fetch_redirect": lambda p, _, __: RenderedLine(
        "DEBUG", "tool", f"Redirect to {_short(p.get('redirect_url'), 80)}"),
    # ---- tool: evidence extraction ----------------------------------------
    "evidence_cache_hit": lambda p, _, __: RenderedLine(
        "DEBUG", "tool", f"Evidence cache hit ({p.get('evidence_chars', '?')} chars)"),
    "evidence_cache_miss": _none,
    "evidence_extraction_started": lambda p, _, __: RenderedLine(
        "DEBUG", "tool",
        f"Evidence rank={p.get('search_rank', '?')} attempt {p.get('attempt', '?')}/"
        f"{p.get('max_attempts', '?')} ({p.get('model', '?')})"),
    "evidence_extraction_succeeded": lambda p, _, __: RenderedLine(
        "INFO", "tool",
        f"Evidence rank={p.get('search_rank', '?')} ok "
        f"({p.get('evidence_chars', '?')} chars, {_seconds(p.get('elapsed_seconds'))})"),
    "evidence_extraction_failed": lambda p, _, __: RenderedLine(
        "WARNING", "tool",
        f"Evidence rank={p.get('search_rank', '?')} attempt {p.get('attempt', '?')}/"
        f"{p.get('max_attempts', '?')} failed: {p.get('error_type', '?')} "
        f"({_seconds(p.get('elapsed_seconds'))})"),
    "evidence_retry_scheduled": lambda p, _, __: RenderedLine(
        "DEBUG", "tool",
        f"Evidence retry in {p.get('delay_seconds', '?')}s (next attempt {p.get('next_attempt', '?')})"),
    "evidence_extraction_exhausted": lambda p, _, __: RenderedLine(
        "WARNING", "tool",
        f"Evidence rank={p.get('search_rank', '?')} exhausted ({p.get('attempts', '?')} attempts)"),
}


def render_event(event: str, payload: dict[str, Any], *,
                 question_number: int = 0, attempt: int = 0) -> RenderedLine | None:
    """Return the human-log line for an event, or None when it is jsonl-only."""
    renderer = _RENDERERS.get(event)
    if renderer is None:
        return None
    try:
        return renderer(payload, question_number, attempt)
    except Exception:
        # Rendering must never fail a rollout; the jsonl record still carries everything.
        return None
