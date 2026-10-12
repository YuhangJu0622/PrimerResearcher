"""Canonical DeepResearch Agent state machine."""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Callable

from agent.contracts.messages import AssistantMessage, ToolMessage, message_to_dict, normalize_trajectory
from agent.contracts.qwen3_codec import Qwen3Codec
from agent.contracts.tools import WEB_SEARCH_TOOL, validate_web_search_arguments


class TerminalState(str, Enum):
    VALID = "valid"
    INVALID_TOOL_JSON = "invalid_tool_json"
    UNKNOWN_TOOL = "unknown_tool"
    TOOL_TIMEOUT = "tool_timeout"
    TOOL_ERROR = "tool_error"
    CONTEXT_OVERFLOW = "context_overflow"
    NO_FINAL_ANSWER = "no_final_answer"
    GENERATION_ERROR = "generation_error"
    MAX_ROUNDS_EXCEEDED = "max_rounds_exceeded"
    FORCED_FINAL_EXHAUSTED = "forced_final_exhausted"


@dataclass
class AgentResult:
    terminal_state: str
    messages: list[dict[str, Any]]
    events: list[dict[str, Any]] = field(default_factory=list)
    raw_response_hashes: list[str] = field(default_factory=list)
    tool_rounds: int = 0
    elapsed_seconds: float = 0.0
    token_usage: dict[str, int] = field(default_factory=dict)
    forced_final_reasons: list[str] = field(default_factory=list)
    forced_final_attempts: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class AgentLoop:
    def __init__(
        self, model: Any, web_search: Callable[..., Any], *, codec: Qwen3Codec | None = None,
        max_tool_rounds: int = 10, max_context_tokens: int | None = None,
        token_counter: Callable[[list[dict[str, Any]]], int] | None = None,
        max_output_tokens: int | None = None, forced_final_trigger_tokens: int | None = None,
        forced_final_attempts: int = 2, forced_final_answer_reserve_tokens: int = 1024,
        forced_final_overhead_tokens: int = 1024,
        tool_timeout: float = 30.0, tool_concurrency: int = 3,
        as_of_time: str | None = None,
        event_sink: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        if not 1 <= max_tool_rounds <= 10:
            raise ValueError("max_tool_rounds must be in 1..10")
        if forced_final_attempts < 1:
            raise ValueError("forced_final_attempts must be positive")
        if forced_final_answer_reserve_tokens < 1 or forced_final_overhead_tokens < 1:
            raise ValueError("forced-final token reserves must be positive")
        if max_output_tokens is not None and max_output_tokens < 1:
            raise ValueError("max_output_tokens must be positive when configured")
        if max_context_tokens is not None:
            if token_counter is None:
                raise ValueError("max_context_tokens requires a tokenizer-backed token_counter")
            if forced_final_answer_reserve_tokens + forced_final_overhead_tokens >= max_context_tokens:
                raise ValueError("forced-final token reserves must fit within max_context_tokens")
            if forced_final_trigger_tokens is not None:
                safe_trigger = max_context_tokens - forced_final_answer_reserve_tokens - forced_final_overhead_tokens
                if forced_final_trigger_tokens > safe_trigger:
                    raise ValueError("forced_final_trigger_tokens leaves insufficient final-answer space")
        self.model = model
        self.web_search = web_search
        self.codec = codec or Qwen3Codec()
        self.max_tool_rounds = max_tool_rounds
        self.max_context_tokens = max_context_tokens
        self.token_counter = token_counter
        self.max_output_tokens = max_output_tokens
        self.forced_final_trigger_tokens = forced_final_trigger_tokens
        self.forced_final_attempts = forced_final_attempts
        self.forced_final_answer_reserve_tokens = forced_final_answer_reserve_tokens
        self.forced_final_overhead_tokens = forced_final_overhead_tokens
        self.tool_timeout = tool_timeout
        self.tool_concurrency = tool_concurrency
        self.as_of_time = as_of_time
        self.event_sink = event_sink

    async def run(self, query: str, system_prompt: str) -> AgentResult:
        started = time.monotonic()
        if self.as_of_time:
            system_prompt += f"\n\nUse this frozen as-of time: {self.as_of_time}."
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ]
        events: list[dict[str, Any]] = []
        hashes: list[str] = []
        usage: dict[str, int] = {}
        tool_rounds = 0
        forced_final_reasons: list[str] = []
        forced_final_attempt_count = 0

        self._emit(events, {
            "type": "agent_run_started", "query": query, "query_chars": len(query),
            "max_tool_rounds": self.max_tool_rounds, "tool_timeout_seconds": self.tool_timeout,
            "tool_concurrency": self.tool_concurrency, "max_context_tokens": self.max_context_tokens,
            "max_output_tokens": self.max_output_tokens,
        })

        def finish(state: TerminalState) -> AgentResult:
            self._emit(events, {
                "type": "agent_finished", "terminal_state": state.value,
                "tool_rounds": tool_rounds,
                "forced_final_reasons": forced_final_reasons,
                "forced_final_attempts": forced_final_attempt_count,
                "elapsed_seconds": round(time.monotonic() - started, 6),
            })
            return self._finish(
                state, messages, events, hashes, tool_rounds, usage, started,
                forced_final_reasons, forced_final_attempt_count,
            )

        def activate_forced_final(reasons: list[str], *, context_token_count: int | None) -> None:
            new_reasons = [reason for reason in reasons if reason not in forced_final_reasons]
            if not new_reasons:
                return
            forced_final_reasons.extend(new_reasons)
            self._emit(events, {
                "type": "forced_final_entered" if len(forced_final_reasons) == len(new_reasons)
                else "forced_final_trigger_reason_added",
                "trigger_reasons": list(forced_final_reasons),
                "max_tool_rounds": self.max_tool_rounds,
                "tool_rounds": tool_rounds,
                "max_context_tokens": self.max_context_tokens,
                "context_token_count": context_token_count,
                "forced_final_attempts_allowed": self.forced_final_attempts,
            })

        def reject_forced_final(reason: str, **details: Any) -> bool:
            self._emit(events, {
                "type": "forced_final_attempt_failed",
                "attempt": forced_final_attempt_count,
                "attempt_limit": self.forced_final_attempts,
                "failure_reason": reason,
                "trigger_reasons": list(forced_final_reasons),
                **details,
            })
            if forced_final_attempt_count < self.forced_final_attempts:
                return False
            self._emit(events, {
                "type": "forced_final_exhausted",
                "attempts": forced_final_attempt_count,
                "trigger_reasons": list(forced_final_reasons),
                "last_failure": reason,
            })
            return True

        def build_request() -> list[dict[str, Any]]:
            """Return the stable, persisted conversation without runtime prompt mutation."""
            return [dict(x) for x in messages]

        def require_final_answer(reasons: list[str]) -> bool:
            """Attach the runtime budget signal to the most recent real tool-result batch."""
            recent_tools: list[dict[str, Any]] = []
            for message in reversed(messages):
                if message.get("role") == "tool":
                    recent_tools.append(message)
                    continue
                break
            if not recent_tools:
                return False

            changed_call_ids: list[str] = []
            for message in recent_tools:
                try:
                    observation = json.loads(str(message["content"]))
                except (KeyError, TypeError, json.JSONDecodeError):
                    continue
                if not isinstance(observation, dict) or "tool_result" not in observation:
                    continue
                runtime = observation.get("runtime")
                if not isinstance(runtime, dict):
                    runtime = {}
                    observation["runtime"] = runtime
                current_reasons = runtime.get("final_answer_reasons")
                current_reasons = current_reasons if isinstance(current_reasons, list) else []
                merged_reasons = list(dict.fromkeys([*current_reasons, *reasons]))
                if not runtime.get("final_answer_required") or current_reasons != merged_reasons:
                    runtime["final_answer_required"] = True
                    runtime["final_answer_reasons"] = merged_reasons
                    message["content"] = json.dumps(observation, ensure_ascii=False, separators=(",", ":"))
                    changed_call_ids.append(str(message.get("tool_call_id", "")))
            if changed_call_ids:
                self._emit(events, {
                    "type": "runtime_final_answer_required",
                    "call_ids": list(reversed(changed_call_ids)),
                    "trigger_reasons": list(reasons),
                    "tool_rounds": tool_rounds,
                })
            return bool(recent_tools)

        while True:
            context_token_count: int | None = None
            trigger_reasons: list[str] = []
            if self.max_context_tokens is not None and self.token_counter is not None:
                # Count precisely what will be sent: the Qwen chat template, frozen tool
                # schema and runtime system prompt are all included by token_counter.
                context_token_count = self.token_counter(build_request())
                if context_token_count >= self.max_context_tokens:
                    self._emit(events, {
                        "type": "context_limit_exceeded",
                        "token_count": context_token_count,
                        "max_context_tokens": self.max_context_tokens,
                    })
                    return finish(TerminalState.CONTEXT_OVERFLOW)
                force_threshold = self.forced_final_trigger_tokens or (
                    self.max_context_tokens - self.forced_final_answer_reserve_tokens
                    - self.forced_final_overhead_tokens
                )
                if context_token_count >= force_threshold:
                    trigger_reasons.append("context_budget")
            if tool_rounds >= self.max_tool_rounds:
                trigger_reasons.append("max_tool_rounds")
            if trigger_reasons and not require_final_answer(trigger_reasons):
                # A tool observation is the only model-visible carrier for the runtime
                # finalization signal.  Do not fabricate a tool result for an oversized
                # initial system/user prompt.
                self._emit(events, {
                    "type": "context_limit_exceeded",
                    "token_count": context_token_count,
                    "max_context_tokens": self.max_context_tokens,
                    "reason": "finalization_requires_a_tool_observation",
                })
                return finish(TerminalState.CONTEXT_OVERFLOW)
            activate_forced_final(trigger_reasons, context_token_count=context_token_count)
            force_final = bool(forced_final_reasons)
            request_messages = build_request()
            request_token_count: int | None = None
            request_max_tokens = self.forced_final_answer_reserve_tokens if force_final else self.max_output_tokens
            if self.max_context_tokens is not None and self.token_counter is not None:
                request_token_count = self.token_counter(request_messages)
                available_output_tokens = self.max_context_tokens - request_token_count
                if available_output_tokens < 1:
                    self._emit(events, {
                        "type": "context_limit_exceeded", "token_count": request_token_count,
                        "max_context_tokens": self.max_context_tokens, "force_final": force_final,
                    })
                    return finish(TerminalState.CONTEXT_OVERFLOW)
                if request_max_tokens is not None:
                    request_max_tokens = min(request_max_tokens, available_output_tokens)
            try:
                model_started = time.monotonic()
                self._emit(events, {"type": "model_request_started", "tool_round": tool_rounds,
                                    "message_count": len(request_messages), "force_final": force_final,
                                    "forced_final_attempt": forced_final_attempt_count + 1 if force_final else None,
                                    "prompt_token_count": request_token_count,
                                    "max_output_tokens": request_max_tokens})
                raw, response_usage = await self._generate(
                    request_messages, max_tokens=request_max_tokens,
                    event_sink=lambda event: self._emit(events, {
                        **event, "tool_round": tool_rounds, "force_final": force_final,
                    }),
                )
                response_hash = hashlib.sha256(raw.encode("utf-8")).hexdigest()
                hashes.append(response_hash)
                for key, value in response_usage.items():
                    if isinstance(value, (int, float)) and not isinstance(value, bool):
                        usage[key] = usage.get(key, 0) + int(value)
                self._emit(events, {"type": "model_request_succeeded", "tool_round": tool_rounds,
                                    "response_sha256": response_hash, "usage": response_usage,
                                    "elapsed_seconds": round(time.monotonic() - model_started, 6)})
                self._emit(events, {"type": "model_output", "tool_round": tool_rounds, "raw": raw})
            except Exception as exc:
                self._emit(events, {"type": "generation_error", "error": type(exc).__name__,
                                    "error_message": str(exc)[:240], "force_final": force_final,
                                    "elapsed_seconds": round(time.monotonic() - model_started, 6)})
                return finish(TerminalState.GENERATION_ERROR)
            if force_final:
                forced_final_attempt_count += 1
            try:
                assistant = self.codec.parse_assistant(raw, self._next_call_index(messages))
            except Exception as exc:
                if force_final:
                    failure_reason = "empty_response" if not raw.strip() else "parse_error"
                    if reject_forced_final(failure_reason, error=type(exc).__name__):
                        return finish(TerminalState.FORCED_FINAL_EXHAUSTED)
                    continue
                state = self._parse_error_state(raw)
                self._emit(events, {"type": state.value, "error": type(exc).__name__})
                return finish(state)
            self._emit(events, {"type": "assistant_message_parsed", "tool_call_count": len(assistant.tool_calls or []),
                                "has_final_content": bool(assistant.content and assistant.content.strip())})
            if assistant.tool_calls:
                if force_final:
                    messages.append(message_to_dict(assistant))
                    for call in assistant.tool_calls:
                        content = json.dumps({
                            "status": "error",
                            "error": {
                                "code": "tool_budget_exhausted",
                                "message": "Tool execution is unavailable because finalization mode is active. "
                                           "Provide the best final answer from the existing context.",
                            },
                        }, ensure_ascii=False, separators=(",", ":"))
                        messages.append(message_to_dict(ToolMessage(role="tool", tool_call_id=call.id, content=content)))
                    self._emit(events, {
                        "type": "forced_final_tool_calls_rejected",
                        "count": len(assistant.tool_calls),
                        "reason": "tool_budget_exhausted",
                        "attempt": forced_final_attempt_count,
                    })
                    if reject_forced_final("tool_call", tool_call_count=len(assistant.tool_calls)):
                        return finish(TerminalState.FORCED_FINAL_EXHAUSTED)
                    continue
                messages.append(message_to_dict(assistant))
                tool_rounds += 1
                runtime = {
                    "tool_rounds_used": tool_rounds,
                    "tool_rounds_remaining": max(0, self.max_tool_rounds - tool_rounds),
                    "final_answer_required": tool_rounds >= self.max_tool_rounds,
                    "final_answer_reasons": ["max_tool_rounds"] if tool_rounds >= self.max_tool_rounds else [],
                }
                semaphore = asyncio.Semaphore(self.tool_concurrency)
                async def bounded(call: Any) -> tuple[str, dict[str, Any]]:
                    async with semaphore:
                        raw_arguments = call.function.arguments
                        try:
                            logged_arguments = json.loads(raw_arguments) if isinstance(raw_arguments, str) else raw_arguments
                        except json.JSONDecodeError:
                            logged_arguments = raw_arguments
                        self._emit(events, {"type": "tool_call_started", "call_id": call.id,
                                            "tool_name": call.function.name, "tool_round": tool_rounds,
                                            "arguments": logged_arguments,
                                            "arguments_raw": raw_arguments if isinstance(raw_arguments, str) else None})
                        return await self._execute_tool(
                            call, runtime=runtime,
                            event_sink=lambda event: self._emit(events, {
                                **event, "call_id": call.id, "tool_name": call.function.name,
                                "tool_round": tool_rounds,
                            }),
                        )
                observations = await asyncio.gather(*(bounded(call) for call in assistant.tool_calls))
                for call, (content, event) in zip(assistant.tool_calls, observations):
                    messages.append(message_to_dict(ToolMessage(role="tool", tool_call_id=call.id, content=content)))
                    self._emit(events, event)
                continue
            messages.append(message_to_dict(assistant))
            if assistant.content and assistant.content.strip():
                if force_final:
                    self._emit(events, {
                        "type": "forced_final_answer_accepted",
                        "attempt": forced_final_attempt_count,
                        "trigger_reasons": list(forced_final_reasons),
                    })
                normalized = normalize_trajectory(messages)
                messages = normalized
                return finish(TerminalState.VALID)
            return finish(TerminalState.NO_FINAL_ANSWER)

    async def _generate(
        self, messages: list[dict[str, Any]], *, max_tokens: int | None = None,
        event_sink: Callable[[dict[str, Any]], None] | None = None,
    ) -> tuple[str, dict[str, int]]:
        method = getattr(self.model, "complete", self.model)
        kwargs: dict[str, Any] = {}
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        if self._accepts_keyword(method, "event_sink"):
            kwargs["event_sink"] = event_sink
        value = method(messages, [WEB_SEARCH_TOOL], **kwargs)
        if inspect.isawaitable(value):
            value = await value
        if isinstance(value, str):
            return value, {}
        if isinstance(value, dict):
            return str(value["content"]), dict(value.get("usage") or {})
        # OpenAI ChatCompletion-compatible response.
        message = value.choices[0].message
        reasoning = (
            getattr(message, "reasoning", None)
            or getattr(message, "reasoning_content", None)
        )
        if message.tool_calls:
            calls = []
            for item in message.tool_calls:
                try:
                    arguments = json.loads(item.function.arguments)
                except json.JSONDecodeError:
                    arguments = item.function.arguments
                calls.append({"name": item.function.name, "arguments": arguments})
            reasoning_block = f"<think>{reasoning}</think>" if reasoning else ""
            if not reasoning_block:
                # With Qwen3 reasoning + Hermes tool parsing, vLLM can return
                # a structured ``tool_calls`` value while leaving the literal
                # reasoning block in ``message.content`` on post-tool turns.
                # Do not discard that block merely because a tool call exists.
                content = message.content or ""
                think_start = content.find("<think>")
                think_end = content.find("</think>", think_start + len("<think>"))
                if think_start >= 0 and think_end >= 0:
                    reasoning_block = content[
                        think_start:think_end + len("</think>")
                    ]
            raw = reasoning_block + "".join(
                f"<tool_call>{json.dumps(x, ensure_ascii=False)}</tool_call>" for x in calls
            )
        else:
            content = message.content or ""
            # Some endpoints keep Qwen's literal <think> block in content,
            # while reasoning-parser deployments return it in a separate
            # reasoning/reasoning_content field.  Preserve either form without
            # duplicating an already embedded block.
            raw = f"<think>{reasoning}</think>{content}" if reasoning and "<think>" not in content else content
        raw_usage = getattr(value, "usage", None)
        usage = raw_usage.model_dump(exclude_none=True) if raw_usage and hasattr(raw_usage, "model_dump") else {}
        return raw, usage

    async def _execute_tool(
        self, call: Any, *, runtime: dict[str, Any],
        event_sink: Callable[[dict[str, Any]], None] | None = None,
    ) -> tuple[str, dict[str, Any]]:
        started = time.monotonic()
        value: Any = None
        try:
            arguments = validate_web_search_arguments(call.function.arguments)
            if event_sink is not None:
                event_sink({"type": "tool_arguments_validated", "arguments": arguments})
            tool_kwargs: dict[str, Any] = {}
            if self._accepts_keyword(self.web_search, "event_sink"):
                tool_kwargs["event_sink"] = event_sink
            value = self.web_search(arguments, **tool_kwargs)
            if inspect.isawaitable(value):
                value = await asyncio.wait_for(value, timeout=self.tool_timeout)
            status = "success" if value.get("status") == "ok" else "error"
            code = (value.get("error") or {}).get("code")
        except asyncio.TimeoutError:
            code, status = "tool_timeout", "error"
            value = {"status": "error", "error": {"code": code, "message": "tool timeout"}}
            if event_sink is not None:
                event_sink({"type": "tool_execution_timeout", "timeout_seconds": self.tool_timeout})
        except Exception as exc:
            code, status = "tool_error", "error"
            value = {"status": "error", "error": {"code": code, "message": type(exc).__name__}}
            if event_sink is not None:
                event_sink({"type": "tool_execution_failed", "error_type": type(exc).__name__,
                            "error_message": str(exc)[:240]})
        content = json.dumps({"tool_result": value, "runtime": runtime}, ensure_ascii=False, separators=(",", ":"))
        event = {"type": "tool_call_finished", "call_id": call.id, "tool_name": call.function.name,
                 "status": status, "error_code": code,
                 "elapsed_seconds": round(time.monotonic() - started, 6)}
        if isinstance(value, dict):
            # Keep the log useful for recovery diagnosis without duplicating query text or webpage bodies.
            event["search_result_count"] = len(value.get("search_results") or [])
            event["evidence_count"] = len(value.get("evidence") or [])
            event["fetch_attempts"] = [
                {key: item.get(key) for key in ("search_rank", "status", "error_code")}
                for item in value.get("fetch_attempts") or [] if isinstance(item, dict)
            ]
            event["truncated"] = bool(value.get("truncated", False))
        return content, event

    @staticmethod
    def _accepts_keyword(function: Callable[..., Any], keyword: str) -> bool:
        """Support lightweight test/custom callables while tracing production clients."""
        try:
            signature = inspect.signature(function)
        except (TypeError, ValueError):
            return False
        return keyword in signature.parameters or any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        )

    def _emit(self, events: list[dict[str, Any]], event: dict[str, Any]) -> None:
        """Record an auditable event without allowing an optional log sink to break a rollout."""
        events.append(event)
        if self.event_sink is not None:
            try:
                self.event_sink(dict(event))
            except Exception:
                # Observability failures must not turn a valid model answer into a failed sample.
                pass

    @staticmethod
    def _next_call_index(messages: list[dict[str, Any]]) -> int:
        return 1 + sum(len(x.get("tool_calls") or []) for x in messages if x.get("role") == "assistant")

    @staticmethod
    def _parse_error_state(raw: str) -> TerminalState:
        if "<tool_call>" in raw:
            match = _first_tool_payload(raw)
            if match is not None:
                try:
                    value = json.loads(match)
                    if isinstance(value, dict) and value.get("name") != "web_search":
                        return TerminalState.UNKNOWN_TOOL
                except json.JSONDecodeError:
                    pass
            return TerminalState.INVALID_TOOL_JSON
        return TerminalState.NO_FINAL_ANSWER

    @staticmethod
    def _finish(state: TerminalState, messages: list[dict[str, Any]], events: list[dict[str, Any]],
                hashes: list[str], rounds: int, usage: dict[str, int], started: float,
                forced_final_reasons: list[str], forced_final_attempts: int) -> AgentResult:
        return AgentResult(
            state.value, messages, events, hashes, rounds,
            round(time.monotonic() - started, 6), usage,
            list(forced_final_reasons), forced_final_attempts,
        )


def _first_tool_payload(raw: str) -> str | None:
    start = raw.find("<tool_call>")
    end = raw.find("</tool_call>", start + 11)
    return raw[start + 11:end].strip() if start >= 0 and end > start else None
