"""Lifecycle-owned OpenAI-compatible model client.

The model call is streamed (`stream=True`) so that `timeout` bounds the gap
between chunks rather than total generation time.  A thinking model that takes
>timeout seconds to finish a long completion still streams token deltas steadily
within that window, so streaming removes the spurious "deterministic 60s
timeout on long thinking rounds" failure mode while still catching genuine
stalls.  The streamed chunks are reassembled into a ChatCompletion-shaped
object so callers (``AgentLoop._generate``) can keep using
``value.choices[0].message`` / ``value.usage`` unchanged.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

import httpx

from openai import AsyncOpenAI


@dataclass
class _Function:
    name: str = ""
    arguments: str = ""


@dataclass
class _ToolCall:
    id: str = ""
    type: str = "function"
    function: _Function = field(default_factory=_Function)


@dataclass
class _Message:
    content: str | None = ""
    tool_calls: list[_ToolCall] | None = None
    reasoning_content: str | None = None


@dataclass
class _Choice:
    message: _Message = field(default_factory=_Message)


@dataclass
class _Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    def model_dump(self, *, exclude_none: bool = False) -> dict[str, int]:
        data = {"prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "total_tokens": self.total_tokens}
        if exclude_none:
            data = {k: v for k, v in data.items() if v is not None}
        return data


@dataclass
class _ChatCompletion:
    choices: list[_Choice] = field(default_factory=lambda: [_Choice()])
    usage: _Usage | None = None


class OpenAIModelClient:
    def __init__(
        self, *, base_url: str, api_key: str, model: str, timeout: float = 60,
        max_retries: int = 2, enable_thinking: bool = True,
        debug_response_logging: bool = False, **parameters,
    ):
        # Retry here rather than inside the SDK so every retry is observable in
        # the per-question trace.  The SDK's retry loop otherwise hides the
        # timeout / status that prompted it.
        # Structured timeout: connect/pool fail fast (10s) so a dead endpoint
        # surfaces quickly, while read tolerates the long inter-chunk gaps that
        # thinking models and a shared upstream endpoint produce under load.
        # `timeout` here is treated as the read budget; connect stays short.
        self.client = AsyncOpenAI(
            base_url=base_url, api_key=api_key,
            timeout=httpx.Timeout(connect=10.0, read=timeout, write=30.0, pool=30.0),
            max_retries=0,
        )
        self.model = model
        self.parameters = parameters
        self.enable_thinking = enable_thinking
        self.debug_response_logging = debug_response_logging
        self.timeout = timeout
        self.max_retries = max(0, max_retries)

    async def complete(
        self, messages, tools, *, max_tokens: int | None = None,
        event_sink: Callable[[dict[str, Any]], None] | None = None,
    ):
        parameters = dict(self.parameters)
        if max_tokens is not None:
            parameters["max_tokens"] = max_tokens

        def emit(event_type: str, **payload: Any) -> None:
            if event_sink is None:
                return
            try:
                event_sink({"type": event_type, **payload})
            except Exception:
                # Logging must never change a model request's outcome.
                pass

        for attempt in range(1, self.max_retries + 2):
            started = time.monotonic()
            emit("model_api_attempt_started", model=self.model, attempt=attempt,
                 max_attempts=self.max_retries + 1, timeout_seconds=self.timeout)
            try:
                response = await self._stream_completion(
                    model=self.model, messages=messages, tools=tools,
                    enable_thinking=self.enable_thinking,
                    response_event_sink=(
                        (lambda payload: emit("model_api_response", **payload))
                        if self.debug_response_logging else None
                    ),
                    **parameters,
                )
                emit("model_api_attempt_succeeded", model=self.model, attempt=attempt,
                     elapsed_seconds=round(time.monotonic() - started, 6))
                return response
            except Exception as exc:
                status_code = getattr(exc, "status_code", None)
                retryable = self._is_retryable(exc, status_code)
                elapsed = round(time.monotonic() - started, 6)
                emit("model_api_attempt_failed", model=self.model, attempt=attempt,
                     max_attempts=self.max_retries + 1, elapsed_seconds=elapsed,
                     error_type=type(exc).__name__, status_code=status_code,
                     retryable=retryable)
                if not retryable or attempt > self.max_retries:
                    raise
                delay = min(0.5 * (2 ** (attempt - 1)), 4.0)
                emit("model_api_retry_scheduled", model=self.model, attempt=attempt,
                     next_attempt=attempt + 1, delay_seconds=delay,
                     error_type=type(exc).__name__, status_code=status_code)
                await asyncio.sleep(delay)

    async def _stream_completion(
        self, *, model: str, messages, tools, enable_thinking: bool,
        response_event_sink: Callable[[dict[str, Any]], None] | None = None,
        **parameters,
    ) -> _ChatCompletion:
        """Reassemble a streamed chat completion into a non-stream-shaped object.

        ``stream_options={"include_usage": True}`` asks the endpoint to emit a
        final chunk carrying aggregate usage; absent that, usage stays zeroed
        (callers tolerate a missing usage dict).
        """
        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        api_reasoning_parts: list[str] = []
        api_reasoning_content_parts: list[str] = []
        tool_calls: dict[int, _ToolCall] = {}
        usage = _Usage()
        finish_reason: str | None = None
        response_metadata: dict[str, Any] = {}

        stream = await self.client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            stream=True,
            stream_options={"include_usage": True},
            # Qwen's OpenAI-compatible API accepts ``enable_thinking`` as a
            # top-level extension field.  Keep the wire shape consistent for
            # every model endpoint used by this agent.
            extra_body={"enable_thinking": enable_thinking},
            **parameters,
        )
        chunk_index = 0
        async for chunk in stream:
            chunk_index += 1
            if response_event_sink is not None:
                chunk_value = self._json_value(chunk)
                if isinstance(chunk_value, dict):
                    response_metadata = {
                        key: value for key, value in chunk_value.items()
                        if key not in {"choices", "usage"}
                    }
                for choice in getattr(chunk, "choices", None) or []:
                    if getattr(choice, "finish_reason", None):
                        finish_reason = choice.finish_reason
                    delta = getattr(choice, "delta", None)
                    if delta is None:
                        continue
                    current_reasoning = getattr(delta, "reasoning", None)
                    legacy_reasoning = getattr(delta, "reasoning_content", None)
                    if current_reasoning:
                        api_reasoning_parts.append(current_reasoning)
                    if legacy_reasoning:
                        api_reasoning_content_parts.append(legacy_reasoning)
            self._merge_chunk(
                chunk, content_parts, reasoning_parts, tool_calls, usage,
            )

        message = _Message(
            content="".join(content_parts) or None,
            tool_calls=list(tool_calls.values()) or None if tool_calls else None,
            reasoning_content="".join(reasoning_parts) or None,
        )
        if response_event_sink is not None:
            response_event_sink({
                "stream_chunk_count": chunk_index,
                "response": {
                    **response_metadata,
                    "object": "chat.completion.aggregated_from_stream",
                    "choices": [{
                        "index": 0,
                        "finish_reason": finish_reason,
                        "message": {
                            "role": "assistant",
                            "content": "".join(content_parts) or None,
                            "reasoning": "".join(api_reasoning_parts) or None,
                            "reasoning_content": (
                                "".join(api_reasoning_content_parts) or None
                            ),
                            "tool_calls": self._json_value(message.tool_calls),
                        },
                    }],
                    "usage": self._json_value(usage),
                },
            })
        return _ChatCompletion(
            choices=[_Choice(message=message)],
            usage=usage if (usage.prompt_tokens or usage.completion_tokens) else None,
        )

    @staticmethod
    def _json_value(value: Any) -> Any:
        """Serialize the complete SDK object, including model-extra fields."""
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, dict):
            return {
                str(key): OpenAIModelClient._json_value(item)
                for key, item in value.items()
            }
        if isinstance(value, (list, tuple)):
            return [OpenAIModelClient._json_value(item) for item in value]
        if hasattr(value, "model_dump"):
            try:
                return OpenAIModelClient._json_value(
                    value.model_dump(mode="json", exclude_none=False)
                )
            except (TypeError, ValueError):
                return OpenAIModelClient._json_value(
                    value.model_dump(exclude_none=False)
                )
        if hasattr(value, "__dict__"):
            return {
                str(key): OpenAIModelClient._json_value(item)
                for key, item in vars(value).items()
                if not str(key).startswith("_")
            }
        return str(value)

    @staticmethod
    def _merge_chunk(
        chunk, content_parts: list[str], reasoning_parts: list[str],
        tool_calls: dict[int, _ToolCall], usage: _Usage,
    ) -> None:
        chunk_usage = getattr(chunk, "usage", None)
        if chunk_usage is not None:
            usage.prompt_tokens = getattr(chunk_usage, "prompt_tokens", 0) or usage.prompt_tokens
            usage.completion_tokens = getattr(chunk_usage, "completion_tokens", 0) or usage.completion_tokens
            usage.total_tokens = getattr(chunk_usage, "total_tokens", 0) or usage.total_tokens

        choices: Iterable[Any] = getattr(chunk, "choices", None) or []
        for choice in choices:
            delta = getattr(choice, "delta", None)
            if delta is None:
                continue
            content = getattr(delta, "content", None)
            if content:
                content_parts.append(content)
            # vLLM renamed the streamed reasoning field from
            # ``reasoning_content`` to ``reasoning``.  Normalize both wire
            # formats into our stable internal ``reasoning_content`` field.
            reasoning = (
                getattr(delta, "reasoning", None)
                or getattr(delta, "reasoning_content", None)
            )
            if reasoning:
                reasoning_parts.append(reasoning)
            for call_delta in getattr(delta, "tool_calls", None) or []:
                idx = getattr(call_delta, "index", 0) or 0
                slot = tool_calls.get(idx)
                if slot is None:
                    slot = _ToolCall()
                    tool_calls[idx] = slot
                cid = getattr(call_delta, "id", None)
                if cid:
                    slot.id = cid
                ctype = getattr(call_delta, "type", None)
                if ctype:
                    slot.type = ctype
                fn = getattr(call_delta, "function", None)
                if fn is not None:
                    name = getattr(fn, "name", None)
                    if name:
                        slot.function.name = name
                    args = getattr(fn, "arguments", None)
                    if args:
                        slot.function.arguments += args

    @staticmethod
    def _is_retryable(exc: Exception, status_code: object) -> bool:
        if isinstance(exc, (TimeoutError, ConnectionError, asyncio.TimeoutError)):
            return True
        if isinstance(status_code, int):
            return status_code == 408 or status_code == 429 or status_code >= 500
        return type(exc).__name__ in {"APITimeoutError", "APIConnectionError", "InternalServerError"}

    async def close(self) -> None:
        await self.client.close()
