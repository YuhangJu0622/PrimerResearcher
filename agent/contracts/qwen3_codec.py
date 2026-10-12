"""Qwen3 textual tool-call parser and optional tokenizer-backed renderer."""

from __future__ import annotations

import json
import re
from typing import Any

from .messages import AssistantMessage, ContractError, Message, normalize_trajectory
from .tools import WEB_SEARCH_TOOL

_THINK = re.compile(r"<think>(.*?)</think>", re.DOTALL)
_TOOL = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)


class Qwen3Codec:
    """Keep contract parsing usable without installing Transformers in the Agent env."""

    version = "qwen3-codec/v1"

    def __init__(self, tokenizer: Any | None = None) -> None:
        self.tokenizer = tokenizer

    def parse_assistant(self, raw: str, next_call_index: int = 1) -> AssistantMessage:
        if not isinstance(raw, str) or not raw.strip():
            raise ContractError("assistant output is empty")
        reasoning_match = _THINK.search(raw)
        reasoning = reasoning_match.group(1).strip() if reasoning_match else None
        remainder = _THINK.sub("", raw).replace("<|im_end|>", "").strip()
        calls: list[dict[str, Any]] = []
        for offset, match in enumerate(_TOOL.finditer(remainder)):
            try:
                value = json.loads(match.group(1).strip())
            except json.JSONDecodeError as exc:
                raise ContractError("tool call is not valid JSON") from exc
            if not isinstance(value, dict) or set(value) != {"name", "arguments"}:
                raise ContractError("tool call must contain only name and arguments")
            arguments = value["arguments"]
            if isinstance(arguments, dict):
                arguments = json.dumps(arguments, ensure_ascii=False, separators=(",", ":"))
            calls.append({
                "id": f"call_{next_call_index + offset}",
                "type": "function",
                "function": {"name": value["name"], "arguments": arguments},
            })
        content = _TOOL.sub("", remainder).strip()
        try:
            return AssistantMessage.model_validate({
                "role": "assistant",
                "content": content or None,
                "reasoning_content": reasoning,
                "tool_calls": calls or None,
            })
        except Exception as exc:
            raise ContractError("assistant output violates the Qwen3 contract") from exc

    def render_prompt(self, messages: list[Message | dict[str, Any]]) -> Any:
        normalized = normalize_trajectory(messages, require_final=False)
        if self.tokenizer is None:
            return json.dumps(normalized, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        return self.tokenizer.apply_chat_template(
            normalized, tools=[WEB_SEARCH_TOOL], tokenize=True, add_generation_prompt=True
        )

    def count_prompt_tokens(self, messages: list[Message | dict[str, Any]]) -> int:
        """Count the exact Qwen chat-template input, including tool schema tokens."""
        if self.tokenizer is None:
            raise RuntimeError("a Qwen tokenizer is required for exact prompt token counting")
        return len(self.render_prompt(messages))

    def normalize_trajectory(self, messages: list[Message | dict[str, Any]], *, require_final: bool = True) -> list[dict]:
        return normalize_trajectory(messages, require_final=require_final)
