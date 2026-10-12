"""Structured message models and full-trajectory semantic validation."""

from __future__ import annotations

import json
from typing import Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError, field_validator, model_validator

from .errors import ErrorCode
from .tools import validate_web_search_arguments


class ContractError(ValueError):
    code = ErrorCode.INVALID_TRAJECTORY


class ToolCallFunction(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    name: Literal["web_search"]
    arguments: str

    @field_validator("arguments")
    @classmethod
    def arguments_are_json_object(cls, value: str) -> str:
        validate_web_search_arguments(value)
        return value


class ToolCall(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    id: str = Field(min_length=1)
    type: Literal["function"] = "function"
    function: ToolCallFunction


class SystemMessage(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    role: Literal["system"]
    content: str = Field(min_length=1)


class UserMessage(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    role: Literal["user"]
    content: str = Field(min_length=1)


class AssistantMessage(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    role: Literal["assistant"]
    content: str | None = None
    reasoning_content: str | None = None
    tool_calls: list[ToolCall] | None = None

    @model_validator(mode="after")
    def is_tool_turn_or_final_turn(self) -> "AssistantMessage":
        if self.tool_calls:
            if self.content not in (None, ""):
                raise ValueError("tool-turn assistant message cannot include final content")
            return self
        if not self.content or not self.content.strip():
            raise ValueError("final assistant message needs non-empty content")
        return self


class ToolMessage(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    role: Literal["tool"]
    tool_call_id: str = Field(min_length=1)
    content: str = Field(min_length=1)


Message = Annotated[
    Union[SystemMessage, UserMessage, AssistantMessage, ToolMessage], Field(discriminator="role")
]
_MESSAGE_ADAPTER = TypeAdapter(Message)


def parse_message(value: Message | dict) -> Message:
    try:
        return _MESSAGE_ADAPTER.validate_python(value)
    except ValidationError as exc:
        raise ContractError(exc.json(include_url=False)) from exc


def message_to_dict(message: Message) -> dict:
    return message.model_dump(exclude_none=True)


def validate_trajectory(values: list[Message | dict], *, require_final: bool = True) -> list[Message]:
    """Ensure calls are unique and ordered; a live prefix may omit the final answer."""
    if not values:
        raise ContractError("trajectory must not be empty")
    messages = [parse_message(item) for item in values]
    if not isinstance(messages[0], SystemMessage):
        raise ContractError("trajectory must start with a system message")
    if len(messages) < 2 or not isinstance(messages[1], UserMessage):
        raise ContractError("system message must be followed by a user message")

    pending: dict[str, ToolCall] = {}
    pending_order: list[str] = []
    seen_calls: set[str] = set()
    final_seen = False
    for index, message in enumerate(messages):
        if final_seen:
            raise ContractError("no messages are allowed after a final answer")
        if isinstance(message, AssistantMessage):
            if pending:
                raise ContractError("assistant response arrived before all tool observations")
            if message.tool_calls:
                for call in message.tool_calls:
                    if call.id in seen_calls:
                        raise ContractError(f"duplicate tool call id: {call.id}")
                    seen_calls.add(call.id)
                    pending[call.id] = call
                    pending_order.append(call.id)
            else:
                final_seen = True
        elif isinstance(message, ToolMessage):
            if message.tool_call_id not in pending:
                raise ContractError(f"unknown or already observed tool call id: {message.tool_call_id}")
            if message.tool_call_id != pending_order[0]:
                raise ContractError("tool observations must preserve tool call order")
            del pending[message.tool_call_id]
            pending_order.pop(0)
        elif index > 1:
            raise ContractError("system and user messages are only allowed at trajectory start")
    if pending:
        raise ContractError("trajectory has unobserved tool calls")
    if require_final and not final_seen:
        raise ContractError("trajectory has no final assistant answer")
    return messages


def normalize_trajectory(values: list[Message | dict], *, require_final: bool = True) -> list[dict]:
    """Globally renumber call ids as call_1...call_n while preserving semantics."""
    messages = [parse_message(item) for item in values]
    call_ids: dict[str, str] = {}
    next_index = 1
    normalized: list[dict] = []
    for message in messages:
        data = message_to_dict(message)
        if isinstance(message, AssistantMessage) and message.tool_calls:
            calls = []
            for call in message.tool_calls:
                new_id = f"call_{next_index}"
                next_index += 1
                call_ids[call.id] = new_id
                call_data = call.model_dump()
                call_data["id"] = new_id
                call_data["function"]["arguments"] = json.dumps(
                    validate_web_search_arguments(call.function.arguments),
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                calls.append(call_data)
            data["tool_calls"] = calls
        elif isinstance(message, ToolMessage):
            try:
                data["tool_call_id"] = call_ids[message.tool_call_id]
            except KeyError as exc:
                raise ContractError(f"tool result appears before matching call: {message.tool_call_id}") from exc
        normalized.append(data)
    validate_trajectory(normalized, require_final=require_final)
    return normalized
