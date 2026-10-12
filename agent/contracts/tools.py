"""The sole model-visible tool and its strict, versioned validation."""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from .errors import ErrorCode

TOOL_SCHEMA_VERSION = "web_search/v1"

WEB_SEARCH_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "Search the web for one query and return evidence summaries from up to three webpages.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string", "minLength": 1}},
            "required": ["query"],
            "additionalProperties": False,
        },
    },
}


class WebSearchArguments(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    query: str = Field(min_length=1)

    @field_validator("query")
    @classmethod
    def query_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("query must not be blank")
        return value


class ToolArgumentError(ValueError):
    code = ErrorCode.INVALID_TOOL_ARGUMENTS


def validate_web_search_arguments(arguments: str | dict[str, Any]) -> dict[str, str]:
    """Parse one JSON object without accepting arrays, guesses, or extra keys."""
    try:
        value = json.loads(arguments) if isinstance(arguments, str) else arguments
    except json.JSONDecodeError as exc:
        raise ToolArgumentError("arguments must be valid JSON") from exc
    if not isinstance(value, dict):
        raise ToolArgumentError("arguments must be a JSON object")
    try:
        return WebSearchArguments.model_validate(value).model_dump()
    except ValidationError as exc:
        raise ToolArgumentError(exc.json(include_url=False)) from exc


def canonical_tool_schema_json() -> str:
    return json.dumps(WEB_SEARCH_TOOL, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
