"""Stable errors exposed by the Agent contract (not provider-specific errors)."""

from enum import Enum


CONTRACT_VERSION = "deepresearch-agent-contract/v1"


class ErrorCode(str, Enum):
    INVALID_MESSAGE = "invalid_message"
    INVALID_TRAJECTORY = "invalid_trajectory"
    INVALID_TOOL_NAME = "invalid_tool_name"
    INVALID_TOOL_ARGUMENTS = "invalid_tool_arguments"
    INVALID_ASSISTANT_OUTPUT = "invalid_assistant_output"
    TOOL_TIMEOUT = "tool_timeout"
    TOOL_UNAVAILABLE = "tool_unavailable"
    TOOL_BUDGET_EXHAUSTED = "tool_budget_exhausted"
    MAX_TOOL_ROUNDS = "max_tool_rounds"
    CONTEXT_LIMIT = "context_limit"
    MODEL_UNAVAILABLE = "model_unavailable"
    CACHE_MISS = "cache_miss"
    NO_FINAL_ANSWER = "no_final_answer"
    GENERATION_ERROR = "generation_error"
    MAX_ROUNDS_EXCEEDED = "max_rounds_exceeded"
    CONTEXT_OVERFLOW = "context_overflow"
    FORCED_FINAL_EXHAUSTED = "forced_final_exhausted"
