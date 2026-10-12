"""P0 configuration loading with explicit protection against secret values in YAML."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .contracts.tools import TOOL_SCHEMA_VERSION

_SECRET_KEYS = {"api_key", "api_keys", "secret", "password", "credential", "credentials"}


class ConfigError(ValueError):
    pass


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = "Qwen/Qwen3-8B"
    revision: str | None = None
    timeout_seconds: float = Field(default=120, gt=0)
    max_retries: int = Field(default=5, ge=0)
    temperature: float = Field(default=0.6, ge=0)
    top_p: float = Field(default=0.95, gt=0, le=1)
    max_output_tokens: int = Field(default=4096, gt=0)
    enable_thinking: bool = True


class ToolConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = "web_search"
    schema_version: str = TOOL_SCHEMA_VERSION
    cache_mode: Literal["read_write", "read_only", "refresh", "disabled"] = "read_write"
    cache_namespace: str = "agent-dev/v1"
    cache_path: str = "cache/agent/web_search.sqlite3"
    timeout_seconds: float = Field(default=60, gt=0)
    connect_timeout_seconds: float = Field(default=5, gt=0)
    read_timeout_seconds: float = Field(default=30, gt=0)
    summary_timeout_seconds: float = Field(default=120, gt=0)
    retry_count: int = Field(default=2, ge=0)
    serper_same_key_max_retries: int = Field(default=3, ge=0)
    serper_max_key_switches: int = Field(default=5, ge=0)
    serper_retry_base_seconds: float = Field(default=0.25, ge=0)
    jina_enabled: bool = True
    visit_stage_timeout_seconds: float = Field(default=60, gt=0)
    # Applies only to direct HTTP. Jina calls are single-shot.
    visit_max_attempts: int = Field(default=2, ge=1, le=2)
    visit_retry_delays_seconds: tuple[float, ...] = (0.5,)
    concurrency: int = Field(default=3, ge=1)
    min_page_chars: int = Field(default=100, ge=20)
    max_page_chars: int = Field(default=50000, ge=100)
    max_snippet_chars: int = Field(default=300, ge=50)
    max_evidence_chars: int = Field(default=500, ge=100)
    max_output_chars: int = Field(default=4000, ge=1000)


class RuntimeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    run_root: str = "runs"
    max_tool_rounds: int = Field(default=10, ge=1, le=10)
    log_level: str = "INFO"
    as_of_time: str | None = None
    # This must be the serving endpoint's limit, not tokenizer_config.json's
    # model_max_length.  The latter can be higher than a deployment's limit.
    max_context_tokens: int | None = Field(default=32768, gt=0)
    tokenizer_path: str = "agent/assets/qwen3-tokenizer"
    forced_final_trigger_tokens: int | None = Field(default=30000, gt=0)
    forced_final_attempts: int = Field(default=2, ge=1)
    forced_final_answer_reserve_tokens: int = Field(default=2048, gt=0)
    forced_final_overhead_tokens: int = Field(default=720, gt=0)
    batch_concurrency: int = Field(default=1, ge=1)
    sample_retries: int = Field(default=1, ge=0)


class AgentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    stage: str = "agent"
    model: ModelConfig = ModelConfig()
    tool: ToolConfig = ToolConfig()
    runtime: RuntimeConfig = RuntimeConfig()


def _contains_secret_key(value: Any) -> bool:
    if isinstance(value, dict):
        return any(str(key).lower() in _SECRET_KEYS or _contains_secret_key(item) for key, item in value.items())
    if isinstance(value, list):
        return any(_contains_secret_key(item) for item in value)
    return False


def load_agent_config(path: str | Path) -> AgentConfig:
    source = Path(path)
    try:
        raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ConfigError(f"cannot read config: {source}") from exc
    except yaml.YAMLError as exc:
        raise ConfigError("config is not valid YAML") from exc
    if not isinstance(raw, dict):
        raise ConfigError("config root must be a mapping")
    if _contains_secret_key(raw):
        raise ConfigError("secrets must be injected by environment variables, not YAML")
    try:
        config = AgentConfig.model_validate(raw)
    except ValidationError as exc:
        raise ConfigError(exc.json(include_url=False)) from exc
    if config.tool.name != "web_search" or config.tool.schema_version != TOOL_SCHEMA_VERSION:
        raise ConfigError("P0 permits only web_search/v1")
    return config
