"""Tokenizer-backed Qwen prompt accounting without loading model weights."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from transformers import AutoTokenizer


DEFAULT_TOKENIZER_PATH = Path(__file__).resolve().parents[1] / "assets" / "qwen3-tokenizer"


def load_qwen3_tokenizer(path: str | Path = DEFAULT_TOKENIZER_PATH) -> Any:
    """Load only the tokenizer and its checked-in chat template from a local directory."""
    source = Path(path).resolve()
    if not source.is_dir():
        raise RuntimeError(f"Qwen tokenizer directory does not exist: {source}")
    try:
        return AutoTokenizer.from_pretrained(str(source), local_files_only=True, use_fast=True)
    except Exception as exc:
        raise RuntimeError(f"cannot load local Qwen tokenizer from {source}") from exc
