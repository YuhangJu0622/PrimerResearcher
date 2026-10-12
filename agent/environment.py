"""A secret-safe P0 preflight check for the lightweight Agent environment."""

from __future__ import annotations

import importlib.metadata
import json
import os
import sys
from pathlib import Path

from dotenv import dotenv_values

REQUIRED = ("openai", "httpx", "beautifulsoup4", "cachetools", "pydantic", "PyYAML", "tenacity", "jsonschema", "fastmcp", "pytest", "transformers", "tokenizers", "jinja2")
FORBIDDEN = ("torch", "vllm", "ms-swift", "deepspeed", "peft", "flash-attn")
SECRET_ENV = ("SERPER_API_KEYS", "JINA_API_KEY", "JINA_API_KEYS", "TOOL_SUMMARY_API_KEY", "JUDGE_API_KEY")


def load_project_env(project_root: str | Path = ".") -> None:
    """Load local defaults without overriding explicitly injected variables.

    ``.env.local`` has precedence over ``.env``.  Process environment values
    always win, so deployment-time secrets cannot be accidentally replaced by
    a file in the checkout.
    """
    root = Path(project_root).resolve()
    values: dict[str, str] = {}
    for filename in (".env", ".env.local"):
        for key, value in dotenv_values(root / filename).items():
            if value is not None:
                values[key] = value
    for key, value in values.items():
        os.environ.setdefault(key, value)


def _version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def check_environment() -> tuple[dict, bool]:
    packages = {name: _version(name) for name in REQUIRED}
    forbidden = {name: version for name in FORBIDDEN if (version := _version(name)) is not None}
    configured = {
        name: (len([part for part in os.environ.get(name, "").split(",") if part.strip()]) if name == "SERPER_API_KEYS" else bool(os.environ.get(name)))
        for name in SECRET_ENV
    }
    report = {
        "python": sys.version.split()[0],
        "packages": packages,
        "forbidden_training_packages": forbidden,
        "secret_configuration": configured,
    }
    healthy = all(packages.values()) and not forbidden
    return report, healthy


def main() -> None:
    load_project_env()
    report, healthy = check_environment()
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    raise SystemExit(0 if healthy else 1)


if __name__ == "__main__":
    main()
