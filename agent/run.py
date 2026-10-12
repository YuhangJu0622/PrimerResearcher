"""Create immutable run directories with frozen configuration and provenance."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import secrets
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from .config import AgentConfig, load_agent_config
from .contracts.errors import CONTRACT_VERSION
from .contracts.tools import TOOL_SCHEMA_VERSION, canonical_tool_schema_json


def generate_run_id(stage: str, now: datetime | None = None) -> str:
    if not stage or any(char not in "abcdefghijklmnopqrstuvwxyz0123456789_-" for char in stage.lower()):
        raise ValueError("stage must use lowercase letters, numbers, _ or -")
    timestamp = (now or datetime.now(UTC)).strftime("%Y%m%d-%H%M%S")
    return f"{stage}-{timestamp}-{secrets.token_hex(3)}"


def _sha256(value: str | bytes) -> str:
    data = value.encode("utf-8") if isinstance(value, str) else value
    return hashlib.sha256(data).hexdigest()


def _git_output(root: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-c", f"safe.directory={root.as_posix()}", *args],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


def code_provenance(project_root: Path) -> dict[str, str]:
    commit = _git_output(project_root, "rev-parse", "HEAD") or "unavailable"
    status = _git_output(project_root, "status", "--porcelain")
    if status is None:
        # A Git/LFS permission failure is not evidence of a clean worktree.
        # Untracked files can still be checked without invoking the LFS filter.
        untracked = _git_output(project_root, "ls-files", "--others", "--exclude-standard")
        state = "dirty" if untracked else "unknown"
    else:
        state = "dirty" if status else "clean"
    return {"code_commit": commit, "working_tree": state}


def installed_dependencies() -> dict[str, str]:
    """Snapshot every installed distribution, including pinned transitive packages."""
    return dict(sorted({dist.metadata["Name"]: dist.version for dist in importlib.metadata.distributions()}.items()))


def create_run(config: AgentConfig, project_root: str | Path, run_id: str | None = None) -> Path:
    root = Path(project_root).resolve()
    run_name = run_id or generate_run_id(config.stage)
    run_dir = root / config.runtime.run_root / config.stage / run_name
    # exist_ok=False is deliberately the final check to prevent any overwrite.
    run_dir.mkdir(parents=True, exist_ok=False)
    try:
        frozen_config = yaml.safe_dump(config.model_dump(mode="json"), allow_unicode=True, sort_keys=True)
        (run_dir / "frozen_config.yaml").write_text(frozen_config, encoding="utf-8", newline="\n")
        provenance = code_provenance(root)
        manifest: dict[str, Any] = {
            "run_id": run_name,
            "stage": config.stage,
            **provenance,
            "qqr_version": "0.2.1",
            "contract_version": CONTRACT_VERSION,
            "tool_schema_version": TOOL_SCHEMA_VERSION,
            "tool_schema_sha256": _sha256(canonical_tool_schema_json()),
            "model_id": config.model.id,
            "model_revision": config.model.revision,
            "frozen_config_sha256": _sha256(frozen_config),
            "created_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "python_version": sys.version.split()[0],
            "dependencies": installed_dependencies(),
            "secrets": "environment-only; values omitted",
        }
        (run_dir / "run_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except Exception:
        # Leave the exclusive directory intact for forensic inspection; never reuse it.
        raise
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Create an immutable P0 Agent run directory")
    parser.add_argument("--config", default="agent/configs/agent.yaml")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--run-id", help="optional unique run id; existing directories are rejected")
    args = parser.parse_args()
    run_dir = create_run(load_agent_config(args.config), args.project_root, args.run_id)
    print(run_dir)


if __name__ == "__main__":
    main()
