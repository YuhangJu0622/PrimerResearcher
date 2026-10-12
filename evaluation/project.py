"""Project raw agent/reference messages into a fair, compact judge dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Iterable

from .contracts import (
    DATASET_FORMAT_VERSION,
    EvaluationSample,
    JudgePathStep,
    JudgeToolCall,
    JudgeTrajectory,
    SampleProvenance,
)


class ProjectionError(ValueError):
    pass


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _parse_arguments(value: Any, *, location: str) -> dict[str, Any]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ProjectionError(f"{location}: tool arguments are not valid JSON") from exc
    if not isinstance(value, dict):
        raise ProjectionError(f"{location}: tool arguments must be an object")
    return value


def _project_tool_calls(calls: Any, *, location: str) -> list[JudgeToolCall]:
    if calls in (None, ""):
        return []
    if not isinstance(calls, list):
        raise ProjectionError(f"{location}: tool_calls must be a list")

    projected: list[JudgeToolCall] = []
    for call_index, call in enumerate(calls, 1):
        if not isinstance(call, dict) or not isinstance(call.get("function"), dict):
            raise ProjectionError(f"{location}: malformed tool call {call_index}")
        function = call["function"]
        name = function.get("name")
        if not isinstance(name, str) or not name:
            raise ProjectionError(f"{location}: tool call {call_index} has no name")
        arguments = _parse_arguments(
            function.get("arguments", {}), location=f"{location} tool call {call_index}"
        )

        # Historical references batch several searches in one function call,
        # while the current contract permits one query per call.  Represent
        # both as calls in the same assistant research round.
        queries = arguments.get("query") if name in {"web_search", "search_web"} else None
        if isinstance(queries, list):
            if not queries:
                raise ProjectionError(f"{location}: web_search query list is empty")
            for query in queries:
                if not isinstance(query, str) or not query.strip():
                    raise ProjectionError(f"{location}: web_search query must be non-empty text")
                normalized_name = "web_search" if name == "search_web" else name
                projected.append(JudgeToolCall(name=normalized_name, arguments={"query": query.strip()}))
        else:
            if name in {"web_search", "search_web"}:
                if not isinstance(queries, str) or not queries.strip():
                    raise ProjectionError(f"{location}: web_search query must be non-empty text")
                arguments = dict(arguments)
                arguments["query"] = queries.strip()
                name = "web_search"
            projected.append(JudgeToolCall(name=name, arguments=arguments))
    return projected


def project_messages(messages: Any, *, location: str) -> JudgeTrajectory:
    """Remove runtime-only messages and retain the path plus final answer."""
    if not isinstance(messages, list) or not messages:
        raise ProjectionError(f"{location}: messages must be a non-empty list")

    if not any(isinstance(message, dict) and message.get("role") == "assistant" for message in messages):
        raise ProjectionError(f"{location}: trajectory has no assistant messages")

    # Preserve the reference evaluator's exact boundary: only the final raw
    # message may be the answer.  A failed/incomplete trajectory remains
    # judgeable as "未回复" instead of disappearing from the denominator.
    final = messages[-1]
    answer = final.get("content") if isinstance(final, dict) and final.get("role") == "assistant" else None
    if not isinstance(answer, str) or not answer.strip() or final.get("tool_calls"):
        answer = "未回复"

    path: list[JudgePathStep] = []
    path_messages = [
        message
        for message in messages[:-1]
        if isinstance(message, dict) and message.get("role") == "assistant"
    ]
    for step_number, message in enumerate(path_messages, 1):
        reasoning = message.get("reasoning_content", "")
        if reasoning is None:
            reasoning = ""
        if not isinstance(reasoning, str):
            raise ProjectionError(f"{location}: assistant step {step_number} reasoning must be text")
        path.append(
            JudgePathStep(
                step=step_number,
                reasoning_content=reasoning,
                tool_calls=_project_tool_calls(
                    message.get("tool_calls"), location=f"{location} assistant step {step_number}"
                ),
            )
        )
    return JudgeTrajectory(path=path, answer=answer.strip())


def _read_jsonl(path: Path) -> list[tuple[int, bytes, dict[str, Any]]]:
    rows: list[tuple[int, bytes, dict[str, Any]]] = []
    for line_number, raw_line in enumerate(path.read_bytes().splitlines(), 1):
        if not raw_line.strip():
            continue
        try:
            value = json.loads(raw_line)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ProjectionError(f"{path}:{line_number}: invalid JSON") from exc
        if not isinstance(value, dict):
            raise ProjectionError(f"{path}:{line_number}: row must be an object")
        rows.append((line_number, raw_line, value))
    return rows


def _atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _atomic_write_jsonl(path: Path, values: Iterable[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        for value in values:
            payload = value.model_dump(mode="json") if hasattr(value, "model_dump") else value
            stream.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _numeric_trajectory_files(run_path: Path) -> list[Path]:
    trajectory_dir = run_path / "trajectories"
    if not trajectory_dir.is_dir():
        raise ProjectionError(f"trajectory directory does not exist: {trajectory_dir}")
    trajectory_files = sorted(
        (path for path in trajectory_dir.glob("*.json") if path.stem.isdigit()),
        key=lambda path: int(path.stem),
    )
    if not trajectory_files:
        raise ProjectionError(f"no numeric trajectory files found in {trajectory_dir}")
    return trajectory_files


def _read_trajectory(path: Path) -> tuple[bytes, list[Any]]:
    raw = path.read_bytes()
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProjectionError(f"{path}: invalid JSON") from exc
    if not isinstance(payload, dict):
        raise ProjectionError(f"{path}: root must be an object")
    messages = payload.get("messages")
    if not isinstance(messages, list) or len(messages) < 2:
        raise ProjectionError(f"{path}: messages are missing")
    return raw, messages


def _trajectory_query(messages: list[Any], *, location: str) -> str:
    users = [message for message in messages if isinstance(message, dict) and message.get("role") == "user"]
    if len(users) != 1 or not isinstance(users[0].get("content"), str) or not users[0]["content"].strip():
        raise ProjectionError(f"{location}: trajectory must contain exactly one non-empty user query")
    return users[0]["content"]


def _run_trajectory_map(run_path: Path) -> dict[int, Path]:
    trajectory_files = _numeric_trajectory_files(run_path)
    files_by_number = {int(path.stem): path for path in trajectory_files}
    if len(files_by_number) != len(trajectory_files):
        raise ProjectionError(f"duplicate question number in {run_path / 'trajectories'}")
    return files_by_number


def _safe_path_component(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._-") or "reference"


def default_output_path(run_dir: str | Path, reference_run_dir: str | Path | None = None) -> Path:
    """Return an isolated default dataset path under the candidate run directory."""
    run_path = Path(run_dir)
    comparison_name = (
        "against-test"
        if reference_run_dir is None
        else f"against-{_safe_path_component(Path(reference_run_dir).resolve().name)}"
    )
    return run_path / "evaluation" / comparison_name / "dataset.judge-v1.jsonl"


def build_dataset(
    test_file: str | Path | None,
    run_dir: str | Path,
    output: str | Path,
    *,
    reference_run_dir: str | Path | None = None,
) -> list[EvaluationSample]:
    if (test_file is None) == (reference_run_dir is None):
        raise ProjectionError("provide exactly one of test_file or reference_run_dir")

    run_path = Path(run_dir).resolve()
    candidate_files = _run_trajectory_map(run_path)

    test_path: Path | None = None
    test_rows: list[tuple[int, bytes, dict[str, Any]]] | None = None
    reference_path: Path | None = None
    reference_files: dict[int, Path] | None = None
    if test_file is not None:
        test_path = Path(test_file).resolve()
        if not test_path.is_file():
            raise ProjectionError(f"test file does not exist: {test_path}")
        test_rows = _read_jsonl(test_path)
    else:
        reference_path = Path(reference_run_dir).resolve()
        if reference_path == run_path:
            raise ProjectionError("reference_run_dir must be different from run_dir")
        reference_files = _run_trajectory_map(reference_path)
        if candidate_files.keys() != reference_files.keys():
            candidate_numbers = sorted(candidate_files)
            reference_numbers = sorted(reference_files)
            raise ProjectionError(
                "candidate and reference trajectory question numbers do not match: "
                f"candidate={candidate_numbers}, reference={reference_numbers}"
            )

    samples: list[EvaluationSample] = []
    for question_number, trajectory_file in candidate_files.items():
        trajectory_raw, candidate_messages = _read_trajectory(trajectory_file)
        candidate_query = _trajectory_query(candidate_messages, location=str(trajectory_file))

        if test_rows is not None:
            assert test_path is not None
            if question_number > len(test_rows):
                raise ProjectionError(f"{trajectory_file}: question number is outside the test set")
            test_line, test_raw, row = test_rows[question_number - 1]
            query = row.get("query")
            reference_messages = row.get("reference")
            if not isinstance(query, str) or not query.strip():
                raise ProjectionError(f"{test_path}:{test_line}: query is missing")
            if candidate_query != query:
                raise ProjectionError(
                    f"{trajectory_file}: candidate query does not exactly match test row {question_number}"
                )
            sample_id = str(row.get("id", question_number))
            provenance = SampleProvenance(
                reference_type="test_file",
                test_file=str(test_path),
                test_line=test_line,
                test_row_sha256=_sha256_bytes(test_raw),
                run_dir=str(run_path),
                trajectory_file=str(trajectory_file),
                trajectory_sha256=_sha256_bytes(trajectory_raw),
            )
            reference_location = f"{test_path}:{test_line} reference"
        else:
            assert reference_path is not None and reference_files is not None
            reference_file = reference_files[question_number]
            reference_raw, reference_messages = _read_trajectory(reference_file)
            reference_query = _trajectory_query(reference_messages, location=str(reference_file))
            if candidate_query != reference_query:
                raise ProjectionError(
                    f"{trajectory_file}: candidate query does not exactly match reference trajectory "
                    f"{reference_file}"
                )
            query = candidate_query
            sample_id = str(question_number)
            provenance = SampleProvenance(
                reference_type="run_dir",
                run_dir=str(run_path),
                trajectory_file=str(trajectory_file),
                trajectory_sha256=_sha256_bytes(trajectory_raw),
                reference_run_dir=str(reference_path),
                reference_trajectory_file=str(reference_file),
                reference_trajectory_sha256=_sha256_bytes(reference_raw),
            )
            reference_location = str(reference_file)

        samples.append(
            EvaluationSample(
                id=sample_id,
                question_number=question_number,
                query=query,
                candidate=project_messages(candidate_messages, location=str(trajectory_file)),
                reference=project_messages(reference_messages, location=reference_location),
                provenance=provenance,
            )
        )

    output_path = Path(output).resolve()
    _atomic_write_jsonl(output_path, samples)
    dataset_sha256 = _sha256_bytes(output_path.read_bytes())
    manifest: dict[str, Any] = {
        "format_version": DATASET_FORMAT_VERSION,
        "dataset_file": str(output_path),
        "dataset_sha256": dataset_sha256,
        "sample_count": len(samples),
        "question_numbers": [sample.question_number for sample in samples],
        "run_dir": str(run_path),
    }
    if test_path is not None:
        manifest.update(
            {
                "reference_type": "test_file",
                "test_file": str(test_path),
                "test_file_sha256": _sha256_bytes(test_path.read_bytes()),
            }
        )
    else:
        assert reference_path is not None
        manifest.update({"reference_type": "run_dir", "reference_run_dir": str(reference_path)})
    _atomic_write_json(output_path.with_suffix(".manifest.json"), manifest)
    return samples


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the compact pairwise judge dataset")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--test-file")
    source.add_argument("--reference-run-dir")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument(
        "--output",
        help="defaults to RUN_DIR/evaluation/against-test or against-agent-NAME/dataset.judge-v1.jsonl",
    )
    args = parser.parse_args()
    test_file = args.test_file or (None if args.reference_run_dir else "data/deepresearch/test.jsonl")
    output = args.output or default_output_path(args.run_dir, args.reference_run_dir)
    samples = build_dataset(
        test_file,
        args.run_dir,
        output,
        reference_run_dir=args.reference_run_dir,
    )
    print(json.dumps({"dataset": str(Path(output).resolve()), "samples": len(samples)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
