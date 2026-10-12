"""Resumable, bidirectional pairwise evaluation through an OpenAI-compatible judge."""
# 自定义参考轨迹
# .\.venv\Scripts\python -m evaluation.project --run-dir runs\agent\agent-qwen3-8b-sft-ck180-test-0-30 --reference-run-dir runs\agent\agent-qwen3-8b-test-0-30
# .\.venv\Scripts\python -m evaluation.judge --dataset runs\agent\agent-qwen3-8b-sft-ck180-test-0-30\evaluation\against-agent-qwen3-8b-test-0-30\dataset.judge-v1.jsonl --concurrency 30
# 与test.jsonl
# .\.venv\Scripts\python -m evaluation.project --run-dir runs\agent\agent-qwen3-8b-sft-ck180-test-0-30
# .\.venv\Scripts\python -m evaluation.judge --dataset runs\agent\agent-qwen3-8b-sft-ck180-test-0-30\evaluation\against-test\dataset.judge-v1.jsonl --concurrency 30

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import re
import time
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI

from agent.environment import load_project_env

from .contracts import EvaluationSample, JudgeTrajectory
from .project import ProjectionError, _atomic_write_json, _atomic_write_jsonl
from .prompts import JUDGE_PROMPT_VERSION, JUDGE_SYSTEM_PROMPT

JUDGE_RUN_VERSION = "deepresearch-bidirectional-judge/v1"


class JudgeError(RuntimeError):
    pass


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return _sha256_bytes(encoded)


def _safe_model_name(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model).strip("_") or "judge"


def format_judge_input(query: str, trajectory_a: JudgeTrajectory, trajectory_b: JudgeTrajectory) -> str:
    return (
        f"<USER_QUERY>\n{query}\n</USER_QUERY>\n\n"
        f"<PATH_A>\n{json.dumps([step.model_dump(mode='json') for step in trajectory_a.path], ensure_ascii=False, indent=2)}\n</PATH_A>\n\n"
        f"<PATH_B>\n{json.dumps([step.model_dump(mode='json') for step in trajectory_b.path], ensure_ascii=False, indent=2)}\n</PATH_B>\n\n"
        f"<ANSWER_A>\n{trajectory_a.answer}\n</ANSWER_A>\n\n"
        f"<ANSWER_B>\n{trajectory_b.answer}\n</ANSWER_B>"
    )


def _find_json_object(text: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    for index, character in enumerate(text):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and "combined_scores" in value:
            return value
    raise JudgeError("judge response does not contain a valid result JSON object")


def parse_judge_response(text: str) -> dict[str, Any]:
    try:
        value = _find_json_object(text)
    except JudgeError as json_error:
        # Some compatible endpoints produce correct score fields but fail to
        # escape quotation marks inside a Chinese assessment string.  The
        # preserved QQR evaluator deliberately extracts these fields with
        # regex for this provider behavior; retain that compatibility only as
        # a fallback after strict JSON parsing.
        block = re.search(r'"combined_scores"\s*:\s*\{(?P<scores>.*?)\}', text, re.S)
        winner = re.search(r'"winner"\s*:\s*"(?P<winner>Agent_A|Agent_B|Tie)"', text)
        if not block or not winner:
            raise json_error
        score_a = re.search(r'"Agent_A"\s*:\s*([0-9]+(?:\.[0-9]+)?)', block.group("scores"))
        score_b = re.search(r'"Agent_B"\s*:\s*([0-9]+(?:\.[0-9]+)?)', block.group("scores"))
        if not score_a or not score_b:
            raise json_error
        value = {
            "combined_scores": {"Agent_A": float(score_a.group(1)), "Agent_B": float(score_b.group(1))},
            "winner": winner.group("winner"),
            "parse_mode": "score_regex_fallback",
        }
    scores = value.get("combined_scores")
    if not isinstance(scores, dict):
        raise JudgeError("judge response has no combined_scores object")
    for agent in ("Agent_A", "Agent_B"):
        score = scores.get(agent)
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            raise JudgeError(f"judge response has invalid score for {agent}")
        if not math.isfinite(float(score)) or not 0 <= float(score) <= 10:
            raise JudgeError(f"judge response score for {agent} is outside 0..10")
    winner = value.get("winner")
    if winner not in {"Agent_A", "Agent_B", "Tie"}:
        raise JudgeError("judge response has an invalid winner")
    return value


async def _judge_direction(
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    *,
    model: str,
    query: str,
    trajectory_a: JudgeTrajectory,
    trajectory_b: JudgeTrajectory,
    max_tokens: int,
    max_retries: int,
) -> dict[str, Any]:
    messages = [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": format_judge_input(query, trajectory_a, trajectory_b)},
    ]
    last_error: Exception | None = None
    last_response_preview = ""
    for attempt in range(1, max_retries + 2):
        try:
            async with semaphore:
                response = await client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=0.0,
                    max_tokens=max_tokens,
                    extra_body={"enable_thinking": False},
                )
            content = response.choices[0].message.content or ""
            try:
                parsed = parse_judge_response(content)
            except JudgeError as exc:
                # Keep a bounded diagnostic in the per-item error artifact so
                # a malformed provider response can be repaired without
                # logging the complete prompt, trajectories, or credentials.
                last_response_preview = content[:2000]
                raise exc
            return {"attempts": attempt, "response": parsed}
        except Exception as exc:
            last_error = exc
            if attempt > max_retries:
                break
            await asyncio.sleep(min(0.5 * (2 ** (attempt - 1)), 4.0))
    assert last_error is not None
    suffix = f"; response_preview={last_response_preview!r}" if last_response_preview else ""
    raise JudgeError(f"judge failed after {max_retries + 1} attempts: {type(last_error).__name__}: {last_error}{suffix}") from last_error


def _score(direction: dict[str, Any], agent: str) -> float:
    return float(direction["response"]["combined_scores"][agent])


async def _evaluate_sample(
    sample: EvaluationSample,
    *,
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    model: str,
    evaluation_fingerprint: str,
    items_dir: Path,
    max_tokens: int,
    max_retries: int,
) -> dict[str, Any]:
    sample_payload = sample.model_dump(mode="json")
    sample_fingerprint = _canonical_sha256(sample_payload)
    item_path = items_dir / f"{sample.question_number:06d}.json"
    if item_path.is_file():
        try:
            existing = json.loads(item_path.read_text(encoding="utf-8"))
            if (
                existing.get("status") == "ok"
                and existing.get("evaluation_fingerprint") == evaluation_fingerprint
                and existing.get("sample_fingerprint") == sample_fingerprint
            ):
                existing["resumed"] = True
                return existing
        except (OSError, json.JSONDecodeError):
            pass

    started = time.monotonic()
    common = {
        "judge_run_version": JUDGE_RUN_VERSION,
        "evaluation_fingerprint": evaluation_fingerprint,
        "sample_fingerprint": sample_fingerprint,
        "id": sample.id,
        "question_number": sample.question_number,
        "query_sha256": _sha256_bytes(sample.query.encode("utf-8")),
        "model": model,
    }
    try:
        forward = await _judge_direction(
            client,
            semaphore,
            model=model,
            query=sample.query,
            trajectory_a=sample.candidate,
            trajectory_b=sample.reference,
            max_tokens=max_tokens,
            max_retries=max_retries,
        )
        reverse = await _judge_direction(
            client,
            semaphore,
            model=model,
            query=sample.query,
            trajectory_a=sample.reference,
            trajectory_b=sample.candidate,
            max_tokens=max_tokens,
            max_retries=max_retries,
        )
        candidate_score = (_score(forward, "Agent_A") + _score(reverse, "Agent_B")) / 2
        reference_score = (_score(forward, "Agent_B") + _score(reverse, "Agent_A")) / 2
        outcome = "win" if candidate_score > reference_score else "loss" if candidate_score < reference_score else "tie"
        result = {
            **common,
            "status": "ok",
            "resumed": False,
            "candidate_score": candidate_score,
            "reference_score": reference_score,
            "score_margin": candidate_score - reference_score,
            "outcome": outcome,
            "forward": {"Agent_A": "candidate", "Agent_B": "reference", **forward},
            "reverse": {"Agent_A": "reference", "Agent_B": "candidate", **reverse},
            "elapsed_seconds": round(time.monotonic() - started, 6),
        }
    except Exception as exc:
        result = {
            **common,
            "status": "error",
            "resumed": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "elapsed_seconds": round(time.monotonic() - started, 6),
        }
    _atomic_write_json(item_path, result)
    return result


def _wilson_interval(successes: float, total: int, z: float = 1.959963984540054) -> list[float] | None:
    if total <= 0:
        return None
    proportion = successes / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    radius = z * math.sqrt((proportion * (1 - proportion) + z * z / (4 * total)) / total) / denominator
    return [max(0.0, center - radius), min(1.0, center + radius)]


def summarize_results(results: list[dict[str, Any]], *, model: str, dataset: Path, dataset_sha256: str) -> dict[str, Any]:
    valid = [result for result in results if result.get("status") == "ok"]
    wins = sum(result["outcome"] == "win" for result in valid)
    losses = sum(result["outcome"] == "loss" for result in valid)
    ties = sum(result["outcome"] == "tie" for result in valid)
    judged = len(valid)
    equivalent_wins = wins + 0.5 * ties
    return {
        "judge_run_version": JUDGE_RUN_VERSION,
        "prompt_version": JUDGE_PROMPT_VERSION,
        "model": model,
        "dataset": str(dataset),
        "dataset_sha256": dataset_sha256,
        "total_samples": len(results),
        "judged_samples": judged,
        "failed_samples": len(results) - judged,
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "win_rate": wins / judged if judged else None,
        "win_equivalent_rate": equivalent_wins / judged if judged else None,
        "decisive_win_rate": wins / (wins + losses) if wins + losses else None,
        "win_equivalent_rate_95ci": _wilson_interval(equivalent_wins, judged),
        "average_candidate_score": sum(r["candidate_score"] for r in valid) / judged if judged else None,
        "average_reference_score": sum(r["reference_score"] for r in valid) / judged if judged else None,
        "average_score_margin": sum(r["score_margin"] for r in valid) / judged if judged else None,
    }


def _load_dataset(path: Path) -> list[EvaluationSample]:
    samples: list[EvaluationSample] = []
    seen_numbers: set[int] = set()
    seen_ids: set[str] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            sample = EvaluationSample.model_validate_json(line)
        except Exception as exc:
            raise ProjectionError(f"{path}:{line_number}: invalid evaluation sample") from exc
        if sample.question_number in seen_numbers or sample.id in seen_ids:
            raise ProjectionError(f"{path}:{line_number}: duplicate id or question number")
        seen_numbers.add(sample.question_number)
        seen_ids.add(sample.id)
        samples.append(sample)
    if not samples:
        raise ProjectionError(f"dataset is empty: {path}")
    return samples


async def run_evaluation(args: argparse.Namespace) -> dict[str, Any]:
    load_project_env(args.project_root)
    dataset = Path(args.dataset).resolve()
    if not dataset.is_file():
        raise ProjectionError(f"dataset does not exist: {dataset}")
    dataset_bytes = dataset.read_bytes()
    dataset_sha256 = _sha256_bytes(dataset_bytes)
    samples = _load_dataset(dataset)
    if args.limit > 0:
        samples = samples[: args.limit]

    model = args.model or os.getenv("JUDGE_MODEL") or "qwen3.5-plus"
    base_url = args.base_url or os.getenv("JUDGE_BASE_URL", "")
    api_key = args.api_key or os.getenv("JUDGE_API_KEY", "")
    if not base_url or not api_key:
        raise JudgeError("JUDGE_BASE_URL and JUDGE_API_KEY must be configured")

    fingerprint_payload = {
        "judge_run_version": JUDGE_RUN_VERSION,
        "prompt_version": JUDGE_PROMPT_VERSION,
        "prompt_sha256": _sha256_bytes(JUDGE_SYSTEM_PROMPT.encode("utf-8")),
        "dataset_sha256": dataset_sha256,
        "model": model,
        "max_tokens": args.max_tokens,
        "enable_thinking": False,
    }
    evaluation_fingerprint = _canonical_sha256(fingerprint_payload)
    output_dir = Path(args.output_dir or dataset.parent / f"judge-{_safe_model_name(model)}").resolve()
    items_dir = output_dir / "items"
    items_dir.mkdir(parents=True, exist_ok=True)

    client = AsyncOpenAI(base_url=base_url, api_key=api_key, timeout=args.timeout, max_retries=0)
    semaphore = asyncio.Semaphore(args.concurrency)
    try:
        tasks = [
            asyncio.create_task(
                _evaluate_sample(
                    sample,
                    client=client,
                    semaphore=semaphore,
                    model=model,
                    evaluation_fingerprint=evaluation_fingerprint,
                    items_dir=items_dir,
                    max_tokens=args.max_tokens,
                    max_retries=args.max_retries,
                )
            )
            for sample in samples
        ]
        results: list[dict[str, Any]] = []
        completed = 0
        for future in asyncio.as_completed(tasks):
            result = await future
            results.append(result)
            completed += 1
            state = result.get("outcome", result.get("status"))
            print(json.dumps({"completed": completed, "total": len(tasks), "question_number": result["question_number"], "state": state}, ensure_ascii=False), flush=True)
    finally:
        await client.close()

    results.sort(key=lambda value: int(value["question_number"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write_jsonl(output_dir / "results.jsonl", results)
    summary = summarize_results(results, model=model, dataset=dataset, dataset_sha256=dataset_sha256)
    summary["evaluation_fingerprint"] = evaluation_fingerprint
    summary["output_dir"] = str(output_dir)
    _atomic_write_json(output_dir / "summary.json", summary)
    _atomic_write_json(output_dir / "run_manifest.json", {**fingerprint_payload, "evaluation_fingerprint": evaluation_fingerprint, "output_dir": str(output_dir)})
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Judge candidate trajectories against test references")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--output-dir")
    parser.add_argument("--model")
    parser.add_argument("--base-url", help=argparse.SUPPRESS)
    parser.add_argument("--api-key", help=argparse.SUPPRESS)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    if args.concurrency < 1 or args.max_retries < 0 or args.max_tokens < 1 or args.timeout <= 0 or args.limit < 0:
        parser.error("concurrency/max-tokens/timeout must be positive; retries/limit must be non-negative")
    summary = asyncio.run(run_evaluation(args))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    raise SystemExit(0 if summary["failed_samples"] == 0 else 2)


if __name__ == "__main__":
    main()
