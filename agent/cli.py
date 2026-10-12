"""Runnable P1 single-query and resumable JSONL entry point."""
# .\.venv\Scripts\python -m agent --input data\deepresearch\test.jsonl --start-index 0 --end-index 29

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path

import yaml

from agent.config import load_agent_config
from agent.contracts.qwen3_codec import Qwen3Codec
from agent.environment import load_project_env
from agent.run import create_run
from agent.runtime.batch import QuestionLogger, _write_trajectory, run_jsonl
from agent.runtime.loop import AgentLoop
from agent.runtime.openai_client import OpenAIModelClient
from agent.runtime.token_counter import load_qwen3_tokenizer
from agent.prompts import default_system_prompt
from agent.tools.web_search import WebSearch, WebSearchSettings

async def _main(args: argparse.Namespace) -> int:
    if args.run_id and args.resume_run:
        raise RuntimeError("--run-id and --resume-run are mutually exclusive")
    if args.query and (args.start_index is not None or args.end_index is not None):
        raise RuntimeError("--start-index and --end-index are supported only with --input")
    if args.start_index is not None and args.start_index < 0:
        raise RuntimeError("--start-index must be non-negative")
    if args.end_index is not None and args.end_index < 0:
        raise RuntimeError("--end-index must be non-negative")
    if args.start_index is not None and args.end_index is not None and args.start_index > args.end_index:
        raise RuntimeError("--start-index must be less than or equal to --end-index")
    load_project_env(args.project_root)
    config = load_agent_config(args.config)
    tokenizer_path = Path(args.project_root) / config.runtime.tokenizer_path
    codec = Qwen3Codec(load_qwen3_tokenizer(tokenizer_path)) if config.runtime.max_context_tokens else Qwen3Codec()
    model_base_url = os.getenv("AGENT_BASE_URL", "")
    model_api_key = os.getenv("AGENT_API_KEY", "")
    model_id = os.getenv("AGENT_MODEL", config.model.id)
    if not model_base_url or not model_api_key:
        raise RuntimeError("AGENT_BASE_URL and AGENT_API_KEY must be configured")
    config.model.id = model_id
    if args.resume_run:
        run_dir = Path(args.resume_run).resolve()
        allowed_root = (Path(args.project_root).resolve() / config.runtime.run_root / config.stage).resolve()
        if allowed_root not in run_dir.parents or not run_dir.is_dir():
            raise RuntimeError("resume run must be an existing directory under the configured stage run root")
        frozen_path = run_dir / "frozen_config.yaml"
        manifest_path = run_dir / "run_manifest.json"
        if not frozen_path.is_file() or not manifest_path.is_file():
            raise RuntimeError("resume run is missing frozen config or manifest")
        frozen = yaml.safe_load(frozen_path.read_text(encoding="utf-8"))
        current_config = config.model_dump(mode="json")
        # Batch concurrency affects only local scheduling, not an individual
        # question's agent behavior or its recorded trajectory.  Allow it to
        # be adjusted when resuming while preserving strict validation of the
        # run's model, tool, and runtime semantics.
        frozen["runtime"].pop("batch_concurrency", None)
        current_config["runtime"].pop("batch_concurrency", None)
        if frozen != current_config:
            raise RuntimeError("current config does not match the resumed run's frozen config")
        if args.query:
            raise RuntimeError("--resume-run is supported only with --input")
    else:
        run_dir = create_run(config, args.project_root, args.run_id)
    model = OpenAIModelClient(
        base_url=model_base_url, api_key=model_api_key, model=model_id,
        timeout=config.model.timeout_seconds, max_retries=config.model.max_retries,
        temperature=config.model.temperature, top_p=config.model.top_p,
        max_tokens=config.model.max_output_tokens,
        enable_thinking=config.model.enable_thinking,
        debug_response_logging=config.runtime.log_level.upper() == "DEBUG",
    )
    tool_settings = WebSearchSettings.from_env(
        cache_path=str(Path(args.project_root) / config.tool.cache_path),
        cache_namespace=config.tool.cache_namespace, cache_mode=config.tool.cache_mode,
        connect_timeout=config.tool.connect_timeout_seconds,
        read_timeout=config.tool.read_timeout_seconds, total_timeout=config.tool.timeout_seconds,
        summary_timeout=config.tool.summary_timeout_seconds,
        min_page_chars=config.tool.min_page_chars, max_page_chars=config.tool.max_page_chars,
        max_snippet_chars=config.tool.max_snippet_chars,
        max_evidence_chars=config.tool.max_evidence_chars,
        max_output_chars=config.tool.max_output_chars,
        retry_count=config.tool.retry_count,
        serper_same_key_max_retries=config.tool.serper_same_key_max_retries,
        serper_max_key_switches=config.tool.serper_max_key_switches,
        serper_retry_base_seconds=config.tool.serper_retry_base_seconds,
        jina_enabled=config.tool.jina_enabled,
        visit_stage_timeout_seconds=config.tool.visit_stage_timeout_seconds,
        visit_max_attempts=config.tool.visit_max_attempts,
        visit_retry_delays_seconds=config.tool.visit_retry_delays_seconds,
    )
    try:
        async with WebSearch(tool_settings) as tool:
            def factory(*, event_sink=None) -> AgentLoop:
                return AgentLoop(
                    model, tool, codec=codec, max_tool_rounds=config.runtime.max_tool_rounds,
                    max_context_tokens=config.runtime.max_context_tokens,
                    token_counter=codec.count_prompt_tokens if config.runtime.max_context_tokens else None,
                    max_output_tokens=config.model.max_output_tokens,
                    forced_final_trigger_tokens=config.runtime.forced_final_trigger_tokens,
                    forced_final_attempts=config.runtime.forced_final_attempts,
                    forced_final_answer_reserve_tokens=config.runtime.forced_final_answer_reserve_tokens,
                    forced_final_overhead_tokens=config.runtime.forced_final_overhead_tokens,
                    tool_timeout=config.tool.timeout_seconds, tool_concurrency=config.tool.concurrency,
                    as_of_time=config.runtime.as_of_time,
                    event_sink=event_sink,
                )
            if args.query:
                system_prompt = args.system_prompt or default_system_prompt(args.query)
                logger = QuestionLogger(
                    run_dir / "logs" / "000001.log", question_number=1,
                    question_id="query", level=config.runtime.log_level,
                )
                logger.write("question_started", phase="question", query=args.query, query_chars=len(args.query),
                             max_attempts=config.runtime.sample_retries + 1)
                result = None
                for attempt in range(config.runtime.sample_retries + 1):
                    logger.attempt = attempt + 1
                    logger.write("attempt_started", phase="attempt")
                    result = await factory(event_sink=logger.emit_loop_event).run(args.query, system_prompt)
                    logger.write("attempt_finished", phase="attempt", terminal_state=result.terminal_state,
                                 elapsed_seconds=result.elapsed_seconds, tool_rounds=result.tool_rounds)
                    if result.terminal_state == "valid":
                        break
                    if attempt < config.runtime.sample_retries:
                        logger.write("retry_scheduled", level="WARNING", phase="recovery",
                                     previous_terminal_state=result.terminal_state, next_attempt=attempt + 2)
                assert result is not None
                _write_trajectory(run_dir / "trajectories" / "000001.json", messages=result.messages)
                logger.write("question_finished", level="INFO" if result.terminal_state == "valid" else "ERROR",
                             phase="question", terminal_state=result.terminal_state, attempts=attempt + 1)
                print(json.dumps({"run_dir": str(run_dir), "terminal_state": result.terminal_state}, ensure_ascii=False))
                return 0 if result.terminal_state == "valid" else 2
            counts = await run_jsonl(
                args.input, run_dir / "logs", run_dir / "trajectories", factory,
                system_prompt=args.system_prompt or default_system_prompt, max_retries=config.runtime.sample_retries,
                concurrency=config.runtime.batch_concurrency,
                start_index=args.start_index, end_index=args.end_index,
                log_level=config.runtime.log_level,
            )
            print(json.dumps({"run_dir": str(run_dir), **counts}, ensure_ascii=False))
            return 0 if counts["failed"] == 0 else 2
    finally:
        await model.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the P1 DeepResearch Agent")
    parser.add_argument("--config", default="agent/configs/agent.yaml")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--run-id")
    parser.add_argument("--resume-run", help="resume an existing batch run directory")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--query")
    group.add_argument("--input", help="JSONL containing query/question and optional id")
    parser.add_argument(
        "--start-index", type=int,
        help="first nonblank input record to run (0-based, inclusive; batch input only)",
    )
    parser.add_argument(
        "--end-index", type=int,
        help="last nonblank input record to run (0-based, inclusive; batch input only)",
    )
    parser.add_argument("--system-prompt", help="override the language-aware default system prompt")
    args = parser.parse_args()
    raise SystemExit(asyncio.run(_main(args)))


if __name__ == "__main__":
    main()
