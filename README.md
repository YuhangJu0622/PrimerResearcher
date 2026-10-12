<div align="center">
  <h1>PrimerResearcher</h1>
  <h3>An Open-Domain DeepResearch Agent Post-Training Model</h3>
  <p>Agent · Cold-Start SFT · QQR / ArenaRL · Multi-Dimensional Trajectory Evaluation</p>
  <p>English | <a href="README_zh.md">简体中文</a></p>
  <hr />
  <p>
    <img src="https://img.shields.io/badge/Base%20Model-Qwen3--8B-5B8FF9?style=for-the-badge" alt="Base Model" />
    <img src="https://img.shields.io/badge/Context-32K-8E5BEF?style=for-the-badge" alt="Context" />
    <img src="https://img.shields.io/badge/RL-QQR%20%2F%20ArenaRL-F97316?style=for-the-badge" alt="RL" />
    <img src="https://img.shields.io/badge/Evaluation-7%20Dimensions-00A67E?style=for-the-badge" alt="Evaluation" />
  </p>
</div>

## Overview

PrimerResearcher builds autonomous agents that produce reports for open-domain research tasks. It presents an end-to-end methodology covering high-quality trajectory collection, cold-start supervised fine-tuning (SFT), QQR / ArenaRL post-training with relative-ranking rewards, and multi-dimensional automated evaluation.

Unlike tasks with a single verifiable answer, deep research requires an agent to search, read, synthesize evidence, and produce a structured report over multiple turns. PrimerResearcher turns that external workflow into learnable long trajectories: SFT first teaches reliable tool use and report writing, then relative-ranking reinforcement learning improves research, reasoning, and response quality on complex open-ended tasks.

## Results

On Tongyi Lab's in-house DeepResearch benchmark, the project fine-tunes Qwen3-8B. Adding a trajectory filter based on similarity between tool-response results reduces cross-turn duplicate tool calls from 40% to 8%. The SFT model improves its win rate against the base model from 33.3% to 68.9%; subsequent RL training further raises it to 76.6%.

| Stage | Win rate against Base | Description |
| --- | ---: | --- |
| SFT | 68.9% | Cold-start fine-tuning on curated long trajectories |
| RL | 76.6% | Post-training with QQR / ArenaRL relative-ranking rewards |

> These scores are produced by bidirectional LLM-judge comparisons of both trajectories and final answers. 

## Public Resources

- **Training dataset:** [PrimerResearcher-Data on ModelScope](https://www.modelscope.cn/datasets/YuhangJu/PrimerResearcher-Data)
- **SFT model:** [PrimerResearcher-SFT-8B on ModelScope](https://www.modelscope.cn/models/YuhangJu/PrimerResearcher-SFT-8B)

Please refer to the respective ModelScope pages for license, intended use, and version details.

## Highlights

### Long-horizon DeepResearch Agent

- **Unified message and tool protocol.** Standardized messages, tool calls, and tool observations make online trajectories directly reusable for training and evaluation.
- **Explicit convergence control.** Tool observations expose a forced-final-answer flag and remaining-turn information. The agent stops calling tools at configured round or context limits and answers from the collected evidence.
- **Tiered fault tolerance.** Search, webpage retrieval, and evidence extraction are supported. Search can rotate API keys and simplify queries; webpage access can fall back across direct HTTP, anonymous Jina Reader, and authenticated Jina Reader.
- **Reproducible runs.** Each run stores frozen configuration, a manifest, per-question trajectories, logs, and tool events; batch jobs can resume after interruption.

### Curated Trajectories and Cold-Start SFT

- Deep-research trajectories are collected with Qwen3-Plus and a custom agent framework, then filtered for format, reasoning, and outcome quality.
- A **tool-response similarity** rule removes ineffective repeated calls that retrieve substantially the same information in different turns.
- About 1K trajectories are used to LoRA fine-tune Qwen3-8B at a 32K context window for 260 training steps.
- The recipe uses all-linear LoRA, BF16, FlashAttention, sequence packing, gradient checkpointing, and Liger Kernel for efficient long-trajectory training.

### QQR / ArenaRL: Relative Ranking for Open-Ended Tasks

Open-domain reports do not have a single ground-truth answer. Pointwise scalar scoring often fails to reliably distinguish trajectories of similar quality, causing reward noise to dominate optimization. QQR adopts the ArenaRL approach, replacing per-trajectory scalar rewards with within-group relative ranking:

1. Sample multiple complete agent trajectories for the same query.
2. Use a process-aware pairwise judge to assess the research path, tool calls, and final answer. Evaluate both presentation orders to reduce position bias.
3. Use a deterministic greedy-decoded trajectory as a quality anchor for preliminary ranking.
4. Build a seeded single-elimination tournament: high seeds face low seeds, yielding stable within-group ranks.
5. Map ranks to normalized advantages and perform limited off-policy updates with a KL constraint.

The combination of anchor-based seeding and seeded single elimination requires roughly `2N - 2` pairwise comparisons, providing linear comparison cost while retaining much of the ranking quality of a full round-robin tournament.

### Seven-Dimension, Bidirectional Evaluation

Evaluation scores both the research process and final report, rather than comparing only final text. Each dimension is scored from 0 to 10.

| Target | Dimension | What it measures |
| --- | --- | --- |
| Research path | Framework | Completeness, clarity, and progression of the research plan |
| Research path | Tool Usage | Specificity, diversity, efficiency, and ordering of retrieval calls |
| Research path | Coverage | Whether research directions cover user requirements and support the response |
| Final answer | Relevance | Coverage of all user questions and constraints |
| Final answer | Accuracy | Evidence support, factual correctness, and internal consistency |
| Final answer | Depth | Quality of analysis, comparison, reasoning, and logical links |
| Final answer | Clarity | Organization, wording, readability, and practical usability |

Candidate and reference trajectories are judged in both A/B and B/A order. Scores are averaged to determine a win, loss, or tie.

## End-to-End Workflow

```text
Open-domain research query
          |
          v
Multi-turn DeepResearch Agent --> tool calls, evidence, and full trajectories
          |                                      |
          |                                      +--> trajectory data --> cold-start SFT
          |                                      |
          |                                      +--> unlabeled queries --> QQR / ArenaRL
          |                                                                  |
          +------------------------------------------------------------------+
                                                                     |
                                                                     v
                                 seven-dimension bidirectional LLM-judge evaluation
```

## Repository Layout

```text
PrimerResearcher/
├── agent/                  # Multi-turn Agent, tool protocol, and runtime
│   ├── configs/            # Agent configuration
│   ├── contracts/          # Message, tool-call, and Qwen3 codec contracts
│   ├── runtime/            # Agent loop, batch execution, logs, context control
│   └── tools/              # Search, webpage access, evidence extraction, cache
├── sft/                    # Qwen3-8B LoRA cold-start training script
├── evaluation/             # Trajectory projection, bidirectional judge, summaries
├── deployment/             # vLLM deployment and LoRA hot-switching scripts
├── data/                   # Local data directory; large datasets are not committed
└── .env.example            # Environment-variable template for Agent and Judge
```

> The current workspace does not contain the `rl/` directory or a QQR/ArenaRL training launcher. RL-related code will be released soon.

## Quick Start

### 0. Prerequisites

Agent, SFT, and RL use different environments. The Agent can run in a local Python environment; SFT and RL are intended for a Linux environment with NVIDIA GPUs.

Create a local environment file from the template:

```bash
cp .env.example .env.local
```

Configure at least the following variables for the Agent:

| Variable | Purpose |
| --- | --- |
| `AGENT_BASE_URL` | OpenAI-compatible endpoint for the Agent model |
| `AGENT_API_KEY` | API key for the Agent model service |
| `AGENT_MODEL` | Agent model name; use `my_lora` for the deployed LoRA service |
| `SERPER_API_KEYS` | Serper Search API keys, separated by commas |
| `TOOL_SUMMARY_BASE_URL` / `TOOL_SUMMARY_API_KEY` / `TOOL_SUMMARY_MODEL` | Optional evidence-extraction model configuration |
| `JUDGE_BASE_URL` / `JUDGE_API_KEY` / `JUDGE_MODEL` | Trajectory-judge configuration |

### 1. Run One Agent Query

```bash
python -m agent \
  --query "Compare two mainstream long-context training methods and recommend when to use each." \
  --project-root .
```

Runs are written to `runs/agent/<run-id>/`, including frozen configuration, manifest, per-question logs, and complete dialogue trajectories.

### 2. Batch Execution and Resume

Each input JSONL row needs at least `query`; `id` is optional:

```json
{"id":"example-001","query":"Research a topic and produce a structured report with sources."}
```

Run a batch:

```bash
python -m agent \
  --input data/deepresearch/test.jsonl \
  --project-root . \
  --run-id baseline
```

Resume it:

```bash
python -m agent \
  --input data/deepresearch/test.jsonl \
  --project-root . \
  --resume-run runs/agent/baseline
```

### 3. Cold-Start SFT

The SFT launcher requires MS-Swift, FlashAttention, and a GPU environment. Set or edit `MODEL_DIR`, `TRAIN_FILE`, `VAL_FILE`, and `OUTPUT_DIR` before launching:

```bash
bash sft/train_qwen3_8b_lora_all_linear.sh
```

The default recipe uses Qwen3-8B, a 32,768-token context window, all-linear LoRA (rank 16), BF16, packing, and gradient checkpointing.

### 4. Build an Evaluation Set and Run the Judge

Project a candidate run and the test-set reference trajectories into a compact judge dataset:

```bash
python -m evaluation.project \
  --run-dir runs/agent/candidate \
  --test-file data/deepresearch/test.jsonl
```

Run the bidirectional pairwise judge:

```bash
python -m evaluation.judge \
  --dataset runs/agent/candidate/evaluation/against-test/dataset.judge-v1.jsonl \
  --project-root .
```

The evaluation writes per-item `results.jsonl` and a `summary.json` with wins, losses, ties, win rates, confidence intervals, and average score margins.

### 5. Deploy the Base or LoRA Model

The deployment scripts use vLLM. The examples assume two GPUs, a 32K context window, and the Qwen3 tool-call parser. Adjust model paths, port, and parallelism for your environment:

```bash
bash deployment/deploy_base_model.sh
# or
bash deployment/deploy_lora_model.sh
```

The LoRA service supports runtime hot-switching under the same adapter name:

```bash
bash deployment/switch_lora.sh /path/to/new-checkpoint
```

## Citation

The relative-ranking RL method used by this project is based on ArenaRL:

```bibtex
@article{zhang2026arenarl,
  title={ArenaRL: Scaling RL for Open-Ended Agents via Tournament-based Relative Ranking},
  author={Zhang, Qiang and Chen, Boli and Zhang, Fanrui and others},
  journal={arXiv preprint arXiv:2601.06487},
  year={2026}
}
```
