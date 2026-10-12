<div align="center">
  <h1>PrimerResearcher</h1>
  <h3>面向开放域深度研究报告生成的 Agent 后训练模型</h3>
  <p>Agent · 冷启动 SFT · QQR / ArenaRL · 多维轨迹评测</p>
  <p><a href="README.md">English</a> | 简体中文</p>
  <hr />
  <p>
    <img src="https://img.shields.io/badge/Base%20Model-Qwen3--8B-5B8FF9?style=for-the-badge" alt="Base Model" />
    <img src="https://img.shields.io/badge/Context-32K-8E5BEF?style=for-the-badge" alt="Context" />
    <img src="https://img.shields.io/badge/RL-QQR%20%2F%20ArenaRL-F97316?style=for-the-badge" alt="RL" />
    <img src="https://img.shields.io/badge/Evaluation-7%20Dimensions-00A67E?style=for-the-badge" alt="Evaluation" />
  </p>
</div>

## 简介

PrimerResearcher 面向开放域研究任务构建自主报告生成 Agent，并提供从高质量轨迹采集、冷启动监督微调（SFT）、基于相对排序奖励的 QQR / ArenaRL 后训练，到多维度自动评测的完整方法框架。

与具有唯一标准答案的任务不同，深度研究通常需要 Agent 在多轮交互中主动检索、阅读和整合证据，再形成结构化报告。项目将这条外部工作流转化为可学习的长轨迹：先由 SFT 学习可靠的工具调用与报告生成范式，再通过以轨迹间相对优劣为核心的强化学习，持续优化复杂开放域任务中的研究、推理与作答能力。

## 研究结果

在 Tongyi Lab 自建 DeepResearch 评测集上，项目以 Qwen3-8B 为基座模型完成微调。额外引入基于工具响应结果相似度的轨迹筛选规则后，SFT 模型跨轮次工具调用重复率由 40% 降至 8%；模型相对 Base 的胜率由 33.3% 提升至 68.9%。在此基础上继续进行 RL 微调后，胜率达到 76.6%。

| 阶段 | 相对 Base 胜率 | 说明 |
| --- | ---: | --- |
| SFT | 68.9% | 高质量长轨迹冷启动微调 |
| RL | 76.6% | 基于 QQR / ArenaRL 相对排序奖励的后训练 |

> 以上胜率来自双向 LLM Judge 的轨迹与最终答案对比

## 公开资源

- **训练数据集：** [PrimerResearcher-Data（ModelScope）](https://www.modelscope.cn/datasets/YuhangJu/PrimerResearcher-Data)
- **SFT 模型：** [PrimerResearcher-SFT-8B（ModelScope）](https://www.modelscope.cn/models/YuhangJu/PrimerResearcher-SFT-8B)

数据集与模型的具体许可、使用范围和版本信息以对应 ModelScope 页面为准。

## 核心亮点

### 1. 面向长程研究的 Agent

- **统一的消息与工具协议。** Agent 使用统一的消息、工具调用及工具响应格式，便于将线上执行轨迹直接投影为训练或评测样本。
- **显式的收敛控制。** 工具响应包含强制作答标记与剩余轮次信息；达到轮数或上下文阈值后，Agent 停止调用工具，基于已收集证据完成最终回答。
- **分级容错的检索链路。** 支持 Search、网页访问和证据抽取；检索可轮换 Key 并简化查询，网页访问可在直接抓取、匿名 Jina Reader 与认证 Jina Reader 间分级降级。
- **可复现运行记录。** 每次运行冻结配置并保存 manifest、逐题轨迹、日志和工具事件，支持中断后的批处理恢复与审计。

### 2. 高质量轨迹数据与冷启动 SFT

- 使用 Qwen3-Plus 与自定义 Agent 框架采集深度研究轨迹，并根据格式、逻辑、结果质量筛选样本。
- 额外以**工具响应结果相似度**筛除无效重复调用，减少“换一种 query 却获得同类信息”的跨轮冗余。
- 在 32K 上下文窗口下，基于约 1K 条轨迹对 Qwen3-8B 进行 LoRA 高效微调，共完成 260 steps 训练。
- 训练配方采用 all-linear LoRA、BF16、FlashAttention、sequence packing、梯度检查点与 Liger Kernel，以支持长轨迹训练。

### 3. QQR / ArenaRL：面向开放域任务的相对排序奖励

开放域研究报告没有唯一的标准答案。逐条轨迹打标量分数时，LLM Judge 往往难以稳定区分质量接近的候选，导致奖励信号被评分噪声淹没。QQR 所采用的 ArenaRL 思路将奖励从“逐条打分”改为“组内相对排序”：

1. 对同一问题采样多条完整 Agent 轨迹。
2. 使用过程感知的 Pairwise Judge 联合评估研究路径、工具调用和最终回答；正反两个展示顺序各评一次，以降低位置偏差。
3. 以贪心解码轨迹作为质量锚点（anchor）对候选进行预排序。
4. 按种子构造单败淘汰赛（seeded single-elimination），让高种子与低种子配对，得到稳定的组内排名。
5. 将名次映射为归一化 advantage，进行有限步 Off-Policy 策略更新，并通过 KL 约束控制策略漂移。

这种“锚点预排序 + 种子单败淘汰”的机制以约 `2N - 2` 次两两比较获得线性的比较开销，同时尽量保留完整循环赛的排名质量，避免优质轨迹因随机首轮配对过早淘汰。

### 4. 7 维、双向的轨迹级评测

评测同时检查 Agent 的研究过程与最终报告，而不是只比较最终文本。每个维度按 0–10 分评分：

| 对象 | 维度 | 关注点 |
| --- | --- | --- |
| 研究路径 | Framework | 研究框架是否完整、清晰且递进合理 |
| 研究路径 | Tool Usage | 检索是否有针对性、多样且无明显冗余 |
| 研究路径 | Coverage | 检索方向是否覆盖用户要求并支撑回答 |
| 最终回答 | Relevance | 是否完整回应用户问题与约束 |
| 最终回答 | Accuracy | 结论是否有依据、无明显矛盾或错误 |
| 最终回答 | Depth | 是否具有足够的分析深度与逻辑链条 |
| 最终回答 | Clarity | 结构、表达和可用性是否清晰 |

候选轨迹与参考轨迹会以 A/B、B/A 两个顺序分别交给 Judge，取双方得分均值并判定胜、负或平局。

## 端到端流程

```text
开放域研究问题
      │
      ▼
多轮 DeepResearch Agent ──► 工具调用、证据与完整轨迹
      │                                  │
      │                                  ├──► 轨迹数据 ──► 冷启动 SFT
      │                                  │
      │                                  └──► 未标注问题 ──► QQR / ArenaRL
      │                                                       │
      └───────────────────────────────────────────────────────┤
                                                              ▼
                                  7 维双向 LLM Judge 与胜率评测
```

## 仓库结构

```text
PrimerResearcher/
├── agent/                  # 多轮 DeepResearch Agent、工具协议与运行时
│   ├── configs/            # Agent 配置
│   ├── contracts/          # 消息、工具调用与 Qwen3 编解码约定
│   ├── runtime/            # Agent Loop、批处理、日志和上下文控制
│   └── tools/              # Search、网页访问、证据抽取与缓存
├── sft/                    # Qwen3-8B LoRA 冷启动训练脚本
├── evaluation/             # 轨迹投影、双向 Pairwise Judge 与结果汇总
├── deployment/             # Base / LoRA vLLM 部署及 LoRA 热切换脚本
├── data/                   # 本地数据目录（默认不提交大规模数据）
└── .env.example            # Agent 与 Judge 的环境变量模板
```

> 当前工作区尚未包含 `rl/` 目录及 QQR/ArenaRL 训练启动脚本。RL 相关代码文件即将发布。

## 快速开始

### 0. 前置准备

Agent、SFT 和 RL 的运行环境不同。Agent 可在本地 Python 环境运行；SFT 与 RL 建议在具备 Linux 与 NVIDIA GPU 的训练环境中执行。

从模板创建本地环境变量文件：

```bash
cp .env.example .env.local
```

至少配置 Agent 使用的以下变量：

| 变量 | 用途 |
| --- | --- |
| `AGENT_BASE_URL` | OpenAI 兼容 Agent 模型服务地址 |
| `AGENT_API_KEY` | Agent 模型服务密钥 |
| `AGENT_MODEL` | Agent 模型名；部署 LoRA 时可设为 `my_lora` |
| `SERPER_API_KEYS` | Serper 检索 API Key，多个 Key 用逗号分隔 |
| `TOOL_SUMMARY_BASE_URL` / `TOOL_SUMMARY_API_KEY` / `TOOL_SUMMARY_MODEL` | 可选的证据抽取模型配置 |
| `JUDGE_BASE_URL` / `JUDGE_API_KEY` / `JUDGE_MODEL` | 轨迹评测 Judge 配置 |

### 1. 运行单条 Agent 任务

```bash
python -m agent \
  --query "比较两种主流长上下文训练方法的适用场景，并给出技术选型建议" \
  --project-root .
```

运行目录默认写入 `runs/agent/<run-id>/`，其中包含冻结配置、运行 manifest、逐题日志与完整对话轨迹。

### 2. 批量执行与断点恢复

输入 JSONL 的每一行至少包含 `query`，也可以附带 `id`：

```json
{"id":"example-001","query":"请调研某一主题，并给出带来源的结构化报告。"}
```

批量运行：

```bash
python -m agent \
  --input data/deepresearch/test.jsonl \
  --project-root . \
  --run-id baseline
```

恢复同一次运行：

```bash
python -m agent \
  --input data/deepresearch/test.jsonl \
  --project-root . \
  --resume-run runs/agent/baseline
```

### 3. 冷启动 SFT

SFT 脚本依赖 MS-Swift、FlashAttention 和 GPU 环境。执行前修改或通过环境变量指定 `MODEL_DIR`、`TRAIN_FILE`、`VAL_FILE` 与 `OUTPUT_DIR`：

```bash
bash sft/train_qwen3_8b_lora_all_linear.sh
```

默认配方使用 Qwen3-8B、32,768 token 上下文、all-linear LoRA（rank 16）、BF16、packing 和梯度检查点。

### 4. 构建评测集并运行双向 Judge

将候选运行与测试集中的参考轨迹投影为精简评测集：

```bash
python -m evaluation.project \
  --run-dir runs/agent/candidate \
  --test-file data/deepresearch/test.jsonl
```

执行双向 Pairwise Judge：

```bash
python -m evaluation.judge \
  --dataset runs/agent/candidate/evaluation/against-test/dataset.judge-v1.jsonl \
  --project-root .
```

结果会保存为逐题 `results.jsonl` 和汇总 `summary.json`，包括胜负平数量、胜率、等价胜率、置信区间与平均得分差。

### 5. 部署 Base 或 LoRA 模型

部署脚本基于 vLLM，示例默认使用两张 GPU、32K 上下文和 Qwen3 tool-call parser。根据机器环境调整脚本中的模型路径、端口与并行度：

```bash
bash deployment/deploy_base_model.sh
# 或
bash deployment/deploy_lora_model.sh
```

LoRA 服务支持运行时同名热切换：

```bash
bash deployment/switch_lora.sh /path/to/new-checkpoint
```

## 引用

本项目的相对排序 RL 方法参考 ArenaRL：

```bibtex
@article{zhang2026arenarl,
  title={ArenaRL: Scaling RL for Open-Ended Agents via Tournament-based Relative Ranking},
  author={Zhang, Qiang and Chen, Boli and Zhang, Fanrui and others},
  journal={arXiv preprint arXiv:2601.06487},
  year={2026}
}
```
