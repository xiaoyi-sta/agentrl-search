# 项目报告：小模型多跳检索 Agent 的任务级后训练

**任务**：让 Qwen3-1.7B 学会调用检索工具完成多跳问答
**约束**：单卡 24G、总预算 ≤ ¥100、可复现
**结果**：EM 0.140 → 0.405（+26.5 点），总花费 ≈ ¥25

---

## 目录

1. 问题定义与选型
2. 系统架构
3. 数据与语料
4. 检索器设计
5. 工具环境与输出协议
6. 评测方法学
7. 基线
8. SFT
9. RFT
10. GRPO
11. 结果汇总
12. 工程问题记录
13. 局限与未来工作
14. 复现指南

---

## 1. 问题定义与选型

### 1.1 任务

给定一个多跳问题（如“《卧虎藏龙》的导演还导演过哪部获奥斯卡最佳外语片的电影？”），模型必须：

1. 生成检索查询；
2. 读取检索结果；
3. 必要时再检索；
4. 用 `<answer>` 输出最终答案。

### 1.2 为什么选“工具调用 + 多跳问答”

| 候选方向 | 判断 |
|---|---|
| 纯数学推理 RL（GSM8K） | 同质化严重，面试官见太多 |
| 代码生成 Agent | 环境依赖复杂，单卡跑不动 |
| **多跳检索 Agent** | 有明确环境、可验证奖励、单卡可训，且是 2026 年招聘热点 |
| 多智能体协作 | 评测困难，成本高 |

选择 HotpotQA 的理由：

- 答案可精确匹配（EM/F1），不需要奖励模型；
- 自带 supporting facts，可以做检索诊断；
- bridge 类问题天然需要两次检索，能体现 agent 行为。

### 1.3 为什么是 1.7B 而不是 7B

| 模型 | 全参显存 | LoRA 显存 | 单卡 24G |
|---|---|---|---|
| 1.7B | ~23G（OOM） | 7–9G | ✅ |
| 7B | ~90G | ~20G | ⚠️ 勉强，训练慢 4 倍 |

预算 ¥100 下，7B 的迭代次数会少一个量级，不适合“多轮实验”的项目目标。1.7B 能在 30 分钟内完成一轮 SFT，允许快速试错。

---

## 2. 系统架构

```
HotpotQA (parquet)
   │ normalize（三种 context 结构归一）
   ▼
corpus.jsonl (66,581 段)  ──►  BM25 索引
   │                              │
questions.jsonl                   │
   │  split(eval200/rl1200/sft2000)
   ▼                              ▼
SFT 数据生成 ──► LoRA SFT ──► 评测 ◄── SearchEnv（search / calculator）
                    │
                    ├──► RFT（拒绝采样 → 再 SFT）
                    └──► GRPO（Dr.GRPO + 工具预算奖励）
```

关键设计：**训练、评测、数据生成共用同一套 `prompts.py` / `SearchEnv` / parser**，任何格式改动只改一处。

---

## 3. 数据与语料

### 3.1 语料的构建

**方法**：把 dev 集每题 10 个 distractor 段落按标题去重，标题+正文拼接。

**为什么不去下全量 Wikipedia**：

- 全量索引 600 万段，单机 BM25 内存吃不下；
- dev 段落本身构成一个“闭卷”语料，评测更可控；
- 代价是检索难度被低估，报告中按相对比较解读。

**遇到的问题**：parquet 的 `context` 字段是**列式 struct**（`{"title": [...], "sentences": [[...]]}`），不是常见的 list-of-dict。

**解决**：写归一化层 `normalize.py`，同时兼容三种形态（列式 struct / list of struct / 原始 JSON 二维数组），下游只认 `{"title","sentences"}`。

### 3.2 数据切分

| split | 数量 | 用途 |
|---|---|---|
| eval200 | 200 | 全部评测，永不训练 |
| rl1200 | 1200 | RL / RFT 采样池 |
| sft2000 | 2000 | 预留 SFT 池 |
| 其余 | ~4000 | 未使用 |

切分后断言三组 qid 两两不交。

**已知数据问题**：该 parquet 的 `level` 列全为 `hard`（7405/7405），说明导出时损坏；因此分层只按 `type`（bridge/comparison）做。

---

## 4. 检索器设计

### 4.1 为什么用 BM25

| 方案 | 成本 | 可复现 | 诊断性 |
|---|---|---|---|
| BM25 | ¥0 | 完全确定 | 词面匹配失败可解释 |
| 稠密检索（bge/e5） | 需 GPU + 索引 | 依赖模型版本 | 失败难归因 |
| 在线搜索 API | 按次计费 | 不稳定 | 不可复现 |

BM25 的“同义改写就搜不到”恰好是 bad case 的主要来源，便于分析。

### 4.2 分词器的三次迭代

| 版本 | 分词 | 问题 | 结果 R@5 |
|---|---|---|---|
| v0 | `[a-z0-9]+`，无标题 | `Motörhead` 被切成 `mot`+`rhead` | 0.613 |
| v1 | 加标题入索引 + 查询去停用词 | 变音符仍丢 | 0.695（旧分词） |
| v2 | `[^\W_]+`（Unicode） | — | **0.695** |

**关键决策**：文档侧保留全部词（让 IDF 更准），查询侧去停用词（避免 `what/the` 稀释权重）。

**验证**：v1→v2 的改动只影响 200 题中的 7 题（3.5%），但修复了带变音符实体永远搜不到的问题。

### 4.3 两跳检索的天然上限

| 指标 | 值 | 解释 |
|---|---|---|
| Recall@3 | 0.610 | top-3 覆盖的 gold 比例 |
| AllHit@3 | 0.315 | 两段 gold 同时进 top-3 |
| Recall@5 | 0.695 | |
| AllHit@5 | 0.440 | |

**为什么 AllHit@5 只有 0.44**：bridge 题的第二跳实体名不在问题里（例：问“某乐队首张未发行录音”，答案段是 “Motörhead”，但问题里没有这个词），单次检索在词面上不可能命中。

**这个数字是项目的核心论据**：它证明了“为什么需要多轮 agent”——不是检索器不够好，而是查询必须由模型改写。

---

## 5. 工具环境与输出协议

### 5.1 文本协议

```text
<tool_call>{"name": "search", "arguments": {"query": "Motörhead band"}}</tool_call>
<answer>Song Il-gon</answer>
```

**为什么不直接用 OpenAI function calling**：

- 文本协议对模型更自然（预训练见过 `<tool_call>` 形式）；
- 评测、训练、数据生成可以共用同一套 parser；
- 不依赖特定推理框架的工具抽象。

**代价**：与 TRL 的结构化 `tool_calls` 不兼容——这是后来 GRPO 失败的直接原因（见第 10 节）。

### 5.2 SearchEnv

```python
class SearchEnv:
    def reset(self, question, answer, supporting_titles=None): ...
    def search(self, query: str) -> str:      # top-3，每条 ≤60 词
    def calculator(self, expression: str) -> str:
```

- 索引是**类级单例**，避免每次 rollout 重建（重建一次 30 秒）；
- 观测截断到 60 词 × 3 条，控制上下文长度；
- 工具返回错误时以文本返回，让模型自己重试，而不是抛异常中断轨迹。

### 5.3 奖励设计

```
R = 0.7·EM + 0.2·F1 + 0.1·format − λ·(calls / max_calls)
```

| 项 | 权重 | 目的 |
|---|---|---|
| EM | 0.7 | 主目标 |
| F1 | 0.2 | 给部分正确的梯度 |
| format | 0.1 | 保证 `<answer>` 存在且不夹带工具调用 |
| budget | λ=0.02–0.1 | 抑制无效检索 |

**设计反思**：λ 取 0.1 时，模型学会了“少搜换奖励”（avg_calls 2.21→2.03，EM −2 点）。后续改为 0.02。
**教训**：辅助奖励项的权重不能接近主项，否则模型会优化辅助项。

---

## 6. 评测方法学

### 6.1 指标

| 指标 | 定义 | 用途 |
|---|---|---|
| EM | 归一化后精确匹配 | 主指标 |
| F1 | 词级重叠 | 辅助 |
| em_contains | 预测文本包含 gold | 诊断“会答但格式不对” |
| tag_rate | 有 `<answer>` 的比例 | 协议合规 |
| avg_calls | 平均工具调用次数 | 行为 |
| tool_hit_rate | 检索结果含 gold 标题 | 检索质量 |
| no_answer_rate | 无答案比例 | 失败模式 |

### 6.2 2×2 分解

| | 答对 | 答错 |
|---|---|---|
| 检索到 gold | hit_correct | hit_wrong |
| 未检索到 | miss_correct | miss_wrong |

这个表把“检索问题”和“推理问题”分开。基线时 `hit_wrong=157`，说明瓶颈在作答；SFT 后降到 111，仍是主要矛盾。

### 6.3 统计可靠性

200 题的 95% 置信区间 ≈ **±7 个点**（SE≈3.5%）。
因此：

- 1–2 点的差异**不能宣称显著**；
- 报告统一给点估计 + 区间说明；
- 训练/评测必须 greedy（`do_sample=False`）保证可复现。

### 6.4 泄漏检查

```python
assert not (train_qids & eval_qids)
assert not (val_qids & eval_qids)
```

实测交集为 0。生成脚本在采样前按 split 文件排除 qid，双重保险。

---

## 7. 基线

### 7.1 0.6B（管道验证）

| 指标 | 值 |
|---|---|
| EM | 0.040 |
| tag_rate | 0.200 |
| avg_calls | 0.95 |
| tool_hit | 0.840 |

结论：0.6B 能搜到（84%），但不会按协议交付（tag 0.20）、不会做第二跳（0.95）。

### 7.2 1.7B（正式基线）

| 指标 | 值 |
|---|---|
| EM / F1 | 0.140 / 0.225 |
| em_contains | 0.345 |
| tag_rate | 0.745 |
| avg_calls | 1.62 |
| tool_hit | 0.920 |
| 耗时 | 502.5s（200 题） |

2×2：27 / 157 / 1 / 15 → **检索不是瓶颈，作答才是**。

**为什么 1.7B 的 EM 仍然低**：
1. 27% 的输出包含正确答案但没放进 `<answer>`（`em_contains 0.345`）；
2. 单跳惯性：平均 1.62 次调用，远低于 bridge 题需要的 2 次；
3. 严格 EM 惩罚别名（`Alan Turing` vs `Alan Mathison Turing`）。

---

## 8. SFT

### 8.1 数据生成

**教师**：DeepSeek-V4.1-Flash（关闭思考模式，temperature 0.8）。
**工具**：用真实的 `SearchEnv` 执行，而不是让教师编造检索结果。

| 类型 | 生成 | 保留 | 用途 |
|---|---|---|---|
| A 格式 | 1200 | 287 | comparison 题，教协议 |
| B 两跳 | 2600 | 1367 | bridge 题，50% oracle 首跳 |
| C 弃答 | 400 | 187 | 假实体，教 unknown |
| 合计 | 4200 | **1937** | 成本 ¥5.5 |

**oracle 首跳**：对一半的 B 类样本，人工注入“正确第一跳查询 + 真实检索结果”，让教师只需要学“第二跳怎么问”。这把通过率从 ~40% 提到 65%。

**三级过滤**：
1. 回放校验：重新执行工具调用，observation 必须逐字一致；
2. 答案匹配：EM 或 F1≥0.8 或“包含 gold 且长度受限”；
3. 比例控制：C 类封顶 10%，B 类要求 ≥2 次调用。

最终 train 1841 / val 96，平均 2.06 轮。

### 8.2 训练

| 项 | 值 |
|---|---|
| LoRA | r=32，α=64，7 个投影层 |
| epoch | 2 |
| 有效 batch | 8（1×8 累积） |
| lr | 1e-4 cosine |
| max_len | 3072 |
| 损失 | 只在 assistant token 上（`assistant_only_loss`） |
| 耗时 | 462 步 / 28 分钟 / ¥1.2 |

**关键点：只在 assistant token 上算 loss**。
工具返回的段落是“外部文本”，若参与训练，模型会学成复述检索结果。TRL 通过 `{% generation %}` 标记实现，脚本里加了 preflight 检查掩码非空。

### 8.3 结果

| 指标 | 基线 | SFT | Δ |
|---|---|---|---|
| EM | 0.140 | **0.405** | **+0.265** |
| F1 | 0.225 | 0.531 | +0.306 |
| tag_rate | 0.745 | 0.960 | +0.215 |
| avg_calls | 1.62 | 2.21 | +0.59 |
| no_answer | 0.255 | 0.040 | −0.215 |

训练侧：train_loss 0.289，eval_loss 0.2575（低于 train，无过拟合），eval_entropy 0.21。

**新问题**：4% 的题跑满 5 轮（max_turns_hit 8）。SFT 教会了“多搜”，没教“何时停”。

---

## 9. RFT（拒绝采样微调）

### 9.1 方法

1. 用 SFT 模型对 500 道 rl 题各采 3 条（temperature 0.8，HF generate）；
2. 过滤：答案正确 + 格式合法 + ≥2 次调用；
3. 592 条通过，与全量 SFT 数据（1841）按 1:3 混合成 2433 条；
4. 在 merged SFT 上再挂 LoRA，1 epoch，lr 5e-5。

### 9.2 结果

| 指标 | SFT | RFT |
|---|---|---|
| EM | 0.405 | 0.395 |
| F1 | 0.531 | 0.525 |
| tag_rate | 0.960 | **0.980** |
| avg_calls | 2.21 | 2.19 |

**为什么没涨**：
- 592 条只来自约 200 道不同的题（同题多分支），有效多样性低；
- 拒绝采样是“模仿自己的好样本”，信息量上限就是采样分布；
- eval_loss 0.125 偏低，说明模型对 RFT 数据拟合很紧，但没有泛化增益。

**结论**：RFT 的收益取决于采样多样性（题目数 × 温度），不是采样数量。

---

## 10. GRPO

### 10.1 动机

SFT 只模仿，不会探索；GRPO 用组内相对优势优化最终奖励，理论上能学出示范外策略。
另外，工具预算奖励正好可以治“4% 跑满轮”的问题。

### 10.2 三次失败与修复

**第一次（E8）**：reward 恒为 −0.025，`answer_reward=0`。

- 现象：模型第一轮正常调用工具，第二轮输出 `<tool_call>{"` 截断。
- 定位：读 TRL 源码发现 **工具返回内容也计入 `max_completion_length`**（默认 256）。
  一次检索返回 3 段 × 60 词 ≈ 240 token，加上调用本身直接超预算 → TRL 丢弃工具结果 → 模型没看到资料。
- 修复：`max_completion_length=1024`。

**第二次**：reward 有值，但 50 步内从 0.41 掉到 0.37，calls 2.21→2.03。

- 定位：lr 5e-6 太大 + λ=0.1 让“少搜”比“答对”更划算。
- 修复：lr 1e-6、λ 0.02、temperature 0.8。

**第三次**：reward 仍下降（0.44→0.29），`clipped_ratio=0.375`。

- 定位：37.5% 的 completion 被截断，坏样本污染梯度；除 std 有难度偏置。
- 修复：`max_completion=2048` + `mask_truncated_completions=True` + `loss_type="dr_grpo"` + `entropy_coef=0.001` + `max_turns=3`。

### 10.3 最终结果

40 步，400 题 × group 4：

| 指标 | SFT | GRPO-40 |
|---|---|---|
| EM | 0.405 | 0.405 |
| F1 | 0.531 | **0.536** |
| tag_rate | 0.960 | 0.965 |
| avg_calls | 2.21 | 2.19 |

训练侧：reward 前 8 步 0.416 → 后 8 步 0.462（+0.046），clipped 0.375 → 0.00。

**结论**：GRPO 收敛到与 SFT 持平。归因：

1. 单卡 batch=16，组内方差估计噪声大；
2. 轨迹级奖励没有 step-level 信用分配；
3. 1.7B 容量在该任务上接近上限。

---

## 11. 结果汇总

| 阶段 | EM | F1 | tag_rate | avg_calls |
|---|---|---|---|---|
| 1.7B 基线 | 0.140 | 0.225 | 0.745 | 1.62 |
| SFT | **0.405** | 0.531 | 0.960 | 2.21 |
| SFT + RFT | 0.395 | 0.525 | 0.980 | 2.19 |
| SFT + GRPO | 0.405 | 0.536 | 0.965 | 2.19 |

**主结论**：SFT 贡献了几乎全部收益（+26.5 点）；RFT 与 GRPO 在 200 题评测下无统计显著差异（CI ±7 点）。

**负结果的解释**：
- 评测方差大于方法差异；
- 单卡 batch 限制了 GRPO 的组内估计；
- 轨迹级奖励对多轮任务的信用分配过粗；
- 1.7B 的表征能力是硬约束。

---

## 12. 工程问题记录

| 问题 | 现象 | 解决 |
|---|---|---|
| HuggingFace Xet 协议 | 下载 0% 报 401（cas-server） | `HF_HUB_DISABLE_XET=1`，或改用 ModelScope |
| DeepSeek 工具消息校验 | 422 `missing tool_call_id` | 教师侧观测改用 `role=user` + `[TOOL OUTPUT]` 前缀；落盘仍是 `role=tool` |
| TRL 版本差异 | `SFTConfig` 不认 `warmup_ratio` | 按函数签名过滤未知参数 |
| 镜像无 tmux | `tmux: command not found` | `apt install tmux`，或 `nohup` |
| 分词器不一致 | 评测/训练两套 tokenizer | 统一为 Unicode 分词，实测影响 7/200 题 |
| 观测格式不一致 | eval 带 `" ..."`，训练不带 | 统一为训练环境格式 |

---

## 13. 局限与未来工作

**局限**

1. 语料只覆盖 dev 的 distractor 段落，检索难度被低估；
2. 200 题评测方差大，1–2 点差异不可辨；
3. RL 只做轨迹级奖励，无 step-level 信用分配；
4. 单卡 batch 16，GRPO 的组内估计不稳；
5. 未做 pass@k / 多样性分析。

**如果给 8 卡**

1. batch 提到 256，稳定 GRPO；
2. 换 7B 验证容量假设；
3. 加 step-level 奖励（每轮检索是否命中 gold）；
4. 用 vLLM 异步 rollout，把 40 步压到 10 分钟；
5. 扩到全量 Wikipedia，用稠密检索对比。

---

## 14. 复现指南

```bash
# 0. 环境
python3 -m venv .venv
.venv/bin/pip install torch transformers trl peft datasets accelerate pandas pyarrow rank_bm25

# 1. 数据
.venv/bin/python scripts/build_data.py --parquet data/raw/hotpot_dev.parquet --out data/processed
.venv/bin/python scripts/eval_retrieval.py          # Recall@5 ≈ 0.695

# 2. SFT 数据（需 DEEPSEEK_API_KEY）
.venv/bin/python scripts/gen_sft_deepseek.py --type B --n 2600 --out data/sft/raw_b.jsonl
.venv/bin/python scripts/validate_sft.py --inputs data/sft/raw_*.jsonl

# 3. SFT
.venv/bin/python scripts/train_sft.py --model Qwen/Qwen3-1.7B --out artifacts/sft_1p7b

# 4. 评测
.venv/bin/python scripts/merge_lora.py --base Qwen/Qwen3-1.7B \
  --adapter artifacts/sft_1p7b/final --out artifacts/sft_1p7b/merged
.venv/bin/python scripts/evaluate.py --model artifacts/sft_1p7b/merged \
  --out artifacts/eval_sft_17b_200
```

---

## 附录：实验编号索引

| ID | 内容 | 关键结果 |
|---|---|---|
| E0 | 数据构建 | 66,581 段 / 7,405 题 |
| E1a/b | 检索闸门 | R@5 0.613 → 0.695 |
| E2/E3 | 0.6B 基线 | EM 0.04 |
| E4 | SFT 数据 | 4,200 → 1,937 |
| E5 | 1.7B 基线 | EM 0.140 |
| E6/E7 | SFT | EM 0.405 |
| E8 | GRPO 首次 | reward 恒 −0.025（工具预算截断） |
| E9 | GRPO v3 | EM 0.405 |
| E10 | RFT | EM 0.395 |

详细数据见 `docs/experiments.md`；简历材料见 `docs/resume.md`。
