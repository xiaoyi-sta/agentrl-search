# 全量操作日志 · agentrl-search

> 按时间顺序记录每一次操作：目的、命令、实际输出、读数、判断、花费。
> 与 `experiments.md`（按实验编号）和 `report.md`（按设计维度）互补：这份是**流水账**。
> 数据来源：终端输出截图 + 落盘 JSON。未能确认的数字标注「未记录」，不臆造。
> 时间跨度：2026-10-29 → 2026-11-17

---

## 目录

- [阶段 0 · 选型与预算（10-29）](#阶段-0--选型与预算10-29)
- [阶段 1 · 数据地基（10-29）](#阶段-1--数据地基10-29)
- [阶段 2 · 检索器（10-30）](#阶段-2--检索器10-30)
- [阶段 3 · 协议与工具环境（10-30）](#阶段-3--协议与工具环境10-30)
- [阶段 4 · 基线（10-30 → 10-31）](#阶段-4--基线10-30--10-31)
- [阶段 5 · SFT 数据（10-31）](#阶段-5--sft-数据10-31)
- [阶段 6 · SFT 训练（11-01）](#阶段-6--sft-训练11-01)
- [阶段 7 · GRPO（11-01 → 11-02）](#阶段-7--grpo11-01--11-02)
- [阶段 8 · RFT（11-02）](#阶段-8--rft11-02)
- [阶段 9 · 收尾与展示（11-02 → 11-17）](#阶段-9--收尾与展示11-02--11-17)
- [附录 A · 错误全集](#附录-a--错误全集)
- [附录 B · 指标口径](#附录-b--指标口径)
- [附录 C · 花费台账](#附录-c--花费台账)
- [附录 D · 命令速查](#附录-d--命令速查)

---

## 阶段 0 · 选型与预算（10-29）

### 0.1 环境事实

| 项 | 值 | 来源 |
|---|---|---|
| 本机 | Apple M4 / 10 核 / 16 GB RAM / 258 GB 可用 | `sysctl`、`df -h` |
| 是否有 NVIDIA | 否 | `system_profiler` |
| Python | 3.13.9（miniconda base） | `python3 -V` |
| 网络 | huggingface.co 直连 000；hf-mirror 200；github 200；pypi 200 | `curl -o /dev/null -w %{http_code}` |

**判断**：本机只能做数据与 CPU 评测，训练必须租卡。

### 0.2 候选仓库盘点（GitHub API，2026-10-29 抓取）

| 仓库 | stars | 最近推送 | 定位 |
|---|---|---|---|
| verl-project/verl | 24,049 | 2026-10-29 | 工业级 RL 训练底座 |
| microsoft/agent-lightning | 18,836 | 2026-10-20 | 给 Agent 接 RL |
| Jiayi-Pan/TinyZero | 13,256 | 2026-02-27 | 极简 GRPO |
| rllm-org/rllm | 5,901 | 2026-10-16 | Agent RL 框架 |
| PeterGriffinJin/Search-R1 | 5,549 | 2025-11-13 | 搜索工具 RL |
| hkust-nlp/simpleRL-reason | 3,880 | 2025-12-23 | 推理 RL |
| alibaba/ROLL | 3,442 | 2026-10-29 | 大规模 RL |
| mll-lab-nu/RAGEN | 2,831 | 2026-10-28 | 多轮 StarPO |
| NovaSky-AI/SkyRL | 2,472 | 2026-10-29 | 全栈 RL |
| AgentR1/Agent-R1 | 1,709 | 2026-10-26 | step-level 多轮 agent RL |
| WooooDyy/AgentGym | 853 | 2026-05-30 | Agent 环境集 |
| agentscope-ai/Trinity-RFT | 710 | 2026-10-08 | 单卡 colocate RFT |
| qiancheng0/ToolRL | 527 | 2025-10-16 | 奖励设计 |
| ReTool-RL/ReTool | 398 | 2025-08-12 | 代码工具 RL |
| THUDM/slime | 8,757 | 2026-10-27 | 大规模 RL |

**决策**：放弃 verl/ROLL/slime/SkyRL（面向 8 卡以上），选 `TRL GRPOTrainer + environment_factory`（单卡、LoRA、多轮工具）。

### 0.3 预算测算

| 项 | 单价 | 用量 | 小计 |
|---|---|---|---|
| AutoDL 3090 | ¥1.6/h | ~28 h | ¥45 |
| AutoDL 4090 | ¥2.9/h | 备选 | — |
| DeepSeek Flash（空闲） | ¥1/M 输入、¥4/M 输出 | ~820 万输入 / 56 万输出 | ¥10–15 |

**计划总预算 ¥60–80，上限 ¥100。**

---

## 阶段 1 · 数据地基（10-29）

### 1.1 下载数据

```bash
curl -L --retry 3 -o data/raw/hotpot_dev.parquet \
  "https://hf-mirror.com/datasets/hotpotqa/hotpot_qa/resolve/main/distractor/validation-00000-of-00001.parquet"
ls -lh data/raw/hotpot_dev.parquet
```

**实际返回**：文件 27,452,575 字节（约 26M）。

### 1.2 结构探查

```python
df = pd.read_parquet(...)
print(df.shape)            # (7405, 7)
print(df.columns.tolist()) # ['id','question','answer','type','level','supporting_facts','context']
```

**关键输出**：

```python
type(df["context"].iloc[0])          # dict
list(df["context"].iloc[0].keys())   # ['title', 'sentences']
df["context"].iloc[0]["title"].shape # (10,)
df["context"].iloc[0]["sentences"][0] # array(['Ed Wood is a 1994 ...'], dtype=object)
df["level"].value_counts()           # {'hard': 7405}
```

**踩坑**：`ctx[0]` 报 `KeyError: 0`——以为是 list，实际是**列式 struct**（title 数组 + sentences 二维数组）。
**解决**：写 `normalize.py`，三种形态兼容。

### 1.3 构建语料

```bash
.venv/bin/python scripts/build_data.py \
  --parquet data/raw/hotpot_dev.parquet --out data/processed
```

**实际返回（完整 JSON）**：

```json
{
  "questions_total": 7405,
  "questions_clean": 7405,
  "support_not_covered": 0,
  "corpus_docs": 66581,
  "avg_para_words": 87.2,
  "avg_paras_per_question": 9.95,
  "avg_supporting": 2.43,
  "type_dist": {"comparison": 1487, "bridge": 5918},
  "level_dist": {"hard": 7405},
  "eval_type_dist": {"bridge": 160, "comparison": 40},
  "eval_level_dist": {"hard": 200},
  "splits": {"eval": 200, "rl": 1200, "sft": 2000},
  "corpus_written": 66581
}
```

**读数**：

- 66,581 段语料，平均 87 词——和 HotpotQA 段落的预期一致；
- `support_not_covered = 0`：所有支持事实都能在上下文里找到；
- `level` 全 hard → 已知数据缺陷，分层只按 type；
- eval200 的 bridge/comparison = 160/40，按题型分层成功。

### 1.4 产物

| 文件 | 大小 | 内容 |
|---|---|---|
| `data/processed/corpus.jsonl` | 38 MB | 66,581 段 |
| `data/processed/questions.jsonl` | 1.9 MB | 7,405 题 |
| `data/processed/splits/eval200.jsonl` | 54 KB | 200 题 |
| `data/processed/splits/rl1200.jsonl` | 324 KB | 1,200 题 |
| `data/processed/splits/sft2000.jsonl` | 537 KB | 2,000 题 |

---

## 阶段 2 · 检索器（10-30）

### 2.1 第一版闸门（v1）

配置：`[a-z0-9]+` 分词、文档只用正文、查询不去停用词。

```bash
.venv/bin/python scripts/eval_retrieval.py
```

**实际返回**：

```json
{
  "corpus_docs": 66581,
  "n_questions": 200,
  "k=1": {"k": 1, "n": 200, "recall_at_k": 0.3325, "all_hit_at_k": 0.0},
  "k=3": {"k": 3, "n": 200, "recall_at_k": 0.5175, "all_hit_at_k": 0.205},
  "k=5": {"k": 5, "n": 200, "recall_at_k": 0.6125, "all_hit_at_k": 0.325}
}
```

**未达闸门**（目标 R@5 ≥ 0.65、AllHit@5 ≥ 0.40）。

### 2.2 诊断：三条 MISS 样例

```
MISS: What organization does Rosyth Dockyard and Queen Elizabeth-class aircraft carrier ...
      缺 ['Queen Elizabeth-class aircraft carrier']
MISS: When was the English rock band formed that's first recording remained unreleased
      缺 ['Motörhead']
MISS: The large subunit and small subunit that use two types of RNA ...
      缺 ['Short interspersed nuclear elements (SINEs)']
```

**判断**：不是 bug，是两跳问题的固有难度——第二跳实体名不在问题里。同时暴露一个真问题：`Motörhead` 被切成 `mot` + `rhead`（ö 被当分隔符）。

### 2.3 四个变体实验（各 200 题）

| 变体 | R@1 | R@3 | R@5 | All@3 | All@5 | 耗时 |
|---|---|---|---|---|---|---|
| A 基线 `[a-z0-9]+` | 0.333 | 0.517 | 0.613 | 0.205 | 0.325 | 29s |
| B Unicode 分词 | 0.333 | 0.517 | 0.610 | 0.205 | 0.320 | 29s |
| C Unicode + 标题入索引 | 0.385 | 0.580 | 0.655 | 0.285 | 0.370 | 30s |
| D C + 查询去停用词 | **0.393** | **0.610** | **0.695** | **0.315** | **0.440** | 18s |
| E C + b=0.5 | 0.367 | 0.573 | 0.630 | 0.280 | 0.350 | 30s |

**决策**：采用 D（Unicode + 标题 + 停用词），`b` 保持默认 0.75。

### 2.4 采用后的正式闸门（v2）

```json
{
  "k=3": {"recall_at_k": 0.610, "all_hit_at_k": 0.315},
  "k=5": {"recall_at_k": 0.695, "all_hit_at_k": 0.440}
}
```

**过闸**。

### 2.5 旧 vs 新分词器的差异量化

| 分词器 | R@3 | All@3 | R@5 | All@5 |
|---|---|---|---|---|
| 旧 `[a-z0-9]+` | 0.613 | 0.315 | 0.693 | 0.435 |
| 新 Unicode | 0.610 | 0.315 | 0.695 | 0.440 |

**top-5 发生变化的题：7/200（3.5%）**。

**决策**：虽影响小，但为了训练/评测一致性，正式基线必须用 v2 重跑。

---

## 阶段 3 · 协议与工具环境（10-30）

### 3.1 协议定稿

```text
<tool_call>{"name": "search", "arguments": {"query": "Motörhead band"}}</tool_call>
<answer>Song Il-gon</answer>
```

写入 `src/agentrl/prompts.py`：`SYSTEM_PROMPT` + `TOOL_SCHEMAS`，训练/评测/生成三处共用。

### 3.2 SearchEnv 规格

| 参数 | 值 |
|---|---|
| top_k | 3 |
| 每条观测 | ≤ 60 词 |
| 索引 | 类级单例（避免每次 rollout 重建 30 秒） |
| reset 返回 | `None`（返回字符串会覆盖 dataset 的 prompt） |
| 工具 | `search`、`calculator` |

### 3.3 训练环境与评测工具的观测一致性验证

```python
for q in ["Rosyth Dockyard", "Motörhead band", "Ed Wood director"]:
    a = env.search(q); b = tools.execute("search", {"query": q})
    print(q, a == b)
```

**实际返回**：

```
Rosyth Dockyard | identical: True
Motörhead band  | identical: True
Ed Wood director | identical: True
```

**修复**：统一前评测工具会加 `" ..."` 截断后缀、训练环境不加；统一为训练环境格式。

### 3.4 第一版评测器

`runner.py`：greedy 多轮循环，最多 5 轮，`max_new_tokens=256`；
parser 兼容 `<tool_call>{json}` 与 `<search query="..."/>`（评测宽容，训练严格）。

---

## 阶段 4 · 基线（10-30 → 10-31）

### 4.1 0.6B × 20 题（5 次迭代）

| 轮次 | EM | tag_rate | avg_calls | tool_hit | 说明 |
|---|---|---|---|---|---|
| smoke | 0.00 | 未记录 | 0.00 | 0.00 | parser 未接上 |
| smoke2 | 0.00 | 未记录 | 0.95 | 0.10 | tool_hit 统计错位 |
| smoke3 | 0.05 | 未记录 | 0.95 | 0.95 | 指标修正 |
| smoke4 | 0.05 | 0.10 | 0.95 | 0.95 | 加 tag_rate |
| smoke5 | 0.05 | 0.10 | 0.95 | 0.95 | 加 lenient 口径 |

**smoke5 关键读数**：`em_contains 0.27`、`no_answer 0.80`、`seconds 40.9`（20 题）。

### 4.2 0.6B × 200 题（E3）

```bash
.venv/bin/python scripts/evaluate.py --model Qwen/Qwen3-0.6B --limit 200 \
  --out artifacts/eval_base_06b_200
```

**实际返回（完整）**：

```json
{
  "n": 200,
  "em": 0.04, "f1": 0.0497,
  "em_lenient": 0.04, "f1_lenient": 0.1484,
  "tag_rate": 0.2, "em_contains": 0.27,
  "avg_calls": 0.95,
  "invalid_call_rate": 0.0,
  "strict_call_rate": 0.5707,
  "no_answer_rate": 0.8,
  "stopped_early_rate": 0.8,
  "max_turns_rate": 0.0,
  "tool_hit_rate": 0.84,
  "avg_out_tokens": 45.5,
  "badcase": {"no_answer": 152, "reasoning_error": 24, "premature_answer": 9, "correct": 8, "retrieval_miss": 7},
  "confusion_2x2": {"hit_correct": 8, "hit_wrong": 160, "miss_correct": 0, "miss_wrong": 32},
  "seconds": 423.6
}
```

**读数**：

- `em_contains 0.27` 远高于 EM 0.04 → 27% 的答案其实对，只是没标签；
- `tag_rate 0.20`、`no_answer 0.80` → 协议是主要矛盾；
- `avg_calls 0.95` → 几乎不搜第二次；
- 2×2：检索到但答错 160，未检索到 32 → 检索不是瓶颈。

### 4.3 1.7B × 200 题（E5）

```bash
.venv/bin/python scripts/evaluate.py \
  --model /root/autodl-tmp/models/Qwen3-1.7B-ms \
  --out artifacts/eval_base_17b_200
```

**实际返回**：

```json
{
  "n": 200,
  "em": 0.14, "f1": 0.2253,
  "em_lenient": 0.14, "f1_lenient": 0.247,
  "tag_rate": 0.745, "em_contains": 0.345,
  "avg_calls": 1.62,
  "invalid_call_rate": 0.0,
  "strict_call_rate": 1.0,
  "no_answer_rate": 0.255,
  "stopped_early_rate": 0.255,
  "max_turns_rate": 0.0,
  "tool_hit_rate": 0.92,
  "avg_out_tokens": 70.5,
  "badcase": {"no_answer": 51, "reasoning_error": 111, "correct": 28, "retrieval_miss": 10},
  "confusion_2x2": {"hit_correct": 27, "hit_wrong": 157, "miss_correct": 1, "miss_wrong": 15},
  "seconds": 502.5
}
```

**对照 0.6B**：

| 指标 | 0.6B | 1.7B | Δ |
|---|---|---|---|
| EM | 0.040 | 0.140 | +0.100 |
| tag_rate | 0.200 | 0.745 | +0.545 |
| avg_calls | 0.95 | 1.62 | +0.67 |
| strict_call_rate | 0.571 | 1.000 | +0.429 |
| tool_hit | 0.840 | 0.920 | +0.080 |
| no_answer | 0.800 | 0.255 | −0.545 |

**结论**：1.7B 把协议和第二跳基本学会了，瓶颈转为“检索到了但答不对”（157/168）。

### 4.4 成本

1.7B 评测 502.5s ≈ 8.4 min，4090 按 ¥2.9/h ≈ **¥0.4**。

---

## 阶段 5 · SFT 数据（10-31）

### 5.1 生成脚本的离线验证

```bash
.venv/bin/python scripts/gen_sft_deepseek.py --type B --n 6 --fake --workers 1 --out /tmp/fake_b.jsonl
.venv/bin/python scripts/validate_sft.py --inputs /tmp/fake_b.jsonl
```

**实际返回**：

```
saved 6 | drops: {'ok': 6}
kept=6 train=5 val=1
dropped: {}
avg_calls=3.00
```

**意义**：不花一分钱验证了「生成 → 校验」全链路。

### 5.2 20 条真实试水

```bash
.venv/bin/python scripts/gen_sft_deepseek.py --type B --n 20 --workers 4 \
  --out data/sft/smoke_b.jsonl --budget 2
```

**实际返回**：

```
saved 20 | elapsed 0.3 min
tokens in=41,728 out=1,291 hit=21,248 miss=20,480 errors=0
cost: off-peak ¥0.03 | peak ¥0.05
drops: {'ok': 20}
```

**读数**：

- 缓存命中 21,248 / 41,728 ≈ **51%**，成本降一半；
- 20 条 ¥0.03 → 4,200 条约 **¥6**，与预算测算一致。

### 5.3 校验发现：严格 EM 误杀

```bash
.venv/bin/python scripts/validate_sft.py --inputs data/sft/smoke_b.jsonl
# kept=8 train=7 val=1
# dropped: {'wrong_answer': 10, 'one_hop': 2}
```

逐条检查被丢的 10 条：

| gold | pred | F1 | contains |
|---|---|---|---|
| `25 April` | `25 April 2017` | 0.80 | ✓ |
| `3,000` | `Over 3,000 people` | 0.50 | ✓ |
| `over 30 million` | `Scooter have sold over 30 million records.` | 0.60 | ✓ |
| `Alfredo "Alfred" Molina` | `Alfred Molina` | 0.80 | ✗ |
| `Bengali` | `India` | 0.00 | ✗ |
| `PewDiePie` | `Bill Hader` | 0.00 | ✗ |
| `24800 mi` | `Approximately 700 miles ...` | 0.00 | ✗ |
| `ninth` | `"Lost Our Lisa" (season 9, episode 24)` | 0.00 | ✗ |
| `Dougray Scott` | `Dougray Scott and Jessica De Gouw ...` | 0.29 | ✓ |
| `5.3 million` | `The sources do not give a specific figure ...` | 0.05 | ✗ |

**判断**：5 条是别名/啰嗦但正确，5 条是真错。

### 5.4 放宽过滤器

规则改为：`EM==1` 或 `F1≥0.8` 或「包含 gold 且长度 ≤ max(6, 3×gold 词数)」。

**重跑**：

```
kept=12 train=11 val=1
dropped: {'wrong_answer': 6, 'one_hop': 2}
avg_calls=2.33
```

保留率 40% → **60%**。

### 5.5 线上报错与修复（三次）

**① 422**

```
422 Failed to deserialize ... messages[3]: missing field `tool_call_id`
```

原因：DeepSeek 要求 `role=tool` 必须配结构化 tool_calls。
修法：教师侧观测改 `role=user` + `[TOOL OUTPUT]` 前缀；**落盘仍是 `role=tool`**。

**② 401 与 TypeError**

```
ERR ... auth failed ... Error code: 401 ... api key is invalid
TypeError: cannot unpack non-iterable NoneType object
```

原因 1：401 判断用字符串匹配，request_id 里的 `403e` 导致误判；
原因 2：`work()` 异常分支 `return None`，主循环按二元组解包。
修法：改用 `exc.status_code`；所有分支返回 `(rec, why)`；鉴权失败立即停止。

**③ 无预算保护**

修法：`--budget`（默认 ¥20，按空闲价估算），预检 `max_tokens=1` 验证 key。

### 5.6 全量生成

```bash
bash scripts/gen_all.sh     # A 1200 / B 2600 / C 400
```

**实际返回**：

```
A: saved 1200?（未记录总数），cost ≈ ¥1.5
B: saved 2396 | elapsed 8.3 min
   tokens in=4,762,105 out=210,621 hit=2,516,218 miss=2,245,887
   cost: off-peak ¥3.14 | peak ¥6.28
   drops: {'ok': 2396, 'no_answer_tag': 204}
C: saved 395 | elapsed 12.6 min
   tokens in=442,771 out=20,344 hit=173,567 miss=269,204
   cost: off-peak ¥0.35 | peak ¥0.71
   drops: {'ok': 395, 'no_answer_tag': 5}
```

**校验汇总**：

```
kept=2139 train=2033 val=106
dropped: {'wrong_answer': 1149, 'one_hop': 250}
avg_calls=2.00
```

### 5.7 二次清理

发现 C 类占比 17%（372/2139），可能让模型过度拒答；且有一条 0 调用样本。
修法：C 类要求 ≥1 次调用，并封顶到总数据的 10%。

**重跑后**：

```
kept=1937 train=1841 val=96
dropped: {'wrong_answer': 1149, 'one_hop': 250, 'c_no_search': 1, 'c_downsized': 201}
avg_calls=2.06
train type: {'B': 1367, 'A': 287, 'C': 187}
```

### 5.8 数据画像

| 指标 | 值 |
|---|---|
| 总量 | 1,937（train 1,841 / val 96） |
| 类型 | B 74% / A 16% / C 10% |
| 平均调用 | 2.06 |
| token 长度 | p50 849 / p95 1,267 / max 2,147 |
| 超 3,072 | 0 |
| 重复 qid | 0 |
| match 分布 | exact 1,200 / contains 299 / f1 162（C 类无 gold） |

### 5.9 花费

A ≈ ¥1.5 + B ¥3.14 + C ¥0.35 = **≈ ¥5.5**。

---

## 阶段 6 · SFT 训练（11-01）

### 6.1 环境

| 项 | 值 |
|---|---|
| GPU | RTX 4090 24G（AutoDL） |
| torch | 2.15.0 + cu132 |
| transformers | 5.21.x |
| Python | 3.12.3 |
| tmux | 需 `apt install`（镜像缺失） |

### 6.2 第一次报错：`warmup_ratio`

```
TypeError: SFTConfig.__init__() got an unexpected keyword argument 'warmup_ratio'
```

原因：transformers v5 移除该参数。
修法：按 `inspect.signature(SFTConfig.__init__)` 过滤未知参数，并打印被丢弃项。

### 6.3 训练输出

```
train: 1841  val: 96
Dropping fully masked examples: 1841/1841（扫描，非全丢）
assistant_only_loss = True
preflight: 76/777 tokens carry loss
{'loss': 1.4xx} → 0.27–0.29 平台
462/462 [28:13, 3.67s/it]
{'eval_loss': 0.2575, 'eval_entropy': 0.2101, 'eval_mean_token_accuracy': 0.9334}
{'train_runtime': 1694, 'train_loss': 0.289}
saved to artifacts/sft_1p7b/final
```

### 6.4 训练侧读数

| 指标 | 值 | 判断 |
|---|---|---|
| 步数 | 462 | 1,841 × 2 ÷ 8 ≈ 460 |
| 单步 | 3.67s | 可接受 |
| train_loss | 0.289 | 从 1.4 收敛 |
| eval_loss | 0.2575 | 低于 train，无过拟合 |
| token accuracy | 0.9334 | 格式 token 基本学会 |
| entropy | 0.2101 | 未塌缩 |

### 6.5 合并与评测

```bash
merge_lora.py --base Qwen/Qwen3-1.7B --adapter artifacts/sft_1p7b/final --out .../merged
evaluate.py --model .../merged --out artifacts/eval_sft_17b_200
```

**实际返回**：

```json
{
  "n": 200,
  "em": 0.405, "f1": 0.5312,
  "em_lenient": 0.405, "f1_lenient": 0.5312,
  "tag_rate": 0.96, "em_contains": 0.52,
  "avg_calls": 2.21,
  "invalid_call_rate": 0.0181,
  "strict_call_rate": 0.9977,
  "no_answer_rate": 0.04,
  "stopped_early_rate": 0.0,
  "max_turns_rate": 0.04,
  "tool_hit_rate": 0.955,
  "avg_out_tokens": 75.8,
  "badcase": {"reasoning_error": 106, "correct": 81, "max_turns_hit": 8, "retrieval_miss": 5},
  "confusion_2x2": {"hit_correct": 80, "hit_wrong": 111, "miss_correct": 1, "miss_wrong": 8},
  "seconds": 537.5
}
```

### 6.6 与基线对比

| 指标 | 基线 | SFT | Δ |
|---|---|---|---|
| EM | 0.140 | **0.405** | **+0.265** |
| F1 | 0.225 | 0.531 | +0.306 |
| tag_rate | 0.745 | 0.960 | +0.215 |
| avg_calls | 1.62 | 2.21 | +0.59 |
| no_answer | 0.255 | 0.040 | −0.215 |

### 6.7 泄漏检查

```python
train ∩ eval200 = 0
val   ∩ eval200 = 0
train ∩ val    = 0
```

### 6.8 新问题

`max_turns_hit 8`（4%）——学会了“多搜”，没学会“何时停”。

### 6.9 成本

28 min × ¥2.9/h ≈ **¥1.2**；评测 537.5s ≈ ¥0.4。

---

## 阶段 7 · GRPO（11-01 → 11-02）

### 7.1 第一次尝试（E8）：reward 恒为 −0.025

```bash
.venv/bin/python scripts/train_grpo.py --model artifacts/sft_1p7b/merged \
  --steps 20 --limit 200 --no-vllm --out artifacts/grpo_smoke
```

**实际返回**：

```
{'loss': '0', 'grad_norm': '0', 'learning_rate': '1e-06',
 'completions/mean_length': '23.25', 'completions/clipped_ratio': '0',
 'tools/call_frequency': '1', 'tools/failure_frequency': '0',
 'rewards/answer_reward/mean': '0', 'rewards/format_reward/mean': '0',
 'rewards/budget_with_lam/mean': '-0.025', 'reward': '-0.025'}
```

### 7.2 定位：打印 sample

```
[assistant tool_calls=[{'function': {'name': 'search', ...}}] content='']
[tool name=search] [1] Scout Tufankjian: ...
[assistant content='<tool_call>{']
```

**根因**（读 TRL 源码，`grpo_trainer.py` 约 2019 行与 2040 行）：

1. TRL 把工具调用解析成结构化 `tool_calls`、工具结果以 `{"role":"tool","name":...}` 注入；
2. **工具返回内容计入 `max_completion_length`**，默认 256 装不下 3 段 × 60 词（≈240 token），
   TRL 直接丢弃工具结果并结束该 sample。

于是模型没看到资料、也没机会作答 → `answer_reward=0`。

### 7.3 修复一：`max_completion_length=1024`

**sanity 20 步输出**（reward 序列）：

```
0.407 0.327 0.216 0.491 0.228 0.207 0.394 0.120 0.168 0.136
0.072 0.381 0.077 0.484 0.543 0.397 0.229 0.278 0.352 0.655
```

| 指标 | 值 |
|---|---|
| answer_reward | 0.03–0.60 |
| format_reward | 0.06–0.10 |
| calls | 1.5–2.6 |
| 耗时 | 97s / 20 步 |

**机制跑通**，但趋势不明。

### 7.4 第二次尝试：50 步（lr 5e-6、λ 0.1）

```bash
--steps 120 → 实际跑到 60 步停止
--lr 5e-6 --beta 0.02 --temperature 0.8 --lam 0.1
```

**实际返回**：

```
前 10 均值 0.413 → 后 10 均值 0.372（下降）
最近 5: 0.0677 0.0469 0.4077 0.5405 0.126
```

评测 `checkpoint-50`：

| 指标 | SFT | GRPO-50 |
|---|---|---|
| EM | 0.405 | 0.385 |
| F1 | 0.531 | 0.509 |
| em_contains | 0.520 | 0.490 |
| tag_rate | 0.960 | 0.965 |
| avg_calls | 2.21 | 2.03 |

**归因**：lr 太大 + λ=0.1 让“少搜”比“答对”更划算（calls 降 0.18，EM 掉 2 点）。

### 7.5 第三次尝试：lr 1e-6、λ 0.02、temp 0.8（32 步）

**实际返回**：

```
reward 前10 0.44 → 后10 0.294
answer 前5 [0.34 0.40 0.45 0.08 0.11] → 后5 [0.34 0.68 0.13 0.34 0.23]
format 稳定 0.09–0.10
calls  1.88 → 2.88（反而变多）
```

**判断**：不是崩，是没学到东西；answer 噪声大、calls 上升。

### 7.6 第四次尝试（E9）：三个稳定性开关

```bash
.venv/bin/python scripts/train_grpo.py \
  --model artifacts/sft_1p7b/merged \
  --steps 40 --limit 400 --no-vllm \
  --max-completion 2048 --max-turns 3 \
  --lr 1e-6 --beta 0.02 --temperature 0.8 --lam 0.02 \
  --mask-truncated --loss-type dr_grpo --entropy-coef 0.001 \
  --out artifacts/grpo_v3
```

**16 步检查**：

```
n=16  前10 0.451  后10 0.422
format 后5 [0.09 0.10 0.09 0.09 0.09]
clipped 后5 [0.00 0.00 0.00 0.00 0.00]
```

**40 步最终**：

```
步数=40
前8均值=0.416  后8均值=0.462
reward 曲线 [0.43 0.09 0.69 0.54 0.08 0.53 0.54 0.43 0.62 0.20]
format 均值=0.087  calls 均值=2.14
checkpoints: checkpoint-30, checkpoint-40
```

**评测对比**：

| 指标 | SFT | GRPO-30 | GRPO-40 |
|---|---|---|---|
| EM | 0.405 | 0.395 | 0.405 |
| F1 | 0.531 | 0.526 | **0.536** |
| em_contains | 0.520 | 0.520 | 0.525 |
| tag_rate | 0.960 | 0.970 | 0.965 |
| avg_calls | 2.21 | 2.19 | 2.19 |
| tool_hit | 0.955 | 0.955 | 0.955 |
| no_answer | 0.040 | 0.030 | 0.035 |

**结论**：持平；F1 +0.005，检索次数 −0.02。

### 7.7 成本

E8 约 ¥0.3 + 50 步约 ¥1.2 + 32 步约 ¥0.8 + 40 步约 ¥2 + 两次评测 ¥1 ≈ **¥5.3**。

---

## 阶段 8 · RFT（11-02）

### 8.1 采样

```bash
.venv/bin/python scripts/rft_generate.py \
  --model /root/autodl-tmp/agentrl-search/artifacts/sft_1p7b/merged \
  --out data/rft/raw.jsonl --k 3 --limit 500
```

500 题 × 3 条 = 1,500 条轨迹（HF generate，无 vLLM）。

### 8.2 过滤

```
kept ≈ 592（train 部分）
```

### 8.3 混合

```
mixed: 1184 | rft: 592 | sft: 592
```

随后改为全量 SFT 混合：

```
mixed: 2433 | rft: 592 | sft: 1841
```

### 8.4 训练

```bash
train_sft.py --model .../sft_1p7b/merged \
  --train data/rft/train_mixed.jsonl --val data/rft/val.jsonl \
  --out artifacts/rft_1p7b --epochs 1 --lr 5e-5
```

**实际返回**：

```
305/305 [14:56, 2.94s/it]
{'eval_loss': 0.1251, 'eval_entropy': 0.1936, 'eval_mean_token_accuracy': 0.9574}
{'loss': 0.1388, 'grad_norm': 3.414, 'learning_rate': 1.326e-09}
{'train_runtime': 896.2, 'train_loss': 0.1737}
saved to artifacts/rft_1p7b/final
```

### 8.5 评测

```json
{"em": 0.395, "f1": 0.525, "tag_rate": 0.980, "avg_calls": 2.19}
```

### 8.6 对比与结论

| 指标 | SFT | RFT |
|---|---|---|
| EM | 0.405 | 0.395 |
| F1 | 0.531 | 0.525 |
| tag_rate | 0.960 | **0.980** |
| avg_calls | 2.21 | 2.19 |

**1 个点 = 200 题里的 2 道题**，在 ±7 点 CI 内，统计上不可区分。
原因：592 条来自约 200 道题（同题多分支），有效多样性低。

### 8.7 成本

采样约 ¥1.5 + 训练 15 min ≈ ¥0.7 + 评测 ¥0.5 ≈ **¥2.7**。

---

## 阶段 9 · 收尾与展示（11-02 → 11-17）

### 9.1 结果拉回本地

```bash
rsync -avz -e "ssh -p 30250" \
  root@connect.bjb3.seetacloud.com:/root/autodl-tmp/agentrl-search/artifacts/eval_sft_17b_200 \
  ~/agentrl-search/artifacts/
```

拉回清单：`eval_base_17b_200`、`eval_sft_17b_200`、`eval_grpo_v3`、`eval_rft_17b_200`、
`sft_1p7b/final`、`rft_1p7b/final`。

### 9.2 artifacts 清理

| 操作 | 前 | 后 |
|---|---|---|
| 删 `smoke_grpo`（6.7G 全量 checkpoint） | 7.4G | 717M |
| 删 `*/merged`（3.4G/个） | — | — |

### 9.3 文档产出

| 文件 | 行数 | 内容 |
|---|---|---|
| `README.md` | 197 | 展示层：结果图、Quickstart、设计决策 |
| `docs/report.md` | 502 | 设计层：14 章 |
| `docs/experiments.md` | 307 | 证据层：E0–E10 |
| `docs/process.md` | 445 | 过程层：按时间复盘 |
| `docs/resume.md` | 145 | 面试层：bullet + 15 Q&A |
| `docs/full-log.md` | 本文件 | 流水层：每一步的命令与输出 |

### 9.4 Git 与 GitHub

第一次 `git count-objects -vH`：

```
size: 4.01 MiB
size-garbage: 331.75 MiB   ← .git/objects/pack/tmp_pack_jl64MV
```

删除临时 pack 后 `.git` 从 **336M → 4.1M**。

推送结果：

```
Enumerating objects: 96, done.
Writing objects: 100% (96/96), 71.20 KiB
* [new branch] main -> main
```

仓库：`https://github.com/xiaoyi-sta/agentrl-search`，37 个文件、236 KB。

### 9.5 已知数据缺陷复核

```json
{"全量 level": {"hard": 7405}, "eval level": {"hard": 200}}
```

结论：上游 parquet 的 level 列损坏；本项目不使用 level 做任何分析。

---

## 附录 A · 错误全集

| # | 现象 | 根因 | 修法 | 耗时 |
|---|---|---|---|---|
| 1 | `KeyError: 0`（读 context） | 列式 struct 被当 list | 归一化层兼容三种 schema | 20 min |
| 2 | `ModuleNotFoundError: numpy`（安装） | pip 写入 `~/.local` 被沙箱拒绝 | 改用项目 venv | 10 min |
| 3 | `dquote>` 卡住 | 粘贴时中文引号 | 重打命令 | 2 min |
| 4 | `Permission denied`（ssh） | 密码手误 | 重输 | 2 min |
| 5 | `nvidia-smi: command not found` | 在 Mac 上跑 GPU 命令 | 区分实例/Mac | 5 min |
| 6 | `rsync: Could not resolve hostname :` | 域名与冒号之间多了空格 | 修正命令 | 3 min |
| 7 | `--info=progress2` 不识别 | macOS rsync 2.6.9 | 去掉参数 | 2 min |
| 8 | `HFValidationError: Repo id must be ...` | 本地路径不存在，被当 repo id | 先合并再评测 | 5 min |
| 9 | Xet 401 `cas-server.xethub.hf.co` | 镜像站不代理 Xet CAS | `HF_HUB_DISABLE_XET=1` / ModelScope | 25 min |
| 10 | `422 missing tool_call_id` | DeepSeek 校验 tool 消息 | 教师侧改 `user` 角色 | 15 min |
| 11 | `TypeError: cannot unpack NoneType` | `work()` 异常分支返回 None | 统一返回二元组 | 10 min |
| 12 | `warmup_ratio` 不认 | transformers v5 移除 | 按签名过滤参数 | 10 min |
| 13 | `tmux: command not found` | 镜像未安装 | `apt install tmux` | 5 min |
| 14 | GRPO reward 恒 −0.025 | 工具结果超 completion 预算被丢 | `max_completion=2048` | 2 h |
| 15 | `clipped_ratio 0.375` | 截断样本污染梯度 | `mask_truncated_completions` | 40 min |
| 16 | vLLM 安装卡住 | 下载/编译慢 | 放弃，用 HF generate | 30 min |

---

## 附录 B · 指标口径

| 指标 | 定义 | 用途 |
|---|---|---|
| EM | 归一化（小写、去标点、去冠词）后完全相等 | 主指标 |
| F1 | 词级重叠的调和平均 | 部分正确 |
| em_contains | 预测文本是否包含 gold 短语 | 诊断“会答但格式错” |
| tag_rate | 输出含 `<answer>` 的比例 | 协议合规 |
| avg_calls | 平均工具调用次数 | 行为 |
| strict_call_rate | 用规范 `<tool_call>{json}` 的比例 | 协议一致性 |
| tool_hit_rate | 检索结果含任一 gold 标题的题占比 | 检索质量 |
| no_answer_rate | 无 `<answer>` 的比例 | 失败模式 |
| max_turns_rate | 跑满 5 轮的比例 | 过度检索 |
| 2×2 | 检索 × 正确 的四格 | 分离检索/推理瓶颈 |
| CI | 二项分布 95% 区间，n=200 时 ±7 点 | 显著性判断 |

---

## 附录 C · 花费台账

| 项 | 明细 | 金额 |
|---|---|---|
| SFT 数据 | A ¥1.5 + B ¥3.14 + C ¥0.35 | ¥5.0 |
| 基线 GPU | 4090 约 40 min | ¥1.4 |
| SFT GPU | 28 min + 评测 9 min | ¥1.8 |
| GRPO GPU | 四次尝试 + 三次评测 | ¥5.3 |
| RFT GPU | 采样 + 训练 + 评测 | ¥2.7 |
| 环境/调试 | 多实例开机、安装 | ¥13（估） |
| **合计** |  | **≈ ¥30** |

> GPU 单价按 4090 ¥2.9/h、3090 ¥1.6/h 估算；以控制台账单为准。

---

## 附录 D · 命令速查

```bash
# 数据
python scripts/build_data.py --parquet data/raw/hotpot_dev.parquet --out data/processed
python scripts/eval_retrieval.py

# SFT 数据
python scripts/gen_sft_deepseek.py --type B --n 2600 --out data/sft/raw_b.jsonl --budget 8
python scripts/validate_sft.py --inputs data/sft/raw_a.jsonl data/sft/raw_b.jsonl data/sft/raw_c.jsonl

# SFT
python scripts/train_sft.py --model Qwen/Qwen3-1.7B --out artifacts/sft_1p7b
python scripts/merge_lora.py --base Qwen/Qwen3-1.7B --adapter artifacts/sft_1p7b/final --out artifacts/sft_1p7b/merged

# 评测
python scripts/evaluate.py --model artifacts/sft_1p7b/merged --out artifacts/eval_sft

# RFT
python scripts/rft_generate.py --model artifacts/sft_1p7b/merged --k 3 --limit 500
python scripts/train_sft.py --model artifacts/sft_1p7b/merged --epochs 1 --lr 5e-5 --train data/rft/train_mixed.jsonl

# GRPO
python scripts/train_grpo.py --model artifacts/sft_1p7b/merged --steps 40 --no-vllm \
  --max-completion 2048 --max-turns 3 --lr 1e-6 --lam 0.02 \
  --mask-truncated --loss-type dr_grpo --entropy-coef 0.001
```
