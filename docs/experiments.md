# 实验记录 · agentrl-search

> 多跳检索 Agent 的任务级后训练：数据 → SFT → GRPO → 评测。
> 目标模型 Qwen3-1.7B（LoRA），预算 ≤ ¥100，单卡 24G。
> 最后更新：2026-10-31

---

## 0. 术语与固定条件

| 项 | 值 |
|---|---|
| 语料 | HotpotQA distractor dev 段落去重，66,581 段，平均 87 词 |
| 数据集切分 | eval200 / rl1200 / sft2000，qid 两两不交（seed 42） |
| 检索器 | BM25Okapi，文档 = `title + " . " + text`，查询去停用词 |
| 分词器 | `[^\W_]+`（Unicode），小写；v2 起与训练环境统一 |
| 工具 | `search`（top-3，每条 ≤60 词）、`calculator` |
| 评测方式 | greedy（`do_sample=False`），最多 5 轮，`max_new_tokens=256` |

---

## 1. 实验一览

| ID | 日期 | 内容 | 关键结果 | 结论 |
|---|---|---|---|---|
| E0 | 10-29 | 数据构建 | corpus 66,581 / questions 7,405 / 支持事实缺失 0 | 数据可用 |
| E1a | 10-29 | 检索闸门 v1 | R@3 0.518 / All@3 0.205 / R@5 0.613 / All@5 0.325 | 未达标，归因：分词 + 标题未入索引 + 未去停用词 |
| E1b | 10-30 | 检索闸门 v2 | R@3 0.610 / All@3 0.315 / **R@5 0.695 / All@5 0.440** | 过闸（≥0.65 / ≥0.40） |
| E2 | 10-30 | 0.6B × 20 题（5 次迭代） | EM 0.05 / tag_rate 0.10 / tool_hit 0.95 | 管道跑通；瓶颈在输出协议 |
| E3 | 10-31 | **0.6B × 200 题基线** | EM 0.04 / em_contains 0.27 / tag_rate 0.20 / tool_hit 0.84 | 协议遵循与单跳惯性是主要矛盾 |

产物路径：`artifacts/retrieval_report*.json`、`artifacts/eval_smoke5/`、`artifacts/eval_base_06b_200/`。

---

## 2. E3 详细（0.6B × 200，正式小模型参照）

配置：Qwen3-0.6B，M4 本地 MPS，fp32，greedy，5 轮上限，200 题，耗时 423.6s。

| 指标 | 值 | 说明 |
|---|---|---|
| EM / F1 | 0.040 / 0.050 | 严格匹配 |
| EM(contains) | 0.270 | 27% 的输出**包含**正确答案，但没被解析 |
| tag_rate | 0.200 | 只有 20% 输出 `<answer>` |
| no_answer_rate | 0.800 | 八成题目无答案标签 |
| avg_calls | 0.95 | 几乎不搜第二次 |
| strict_call_rate | 0.571 | 57% 用规范 JSON，43% 自造 XML |
| tool_hit_rate | 0.840 | 任意一条 gold 被召回 |
| invalid_call_rate | 0.000 | parser 容错生效 |

Bad case 分布：

| 类别 | 数量 |
|---|---|
| no_answer | 152 |
| reasoning_error | 24 |
| premature_answer | 9 |
| retrieval_miss | 7 |
| correct | 8 |

2×2（检索 × 正确）：

| | 答对 | 答错 |
|---|---|---|
| 检索到 gold | 8 | **160** |
| 未检索到 | 0 | 32 |

**一句话结论**：0.6B 能搜到（84%），但不会按协议交付（tag_rate 0.20）、也不会做第二跳（avg_calls 0.95）。EM 4% 不代表“不知道”，27% 的输出其实含正确答案。

---

## 3. 关键发现

1. **检索不是瓶颈**：tool_hit 84%，2×2 中 hit_wrong 160 vs miss_wrong 32。
2. **协议是最大瓶颈**：tag_rate 0.20，strict_call_rate 0.57；两种工具调用语法并存。
3. **两跳缺失是结构性的**：avg_calls 0.95，而 eval200 中 160/200 是 bridge 题。
4. **两跳检索的天然上限**：仅用问题做单次检索，AllHit@5 只有 0.44——第二跳实体名不在问题里，必须靠多轮改写查询才能拿到。这既是检索的局限，也是 agent 存在的理由。

---

## 4. 变更日志

### 2026-10-31 · 统一分词器与观测格式（影响基线可比性）

- `bm25.py`：`[a-z0-9]+` → `[^\W_]+`（Unicode），修复 `Motörhead` 被切成 `mot`+`rhead`。
- `runner.Tools`：截断不再追加 `" ..."`，与训练环境 `SearchEnv` 的观测逐字一致。
- 实测影响：eval200 中 top-5 仅 **7/200** 题变化；R@5 0.6925 → 0.6950。
- **后果**：E1a/E1b 与 E2/E3 产自不同检索器版本。E1a/E1b 为闸门记录，保留；正式基线（1.7B）必须用 v2 代码重跑，避免“检索器变了”污染 SFT 前后对比。

### 2026-10-31 · 生成脚本离线化验证

- `scripts/gen_sft_deepseek.py` 增加 `--fake`。离线跑通 生成→校验 全链路：6/6 生成、6/6 通过、avg_calls 3.00，零 API 费用。
- 修复：A/B 类 qid 重叠（A 改为 comparison-only）、成本统计缺字段时归零、401 无限重试、无预算上限、丢弃原因不可见。

---

## 5. 已知数据问题

1. **`level` 全为 `hard`**（7405/7405）——该 parquet 的 level 列已损坏，分层只按 `type` 做，报告中不得用 level 分析。
2. **语料仅覆盖 dev 的 distractor 段落**，不是全量 Wikipedia——检索难度被低估，结论按相对比较解读。
3. **评测样本量小**：200 题的 95% 置信区间约 ±7 个点；20 题约 ±15 个点。任何 <3 个点的差异不可宣称显著。

---

## 6. 待办

- [ ] 1.7B × 200 正式基线（GPU，约 15 min，¥0.4）
- [ ] 用 DeepSeek 生成 A/B/C 三类 SFT 数据（预算 ¥20 上限）
- [ ] `validate_sft.py` 校验，保留率目标 ≥50%
- [ ] LoRA SFT（r=32，2 epoch，max_len 3072）
- [ ] SFT 评测：tag_rate >0.9、second_call_rate >0.5、EM +8 点
- [ ] GRPO（0.6B 验证 → 1.7B 主实验）

---

## 7. 复现命令

```bash
# 数据
.venv/bin/python scripts/build_data.py --parquet data/raw/hotpot_dev.parquet --out data/processed

# 检索闸门
.venv/bin/python scripts/eval_retrieval.py --out artifacts/retrieval_report.json

# 基线评测
.venv/bin/python scripts/evaluate.py --model Qwen/Qwen3-1.7B --out artifacts/eval_base_17b_200

# SFT 数据（先 20 条试水）
.venv/bin/python scripts/gen_sft_deepseek.py --type B --n 20 --workers 4 --out data/sft/smoke_b.jsonl --budget 2
.venv/bin/python scripts/validate_sft.py --inputs data/sft/raw_a.jsonl data/sft/raw_b.jsonl data/sft/raw_c.jsonl
```
