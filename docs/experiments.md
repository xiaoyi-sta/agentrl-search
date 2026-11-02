# 实验记录 · agentrl-search

> 多跳检索 Agent 的任务级后训练：数据 → SFT → GRPO → 评测。
> 目标模型 Qwen3-1.7B（LoRA），预算 ≤ ¥100，单卡 24G。
> 完整叙述版见 [report.md](./report.md) ｜ 全过程流水见 [journal.md](./journal.md)
> 最后更新：2026-11-01

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
| E4 | 10-31 | SFT 数据生成（DeepSeek V4.1 Flash） | 生成 4200 → 保留 1937（B 74% / A 16% / C 10%），avg_calls 2.06，¥5.5 | 数据可用，进 SFT |
| E5 | 10-31 | **1.7B × 200 题基线**（4090） | EM 0.140 / F1 0.225 / tag_rate 0.745 / avg_calls 1.62 / tool_hit 0.920 | 协议基本学会，瓶颈转为“检索到了答不对” |
| E6 | 11-01 | **LoRA SFT（1841 条，2 epoch）** | train_loss 0.289 / eval_loss 0.258 / 28 min / ¥1.2 | 收敛，无过拟合 |
| E7 | 11-01 | SFT 模型 × 200 题 | **EM 0.405** / F1 0.531 / tag_rate 0.960 / avg_calls 2.21 / tool_hit 0.955 | 相对基线 **+26.5 点**，链路成立 |
| E10 | 11-02 | **RFT（拒绝采样 + 再 SFT）** | EM 0.395 / F1 0.525 / tag 0.980 / calls 2.19 | 与 SFT 持平（−1 点，噪声内），格式更好 |
| E9 | 11-02 | **GRPO v3（Dr.GRPO + truncation mask）** | 40 步；EM 0.405 / F1 0.536 / tag 0.965 / calls 2.19 | 与 SFT 持平，F1 +0.005 |
| E8 | 11-01 | GRPO（TRL `environment_factory`） | reward 恒为 −0.025，`answer_reward=0`，第二轮输出截断为 `<tool_call>{` | 失败：TRL 结构化 tool_calls 与文本协议不兼容 |

产物路径：`artifacts/retrieval_report*.json`、`artifacts/eval_smoke5/`、`artifacts/eval_base_06b_200/`、`data/sft/`。

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

## 2b. E4 详细（SFT 数据生成，2026-10-31）

生成模型：DeepSeek V4.1 Flash（`deepseek-flash`，思考模式关闭），温度 0.8，工具调用由本地 `SearchEnv` 真实执行。

| 阶段 | 生成量 | 通过 | 通过率 | 花费 |
|---|---|---|---|---|
| A 格式（comparison） | 1200 | 287 | 24% | ≈ ¥1.5 |
| B 两跳（bridge，50% oracle 首跳） | 2600 | 1367 | 53% | ¥3.14 |
| C 弃答（假实体） | 400 | 187（压到 10%） | 47% | ¥0.35 |
| **合计** | 4200 | **1937** | 46% | **≈ ¥5.5** |

最终切分：train 1841 / val 96。

| 指标 | 值 |
|---|---|
| 类型配比 | B 74% / A 16% / C 10% |
| 平均调用轮数 | 2.06 |
| token 长度 | p50 849 / p95 1267 / max 2147（≤3072，无截断） |
| 丢弃原因 | `wrong_answer` 1149、`one_hop` 250、`c_no_search` 1、`c_downsized` 201 |

**校验规则**（`validate_sft.py`）：严格 EM 之外，允许 `F1 ≥ 0.8`（别名/标点差异）和“包含 gold 且长度 ≤ max(6, 3×gold 词数)”（防止啰嗦答案被误杀）。这条规则把 smoke 的保留率从 40% 提到 60%。

**已知偏弱项**：A 类通过率仅 24%，占比 16% 偏低。若 SFT 后 `tag_rate < 0.9`，需补生成 A 类（约 ¥1.5）。

---

## 2c. E5 详细（1.7B × 200 基线，2026-10-31）

配置：Qwen3-1.7B（ModelScope 本地权重），RTX 4090 24G，bf16，greedy，5 轮上限，200 题，502.5s（≈8.4 min，¥0.6）。

| 指标 | 1.7B | 0.6B（E3） | Δ |
|---|---|---|---|
| EM | 0.140 | 0.040 | +0.100 |
| F1 | 0.225 | 0.050 | +0.175 |
| EM(contains) | 0.345 | 0.270 | +0.075 |
| tag_rate | **0.745** | 0.200 | +0.545 |
| avg_calls | **1.62** | 0.95 | +0.67 |
| strict_call_rate | **1.000** | 0.571 | +0.429 |
| tool_hit_rate | 0.920 | 0.840 | +0.080 |
| no_answer_rate | 0.255 | 0.800 | −0.545 |

Bad case：`reasoning_error` 111、`no_answer` 51、`correct` 28、`retrieval_miss` 10。

2×2：

| | 答对 | 答错 |
|---|---|---|
| 检索到 gold | 27 | **157** |
| 未检索到 | 1 | 15 |

**结论**：1.7B 把“协议”和“第二跳”基本学会了（strict_call 1.00、avg_calls 1.62），瓶颈完全转到**“检索到了但答不对”**（157/168）。同时 `em_contains 0.345` 远高于 EM 0.140，说明大量答案语义正确但形式不同（如 `Alan Turing` vs `Alan Mathison Turing`、`Royal Navy` vs `Navy`），**严格 EM 低估了真实能力**，报告中需同时给出 lenient 口径。

---

## 2d. E6/E7 详细（LoRA SFT + 评测，2026-11-01）

训练：Qwen3-1.7B + LoRA（r=32，α=64，7 个投影层），1841 条，2 epoch，max_len 3072，batch 1 × accum 8，lr 1e-4 cosine，bf16 + 梯度检查点。462 步 / 28 分钟 / ¥1.2。

| 训练指标 | 值 |
|---|---|
| train_loss | 0.289 |
| eval_loss | **0.2575**（低于 train，无过拟合） |
| eval_mean_token_accuracy | 0.9334 |
| eval_entropy | 0.2101（未塌缩） |

评测（合并 LoRA 后，eval200，537.5s）：

| 指标 | 基线 E5 | SFT | Δ |
|---|---|---|---|
| **EM** | 0.140 | **0.405** | **+0.265** |
| F1 | 0.225 | 0.531 | +0.306 |
| EM(contains) | 0.345 | 0.520 | +0.175 |
| tag_rate | 0.745 | **0.960** | +0.215 |
| avg_calls | 1.62 | 2.21 | +0.59 |
| tool_hit_rate | 0.920 | 0.955 | +0.035 |
| no_answer_rate | 0.255 | 0.040 | −0.215 |
| invalid_call_rate | 0.000 | 0.018 | +0.018 |

2×2：hit_correct 80 / hit_wrong 111 / miss_correct 1 / miss_wrong 8。Bad case：`reasoning_error` 106、`correct` 81、`max_turns_hit` 8、`retrieval_miss` 5。

**结论**：
1. 相对基线 **+26.5 点 EM**，远超 +8 点的验收线；`tag_rate 0.96`、`no_answer 0.04` 说明协议已学会。
2. 剩余 111 个 hit_wrong 中，有相当比例是**答案形式差异**（`2 acres` vs `2 acre`、`the Royal Navy` vs `Navy`），严格 EM 仍偏保守（`em_contains 0.52`）。
3. **新问题：4% 的题跑满 5 轮**（max_turns_hit 8）——SFT 教会了“多搜”，但没教“何时停”。这正好是 P6 奖励里加工具预算惩罚的动机。
4. 无数据泄漏：train∩eval200 = 0，val∩eval200 = 0。

### E8：GRPO 首次尝试（失败，2026-11-01）

用 TRL `GRPOTrainer + environment_factory` 跑 20 步，reward 恒为 **−0.025**（只有工具预算惩罚）：

```
rewards/answer_reward/mean = 0
rewards/format_reward/mean = 0
completions/mean_length = 23
tools/call_frequency = 1
```

打印采样发现根因：

```
[assistant tool_calls=[search(...)] content='']   ← 第一轮正常
[tool name=search] [1] Scout Tufankjian: ...      ← TRL 注入的工具结果
[assistant content='<tool_call>{']                ← 第二轮直接崩溃，JSON 未完成
```

TRL 把工具调用解析成**结构化 `tool_calls`**、工具结果以 `{"role":"tool","name":...}` 注入；而 SFT 数据是**纯文本协议**（`<tool_call>{json}</tool_call>` + 无 name 的 tool 消息）。模型面对陌生格式退化成半截 JSON 并 EOS，奖励永远拿不到。

**结论**：不是 GRPO 无效，是框架的 tool loop 与自定义文本协议不兼容。下一步改为自写 GRPO loop（复用 `Runner` 的文本协议多轮循环 + PEFT 训练），绕开 TRL 的工具抽象。


---

## 2e. E9 详细（GRPO v3，2026-11-02）

配置：merged SFT 为基座 + 新 LoRA（r=32），400 题 × group 4，40 步，lr 1e-6，β=0.02，λ=0.02，
temperature 0.8，`max_completion_length=2048`，`max_tool_calling_iterations=3`，
`mask_truncated_completions=True`，`loss_type="dr_grpo"`，`entropy_coef=0.001`。

| 指标 | SFT | GRPO-30 | GRPO-40 |
|---|---|---|---|
| EM | 0.405 | 0.395 | **0.405** |
| F1 | 0.531 | 0.526 | **0.536** |
| EM(contains) | 0.520 | 0.520 | 0.525 |
| tag_rate | 0.960 | 0.970 | 0.965 |
| avg_calls | 2.21 | 2.19 | 2.19 |
| tool_hit_rate | 0.955 | 0.955 | 0.955 |
| no_answer_rate | 0.040 | 0.030 | 0.035 |

训练侧：reward 前 8 步均值 0.416 → 后 8 步 0.462（+0.046），format 稳定在 0.087–0.10，
`clipped_ratio` 从 0.375 降到 0.00（`max_completion` 2048 + truncation mask 生效）。

**结论**：GRPO 在单卡 24G、batch 16 的约束下**收敛到与 SFT 持平**（EM 相同，F1 +0.005，检索次数 −0.02）。
前三轮失败原因已定位并逐个修复（工具结果被 completion 预算截断 → 2048；截断样本污染梯度 → mask；
除 std 的难度偏置 → Dr.GRPO）。最终仍无 EM 增益，说明 SFT 已接近该奖励函数的局部最优，
进一步提升需要更大 batch、step-level 奖励或异步 rollout。

---

## 2f. E10 详细（RFT，2026-11-02）

流程：用 SFT 模型对 500 道 rl 题各采 3 条（temperature 0.8，HF generate），过滤后得 592 条正确轨迹；
与全量 SFT 数据（1841 条）按 1:3 混合成 2433 条，在 merged SFT 上再挂 LoRA 训 1 epoch（lr 5e-5，305 步，15 分钟）。

| 指标 | SFT | RFT |
|---|---|---|
| EM | 0.405 | 0.395 |
| F1 | 0.531 | 0.525 |
| tag_rate | 0.960 | **0.980** |
| avg_calls | 2.21 | 2.19 |

**结论**：RFT 与 SFT 在 EM 上相差 1 个点（200 题里 2 道题），**在噪声范围内**（95% CI ≈ ±7 点）；
格式合规率略升（0.96 → 0.98）。说明 592 条自采样轨迹没有提供超出 SFT 的新信息，
拒绝采样的收益被“同题多分支”的低多样性抵消。

至此三种后训练方案（SFT / RFT / GRPO）在 1.7B 单卡、200 题评测下全部落在 0.395–0.405：
**SFT 已是该配置下的有效上限**，进一步提升需要更大 batch、更多样题或 step-level 奖励。

---

## 3. 关键发现

1. **检索不是瓶颈**：tool_hit 84%，2×2 中 hit_wrong 160 vs miss_wrong 32。
2. **协议是最大瓶颈**：tag_rate 0.20，strict_call_rate 0.57；两种工具调用语法并存。
3. **两跳缺失是结构性的**：avg_calls 0.95，而 eval200 中 160/200 是 bridge 题。
4. **数据侧的对齐**：SFT 数据中 74% 是两跳桥接题，与 eval200（80% bridge）分布一致；平均调用 2.06 轮，正好对应“第二跳靠改写查询”的行为。
5. **两跳检索的天然上限**：仅用问题做单次检索，AllHit@5 只有 0.44——第二跳实体名不在问题里，必须靠多轮改写查询才能拿到。这既是检索的局限，也是 agent 存在的理由。

---

## 4. 变更日志

### 2026-10-31 · 统一分词器与观测格式（影响基线可比性）

- `bm25.py`：`[a-z0-9]+` → `[^\W_]+`（Unicode），修复 `Motörhead` 被切成 `mot`+`rhead`。
- `runner.Tools`：截断不再追加 `" ..."`，与训练环境 `SearchEnv` 的观测逐字一致。
- 实测影响：eval200 中 top-5 仅 **7/200** 题变化；R@5 0.6925 → 0.6950。
- **后果**：E1a/E1b 与 E2/E3 产自不同检索器版本。E1a/E1b 为闸门记录，保留；正式基线（1.7B）必须用 v2 代码重跑，避免“检索器变了”污染 SFT 前后对比。

### 2026-10-31 · DeepSeek 接口适配（两次线上报错）

- `422 missing field tool_call_id`：DeepSeek 要求 `role=tool` 必须配结构化 `tool_calls`。修法：发给教师模型的观测改成 `role=user` + `[TOOL OUTPUT]` 前缀；**落盘的 SFT 记录仍是 `role=tool`**，与训练/评测一致。
- `TypeError: cannot unpack NoneType`：`work()` 的异常分支返回 `None`，主循环按二元组解包崩溃。已统一返回 `(rec, why)`；鉴权失败立刻中止，不再空跑。
- 新增启动预检：一条 `max_tokens=1` 的请求验证 key，失败即退出（约 ¥0.0001）。
- 新增 `--budget`：按空闲时段价格估算，超限自动停止。
- 校验器放宽答案匹配（EM → EM/F1≥0.8/包含且不啰嗦），保留率 40% → 60%；C 类封顶 10%，避免模型过度拒答。

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
- [x] 用 DeepSeek 生成 A/B/C 三类 SFT 数据（实花 ¥5.5）
- [x] `validate_sft.py` 校验，保留率 46%，avg_calls 2.06
- [ ] LoRA SFT（r=32，2 epoch，max_len 3072）—— 先 0.6B 冒烟再 1.7B
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
