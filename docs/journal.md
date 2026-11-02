# 项目全过程记录 · agentrl-search

> 从 0 到「SFT 跑通、EM 0.140 → 0.405」的完整流水：决策、命令、数字、踩过的坑。
> 用途：复盘、面试自述、交接。
> 最后更新：2026-11-01

---

## 0. 一页速览

| 项 | 结果 |
|---|---|
| 任务 | HotpotQA 多跳问答，1.7B 模型调用 search 工具作答 |
| 方法 | LoRA SFT（r=32），数据由 DeepSeek 教师生成并规则过滤 |
| 基线 EM | 0.140（Qwen3-1.7B 零样本） |
| SFT 后 EM | **0.405（+26.5 点）** |
| 协议遵循 | `tag_rate` 0.745 → 0.960 |
| 平均检索轮数 | 1.62 → 2.21 |
| 累计花费 | **≈ ¥13**（数据 ¥5.5 + GPU ≈ ¥7） |
| 累计 GPU 时间 | ≈ 2.8 h（有效计算 ≈ 46 min） |
| 代码 | `~/agentrl-search`，git 管理 |

---

## 1. 时间线

### 10-29 · 选题与骨架

- 明确方向：tool use + agentic RL + GRPO，要求完整链路（基线 → 数据 → SFT → badcase → RL 数据/奖励 → RL → 对比）。
- 盘点 GitHub 生态：verl / Agent-R1 / Search-R1 / ReTool / ToolRL / Trinity-RFT / TinyZero / RAGEN 等，最终**放弃多卡框架**，选 `TRL GRPOTrainer + environment_factory`（单卡、LoRA、多轮工具）。
- 建库 `~/agentrl-search`，Python venv，装 pandas / pyarrow / rank_bm25。
- 下载 HotpotQA distractor dev（hf-mirror），归一化 `context` 结构，构建 66,581 段语料、7,405 题。
- 检索闸门 v1：R@5 0.613 / AllHit@5 0.325 —— **未达标**。

### 10-30 · 检索修复与首轮评测

- 定位 v1 三个问题：分词只认 ASCII、标题没进索引、查询没去停用词。
- 修复后 v2：**R@5 0.695 / AllHit@5 0.440**，过闸。
- 搭评测器：greedy 多轮循环、bad case 五类归因、2×2 误差分解。
- 0.6B 冒烟 5 次（20 题）：EM 0.05 / tag_rate 0.10 / tool_hit 0.95。
- 0.6B × 200 正式跑：**EM 0.040 / tag_rate 0.200 / avg_calls 0.95 / tool_hit 0.840**。
- 归因：0.6B 不是“不会查”，是“查到了不会按协议交卷”。

### 10-31 · 统一口径、造数据、上 1.7B

- 统一分词器（`[^\W_]+`）与观测格式，实测只影响 7/200 题，R@5 0.6925 → 0.6950。
- 用 DeepSeek V4.1 Flash 造 SFT 数据：A 格式 / B 两跳（50% oracle 首跳）/ C 弃答。
  - 生成 4200 → 保留 1937（46%），实花 **¥5.5**。
  - 校验规则：EM / F1≥0.8 / 含 gold 且不啰嗦；C 类封顶 10%。
- 租 AutoDL 4090 24G（¥2.5–2.9/h），装环境，ModelScope 下 1.7B 权重。
- 1.7B 基线（200 题，8.4 min）：**EM 0.140 / tag_rate 0.745 / avg_calls 1.62 / tool_hit 0.920**。
- LoRA SFT：1841 条、2 epoch、462 步、28 分 13 秒、¥1.2。
  - train_loss 0.289、eval_loss 0.2575（无过拟合）、mask preflight 76/777。
- 合并 LoRA → 评测（9 min）：**EM 0.405 / F1 0.531 / tag_rate 0.960 / avg_calls 2.21 / tool_hit 0.955 / no_answer 0.040**。
- 拉回 artifacts，关机。

### 11-01 · 文档化

- 写 `docs/report.md`（技术报告）与本文。
- 待办：RFT、GRPO、别名归一化、多种子评测。

---

## 2. 关键决策与理由

| 决策 | 选择 | 理由 |
|---|---|---|
| 训练框架 | TRL `GRPOTrainer` + `environment_factory` | 单卡可跑、原生多轮工具、LoRA 友好；verl 的环境成本会吃掉预算 |
| 基座 | Qwen3-1.7B | 24G 卡 LoRA 舒适；比 0.6B 协议遵循强得多（tag_rate 0.20 → 0.745） |
| 微调方式 | LoRA r=32 | 全参 Adam 状态就 13.6G，24G 必炸；且 1841 条数据全参会过拟合 |
| 检索器 | BM25 + 标题入索引 + 去停用词 | 零成本、可复现；先暴露“查询构造”这一瓶颈，再上向量检索不迟 |
| 数据来源 | DeepSeek 教师 + 真实 `SearchEnv` 执行工具 | 保证观测与推理逐字一致；oracle 首跳把两跳样本通过率从 ~10% 提到 53% |
| 评测 | 固定 eval200、greedy、qid 不交 | 可复现、可比较；所有报告数字同源 |
| 指标 | EM + F1 + `em_contains` | 严格 EM 会误杀别名（`Alan Turing` vs `Alan Mathison Turing`） |

---

## 3. 代码与数据资产

### 3.1 目录

```
agentrl-search/
├── src/agentrl/
│   ├── data/{normalize,build}.py        # 三种 context 结构的归一化 + 语料/切分
│   ├── retrieval/bm25.py                # BM25 + Recall@k
│   ├── env/search_env.py                # TRL 用的 SearchEnv（search/calculator）
│   ├── prompts.py                       # 唯一的 system prompt + 工具 schema
│   ├── eval/{runner,metrics,badcase}.py # 多轮评测、指标、bad case 归因
│   └── reward/                          # 奖励函数（RL 用）
├── scripts/
│   ├── build_data.py                    # parquet → corpus/questions/splits
│   ├── eval_retrieval.py                # 检索闸门
│   ├── evaluate.py                      # 200 题评测（greedy）
│   ├── gen_sft_deepseek.py              # A/B/C 数据生成（可 --fake 离线）
│   ├── gen_all.sh                       # 三类数据串行 + 校验
│   ├── validate_sft.py                  # 回放校验、过滤、切分
│   ├── train_sft.py                     # LoRA SFT（含 mask preflight）
│   └── merge_lora.py                    # adapter → 完整模型
├── data/{raw,processed,sft}/
├── artifacts/                           # eval_*、sft_*/
└── docs/{experiments.md,report.md,journal.md}
```

### 3.2 数据规模

| 文件 | 行数 / 大小 |
|---|---|
| `data/processed/corpus.jsonl` | 66,581 段，38MB |
| `data/processed/questions.jsonl` | 7,405 题 |
| `data/processed/splits/eval200.jsonl` | 200 题 |
| `data/sft/train.jsonl` / `val.jsonl` | 1,841 / 96 条 |
| adapter（`sft_1p7b/final`） | 133MB |
| 1.7B 权重（ModelScope 本地） | 3.9GB（两分片） |

---

## 4. 结果汇总

### 4.1 三阶段对比

| 指标 | 0.6B 基线 | 1.7B 基线 | 1.7B + SFT |
|---|---|---|---|
| EM | 0.040 | 0.140 | **0.405** |
| F1 | 0.050 | 0.225 | 0.531 |
| tag_rate | 0.200 | 0.745 | 0.960 |
| avg_calls | 0.95 | 1.62 | 2.21 |
| tool_hit | 0.840 | 0.920 | 0.955 |
| no_answer | 0.800 | 0.255 | 0.040 |

### 4.2 2×2（1.7B + SFT）

| | 答对 | 答错 |
|---|---|---|
| 检索到 gold | 80 | 111 |
| 未检索到 | 1 | 8 |

### 4.3 结论

1. **SFT 解决“会不会用工具”**：协议、两跳、终止格式全部到位。
2. **剩余错误是“答得对不对”**：111 个 hit_wrong，其中相当比例是答案形式差异。
3. **新问题**：4% 的题跑满 5 轮，SFT 没教“何时停”——这是 RL 的切入点。

---

## 5. 踩过的坑（按时间）

| # | 现象 | 根因 | 修法 |
|---|---|---|---|
| 1 | `KeyError: 0` 读 `context[0]` | parquet 里 context 是**列式 struct** `{title: [...], sentences: [...]}` | 写 `normalize.py` 统一三种结构 |
| 2 | `FileNotFoundError` | notebook 的 cwd 是 `notebooks/` | 顶部加“向上找 data/ 目录”的 ROOT 定位 |
| 3 | `NameError: to_paras` | kernel 重启后函数定义丢失 | 函数下沉到模块，notebook 只 import |
| 4 | `ModuleNotFoundError: rank_bm25` | 脚本用了 conda 的 `python` | 脚本固定 `.venv/bin/python`，并支持 `PYTHON=` 覆盖 |
| 5 | DeepSeek `422 missing tool_call_id` | `role=tool` 必须配结构化 `tool_calls` | 给教师看的观测改成 `role=user` + `[TOOL OUTPUT]`；落盘仍是 `role=tool` |
| 6 | `TypeError: cannot unpack NoneType` | 异常分支返回 `None`，主循环按二元组解包 | 所有分支返回 `(rec, why)`；鉴权失败立即中止 |
| 7 | `401 Your api key is invalid` | key 在截图里泄漏、已被吊销，终端里还是旧值 | 重新 `export` 新 key；用 `read -s` 输入避免进 history |
| 8 | rsync `Could not resolve hostname :` | 在实例里执行、且域名与冒号间有空格 | rsync 只在 Mac 上跑；`host:path` 之间不能有空格 |
| 9 | macOS rsync 不认 `--info=progress2` | 系统自带 rsync 2.6.9 | 去掉该参数，或 `brew install rsync` |
| 10 | 传了 1.59GB | `--exclude` 引号被写成中文引号，`.venv` 没排掉 | 用英文单引号；排除 `.venv`、`.git`、`artifacts` |
| 11 | HF 下载 0% 后 401 | 新版 Xet 协议在镜像站鉴权失败 | `HF_HUB_DISABLE_XET=1` + 卸 `hf_xet`；最终改用 ModelScope |
| 12 | `tmux: command not found` | 镜像未装 | `apt-get install -y tmux`，或 `nohup ... &` |
| 13 | `SFTConfig unexpected keyword 'warmup_ratio'` | 实例上 transformers v5 移除了该字段 | 按 `inspect.signature` 过滤未知参数 |
| 14 | A/B 类 qid 重叠 | 都从同一题池采样 | A 改为 comparison-only，与 B（bridge）不相交 |
| 15 | 严格 EM 误杀正确样本 | `2 acres` vs `2 acre`、别名 | 校验放宽为 EM / F1≥0.8 / contains + 长度约束 |
| 16 | 训练可能只学 tool 观测 | 模板缺 `{% generation %}` 会导致 mask 为空 | 训练前 preflight 检查 mask，空则直接退出 |

---

## 6. 成本台账

| 项 | 用量 | 金额 |
|---|---|---|
| SFT 数据生成 | 4200 条，DeepSeek Flash 空闲时段 | ¥5.5 |
| GPU（4090 24G） | ≈2.8 h（含安装/下载/训练/评测） | ≈ ¥7 |
| **合计** | | **≈ ¥13 / ¥100** |

有效计算：基线 8.4 min + SFT 28.2 min + 评测 9.0 min ≈ **46 min**；其余为环境与下载。

---

## 7. 常用命令速查

```bash
# 本地（Mac）
source .venv/bin/activate
python scripts/build_data.py --out data/processed
python scripts/eval_retrieval.py
python scripts/evaluate.py --model Qwen/Qwen3-0.6B --limit 200 --out artifacts/eval_06b

# 数据生成（本地，走 API）
export DEEPSEEK_API_KEY=...
bash scripts/gen_all.sh
python scripts/validate_sft.py --inputs data/sft/raw_*.jsonl

# 同步到 GPU（Mac 上执行）
rsync -avz --exclude '.venv' --exclude '.git' --exclude 'artifacts' \
  -e "ssh -p <端口>" ~/agentrl-search/ root@<域名>:/root/autodl-tmp/agentrl-search/

# GPU 上
export HF_HOME=/root/autodl-tmp/hf
export HF_ENDPOINT=https://hf-mirror.com
tmux new -s sft
python scripts/train_sft.py --model /root/autodl-tmp/models/Qwen3-1.7B-ms --out artifacts/sft_1p7b
python scripts/merge_lora.py --base /root/autodl-tmp/models/Qwen3-1.7B-ms \
  --adapter artifacts/sft_1p7b/final --out artifacts/sft_1p7b/merged
python scripts/evaluate.py --model artifacts/sft_1p7b/merged --out artifacts/eval_sft_17b_200
```

---

## 8. 待办

- [ ] RFT：用 SFT 模型对 rl1200 采 4 条 → 留答对 → 再 SFT 一轮（~30 min，¥1.5）
- [ ] GRPO：奖励 = EM/F1 + 格式 + 工具预算惩罚，验证“准确率–成本”权衡（~1–2 h，¥4）
- [ ] 别名归一化（`Alan Turing` / `Alan Mathison Turing`），降低 EM 误杀
- [ ] 3 个 seed 重跑 eval200，给置信区间
- [ ] 把 `docs/report.md` 的结论翻成简历 bullet

---

## 9. 经验总结（写给下一次的自己）

1. **环境隔离**：项目一个 venv，永远显式 `.venv/bin/python`；不要污染 conda base。
2. **先冒烟再全量**：5 题冒烟能挡掉 80% 的配置错误，成本几分钱。
3. **数据一致性优先**：训练/评测/造数据必须共用同一套 prompt、parser、工具实现。
4. **便宜的地方别省时间，贵的地方别省钱**：数据生成花 ¥5 省一天；GPU 上挂机一小时比什么都贵。
5. **所有数字都要能复现**：固定 seed、固定 eval200、固定解码参数，报告里的每个数都能一条命令重跑。
6. **先写局限，再写结论**：面试官会挑的刺，自己先说出来。
