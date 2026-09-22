# narde — Architecture Notes (对标 Laya 的复刻蓝图)

> 本文件是 narde 的架构蓝本，逐模块对照 laya (https://github.com/NandhaKishorM/laya, Apache-2.0)
> 的参考实现整理而成。写代码前必须读本文件；改动架构前必须更新本文件。
> 参考源码在本仓 `reference/laya/`（只读，勿改）。

## 1. 一句话

Laya/narde 是**非自回归（NAR）的 System-1 决策引擎**：把「任意状态（文本/邮件/工单/JSON）+
类型化问题」一次性编码，**单次前向**输出带置信度的结构化决策（choice/score/noul），
并用 `Router` 按语言/脚本路由到最合适的 checkpoint。核心卖点是**快**（单次前向、无自回归、
无 token 生成）+ **概率可校准**。

## 2. 关键不变量（golden invariants）

1. **prompt 字节级兼容**：喂给 encoder 的 token 序列必须与 laya 逐 token 一致（见 §5 格式）。
   任何改变 token id 的改动，都要对照参考实现重新验证字节流。
2. **确定性**：评估/默认推理走贪心/温度=1 路径；随机采样默认关闭。
3. **不静默截断**：state 超预算必须显式报错或标记 `truncated=True`。
4. **概率可校准是一等公民**：原始 logit 不是置信度；`confidence` 必须来自校准（温度/isotonic），
   未校准的桶要显式标记 `calibrated=False` 并走 fail-closed。
5. **Apache-2.0 出处**：任何借鉴自 laya 的代码，须在文件头注明来源；不 vendor 其权重。

## 3. 三个原语（Decision Primitives）

| 原语 | 内部 code | 输出 | 典型用途 |
|---|---|---|---|
| `choice` | 0 | argmax 标签 + 各类概率 + 置信度 | 部门路由、意图、主题 |
| `score`  | 1 | 序数 rubric 期望分 + 各级分布 + 置信度 | 紧急度、危害度、挫败感 |
| `noul`   | 2 | `P(true)` 概率（二分类） | 垃圾邮件、越狱、流失风险 |

`QTYPES = {"choice":0, "score":1, "noul":2}`。`noul` = "no output label" 的二分类特例。

## 4. 模型结构（对应 `common.py::DecisionModel`）

```
encoder:  HF AutoModel (ModernBERT-large / mmBERT-base), sdpa, 冻结或低 lr 微调
          └─ last_hidden_state [B, L, D]
type_emb: Embedding(3, D)           # 按 qtype 加到每个 token
head:     TransformerEncoder(2 层, head=nhead=D//64, ffn=4D, norm_first, 带 padding mask)
scorer:   LayerNorm(D)→Linear(D,D)→GELU→Linear(D,1)   # 每个 [MASK] 出一个 logit
act_head: Linear(D+4,256)→GELU→Linear(256,n_act)      # n_act = len(act_costs)+1
```

- 在 `marker_pos`（每个 [MASK] 的位置）gather 出每个选项的表征 → `scorer` 出 logit。
- 非 marker 位置 logit mask 成 -1e4。
- `act_head` 吃 `[CLS] 池化 + [top1, top1-top2, 归一化熵, k/255]`，输出 `act_probability`
  （供下游"是否直接执行/是否需人审"决策）。单选项时用 0 补齐 top2 使 gap=1.0。
- `temperature` 是 buffer，形状 (3,) 按 qtype；`temperature_by_options` 覆盖到
  `temp_bucket(qtype,k)` 粒度（`2 / 3-5 / 6-10 / 11+`）。`clamp_temperature` 限到 [0.5, 5.0]，
  越界会 warn（防过度锐化，如 0.1 会把 0.24 撑成 0.99）。

## 5. Prompt / 序列格式（必须逐字节对齐）

```
[CLS] {type} {instructions} [SEP] [MASK] opt0 [MASK] opt1 ... [MASK] optN-1 [SEP] {state} [SEP]
```

- `render_options`：choice 按 criteria 的 key 顺序渲染 `key: desc`（None/"" 只留 key）；
  score 渲染成 `level i: <desc>`；noul 固定 `["false: ...", "true: ..."]`。
- 每个 option 前缀一个 `[MASK]`，单选项 token 上限 48；若总长超 `head_max_len`，
  按 `max(4,(head_max_len-16)//n)` 重新均分裁剪。
- state 序列化：str 原样；dict/list 走 `json.dumps(ensure_ascii=False)`；再 `[:room]` 或 `[-room:]`
  （`truncate_left`）截断，**必须**在结果里体现是否截断。
- 预算默认：EN 512/192；multilingual/typed 1024/256。

## 6. 路由（对应 `router.py`）

- `Router` 维护 LRU 的 agent 缓存（`max_loaded`），`RLock` 保护生命周期；`predict` 本身不持锁，
  使并发推理可共享同一 checkpoint。
- 路由优先级：显式 `model=` > 显式 `task=` > (可选) 自动识别 workflow > 显式 `lang=` > 脚本/语言检测 > 默认。
- 脚本检测走纯 Python Unicode 区间（`lang.py`），亚毫秒；拉丁文再叠停用词/变音符启发判是否英语。
- **关键事实**：英文 checkpoint 遇非拉丁脚本会「自信地答错」（Khmer 0.000 acc @ 0.952 conf），
  所以路由必须在 forward 之前做，靠脚本而非模型置信度。

## 7. 校准（`calib`）

- `ece_score(conf, correct, bins=15)`：分箱期望校准误差。
- `confidence_from_probs(p,k)=1 - H(p)/log k`（k<2 时 1.0）。
- 温度拟合：对每个 (qtype, 选项数桶) 在 held-out 上拟合一维温度（LBFGS，`clamp [0.5,5]`，
  样本 <10 退回 1.0）。multilingual 出厂无温度，务必先 fit 再用于门控。

## 8. 训练（RLCD，对应 notebook / `proper_reward`）

- 奖励 = `log_score + 0.5*spherical`，score 型再减 `1.0*RPS`（`proper_reward`，log_floor=-9.21）。
- GRPO 式：每题采样 G 组带噪 logits（σ 线性退火 0.4→0.1），组内相对优势，配合 1.0 权重
  的 soft-CE 项共同反传；`detach_encoder` 可选。
- 多轮：`td_lambda_targets` 按 `ep_group` 排序做 λ-return。
- 微调后**必须重新拟合温度**再评估。

## 9. 已知边界（对标时别自我欺骗）

- base checkpoint zero-shot 在 typed-decisions 上≈随机（0.36/0.35 vs 随机 0.318 / 多数类 0.461）；
  0.766 全靠域内微调。复刻后别拿 base 权重直接宣称能打。
- 高基数 choice（>20 选项）默认预算下崩到 0.425；用 shortlist/budget 两阶段。
- `score` 原语最弱（SST-5 0.372）。
- 两个 base 都偏自信，务必温度校准后再谈门控。

## 10. 参考源文件对照

| narde 目标模块 | 参考源（reference/laya/laya/*.py） |
|---|---|
| `prompts.py`  | `common.py`（build_sequence/render_options/serialize/collate_items/QTYPES/temp_bucket/confidence） |
| `model.py`    | `common.py::DecisionModel, build_model, proper_reward, td_lambda_targets` |
| `agent.py`    | `agent.py::Agent, load` |
| `router.py`   | `router.py::Router, RouteDecision, normalise_name` |
| `lang.py`     | `lang.py`（纯 Python，无依赖） |
| `shortlist.py`| `shortlist.py` |
| `presets.py`  | `presets.py` |
| `email.py`    | `email.py`（清洗+email_state） |
| `calib.py`    | 温度拟合 + `ece_score`（参考 notebook §校准 与 common.py） |
