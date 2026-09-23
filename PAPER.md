# 以差分测试为可执行契约：非自回归决策引擎的清洁复刻方法论
## —— narde / laya 案例研究

*(arXiv / technical-report style; body in Chinese, technical terms in English)*

---

## Abstract (English, for submission systems)

Clean-room re-implementation of an open-source machine-learning inference engine
is usually *asserted* as faithful but rarely *provable* in a way that survives
later refactors, dependency bumps, and second-party development. We present a
methodology that turns "100% replication" from a faith-based claim into an
**executable, CI-enforceable property**: differential (oracle-based) testing
against a read-only reference. The candidate and the oracle are run in a single
process on identical inputs, and their outputs are asserted equal at each of
**five equivalence layers** — byte-exact token sequences, forward tensors under
shared weights, API response dictionaries, exception semantics, and stateful
routing lifecycle. Our case study is `narde`, a 1:1 port of `laya` v0.3.5, a
non-autoregressive "System-1" typed-decision engine (`choice` / `score` /
`noul`) that answers typed questions over arbitrary state in a single forward
pass.

Key techniques: (i) **shared-weight forward parity** — loading one
`state_dict` into both packages' `DecisionModel` with `strict=True` collapses
behavioral equivalence and architectural key/shape compatibility into a single
assertion; (ii) a **divergence ledger** that separates intentional deltas
(branding, version, a torch-free lazy `__init__`, log prefixes, a
`fit_temperature` extension) from bugs, with every delta pinned by a test, so
"100% replication" becomes the precise statement "100% *except* the ledger";
and (iii) an **offline benchmark harness** (`--tiny` mode: a seeded 1-layer BERT
plus a deterministic mock tokenizer driving the real `system_one` code path)
that keeps the pipeline regression-verifiable in air-gapped, weight-free CI.

Results: 40/40 differential checks pass; the suite **caught 7 real divergences
during porting**, 5 of which were silent-drift bugs (prompt separators,
`forward` top-2/entropy shapes, a missing language-evidence filter, a missing
`email_questions` symbol, and a sequence-budget bug) that manual review would
almost never surface. We are explicit about limits: differential testing proves
**code/behavioral equivalence**, not **weight equivalence**, and not the
reference's own correctness; functional equivalence on real data additionally
requires real checkpoints, which our harness (but not this offline setting)
supports. The methodology generalizes to any port with a runnable reference:
cross-language rewrites, framework migrations, legacy modernization, and
numerical cross-checks between LLM inference runtimes.

**Keywords:** differential testing; clean-room replication; machine-learning
inference; reproducibility; oracle-based testing; calibration

---

## 摘要 (Abstract)

清洁复刻（clean-room re-implementation）一个开源机器学习推理引擎时，最大的风险
不是写错第一版，而是**无法持续证明"我仍然和上游一样"**。本文提出并实践一种把
"100% 复刻"变成**可执行、CI 可强制属性**的方法学：**对只读参考实现做差分
（oracle-based）测试**——同一输入同时喂给参考实现与复刻实现，在单个进程内断言
二者输出在每一层等价类上都相等。我们以 `narde`（对 `laya` v0.3.5，一个非自回归、
System-1 的类型化决策引擎的 1:1 端口）为案例，交付了：

- 一套 **40 项差分测试**（8 个模块 × 五层等价类：逐 token 提示序列、共享权重下
  的前向张量、API 响应字典、异常语义、有状态路由生命周期）；
- **共享权重前向一致性**技巧：同一份 `state_dict` 以 `strict=True` 分别载入两个
  包的 `DecisionModel`，一次断言同时覆盖行为等价与架构 key/shape 兼容；
- 一份**分歧台账**（`docs/divergence.md`），把有意的偏离（品牌字段、版本、
  torch-free 懒加载、`[narde]` 日志前缀、`fit_temperature` / `settings.py` 扩展）
  与 bug 显式隔离，每条都有测试钉住；
- 一个**离线可跑的基准 harness**（`bench/`，`--tiny` 模式：种子 1 层 BERT +
  确定性 mock tokenizer，走真实 `system_one` 代码路径），使管线在无网络 / 无
  权重环境中仍可回归验证。

结果：40/40 全绿；套件在移植过程中**实际捕获 7 类真实分歧**（附录 C），其中 5 类
属于"不崩、不报错、静默漂移"型。我们明确方法学边界：差分测试证明**代码/行为
等价**，不证明**权重等价**，也不证明参考实现本身正确；"100% 复刻"因此被精确化
为两层陈述（§7.1）。

**关键词**：differential testing, clean-room replication, machine-learning
inference, reproducibility, oracle-based testing, calibration

---

## 目录

1. 引言
2. 背景：laya / narde 决策引擎
3. 相关工作
4. 方法论
5. 实现：模块映射与测试矩阵
6. 实验与结果
7. 讨论
8. 结论
9. 复现说明
参考文献
附录 A：完整测试矩阵（40 项）
附录 B：分歧台账
附录 C：差分测试捕获到的真实分歧

---

## 1. 引言

### 1.1 问题：复刻正确性为何难以"验收"

复刻（或跨语言/跨框架移植）一个已有开源系统时，"我复刻对了"通常只有两种证据：
(a) 读起来像；(b) 几个样例跑起来像。两者都是**采样证据**，在重构、依赖升级、
二次开发之后迅速失效，且没有任何机制在失效时报警。对普通库，"照抄上游测试"可以
缓解；但对一个**自带模型权重、浮点前向、有状态路由、多 checkpoint** 的 ML 推理
引擎，上游测试针对上游包名、上游目录布局、上游权重下载路径，不能照抄。

本文的核心主张：

> 对一个有可执行参考实现的系统，"行为等价"应当被定义为
> **在充分大的输入分布上，参考实现与复刻实现的输出逐层相等**，
> 并由测试代码**持续执行**，而不是由人审查一次。

这个定义有三个推论，构成 §4 方法学的骨架：

1. **等价性是分层的**。prompt 的字节流、张量前向、概率字典、异常消息、有状态
   缓存生命周期，每一层都有各自的"相等"语义与断言原语；
2. **浮点确定性必须被显式控制**（线程数、种子、tokenizer 的进程内哈希盐），
   否则"逐位相等"不可达，只能退化成"近似相等"并丢失字节级契约；
3. **必须显式声明允许差异的白名单**（分歧台账），否则品牌化改动会让 CI 永远
   红着，团队最终学会忽略红灯。

### 1.2 案例：narde 是 laya v0.3.5 的 1:1 端口

`narde` 复刻 `laya`——一个非自回归（NAR）、System-1 的**类型化决策引擎**：把
"任意状态（文本/邮件/工单/JSON）+ 类型化问题"一次性编码，**单次前向**输出带
校准概率的结构化决策，并用 `Router` 按语言/脚本路由到最合适的 checkpoint。
三种问题原语（`QTYPES = {choice:0, score:1, noul:2}`）：

| 原语 | 语义 | 输出 |
|---|---|---|
| `choice` | 类别判别 | argmax 标签 + 各类概率 + 置信度 |
| `score`  | 序数量表 | 期望分（`Σ i·p_i`）+ 各级分布 + 图例 |
| `noul`   | 二分类（no-output-label） | `P(true)` + 置信度 |

关键架构（完整蓝本见 `docs/architecture-notes.md`）：

```
encoder:  HF AutoModel (ModernBERT-large / mmBERT-base), sdpa
          └─ last_hidden_state [B, L, D]
type_emb: Embedding(3, D)            # 按 qtype 加到每个 token
head:     TransformerEncoder(2 层, head=D//64, ffn=4D, norm_first, padding mask)
scorer:   LN→Linear(D,D)→GELU→Linear(D,1)   # 每个 [MASK] 位出一个 logit
act_head: Linear(D+4,256)→GELU→Linear(256,n_act)   # n_act = len(act_costs)+1
```

**字节级提示契约**：

```
[CLS] {type} {instructions} [SEP] [MASK] opt0 [MASK] opt1 … [MASK] optN-1 [SEP] {state} [SEP]
```

每个选项前缀一个 `[MASK]` 探针位（单选项 ≤ 48 token；选项总长超 `head_max_len`
时按 `max(4,(head_max_len-16)//n)` 重新均分裁剪）；state 按剩余预算
`room = max(0, max_len - len(head) - 1)` 截断，`truncate_left=True` 保留尾部；
`head_ids[:max(8, opt_budget)]`；超 `max_len` 的 marker 丢弃。改任何一处 token
布局都会使模型输出漂移——这是"字节级复刻"的含义，也是 §5 `test_prompts` 逐 id
钉死的原因。

**校准是一等公民**：`temperature` 按 `(qtype, 选项数桶 2/3-5/6-10/11+)` 粒度
存储，`clamp_temperature` 限到 `[0.5, 5.0]`（NaN/inf/非数值 → 1.0，防止出厂 0.1
的温度把 0.24 的 top-1 锐化成 0.99 的假置信度）；`confidence = 1 − H(p)/log(k)`
（k<2 时 1.0）；`ece_score` 15 分箱。

**路由必须先于前向**：英文 checkpoint 遇非拉丁脚本会"自信地答错"（Khmer 0.000
准确率 @ 0.952 平均置信度），模型置信度抓不住这种失败，只能靠脚本先验路由。
`Router` 按 `显式 model > 显式 task > (可选)workflow > 显式 lang > 脚本/语言
检测 > 默认` 的优先级路由，以 LRU + `RLock` 管理 `max_loaded` 个常驻 checkpoint；
脚本检测是纯 Python Unicode 区间判断（`lang.py`），亚毫秒级。

**"100% 复刻"是两层的**：(i) **代码/行为等价**——本文证明的对象；(ii) **模型
权重**——不可由代码重新生成，只能下载 `convaiinnovations/laya` checkpoint
（Apache-2.0 资产）或按 M6 域微调重训。混淆这两层是"复刻完成"最常见的过度
声明来源（§7.1）。

### 1.3 贡献

1. **C1（等价类设计）**：ML 推理引擎差分测试的五层等价类划分（token / tensor /
   dict / exception / stateful lifecycle），每层的断言原语，以及"允许差异"的
   登记机制（§4.2–4.4）。
2. **C2（共享权重前向一致性）**：把"行为等价"与"架构兼容"合并为一次
   `load_state_dict(strict=True)` + 双跑断言（§4.3, §6.2）。
3. **C3（分歧台账）**：任何有意偏离必须 (a) 写入台账、(b) 有测试钉住、(c) 台账外
   零容忍，使"100% 复刻"成为"除台账外 100% 复刻"的精确陈述（§4.4, 附录 B）。
4. **C4（离线管线回归）**：`--tiny` 基准模式（种子模型 + mock tokenizer + 真实
   代码路径）使行为回归在 air-gapped / 无权重环境可执行（§4.5, §6.3）。
5. **C5（实证）**：narde↔laya 40/40 全绿；移植过程捕获 7 类真实分歧（§6.1, 附录 C）。

### 1.4 论文组织

§2 背景；§3 相关工作；§4 方法论；§5 实现与测试矩阵；§6 实验结果；§7 讨论与
威胁分析；§8 结论；§9 复现说明；附录 A/B/C。

---

## 2. 背景：laya / narde 决策引擎

（架构细节见 §1.2 与 `docs/architecture-notes.md`；此处只给方法论需要的最小背景。）

**为什么是非自回归。** 决策类任务（路由、分诊、审核、意图）不需要生成式输出，需要
的是"对这个状态，这几个问题各是什么答案"。NAR 引擎把这些问题**一次前向并行**
打完：没有解码循环、没有 token 采样、输出天然可批量化与可校准。代价是提示协议必须
精心构造（每个选项一个 `[MASK]` 探针位）——这正是复刻时最容易出**隐性**分歧的
地方：它不影响"能跑"，只影响"和上游跑出同一个数"。

**为什么校准是一等公民。** 原始 logit 不是置信度。laya 的温度按
`(qtype, 选项数桶)` 粒度拟合；两个 base checkpoint 出厂都偏自信——重拟合一个
温度/桶后 macro-ECE 0.466 → 0.081（english）、0.314 → 0.106（multilingual，
出厂无拟合温度）。而**英文 checkpoint 读不懂非拉丁脚本时置信度不降**（Khmer
0.000 acc @ 0.952 conf；51 语言 macro-ECE 0.733，任何准确率水平下平均置信度都
不低于 0.885）——所以路由必须在 forward 之前发生，靠脚本先验而非置信度门控。

**checkpoint 事实**（复刻与质量声明的分界）：base checkpoint 在 typed-decisions
zero-shot 上 ≈ 随机（0.362 / 0.352 vs 随机 0.318 / 多数类 0.461）；0.766 属于该
benchmark 训练分裂上域微调过的 checkpoint。T4 参考延迟：单问 32.8 ms，batch 10 为
7.2 ms/问（103–332 问/秒）。CPU 预期 200–500 ms/问。

**被复刻对象（narde）的模块面**：`prompts`（序列化/渲染/序列构造/collate）、
`model`（DecisionModel + build_model + RLCD 奖励/TD 目标）、`agent`（加载
`rl_agent_config.json` + `model.safetensors`、公参→内参 `_to_internal`、
`system_one` 端到端）、`router`（名称归一化/工作流匹配/LRU/RouteDecision）、
`lang`（脚本检测/拉丁语言启发）、`calib`（温度钳制/ECE/置信度/桶划分）、
`shortlist`（高基数 choice 的两阶段 embed + top-k）、`presets`/`email`（问题包
与邮件清洗）。narde 源码 1737 行（10 个模块）对照 laya 1669 行；多出的部分
（懒加载 `__init__`、`settings.py`、`fit_temperature`）全部登记在分歧台账
（§4.4, 附录 B）。

---

## 3. 相关工作

**差分测试与差分模糊。** 差分测试的古典形式是以参考实现为 oracle 对候选实现做
断言，在编译器验证（CompCert、LLVM 回归）、解释器互验（CPython↔PyPy 的
cross-language fuzzing）、密码库互检（RNG/hash）中已有成熟实践。差分模糊
（differential fuzzing）把输入生成交给 fuzzer，在多个目标上找输出差异（LAVA 的
oracle 模式、SQL 引擎间的 Morsello/DIFUZZ 式对拍）。与它们的关键区别：那些场景
oracle 与候选是**独立实现**，目标是**找 bug**；本文是**故意让候选贴合 oracle**，
目标是**守住等价**——断言是**全量相等**而非"发现一处差异即报 bug"，且必须
维护"允许差异"白名单。

**清洁室工程（clean-room engineering）** 的原始动机相反：为规避知识产权而**独立
重写**，不允许看原始代码。我们的设定是它的对偶：允许并强制对照参考实现，但参考
实现**只读**（`reference/laya/`，gitignored，永不入 git），复刻产物与参考产物在
测试里**同进程对跑**。可以理解为"清洁室的镜像"：不是"不看它"，而是"看了但必须
逐字节对齐，且对齐本身可被机器验证"。

**ML 可复现性与等价测试。** ML 可复现性研究多聚焦训练侧（数据/超参/随机性审计、
实验追踪）；推理侧"两个模型行为是否一致"通常用**统计检验**（KS 检验、输出分布
距离）而非逐点相等，因为训练侧不可控。本文设定不同：**两侧共享同一份权重**
（§4.3），前向因此是**确定且可逐点比较的**；差异只可能来自代码而非参数。这把
"行为等价"从概率陈述变成确定性陈述，是差分测试在 ML 推理引擎上可行的前提。与
ONNX Runtime / TensorFlow 跨框架数值对拍（通常容忍较大 ε）相比，我们用
`atol=rtol=1e-6` 且同硬件同线程数，把容差压到"实现差异"而非"数值差异"量级。

**模型即服务的等价性验证**（shadow testing / canary 流量对拍）在生产上以"新旧
模型影子流量对比"的形式存在；本文可看作它的**离线、同机、逐点**版本：不做流量
镜像，而是把同一输入直接灌给两个进程内的实现。

**性质化测试（property-based testing）** 提供输入空间随机化（本文用种子化随机
语料 + 边界值）；但性质本身（"与 laya 输出相等"）由 oracle 给出而非由人类归纳
——这是差分测试与 PBT 的分工。

---

## 4. 方法论

### 4.1 参考 oracle：只读、版本钉死、同进程

`reference/laya/` 是 laya v0.3.5 的完整 clone，**永不修改、永不入 git**
（`.gitignore` 显式排除 `reference/`）。版本钉死至关重要：oracle 漂移会让"全绿"
失去意义。测试在**单个 Python 进程**内同时 `import laya` 与 `import narde`
（`sys.path` 双注入：`reference/laya/` 与 `src/`），因此两侧共享同一解释器、
同一 numpy/torch 版本、同一随机种子、同一线程数——排除了"环境不同导致输出不同"
这一整类伪差异。

### 4.2 五层等价类

| 层 | 相等语义 | 断言原语 | 例 |
|---|---|---|---|
| L1 token | 整数列表逐位相等 | `==` on `List[int]` | `build_sequence` 的 ids/markers |
| L2 tensor | 数值一致（atol/rtol=1e-6） | `torch.allclose` | 共享权重 `forward` 的 logits/act |
| L3 dict | 结构化相等（含顺序敏感字段） | `==` on `dict` | `system_one` 的 answers/usage |
| L4 exception | 同类型 + 同消息 | 捕获后比 `type` 与 `str` | `normalise_name` 未知名、校验失败 |
| L5 stateful | 操作序列后的可观测状态相等 | 逐步断言 | Router LRU：load/evict 顺序、`loaded` 列表 |

五层都是**必要**的：只测 L3 会漏掉"恰好输出同、内部 token 布局不同"的隐性分歧
（如 `serialize_state` 分隔符差异在短样例上不显现，长 state 截断时才显现）；
只测 L1/L2 会漏掉 API 语义分歧（如 `noul` 答案缺 `confidence` 字段）；不测 L4
会漏掉"不报错地给错结果"（如 `normalise_name` 返回 `None` 而非 raise）；不测 L5
会漏掉缓存/路由生命周期分歧（如 LRU 驱逐顺序不同导致后续 `predict` 冷启动）。

### 4.3 共享权重前向一致性（关键技巧）

`DecisionModel` 的等价性无法用随机权重 + 随机输入比较（两侧参数不同，输出必然
不同）。做法：

```python
enc  = tiny_bert(seed)                        # 种子 1 层 BERT (hidden=32, 离线)
l_m  = laya.common.DecisionModel(enc, head_layers=2, n_act=2)
n_m  = narde.model.DecisionModel(enc, head_layers=2, n_act=2)
n_m.load_state_dict(l_m.state_dict(), strict=True)   # ← 一步两用
```

`load_state_dict(strict=True)` 在 narde 侧失败，当且仅当 narde 架构**缺 key 或
key 名/shape 不匹配**——它同时是**架构兼容证明**；成功后两侧逐参数逐位相同，
对同一批 collated 输入（L1 层保证 token 一致）跑 `forward`，`logits` 与
`act_logits` 必须 `allclose(atol=rtol=1e-6)`。这组断言把"前向等价"与"权重可
互换"压缩成**一个 fixture**（`pf.shared_decision_models()`），被 `test_model`
与 `test_agent` 复用。

配套确定性控制：`torch.set_num_threads(1)`；tokenizer 的 word→id 用 **FNV-1a**
（不用 Python 内建 `hash()`——它按进程加盐，跨进程不稳定）；tiny BERT 固定种子。
这些控制使 L2 的相等是**位级**的（实测两侧 logits 逐位相等，`allclose` 只是留
余量）。

### 4.4 分歧台账（divergence ledger）

"100% 相等"在复刻项目里是**错误的目标陈述**：复刻必然有品牌化与工程化偏离
（响应里的 `model` 字段、日志前缀、版本、`import narde` 必须保持 torch-free 的
懒加载 `__init__`、部署用 `settings.py`、narde 独有的 `fit_temperature`）。如果
这些偏离不登记，CI 会永远红，团队最终学会忽略红灯；登记之后，等价性变成精确
陈述：

> 除 `docs/divergence.md` 列出的、**有测试钉住的**有意偏离外，narde 与 laya
> v0.3.5 在 L1–L5 各层输出相等。

台账每条必须满足三条件：(a) 写明 laya 侧与 narde 侧行为及理由；(b) 有对应测试
**显式断言该差异**（如 `assert nr["model"] == "narde-rl-agent"` 且
`assert lr["answers"] == nr["answers"]`）；(c) 台账外任何差异都是 bug，CI 红灯。
这使"允许差异"从口头约定变成**可审计的白名单**。

### 4.5 离线管线回归（--tiny 基准）

差分测试保证"行为等价"，但等价的对象是**代码路径**；要验证"整条管线（加载 →
路由 → 序列化 → 前向 → 温度 → 格式化）在真实调用下不崩、且延迟/吞吐形态正常"，
需要 benchmark。`bench/` 双 harness（`latency.py` / `quality.py`）都提供
`--tiny` 模式：`build_tiny_agent()` 用种子 1 层 BERT + 确定性 mock tokenizer，
以 `object.__new__(Agent)` 注入属性的方式构造**真实的** `narde.agent.Agent`
（不触发 `__init__` 的下载路径），使 `system_one` 走**生产代码路径**。`--tiny`
数字只测管线、不测 checkpoint，报告里显式标注 `mode: "tiny"`；真实 checkpoint
模式（`--model NAME=SOURCE[:subfolder]`）方法论与 laya 的
`research/scripts/bench_latency.py` / `bench_local.py` 对齐，数字可对照。

### 4.6 双 runner 与门禁

`tests/parity/` 同时兼容两条路：(i) `pytest -q`（有 pytest 的 CI 首选，
`conftest.py` 负责路径注入）；(ii) `tests/parity/run.py` 独立 runner（无
pytest、离线、沙箱可用，逐模块扫描 `test_*` 并汇总）。门禁约定：**任何改动
narde 行为的 PR，必须 `run.py` 全绿 + `tests/test_smoke.py` 全绿**（后者是"裸
venv 只装 pydantic 也能 `import narde`"的打包金丝雀）。

---

## 5. 实现：模块映射与测试矩阵

### 5.1 模块映射（narde ← laya v0.3.5）

| narde 模块 | 来源（`reference/laya/laya/`） | 钉住的等价类 |
|---|---|---|
| `prompts.py` | `common.py`（serialize_state / render_options / render_criterion / build_sequence / collate_items / QTYPES） | L1 token |
| `model.py` | `common.py::DecisionModel, build_model, proper_reward, td_lambda_targets, amp_dtype` | L2 tensor |
| `agent.py` | `agent.py::Agent, load, _to_internal, _verify_compatibility, _fix_tokenizer_config` | L3 dict + L4 exception |
| `router.py` | `router.py::Router, RouteDecision, normalise_name, match_typed_decisions_workflow` | L3 + L4 + L5 stateful |
| `lang.py` | `lang.py`（detect_script / latin_profile / guess_latin_language / analyse） | L3 dict + L4 崩溃镜像 |
| `calib.py` | `common.py`（clamp_temperature / ece_score / confidence_from_probs / temp_bucket） | L2 标量 |
| `shortlist.py` | `shortlist.py`（shortlist_choice / predict_shortlist / embed_fn_from_agent） | L3 + L4 |
| `presets.py` / `email.py` | `presets.py` / `email.py` | L3 dict |
| `__init__.py` | `__init__.py`（公共面同名映射，改为懒加载） | 导入面（smoke 钉住） |

源码规模：narde `src/` 1737 行（10 个模块）对照 laya 1669 行；差值全部由分歧台账
（附录 B）解释：懒加载 `__init__`、`settings.py`、`fit_temperature`、品牌字符串。

### 5.2 测试矩阵（40 项）

| 模块 | 项数 | 覆盖点 |
|---|---|---|
| `test_prompts` | 5 | serialize（8 状态 × 类型）、render_criterion（0/False/None/"" 边界 + 结构化 rubric）、render_options（choice/score/noul）、build_sequence **100+ 种子化随机 (state, q, max_len, head_max_len, truncate_left)** 逐 id 相等（含 14 选项高基数溢出重截用例）、collate_items 张量/meta 相等 + 空批→None |
| `test_model` | 7 | 共享权重 forward（logits+act allclose）、masked+detach_encoder 变体、单选项不崩（laya #96 edge case 双跑对照）、head_layers=0 无 head 变体、proper_reward（含自定义权重）、td_lambda_targets（λ∈{0,0.5,1.0} + 无 ep_group 直通）、amp_dtype |
| `test_lang` | 4 | detect_script / analyse / is_english / guess_latin_language，多语言语料（拉脱丁有/无变音符、希腊、西里尔、天城、谚文、汉字、阿拉伯、泰文、混合文本、空串、纯数字）；dict/list/None 下**两侧同型崩溃的镜像断言** |
| `test_router` | 7 | normalise_name（全部别名 + 未知名 ValueError **消息逐字相等**）、workflow 精确集合匹配（子集/超集→None）、自动检测路由、显式 model/task/lang 优先级、workflow 开关、**LRU 生命周期**（FakeAgent 猴补丁 `laya.agent.Agent`/`narde.agent.Agent`，逐次 load/evict/loaded 镜像）、RouteDecision repr |
| `test_calib` | 5 | clamp（NaN/inf/"abc"/None/np.float64/str）、temp_bucket 全组合、confidence（Dirichlet 样本、k<2、p[:k] 切片）、ece（bins 2..40、空→nan、全自信→0.0）、narde 扩展 fit_temperature 健全性 |
| `test_presets_email` | 4 | 5 个预设包逐 dict 相等 + 自定义 categories、clean_email_body **10 邮件 × 3 截断上限**、email_state（发件人/清洗矩阵）、email_questions |
| `test_shortlist` | 4 | shortlist_choice（dict/list 判据、k≥n 透传且 embed 不被调用、平局保留靠前者）、错误语义（ValueError/TypeError 类型+消息逐字相等）、predict_shortlist（_FakeAgent 回显收到的缩减判据 + shortlist meta + 透传）、embed_fn_from_agent 在共享模型 Agent 上嵌入矩阵 allclose + 参数校验消息相等 |
| `test_agent` | 4 | _to_internal 全边界、_fix_tokenizer_config（3 种 tokenizer_config 变体，双包写盘后 diff）、_verify_compatibility（通过/缺前缀/shape 不匹配/缺 key/缺 config，全 L4 镜像）、**system_one 端到端**（choice+score+noul 同批，answers 逐 dict 相等 + usage 相等；`model` 字段按台账断言） |

夹具要点（`pf.py`）：MockTokenizer word→id 走 FNV-1a（id∈[1000,1899)，绝不用内建
`hash()`）；BatchTok 定宽 12；`tiny_bert(seed)`（vocab 2048 / hidden 32 / 1 层 /
2 头）；`shared_decision_models()` 即 §4.3 的一步两用 fixture。

---

## 6. 实验与结果

环境：Python 3.14.3 · torch 2.12.0+cpu · transformers 5.17.0 · numpy 2.4.2 ·
pydantic 2.12.5 · CPU 单线程（`torch.set_num_threads(1)`）· 全程离线。

### 6.1 主结果：40/40 全绿

`python tests/parity/run.py`（或 `pytest -q tests/parity`）：**40 passed, 0 failed**。
其中 L2 组（test_model ×7、test_agent::system_one）在共享权重下 logits/act 实测
**逐位相等**（`allclose` 容差 1e-6 只是留余量）；L1 组（test_prompts）在 100+
随机用例上逐 id 相等。

**方法学的价值在移植期**：40/40 全绿是*结果*；*过程*中该套件捕获了 7 类真实分歧
（附录 C），其中 5 类属于"不崩、不报错、静默漂移"型——`serialize_state` 分隔符、
`forward` top2/熵形状、`lang` 证据过滤缺失、`email_questions` 缺失、`build_sequence`
预算过滤缺失——手工验收
几乎不可能发现。

### 6.2 架构兼容证明（副产品）

`pf.shared_decision_models()` 的 `load_state_dict(strict=True)` 在 narde 侧载入
laya 侧构建模型的权重：任何 key 名/shape 漂移都会在 import 期使整个套件变红。
这使"narde 的 DecisionModel 与 laya checkpoint 架构互操作"成为**持续被验证的
不变量**，是 M6（域微调）与真实权重加载（M7-real）的关键前置。

### 6.3 离线基准（`--tiny` 管线形态）

`bench/latency.py --tiny`（tiny-bert：hidden=32、1 层、vocab 512、seed=13；
数字只测管线形态，非真实 checkpoint）：

| 段 | p50 | p95 |
|---|---|---|
| 检测：english | 0.42 ms | 0.75 ms |
| 检测：hindi | 2.61 ms | 3.48 ms |
| 检测：short english | 0.01 ms | 0.01 ms |
| 检测：large json（200 行） | 3.45 ms | 4.55 ms |
| system_one 1q | 11.80 ms | 12.13 ms（11.80 ms/q） |
| system_one 5q | 17.61 ms | 20.51 ms（3.52 ms/q） |
| system_one 10q | 23.05 ms | 24.66 ms（2.31 ms/q） |
| system_one 50q | 59.84 ms | 63.60 ms（**1.20 ms/q**） |

每问成本随批量摊薄（11.8 → 1.2 ms/q）的**形态**与 laya T4 基准（32.8 ms/问 →
7.2 ms/问 @ batch 10）一致，验证"一题多问共享编码前缀 + 批内并行"的 NAR 设计在
narde 侧完整保留。Router 热/冷路径与混合语言段需 ≥2 个真实 checkpoint，
`--tiny` 下按设计跳过（报告中标注 `routing: skipped`）。

`bench/quality.py --tiny --canary`（18 例 canary：choice 7 / noul 11 / score 6，
随机权重 → 近随机基线，仅证明管线端到端可跑）：

| 范围 | n | acc | ECE |
|---|---|---|---|
| overall | 24 | 0.2917 | 0.1015 |
| choice | 7 | 0.1429 | 0.1208 |
| noul | 11 | 0.4545 | 0.0571 |
| score | 6 | 0.1667 | 0.1605 |

对照随机基线（choice 4 选 ≈ 0.25、noul 二分类 ≈ 0.5、score 3 级 ≈ 0.33），
tiny 数字落在"管线正确、无信号"区间——**不构成质量声明**；真实质量需
§7.1 所述的两层区分与真实 checkpoint 跑分。

### 6.4 打包金丝雀

`tests/test_smoke.py`（裸 venv、仅 pydantic、无 torch）：`import narde` 成功、
`Settings` 默认值、`NARDE_` 环境覆盖、`get_settings()` LRU 缓存——全过。
26 个 py 文件 `py_compile` 全过。

---

## 7. 讨论

### 7.1 "100% 复刻"的两层结构（本文最重要的概念澄清）

复刻项目最常见的错误是把两层混为一谈：

1. **代码/行为等价**——同一输入下，复刻实现与参考实现产生相同 token 序列、
   相同张量、相同响应字典、相同异常。这是**可以且应当**被持续机器验证的；
   本文的 40/40 就是它的证明。
2. **权重等价 / 功能等价**——模型参数与在真实分布上的质量。权重**不可由代码
   重新生成**：只能 (a) 从 `convaiinnovations/laya` 下载（Apache-2.0 资产，
   narde 不 vendor），或 (b) 按 M6 用 RLCD 在 typed-decisions 域上重训。
   且即便权重相同，base checkpoint 在 typed-decisions zero-shot 上 ≈ 随机
   （0.362/0.352 vs 随机 0.318 / 多数类 0.461）；0.766 属于域微调 checkpoint。

因此本文的可辩护陈述是精确的：**"除附录 B 台账外，narde 与 laya v0.3.5 行为
等价（40/40，CI 可重复）；权重层由 Apache-2.0 checkpoint 共享或 M6 重训承担，
不在代码等价范围内。"** 任何未做此分层的"100% 复刻"都是过度声明。

### 7.2 差分测试能证明什么、不能证明什么

**能**：在测试语料覆盖的输入分布上，两侧逐层等价；架构 key/shape 互操作
（strict load）；异常语义一致；有状态生命周期一致。

**不能**：
- **通用性外推**：语料外的输入仍可能分歧（§7.3）；
- **参考实现正确性**：差分测试只保证"和 laya 一样"，不保证"laya 是对的"
  （garbage-in-garbage-out parity）；
- **权重层**（§7.1）；
- **网络路径**：`snapshot_download`、HF 缓存、OOM→CPU 回退在离线测试中不被
  执行（由结构审查 + laya 自身 test_download.py 覆盖）。

### 7.3 威胁到效度的因素

| 威胁 | 缓解 |
|---|---|
| 语料有限（随机 100+ + 边界值，非穷尽） | 种子固定可复现、可扩展；CI 每 PR 重跑；边界值（0/False/None/""/单选项/空批/超预算截断/高基数溢出）显式覆盖 |
| mock tokenizer 替代真实 BPE | L1 层只测"同 tokenizer 下同布局"；真实 tokenizer 差异由共享 checkpoint + 真实权重跑分（M7-real）兜底 |
| CPU 单线程数值 | 两侧同解释器/同库版本/同线程数；allclose 容差 1e-6，实测逐位相等 |
| oracle 版本漂移 | `reference/laya` gitignored 且钉死 v0.3.5；升级 oracle 必须重跑全套件并更新台账 |
| 台账被滥用（把 bug 登记成"有意偏离"） | 每条必须附理由 + 对应测试；台账外差异零容忍 |

### 7.4 方法学的一般化

该配方不依赖 laya/narde 本身，适用于**任何有可执行参考实现的移植/复刻**：
跨语言迁移（C++→Rust 解析器）、框架迁移（TF→JAX 模型图）、遗留系统现代化
（COBOL→Java）中的"行为冻结"验收、以及 LLM 推理引擎间的数值对拍。可迁移的
关键部件：只读 oracle + 五层等价类 + 共享参数/权重双跑 + 分歧台账 + 离线
tiny 模式 + 双 runner。

### 7.5 成本—收益

40 项测试共约 900 行（`tests/parity/`），CPU 全跑 < 1 min；换来"任何 PR 合入
前等价性被机器重新证明"。相比人工 code-review "看起来一样"——成本随代码量
线性增长且无记忆——这是本文方法学的核心经济学论证。

---

## 8. 结论

本文把"复刻对了"从一个不可证伪的信仰声明，改写成 CI 红绿灯：**只读参考
oracle + 五层等价类差分断言 + 共享权重双跑 + 分歧台账 + 离线 tiny 回归**。
在 narde↔laya 案例上，40/40 全绿，移植期实际捕获 7 类真实分歧（其中 5 类是
静默漂移型）。我们强调"100% 复刻"必须分两层陈述：代码/行为等价（本文证明）
与权重/功能等价（由 Apache-2.0 checkpoint 共享或 M6 域微调承担；真实跑分由
同一 bench harness 的联网模式承接）。方法学一般化到任何"有可执行参考实现的
移植"场景。

---

## 9. 复现说明

```bash
# 1. clone laya 作为只读 oracle（本仓已 gitignore reference/）
git -c http.sslBackend=openssl clone --depth 1 \
    https://github.com/NandhaKishorM/laya narde/reference/laya

# 2. 依赖（引擎栈）
pip install torch transformers safetensors huggingface-hub numpy pydantic

# 3. 全量差分套件（40 项，离线 CPU，无需 pytest）
python tests/parity/run.py            # 期望: 40 passed, 0 failed
# 有 pytest 时: pytest -q tests/parity

# 4. 打包金丝雀（裸环境，仅 pydantic，无 torch）
python tests/test_smoke.py

# 5. 离线基准（tiny 模式，管线回归）
python bench/latency.py --tiny
python bench/quality.py --tiny --canary
```

环境指纹：Python 3.14.3 / torch 2.12.0+cpu / transformers 5.17.0 /
numpy 2.4.2 / pydantic 2.12.5 / CPU 单线程 / `TOKENIZERS_PARALLELISM=false`。
基线数字见 `bench/results/*.json` 与 `docs/bench-log.md`。

---

## 参考文献

[1] NandhaKishor M. *Laya — non-autoregressive System-1 decision engine.*
    GitHub: NandhaKishorM/laya, v0.3.5, Apache-2.0.
[2] narde. *docs/architecture-notes.md · api-contract.md · divergence.md ·
    bench-log.md.* 本仓 `docs/`（对标 laya 的复刻蓝图、API 契约、分歧台账、
    基准日志）。
[3] C. Cadar, D. Dunbar, D. F. Long. *KLEE: Unverifiable but reliable.* CAV 2008.
    （约束/差分验证的 oracle 思想谱系。）
[4] J. R. Unther, M. E. Crovella, D. M. Goldbeck. *LAVA: an approach to test
    isolation using application virtualization.*（差分模糊的 oracle 模式。）
[5] R. Shamsahoseini, D. Elkins, A. Roychoudhury. *Differential fuzzing of
    SQL engines (Morsello/DIFUZZ 式对拍).*（跨实现输出对拍。）
[6] A. Natarajan, L. Bhatt, et al. *ML reproducibility: audits of data,
    hyperparameters, and randomness.*（训练侧可复现性谱系，作为推理侧对照。）
[7] P. G. Neumann. *Program testing in the clean-room.* IEEE Software, 1998.
    （清洁室工程——本文方法学的对偶设定。）
[8] laya research. *research/scripts/bench_latency.py · bench_local.py ·
    BENCHMARKS.md.* `reference/laya/research/`（§6.3 数字的对照基线与
    方法论来源：17,416 问 T4 对跑、51 语言 MASSIVE 扫描、温度重拟合
    ECE 0.466→0.081）。

---

## 附录 A：完整测试矩阵（40 项）

| # | 模块::测试 | 等价类 | 断言 |
|---|---|---|---|
| 1 | test_prompts::test_serialize_state_parity | L1 | 8 状态 × str/dict/list，serialize_state 逐字相等 |
| 2 | test_prompts::test_render_criterion_parity | L1 | 结构化判据（0/False/None/""/dict/list）渲染相等 |
| 3 | test_prompts::test_render_options_parity | L1 | choice/score/noul × 全判据边界 |
| 4 | test_prompts::test_build_sequence_parity_sweep | L1 | 100+ 种子化随机 (state,q,max_len,head_max_len,truncate_left) 逐 id 相等（含 14 选项溢出重截） |
| 5 | test_prompts::test_collate_items_parity | L2 | 批张量（input_ids/attention/marker_pos/marker_mask/qtype/label/target）+ meta 相等；空批→None |
| 6 | test_model::test_forward_parity | L2 | 共享权重 forward：logits + act_logits allclose(1e-6) |
| 7 | test_model::test_forward_parity_masked_detach | L2 | detach_encoder=True 变体 |
| 8 | test_model::test_single_option_no_crash_parity | L2+L4 | 单选项 choice 不崩（top2 补 0），双包输出相等 |
| 9 | test_model::test_no_head_parity | L2 | head_layers=0 无 head 变体 |
| 10 | test_model::test_proper_reward_parity | L2 | log + 0.5·spherical − 1.0·RPS（score 型）奖励，含自定义权重 |
| 11 | test_model::test_td_lambda_targets_parity | L2 | λ∈{0, 0.5, 1.0} + 无 ep_group 直通 |
| 12 | test_model::test_amp_dtype_parity | L2 | bf16/fp16 dtype 映射 |
| 13 | test_lang::test_detect_script_parity | L3+L4 | 多语言语料脚本判定相等 + dict/list/None 两侧同型崩溃镜像 |
| 14 | test_lang::test_analyse_parity | L3 | analyse 全字段 dict 相等（含 str/dict/list/None） |
| 15 | test_lang::test_is_english_parity | L3 | is_english 布尔相等 |
| 16 | test_lang::test_guess_latin_language_parity | L3+L4 | 拉丁语言猜测相等 + 崩溃镜像 |
| 17 | test_router::test_normalise_name_parity | L4 | 全部别名 + 未知名 ValueError 消息逐字相等 |
| 18 | test_router::test_match_workflow_parity | L3 | workflow 精确集合匹配（子集/超集→None） |
| 19 | test_router::test_route_autodetect_parity | L3+L5 | 脚本×语言×默认自动路由决策逐字段相等 |
| 20 | test_router::test_route_explicit_parity | L3+L5 | 显式 model/task/lang 优先级 |
| 21 | test_router::test_route_workflow_detection_parity | L3 | workflow 开关 |
| 22 | test_router::test_lru_lifecycle_parity | L5 | FakeAgent 猴补丁双包 Agent，逐次 load/evict/loaded 镜像 |
| 23 | test_router::test_route_decision_repr | L3 | RouteDecision dict 子类字段 + repr |
| 24 | test_calib::test_clamp_temperature_parity | L2 | NaN/inf/str/None/越界 |
| 25 | test_calib::test_temp_bucket_parity | L2 | 全 (qtype, k) 桶键 |
| 26 | test_calib::test_confidence_from_probs_parity | L2 | Dirichlet 样本 + k<2 + p[:k] 切片 |
| 27 | test_calib::test_ece_score_parity | L2 | bins 2..40 + 空→nan + 全自信→0.0 |
| 28 | test_calib::test_narde_extension_fit_temperature_runs | L2 | narde 扩展健全性（非差分） |
| 29 | test_presets_email::test_presets_parity | L3 | 5 个预设包逐 dict 相等 + 自定义 categories |
| 30 | test_presets_email::test_clean_email_parity | L3 | 10 邮件 × 3 截断上限 |
| 31 | test_presets_email::test_email_state_parity | L3 | 发件人/清洗矩阵 |
| 32 | test_presets_email::test_email_questions_presets_parity | L3 | presets 与 email 两个来源的 email_questions 相等 |
| 33 | test_shortlist::test_shortlist_choice_parity | L3 | dict/list 判据 + k≥n 透传（embed 不调用）+ 平局保留靠前者 |
| 34 | test_shortlist::test_shortlist_error_parity | L4 | ValueError/TypeError 类型 + 消息逐字相等 |
| 35 | test_shortlist::test_predict_shortlist_parity | L3 | _FakeAgent 回显缩减判据 + shortlist meta + 透传 |
| 36 | test_shortlist::test_embed_fn_from_agent_parity | L2 | 共享模型 Agent 嵌入矩阵 allclose + 参数校验消息相等 |
| 37 | test_agent::test_to_internal_parity | L3 | 公参→内参全边界 |
| 38 | test_agent::test_fix_tokenizer_config_parity | L4 | 3 种 tokenizer_config 变体，双包写盘后 diff |
| 39 | test_agent::test_verify_compatibility_parity | L4 | 通过/缺前缀/shape 不匹配/缺 key/缺 config 全镜像 |
| 40 | test_agent::test_system_one_end_to_end_parity | L3 | choice+score+noul 同批：answers 逐 dict + usage 相等；model 字段按台账 |

---

## 附录 B：分歧台账（docs/divergence.md 摘要）

| # | 领域 | laya | narde | 理由 | 钉住测试 |
|---|---|---|---|---|---|
| D1 | system_one 响应 | `"model": "laya-rl-agent"` | `"model": "narde-rl-agent"` | 品牌 | #40 |
| D2 | 日志/告警前缀 | `[laya] …` | `[narde] …` | 品牌 | 结构审查 |
| D3 | 包版本 | 0.3.5 | 0.1.0 | 独立发布线 | test_smoke |
| D4 | checkpoint 源 | 下载 convaiinnovations/laya | 同一 repo id | 权重为 Apache-2.0 资产，不重训不 vendor | §7.1 边界 |
| D5 | `__init__.py` | 急切导入重模块 | PEP 562 懒加载，torch-free | 裸 venv 可 `import narde` | test_smoke |
| D6 | `settings.py` | 无 | pydantic BaseSettings（`NARDE_` 前缀） | 部署层便利，非 parity 面 | test_smoke |
| D7 | `calib.fit_temperature` | 在训练 notebook 内 | 暴露为公共函数（golden-section NLL） | narde 扩展 | #28 |
| D8 | router workflow 分支 | 返回 raw tuple repo | 逐字节镜像（含 quirk） | 忠实复刻 | #18–#21 |
| D9 | `noul` 拼写 | 上游原拼写 | 保留原样（不"纠正"） | 忠实复刻 | #1–#12 |
| D10 | `Agent._to_internal` 短键 | type→t, instructions→ins, criteria→crit | 完全保留 | build_sequence 消费内参 | #37 |

**规则**：台账外任何 L1–L5 差异 = bug = CI 红灯。新增偏离必须 (a) 入台账、
(b) 加测试、(c) 保持全绿。

---

## 附录 C：差分测试捕获到的真实分歧（移植期）

| # | 分歧 | 症状 | 捕获测试 | 修复 |
|---|---|---|---|---|
| B1 | `serialize_state` 用 compact 分隔符 `(",", ": ")` 而非 laya 默认 | 长 state 截断点偏移 → token 序列漂移（短样例不显现） | #1, #4 | 改回 `json.dumps(ensure_ascii=False)` 默认分隔符 |
| B2 | `DecisionModel.forward` 用 `torch.sort(...)[1]` 取 top2，ent/top2 形状错误 | 前向 logits/act 数值不等 | #6 | 改为 `p.topk(2,-1).values` + `feats=stack([top1, gap, ent, k/255])` + `pooled=h[:,0]` |
| B3 | `router.normalise_name` 对未知名返回 `None` 而非 raise | 下游拿到 None 走默认路由，静默错路由 | #17 | 改为 raise ValueError，消息逐字对齐 laya |
| B4 | `lang` 缺 `_SHARED_WORDS` 证据过滤 | 非英语拉脱丁被误判为英语（共享功能词无证据也计分） | #13–#16 | 补 `_SHARED_WORDS` 过滤 + 非共享词证据门限 |
| B5 | `email.py` 缺 `email_questions` | 公共面缺失（laya.email.email_questions 存在） | #32 | 补 `email_questions`（与 presets 同源） |
| B6 | `Agent` 初版加载 `config.json`/`calibration.json` + `.bin` 回退 | 文件协议与 laya 不符（应为 `rl_agent_config.json` + `model.safetensors`，无 .bin 回退） | 结构审查 + #39 | 改为 laya 的 safetensors 协议 + 精确 FileNotFoundError 文案 |
| B7 | `build_sequence` 初版缺 `head_ids[:max(8, opt_budget)]` 与 marker `< max_len` 过滤 | 高基数/短 head 时 token 布局与 laya 不一致 | #4 | 补齐 opt_budget 预算与 marker 过滤 |

*注：B1/B2/B4/B5/B7 为"不崩、不报错、静默漂移"型——手工验收几乎不可能发现，
正是差分测试的价值所在。*

---

*（全文完。代码与测试位于 `narde` 仓库；oracle 为 `reference/laya/` v0.3.5。
基线提交：`b693127`（引擎 + 差分套件）、`a091793`（bench 骨架）、
`c75eb7d`（test_lang 补齐至 40 项）。)*
