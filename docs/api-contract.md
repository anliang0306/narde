# narde — API 契约（对标 Laya）

目标：让上游 Laya 的调用代码可以「几乎不改」地迁移到 narde。

## 1. 顶层导入（`import narde`）

```python
import narde
narde.Agent
narde.load
narde.Router
narde.RouteDecision
narde.DEFAULT_MODELS
narde.shortlist_choice
narde.predict_shortlist
narde.embed_fn_from_agent
narde.detect_language
narde.detect_script
narde.is_english
narde.clean_email_body
narde.email_questions
narde.guard_questions
narde.moderation_questions
narde.router_questions
narde.triage_questions
narde.proper_reward
narde.td_lambda_targets
narde.ece_score
narde.confidence_from_probs
narde.render_options
narde.QTYPES
narde.QTYPE_NAMES
narde.__version__
```

> 与 Laya 的差异：本包名 `narde`，函数/类名与 laya 保持一致以便平滑迁移。

## 2. `Agent`（对应 laya `agent.Agent`）

```python
agent = narde.Agent(
    model_id_or_path: str = "convaiinnovations/laya",
    device: str | None = None,      # None = 自动探测 (cuda → mps → cpu)
    token: str | None = None,       # HF token，可选
    subfolder: str | None = None,   # "multilingual" | "typed-decisions"
)
res = agent.predict(state, questions)   # 亦可用 agent.system_one(...)
agent.cfg          # dict，可读写 cfg["head_max_len"] 等
agent.tok          # HF tokenizer
agent.model        # DecisionModel
agent.device       # torch.device
```

### `predict(state, questions) -> dict`

| 字段 | 类型 | 说明 |
|---|---|---|
| `model` | str | 固定 `"narde-rl-agent"`（对标 laya 的 `"laya-rl-agent"`） |
| `answers` | dict[qid] | 每个问题的答案对象（见下） |
| `usage` | dict | `{"input_tokens": int, "output_tokens": 0}` |

`answers[qid]`：
- **choice**：`{"type":"choice","choice":label,"probabilities":{label:prob},"confidence":float,"action":{"act_probability":float}}`
- **score**：`{"type":"score","score":float,"legend":{str(i):criterion},"probabilities":{str(i):p},"confidence":float,"action":{...}}`
- **noul**：`{"type":"noul","noul":p_true,"confidence":float,"action":{"act_probability":float}}`

`confidence = 1 - H(p)/log(k)`，k 为有效选项数；k<2 时恒 1.0。

## 3. `Router`

```python
router = narde.Router(preload=True, device="cuda", max_loaded=1)
router.route(state, questions)          # -> RouteDecision (dict 子类)
router.predict(state, questions, model=None, task=None, lang=None)
router.preload(["english","multilingual"])
router.attach("english", agent)
router.unload()
router.loaded  # List[str]
```

`RouteDecision`：`.model`, `.repo`, `.reason`, `.detection`, `.workflow`；可 `dict(decision)` 序列化。

路由优先级：`model > task > auto_workflow(默认关) > lang > script/lang detect > default`。

## 4. `predict_shortlist`

```python
res = narde.predict_shortlist(agent, state, questions, embed_fn=..., k=20)
res["shortlist"][qid] = {"labels":[...], "scores":[...], "k":20, "n":77, "passthrough":bool}
```
- 非 choice 问题原样透传；choice 且 k ≥ 选项数则透传、不调 embed_fn。
- `embed_fn(texts: list[str]) -> (n, dim) ndarray`。

## 5. 纯函数（无模型依赖）

| 函数 | 签名 | 备注 |
|---|---|---|
| `serialize_state(state)` | `str\|dict\|list -> str` | dict/list → compact JSON |
| `render_options(q)` | `dict -> List[str]` | choice/score/noul 三种渲染 |
| `build_sequence(tok, state, q, max_len, head_max_len, option_order=None, truncate_left=False)` | `-> (ids, markers)` | 见 §6 |
| `proper_reward(q, target, qtype, mask, w_sph=0.5, w_rps=1.0, log_floor=-9.21)` | `tensor` | log + 0.5·spherical(− RPS for score) |
| `td_lambda_targets(p_true, batch, lam=1.0)` | `tensor` | 多轮 TD(λ) 目标 |
| `ece_score(conf, correct, bins=15)` | `float` | 期望校准误差 |
| `confidence_from_probs(p, k)` | `float` | 归一化熵 |
| `temp_bucket(qtype, k)` | `str` | `"choice:11+"` 等 |
| `clamp_temperature(t, lo=0.5, hi=5.0)` | `float` | 越界回退 1.0 |
| `detect_script(text)` / `analyse(state)` / `is_english(state)` | — | 见 lang 模块 |

## 6. Prompt 格式（字节级契约）

```
[CLS] {type} {instructions} [SEP] [MASK] opt0 [MASK] opt1 … [SEP] {state} [SEP]
```
- 每个选项前缀一个 `[MASK]`；单选项默认截断 48 token。
- 选项总长超 `head_max_len` 时按 `max(4, (head_max_len-16)//n)` 再截断。
- state 段按剩余预算截断；`truncate_left=True` 保留尾部。
- 全部 token 数不超过 `max_len`，超出的 marker 丢弃。

## 7. 与上游差异登记（divergence log）

> 任何偏离上游的地方都记在这里，附理由与影响。

| 日期 | 改动 | 理由 | 影响 |
|---|---|---|---|
| — | （初始，无改动） | | |
