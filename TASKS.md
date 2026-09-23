# narde — 开发任务拆解

## 里程碑总览

| M | 内容 | 状态 | 完成标准 |
|---|------|------|----------|
| M0 | 脚手架 + AGENTS.md + 占位测试 | ✅ | 打包冒烟通过 |
| M1 | `prompts.py` — 序列构造，与 laya 逐 token 对齐 | ✅ | 差分 100+ 用例全一致 |
| M2 | `model.py` — DecisionModel forward | ✅ | logits/act 与 laya 全一致 |
| M3 | `router.py` + `lang.py` | ✅ | 路由/别名/LRU/多语言全一致 |
| M4 | `calib.py` 温度 + ECE | ✅ | clamp/ECE/confidence 与 laya 一致 |
| M5 | `shortlist.py` + `presets.py` + `email.py` | ✅ | 高基数/预设/邮件与上游一致 |
| M5.5 | 差分测试套件 `tests/parity/` | ✅ | 36 项全绿，CI 可重复 |
| M6（可选） | RLCD 域微调（需 HF 权重 + GPU） | ⬜ | typed-decisions 域 ≥ 0.70 acc |
| M7 | bench 骨架 + 文档收尾 | ✅ | latency/quality 双 harness（`--tiny` 离线可跑）+ bench-log |

## M0 — 脚手架（✅）

- 项目骨架、pyproject.toml、AGENTS.md、.gitignore、NOTICE.md
- `src/narde/settings.py` + 懒加载 `__init__.py` + 打包冒烟 `tests/test_smoke.py`
- clone laya → `reference/laya/`（read-only oracle，gitignored）
- `docs/architecture-notes.md` + `docs/api-contract.md` + `docs/divergence.md`

## M1–M5 — 忠实移植（✅ 差分验证通过）

所有 narde 模块均为 laya v0.3.5 的 1:1 移植（见 `NOTICE.md` 逐文件归属）：

- **M1 prompts**：`serialize_state` / `render_options` / `render_criterion` /
  `build_sequence` / `collate_items` / `QTYPES`。48-token option 上限、溢出重截、
  `truncate_left` 保留尾部，全部与 laya 逐字节一致。
- **M2 model**：`DecisionModel` forward（含单选项 top2 补 0 的 edge case）、
  `build_model`、`proper_reward`、`td_lambda_targets`、`amp_dtype`。
- **M3 router + lang**：`normalise_name`（未知名 raise）、`_ALIASES`、
  `_TYPED_DECISION_WORKFLOWS` 精确集合匹配、LRU 生命周期、`RouteDecision`；
  `detect_script` / `analyse` / `guess_latin_language`（含 `_SHARED_WORDS` 证据过滤）。
- **M4 calib**：`clamp_temperature` / `ece_score` / `confidence_from_probs` /
  `temp_bucket`（narde 扩展：`fit_temperature`）。
- **M5 shortlist + presets + email**：`shortlist_choice` / `predict_shortlist` /
  `embed_fn_from_agent`；5 个预设问题包；`clean_email_body` / `email_state` /
  `email_questions`。

## M5.5 — 差分测试套件（✅ 36/36 全绿）

"100% 复刻"的可执行定义：同输入下 laya 与 narde 输出必须相等
（token 列表精确相等、张量 allclose、dict 相等；权重不重新生成）。

```
tests/parity/
  pf.py                 # 共享 fixtures（MockTokenizer/BatchTok、tiny_bert、
                        # shared_decision_models、确定性语料、expect_raises）
  conftest.py           # pytest 路径引导
  run.py                # 独立 runner（无需 pytest，离线/沙箱可用）
  test_prompts.py       # 5
  test_model.py         # 6（含 shared-weights 前向一致性）
  test_router.py        # 8（含 LRU 生命周期 + 错误消息一致性）
  test_calib.py         # 5
  test_presets_email.py # 4（含 10×3 邮件清洗矩阵）
  test_shortlist.py     # 4
  test_agent.py         # 4（含 system_one 端到端）
```

`shared_decision_models()` 用同一份 `load_state_dict(strict=True)` 权重喂两个
DecisionModel，既验证前向一致，又验证架构 key/shape 兼容。

## M6（可选）— 域微调 ⬜

- 需 HF 权重下载（`convaiinnovations/laya`）+ GPU
- RLCD 训练循环 + 温度重拟合；typed-decisions test 400 cases ≥ 0.70 acc

## M7 — Bench 骨架 + 文档收尾（✅ 离线部分完成）

- [x] `bench/common.py` — 共享工具（env_meta / timed / 报告写出 / tiny 离线 Agent / 双语样例 state）
- [x] `bench/latency.py` — 检测开销、单/批量 system_one、Router 热/冷路径、混合语言负载
- [x] `bench/quality.py` + `bench/canary.jsonl` — 内置 18 例 canary 集，acc / macro-F1 / ECE / brier / nll
- [x] `docs/bench-log.md` 记录首次 tiny 基线；README 补 Benchmarks 章节
- [ ] 真实 checkpoint 跑分（需 `--model convaiinnovations/laya`，联网 + 下载权重）

## 提交规范

- conventional commit：`feat(model): ...`、`test(parity): ...`、`docs: ...`
- 每个 commit 前跑 `python tests/parity/run.py`（有 pytest 时 `pytest -q`）
- `import narde` 必须保持 torch-free（`tests/test_smoke.py` 守门）
