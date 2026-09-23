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
| M5.5 | 差分测试套件 `tests/parity/` | ✅ | 40 项全绿，CI 强制（`.github/workflows/ci.yml`） |
| M6（可选） | RLCD 域微调（需 HF 权重 + GPU） | ⬜ | typed-decisions 域 ≥ 0.70 acc |
| M7 | bench 骨架 + 文档收尾 | ✅ | latency/quality 双 harness（`--tiny` 离线可跑）+ bench-log |
| M8 | CI 门禁 + 许可/发布卫生 | ✅ | GitHub Actions 双 job 全绿；Apache-2.0 与上游一致 |

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

## M5.5 — 差分测试套件（✅ 40/40 全绿）

"100% 复刻"的可执行定义：同输入下 laya 与 narde 输出必须相等
（token 列表精确相等、张量 allclose、dict 相等；权重不重新生成）。

```
tests/parity/
  pf.py                 # 共享 fixtures（MockTokenizer/BatchTok、tiny_bert、
                        # shared_decision_models、确定性语料、expect_raises）
  conftest.py           # pytest 路径引导
  run.py                # 独立 runner（无需 pytest，离线/沙箱可用）
  test_prompts.py       # 5
  test_model.py         # 7（含 shared-weights 前向一致性）
  test_lang.py          # 4（多语言语料 + 同型异常镜像）
  test_router.py        # 7（含 LRU 生命周期 + 错误消息一致性）
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
- [x] `PAPER.md` — 研究论文（差分复刻方法学 + narde/laya 案例，arXiv 风格，中文）
- [ ] 真实 checkpoint 跑分（需 `--model convaiinnovations/laya`，联网 + 下载权重）

## 提交规范

- conventional commit：`feat(model): ...`、`test(parity): ...`、`docs: ...`
- 每个 commit 前跑 `python tests/parity/run.py`（有 pytest 时 `pytest -q`）
- `import narde` 必须保持 torch-free（`tests/test_smoke.py` 守门）
- 推送后 `.github/workflows/ci.yml` 会独立复验以上两条，红灯即阻断

## M8 — CI 门禁 + 许可/发布卫生（✅）

"CI 可强制"从论文里的一句话变成真实门禁：

- [x] `.github/workflows/ci.yml` — 两个刻意分离的 job：
  - **`parity`**：CPU 引擎栈 + laya oracle，**按 commit SHA 钉死** `4170897`
    （= v0.3.5；上游不发 tag，浮动 oracle 会让"40/40 全绿"失去意义）。
    CI 断言取回的 SHA 与 pin 相符、且树内 `version = "0.3.5"`；
    **两个 runner 都跑**（`tests/parity/run.py` + `pytest`）。
  - **`smoke`**：裸 venv（仅 pydantic）× Python 3.10 / 3.12，并且
    **断言 `torch` 不存在**（而不是只 import narde）——这条分离才是真正要守的不变量。
- [x] `LICENSE` — Apache-2.0，与上游 laya **逐字节一致**（sha256 相同）
- [x] `pyproject.toml` — license 由 `MIT` 修正为 `Apache-2.0`（原先与 `NOTICE.md`
  自相矛盾）、补齐真实项目 URL 与 classifiers
- [x] `tests/test_smoke.py` — 原先只有 pytest 风格函数、**无 `__main__`**，因此
  `python tests/test_smoke.py` 会"什么都没跑"却 exit 0（假的绿灯）；现改为无 fixture
  依赖、可独立运行，与 `tests/parity/run.py` 对称
- [x] `.gitattributes` — yml/py/sh/toml/md/json 强制 LF。CRLF 进入 GitHub Actions 的
  `run:` 块会以字面 `\r` 抵达 bash，破坏 heredoc；这条规则把这个失败模式彻底移除
