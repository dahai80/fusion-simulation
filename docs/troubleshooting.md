---
doc_id: fusion-simulation/troubleshooting
product: fusion-simulation
class: app
entry: service
doc_type: troubleshooting
title: { zh: "fusion-simulation 故障排查", en: "fusion-simulation troubleshooting" }
audience: engineer
status: draft
version: 0.1.12
updated: 2026-11-07
owner: "@dahai"
source_of_truth:
  - README.md
  - fusion_simulation/service/openai_api.py
  - fusion_simulation/gui/app.py
---

# 故障排查 / Troubleshooting

## 安装与导入

**`import pybullet` 失败或物理仿真不可用**
PyBullet 是可选依赖。执行 `pip install -e ".[physics]"` 或 `pip install pybullet`。不加 physics extra 时包本身可导入，仅物理仿真不可用——这是设计行为，不是缺陷。

**Python 版本报错**
要求 Python ≥ 3.12（`pyproject.toml` `requires-python`）。用 `python3.12 -m venv .venv` 重建环境。

**Homebrew 安装后 `fusion-sim version` 版本号不对**
v0.1.10 之前存在 formula 与 `__version__` 不同步问题（issue #7d57f11 已修复）；升级到 v0.1.12 或重装 `dahai80/test/fusion-simulation`。

## 服务启动

**服务拒绝启动，提示空 token**
`auth_enabled=True` 是默认值，认证开启且 token 为空时直接拒绝启动。设置非空 `FUSION_CLIENT_KEY` 后重启。

**内核启动时出现 key 缺失 WARNING**
未设置 `FUSION_CLIENT_KEY`（或旧的 `FUSION_MLX_API_KEY`）时输出 WARNING；`FUSION_PROFILE=prod` 下会直接失败。本地开发可忽略 WARNING，生产必须配置。

**端口被占用 / 端口冲突**
默认端口 11447（gRPC）、11456（metrics）、11455（GUI）、11434（OpenAI API）均已在工作区 `architecture/port-registry.yaml` 登记。冲突时先用 `--port` / `--metrics-port` / `--gui-port` 换端口，再到 registry 检查占用方；不要私自占用其他服务的注册端口。

**`service health` 返回异常**
确认服务进程存在（进程名应为 `fusion-simulation ...`）；检查目标端口是否绑定在 `127.0.0.1`。用 `--host 0.0.0.0` 对外暴露时必须同时保证认证开启且 token 非空。

## 推理与训练

**训练 / 推理请求连不上**
推理默认走 Fusion-Gateway `http://localhost:11432/v1`。先确认 fusion-gateway 与 fusion-mlx 已启动（`fusion-mlx` 的 `start.sh status`），再检查 `FUSION_MLX_URL` 是否被覆盖错。仓库策略是经网关访问，不直连 fusion-mlx 的 11434。

**train 输出带 `[SIMULATED]` 标记**
模拟 loss 回退仅在显式 `--allow-simulated` 时发生，且只应作为链路验证手段；真实训练需提供非空数据集（空数据集直接报错退出，退出码非 0）。

**`dataset import` 被拒绝**
名称包含 `/`、`..` 或前导点会被拒绝；路径必须解析到数据根目录之内。这是路径安全校验（防目录穿越），按规则改名重试即可。

**bench `--output` 报错**
输出必须位于报告目录（默认 `~/Library/Fusion/Simulation/reports`，可用 `FUSION_SIM_REPORTS_DIR` 覆盖）内；绝对路径越界或 `..` 穿越会被拒绝。

## REST API / Dashboard

**`/v1/chat/completions` 返回 forbidden**
消息携带的仿真命令不在白名单内。白名单：init / step / reset / pause / resume / load_scene / save_snapshot / restore_snapshot / add_sensor / add_agent / get_observations / status / close。命令 JSON 直接作为 user 消息内容发送（见 get-started 示例）。

**请求被拒并提示 body 过大**
请求体上限 4 MB；Content-Length 超限直接拒绝。相机帧等大负载走 gRPC bytes 通道，不走该 REST 端点。

**Dashboard WebSocket 连接失败**
连接需带 token（query `?token=` 或 `Authorization: Bearer`），且请求 Origin 的 netloc 必须与 Host 一致；浏览器跨源直连会被 Origin 校验拒绝，请从同一 Dashboard 页面内建立连接。

**Dashboard 页面出现未转义内容**
动态文本（智能体 / 场景 / 快照名）在写入 `innerHTML` 前已做 HTML 属性转义；若看到转义符原样显示，属于转义生效，不是 XSS。

## 测试与质量

```bash
pytest                              # 全量测试
pytest tests/test_e2e.py -v         # E2E（可靠性 / 稳定性 / GUI，72 项）
pytest --cov=fusion_simulation --cov-report=term-missing
ruff check .
ruff format --check .
```

`tests/test_e2e.py` 覆盖 72 项 E2E 用例；v0.1.9 基线为 592 项测试全绿、`ruff check` 与 `ruff format --check` 干净。本地测试失败时先排除环境因素（PyBullet 缺失、端口占用、gateway 未启动），再按失败用例定位。

## 仍未解决

- 在 `gh issue view <N> --repo dahai80/fusion-simulation` 查看已知问题，或新建 issue（英文描述，附 `fusion-sim version` 输出与最小复现步骤）。
- 审计背景见 README「Reliability & Determinism (v0.1.9)」一节引用的 audit 报告。
