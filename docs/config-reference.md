---
doc_id: fusion-simulation/config-reference
product: fusion-simulation
class: app
entry: service
doc_type: config-reference
title: { zh: "fusion-simulation 配置参考", en: "fusion-simulation config reference" }
audience: engineer
status: draft
version: 0.1.11
updated: 2026-11-07
owner: @dahai
source_of_truth:
  - README.md
  - fusion_simulation/cli/__init__.py
  - fusion_simulation/gui/app.py
---

# 配置参考 / Config Reference

## 环境变量

| 变量 | 用途 | 默认值 |
|------|------|--------|
| `FUSION_MLX_URL` | Fusion-Gateway 基址（代理 fusion-mlx），`KernelConfig` / `AgentConfig` / `GUIConfig` / CLI `--mlx-url` / `BCTrainer` 读取 | `http://localhost:11432/v1` |
| `FUSION_CLIENT_KEY` | fusion-mlx Bearer 认证 key，经 fusion-core `client_key()` 单点解析；旧名 `FUSION_MLX_API_KEY` 仍作为回退并告警 | 空——内核启动时输出 WARNING；`FUSION_PROFILE=prod` 时直接失败 |
| `FUSION_MLX_MODEL` | 默认模型名，`AgentConfig` / `PolicyClient` / `BCTrainer` / GUI 读取 | `Qwen3.5-4B-bf16` |
| `FUSION_MLX_ROUTE` | `X-Fusion-Route` 路由头 | `mlx` |
| `FUSION_SIM_REPORTS_DIR` | bench 报告目录覆盖（默认 `~/Library/Fusion/Simulation/reports`） | 见左 |
| `FUSION_PROFILE` | 运行档位；`prod` 时强制要求非空 `FUSION_CLIENT_KEY` | 未设置 |

## 服务端口与启动参数

| 参数 / 端口 | 说明 | 默认 |
|-------------|------|------|
| `service start --port` | gRPC + JSON-RPC 监听端口 | 11447 |
| `service start --metrics-port` | MetricsServer 端口（`/health`、`/metrics`） | 11456 |
| `service start --openai-api-port` | OpenAI 兼容 REST API 端口 | 11434 |
| `service start --gui --gui-port` | Web Dashboard 端口 | 11455 |
| `--host` | 绑定地址；默认全部 `127.0.0.1`，绑定 `0.0.0.0` 必须显式传入并保证认证开启 | 127.0.0.1 |

端口注册的唯一事实来源是 `architecture/port-registry.yaml`（仓库外，fusion 工作区根）：11447 / 11455 / 11456 均已登记。

## CLI 顶层参数

| 参数 | 说明 |
|------|------|
| `--mlx-url` | 覆盖 Fusion-Gateway 基址 |
| `--api-key` | 覆盖 `FUSION_CLIENT_KEY` |
| `-v, --verbose` | 调试日志 |

## 主要子命令

| 子命令 | 关键参数 |
|--------|----------|
| `env init` | `--engine`（lerobot 等）、`--scene`、`--headless`、`--physics-dt`、`--render-dt` |
| `scene load` | `--name`、`--builtin` |
| `agent spawn` | `--name`（必填）、`--entity-id`、`--role`、`--action-dim` |
| `sensor add` | `--type`（rgb_camera / imu 等）、`--name` |
| `snapshot save / restore` | `--name` |
| `dataset import` | `--name`、`--path`；名称拒绝 `/`、`..`、前导点 |
| `train` | `--dataset`、`--model-name`、`--epochs`、`--use-kernel`、`--allow-simulated` |
| `test` | `--model`、`--engine`、`--episodes` |
| `bench` | `--model`、`--output`；输出必须位于报告目录内 |
| `service start / stop / health` | 见上表 |
| `gateway register` | `--gateway-url` |
| `kernel run / status` | `--steps`、`--headless` |

## 认证配置

- `auth_enabled` 默认 `True`；认证开启且 token 为空时服务拒绝启动。
- 所有推理请求带 `Authorization: Bearer`（key 来自 `FUSION_CLIENT_KEY`）与 `X-Fusion-Route: mlx`。
- Dashboard WebSocket：`?token=` 或 `Authorization: Bearer` 之一即可，另做 Origin 校验。
