---
doc_id: fusion-simulation/api-reference
product: fusion-simulation
class: app
entry: service
doc_type: api-reference
title: { zh: "fusion-simulation API 参考", en: "fusion-simulation API reference" }
audience: engineer
status: draft
version: 0.1.11
updated: 2026-11-07
owner: @dahai
source_of_truth:
  - fusion_simulation/service/server.py
  - fusion_simulation/service/openai_api.py
  - fusion_simulation/service/metrics_server.py
  - fusion_simulation/gui/app.py
---

# API 参考 / API Reference

fusion-simulation 对外暴露四类接口：gRPC + JSON-RPC、OpenAI 兼容 REST API、Metrics HTTP、Dashboard REST/WebSocket。全部默认绑定 `127.0.0.1`。

## gRPC + JSON-RPC（默认 11447）

`SimulationServer.handle_request(method, params)` 为统一入口，HTTP JSON-RPC 作为 gRPC 的回退通道。RPC 方法共 12 个，测试覆盖见 `tests/test_e2e.py`（Server RPC 10 项）：

| 方法 | 说明 |
|------|------|
| `init` | 初始化内核 |
| `step` | 按 `num_steps` 推进仿真 |
| `reset` | 重置仿真状态 |
| `pause` / `resume` | 暂停 / 恢复 |
| `load_scene` | 加载场景 |
| `save_snapshot` / `restore_snapshot` | 快照保存 / 恢复（真实 `pybullet.saveState`，带 LRU 上限与淘汰） |
| `add_sensor` / `add_agent` | 新增传感器 / 智能体（服务端做 SSRF 过滤，拒绝调用方传入 endpoint/key） |
| `get_observations` | 读取观测 |
| `status` | 查询状态 |
| `close` | 关闭内核并释放资源 |

未知方法返回错误；相机帧以 raw RGB/depth bytes（≤4 MB，gRPC 上限内）编码，不再走 `json.dumps` 文本编码。

## OpenAI 兼容 REST API（默认 11434）

| 端点 | 方法 | 说明 |
|------|------|------|
| `/v1/models` | GET | 模型列表 |
| `/v1/chat/completions` | POST | 聊天补全，支持 SSE 流式；消息体可携带仿真命令 |
| `/v1/health` | GET | 健康检查 |

```bash
curl -X POST http://localhost:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"fusion-sim","messages":[{"role":"user","content":"{\"command\":\"status\"}"}]}'
```

约束：
- 认证默认开启（Bearer token，常数时间比较）；`auth_enabled=True` 且 token 为空时拒绝启动。
- 命令白名单：init / step / reset / pause / resume / load_scene / save_snapshot / restore_snapshot / add_sensor / add_agent / get_observations / status / close；未列出命令返回 `{"error":"command not allowed","error_code":"forbidden"}`。
- 请求体上限 4 MB；Content-Length 超限直接拒绝。
- 该服务与本仓库之外的 fusion-mlx（同为 11434）相互独立，端口注册见 `architecture/port-registry.yaml`。

## MetricsServer（默认 11456）

| 端点 | 方法 | 说明 |
|------|------|------|
| `/health` | GET | JSON 健康状态 |
| `/metrics` | GET | Prometheus 文本格式指标 |

`MetricsCollector` 提供线程安全的 counters / gauges / histograms。

## Dashboard REST + WebSocket（默认 11455）

- REST：`/api/*`（智能体、场景、快照等 CRUD），静态页面由同一 FastAPI 应用提供。
- WebSocket：`/ws/events` 事件流。连接需通过 token 校验（query `?token=` 或 `Authorization: Bearer`），并做 Origin 校验（Origin netloc 必须与 Host 一致）；命令同样受白名单限制。

## 推理出口（对 fusion-gateway 的依赖）

PolicyClient / BCTrainer 的推理请求发往 Fusion-Gateway `http://localhost:11432/v1`（`FUSION_MLX_URL` 可覆盖），携带 `X-Fusion-Route: mlx` 与 `Authorization: Bearer ${FUSION_CLIENT_KEY}`（经 fusion-core `client_key()` 解析）。不直连 fusion-mlx 的 11434。
