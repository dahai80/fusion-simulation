---
doc_id: fusion-simulation/product-page
product: fusion-simulation
class: app
entry: service
doc_type: product-page
title: { zh: "fusion-simulation 产品页", en: "fusion-simulation product page" }
audience: engineer
status: draft
version: 0.1.11
updated: 2026-11-07
owner: @dahai
source_of_truth:
  - README.md
  - pyproject.toml
  - fusion_simulation/core/kernel.py
  - fusion_simulation/service/server.py
  - fusion_simulation/gui/app.py
---

# fusion-simulation

面向 Apple Silicon 的机器人虚拟仿真训练与测试平台。物理仿真使用 PyBullet（可选依赖），神经网络推理全部委托给 fusion-mlx / Fusion-Gateway 的 HTTP API，仓库内不直接导入 torch、CUDA 或 MLX。

## 产品能力

| 能力 | 说明 | 你不再需要做的事 | 约束 |
|------|------|------------------|------|
| 六层仿真内核（SimulationKernel） | 时钟、ECS、事件总线、世界状态、故障隔离、资源配额统一调度 | 不必自建仿真循环、实体系统与崩溃隔离逻辑 | 需 Python ≥ 3.12；MuJoCo 引擎目前为 stub |
| 传感器管线 | RGB / Depth / IMU / Contact / Semantic，注册表管理 + 频率控制 | 不必手写传感器采集与去噪代码 | 无 PyBullet 时仅传感元数据可用，相机渲染走 CPU tiny renderer |
| 多智能体 + 策略推理 | AgentManager + PolicyClient，经 Fusion-Gateway 调 fusion-mlx | 不必自行对接模型服务或管理路由头 | 默认模型 Qwen3.5-4B-bf16；推理经 11432 网关，不直连 11434 |
| 训练 / 评测 / Gym | BCTrainer 数据导出、SimulationEvaluator、FusionGymEnv（四 Manager 分解 MDP） | 不必自己导出数据集给 fusion-mlx 或封装 gym 接口 | 空数据集直接报错退出；模拟 loss 回退需显式 `--allow-simulated` 且输出标记 `[SIMULATED]` |
| gRPC + JSON-RPC 服务 | SimulationServer 提供 12 个 RPC 方法，含 HTTP 回退 | 不必为跨语言调用自写桥接层 | gRPC 默认绑定 127.0.0.1:11447 |
| OpenAI 兼容 REST API | `/v1/models`、`/v1/chat/completions`（SSE）、`/v1/health`，聊天消息可派发仿真命令 | 不必自建对话入口来控制仿真 | 默认 11434 端口；命令白名单之外一律拒绝；请求体上限 4 MB |
| 指标与健康 | MetricsServer 暴露 `/health`（JSON）与 `/metrics`（Prometheus 文本） | 不必自建监控采集端点 | 默认 127.0.0.1:11456 |
| Web Dashboard | 5 页面 GUI（欢迎页 / 工作台 / 智能体编排 / 数据录制 / 设置），FastAPI + 静态页 | 不必写前端即可视化仿真运行 | 默认 127.0.0.1:11455；WebSocket 有 token + Origin 校验 |

## 安全基线

- gRPC / Metrics / GUI / OpenAI API 默认全部绑定 `127.0.0.1`；绑定 `0.0.0.0` 必须显式传 `--host`，并确保 `auth_enabled` 且 token 非空。
- 认证默认开启（`auth_enabled=True`），空 token 拒绝启动；token 比较使用常数时间 `hmac.compare_digest`。
- SSRF 防御：`add_agent` 拒绝调用方提供的 `policy_endpoint` / `api_key`；URL scheme 白名单（http/https），阻断 link-local / loopback。
- 路径安全：数据集名称拒绝 `/`、`..` 与前导点；URDF 解析强制使用 `defusedxml`（XXE 防御）；bench 输出限制在报告目录内。
- 确定性：`core/rng.py` 全局播种 `random` + `numpy`，每回合使用派生 RNG，同种子可复现。

## 安装

```bash
# From source (Apple Silicon, Python 3.12+)
git clone https://github.com/dahai80/fusion-simulation.git
cd fusion-simulation
python -m venv .venv
source .venv/bin/activate
pip install -e ".[test,physics]"

# Or via Homebrew (published release)
brew install dahai80/test/fusion-simulation
```

物理引擎 PyBullet 为可选依赖，不加 `physics` extra 时包可正常导入，仅物理仿真不可用。

## 服务端口

| 端口 | 服务 |
|------|------|
| 11447 | gRPC + JSON-RPC（SimulationServer） |
| 11456 | MetricsServer（`/health`、`/metrics`） |
| 11455 | Web Dashboard（`--gui`） |
| 11434 | 本仓库自带的 OpenAI 兼容 REST API（注意与 fusion-mlx 的 11434 是各自独立的服务） |
| 11432 | Fusion-Gateway（推理出口，属 fusion-gateway） |

## 文档索引

- 快速上手：[get-started.md](get-started.md)
- API 参考：[api-reference.md](api-reference.md)
- 配置参考：[config-reference.md](config-reference.md)
- 架构说明：[architecture.md](architecture.md)
- 故障排查：[troubleshooting.md](troubleshooting.md)
- 变更记录：[CHANGELOG.md](../CHANGELOG.md)
