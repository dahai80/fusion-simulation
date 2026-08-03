# Fusion-Simulation

面向 **Apple Silicon** 的机器人虚拟仿真训练与测试平台。

使用 PyBullet 进行物理仿真，所有神经网络推理通过 [fusion-mlx](https://github.com/fusion-mlx) HTTP API 完成 —— **绝不直接导入 torch、CUDA 或 MLX**。

> 📖 [English Documentation](README.md)

## 架构

受 NVIDIA Isaac Sim 启发的六层架构：

```
L1  Core        SimClock · ECS · EventBus · WorldState · SimulationKernel · FaultIsolation · Plugin · ResourceQuota
L2  Physics     PyBulletEngine (解耦步进) · MuJoCoEngine (stub)
L3  Sensor      SensorManager · SensorRegistry · RGB / Depth / IMU / Contact / Semantic
L4  Agent       AgentManager · PolicyClient (fusion-mlx HTTP) · PromptScheduler · Decimation · JointController · TaskTemplates
L5  Train/Eval  BCTrainer (数据导出) · SimulationEvaluator · FusionGymEnv
L6  Service     gRPC + JSON-RPC · OpenAI 兼容 REST API · MetricsServer · GatewayClient
```

### SimulationKernel

集成所有子系统的核心调度器：

```python
from fusion_simulation.core.kernel import SimulationKernel, KernelConfig

kernel = SimulationKernel(KernelConfig(headless=True))
kernel.init()
state = kernel.step(num_steps=100)
print(state.sim_time, state.frame_count)
kernel.close()
```

### ECS (实体-组件-系统)

所有仿真实体均基于组件：

- **Transform** — 位置、朝向
- **RigidBody** — 质量、摩擦力
- **Articulation** — 关节位置/速度
- **AgentBind** — 绑定的智能体名称
- **CameraSensor / IMUSensor** — 传感器元数据

```python
from fusion_simulation.core.ecs import EntityManager, Transform, RigidBody

mgr = EntityManager()
eid = mgr.create_entity()
mgr.add_component(eid, Transform(position=[0, 0, 1]))
mgr.add_component(eid, RigidBody(mass=1.0))
```

### SensorManager

基于注册表的传感器管理，支持更新频率控制：

```python
from fusion_simulation.sensor.manager import SensorManager
from fusion_simulation.sensor.base import SensorConfig, SensorType

sm = SensorManager()
sm.add_sensor(SensorConfig(sensor_type=SensorType.RGB_CAMERA, name="wrist_cam"))
observations = sm.get_observations()
```

### AgentManager + PolicyClient

多智能体管理，支持降采样和 fusion-mlx 推理：

```python
from fusion_simulation.agent.manager import AgentManager
from fusion_simulation.agent.config import AgentConfig, AgentRole

am = AgentManager()
am.add_agent(
    AgentConfig(
        name="robot0",
        role=AgentRole.ROBOT,
        action_dim=6,
        decimation=4,
        policy_endpoint="http://localhost:11434/v1/chat/completions",
    )
)
```

### FusionGymEnv

对齐 Isaac Lab 架构的 Manager-Based RL 环境：

```python
import numpy as np
from fusion_simulation.train.gym_env import FusionGymEnv, RewardManager

env = FusionGymEnv(max_steps=500, decimation=4)
env._reward_mgr.add_reward_fn("reach", lambda obs, info: 1.0)

obs, info = env.reset()
for _ in range(100):
    obs, reward, terminated, timed_out, info = env.step(np.zeros(6))
    if terminated or timed_out:
        break
env.close()
```

四个管理器实现清晰的 MDP 分解：
- **ObservationManager** — 分组观测（策略/评论）、噪声、历史
- **ActionManager** — 处理 + 应用分离、降采样感知、缩放/裁剪
- **RewardManager** — 可组合奖励函数、累积跟踪
- **TerminationManager** — 终止 + 超时条件

### SimulationServer

gRPC 服务 + HTTP JSON-RPC 回退、MetricsServer、Fusion-Gateway 集成：

```python
from fusion_simulation.service.server import SimulationServer
from fusion_simulation.service.gateway_client import GatewayConfig
from fusion_simulation.service.metrics_server import MetricsConfig

server = SimulationServer(
    gateway_config=GatewayConfig(gateway_url="http://localhost:11432", enabled=True),
    metrics_config=MetricsConfig(port=11456),
)
server.handle_request("init", {})
server.handle_request("step", {"num_steps": 10})
server.handle_request("status", {})
server.handle_request("close", {})
```

### OpenAI 兼容 REST API

默认在端口 11434 暴露 `/v1/models`、`/v1/chat/completions`（SSE 流式）、`/v1/health`。支持 Bearer Token 认证，可从聊天消息中分发仿真命令：

```bash
# 列出模型
curl http://localhost:11434/v1/models

# 聊天补全（分发仿真命令）
curl -X POST http://localhost:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"fusion-sim","messages":[{"role":"user","content":"{\"command\":\"status\"}"}]}'
```

### MetricsServer + GatewayClient

- **MetricsServer** 在端口 11456 暴露 `/health`（JSON）和 `/metrics`（Prometheus 文本格式）
- **GatewayClient** 向 Fusion-Gateway（:11432）注册、发送心跳、关闭时注销
- **MetricsCollector** 提供线程安全的计数器、仪表盘和直方图

## 安装

### 从源码安装

```bash
git clone <repo-url>
cd fusion-simulation
python -m venv .venv
source .venv/bin/activate
pip install -e ".[test]"
```

### 通过 Homebrew (Apple Silicon)

```bash
brew install --formula Formula/fusion-simulation.rb
```

## CLI

```bash
# 版本
fusion-sim version

# 环境
fusion-sim env init --engine=lerobot --headless

# 场景
fusion-sim scene list
fusion-sim scene load --name=pick

# 智能体
fusion-sim agent spawn --name=robot0 --role=robot --action-dim=6
fusion-sim agent list
fusion-sim agent destroy --name=robot0

# 传感器
fusion-sim sensor add --type=rgb_camera --name=wrist_cam
fusion-sim sensor add --type=imu --name=imu0
fusion-sim sensor list

# 快照
fusion-sim snapshot save --name=checkpoint_1
fusion-sim snapshot restore --name=checkpoint_1

# 数据集
fusion-sim dataset list
fusion-sim dataset import --name=mydata --path=/path/to/data

# 训练
fusion-sim train --dataset=mydata --model-name=policy --epochs=10
fusion-sim train --dataset=mydata --use-kernel

# 评估
fusion-sim test --model=policy --engine=lerobot --episodes=5

# 基准测试
fusion-sim bench --model=policy --output=report.md

# 服务
fusion-sim service start --port=11447 --metrics-port=11456 --openai-api-port=11434 --gui
fusion-sim service stop
fusion-sim service health

# 网关
fusion-sim gateway register --gateway-url=http://localhost:11432

# 内核直连
fusion-sim kernel run --steps=100 --headless
fusion-sim kernel status
```

## Web 仪表盘 GUI

按 PRD 第 7 节实现的 5 页 Web 仪表盘，由 FastAPI 在端口 11455 提供服务：

```bash
fusion-sim service start --gui --gui-port 11455
# 打开 http://localhost:11455
```

| 页面 | 功能 |
|------|------|
| Welcome | 环境自动检测（PyBullet/gRPC/MLX/Service）、快速模板 |
| Workstation | 4 区布局：侧边栏 + 视口 + 检查器 + 状态栏、传输控制、实时指标 |
| Agent Orchestration | 智能体增删改查、提示词编辑器、角色配置 |
| Data & Recording | 快照保存/恢复、导出按钮 |
| Settings | 内核/AI/服务配置、环境检查 |

API：REST（`/api/*`）+ WebSocket（`/ws/events`）+ 静态文件。

## 测试

```bash
# 全部测试
pytest

# 单个文件
pytest tests/test_core.py
pytest tests/test_new_arch.py
pytest tests/test_e2e.py

# 仅 E2E 测试（可靠性、稳定性、GUI）
pytest tests/test_e2e.py -v

# 带覆盖率
pytest --cov=fusion_simulation --cov-report=term-missing

# 代码检查
ruff check .
ruff format --check .
```

### E2E 测试覆盖

`tests/test_e2e.py` 覆盖端到端可靠性、稳定性和 GUI 稳定性，共 72 个测试：

| 类别 | 测试数 | 覆盖范围 |
|------|--------|----------|
| Kernel 生命周期 | 8 | 完整生命周期（init→step→pause→resume→stop→close）、重复循环、重置、快照 |
| Server RPC | 10 | 全部 12 个 RPC 方法、未知方法错误、有无内核的健康检查 |
| Metrics HTTP | 10 | /health + /metrics 端点、Prometheus 格式、线程安全、真实 HTTP |
| Gateway Client | 4 | 禁用网关、不可达网关、健康提供者、关闭注销 |
| SimulationEnv | 5 | Init/step/reset/close、重复循环、场景、未初始化错误 |
| FusionGymEnv | 6 | Reset/step/close、观测/动作空间、超时、奖励、终止 |
| Sensor+Agent 流水线 | 5 | 多传感器、多智能体、启用/禁用、带事件的完整流水线 |
| 稳定性/压力 | 7 | 10x 循环、500 步、并发 RPC、100 传感器、50 智能体、1K 实体、10K 事件 |
| GUI 稳定性 | 7 | 健康端点结构、Prometheus 格式、404、事件流、服务启停 |
| Gym 管理器 | 8 | ObservationManager、ActionManager、RewardManager、TerminationManager |
| CLI | 4 | version、scene list、agent spawn、sensor add |

## 关键约束

- 所有模型推理通过 fusion-mlx HTTP API（`http://localhost:11434/v1`）
- 禁止直接导入 torch/CUDA/MLX
- PyBullet 可选 — 代码优雅处理 `ImportError`
- Python ≥3.12
- 默认模型：`qwen3.5-9b`（通过 fusion-mlx）

## 项目结构

```
fusion_simulation/
├── core/           # L1: Kernel, Clock, ECS, EventBus, WorldState, FaultIsolation, Plugin, ResourceQuota
│   ├── clock.py
│   ├── ecs.py
│   ├── event_bus.py
│   ├── fault.py
│   ├── kernel.py
│   ├── plugin.py
│   ├── resource.py
│   └── world_state.py
├── physics/        # L2: 物理引擎抽象
│   ├── base.py
│   ├── mujoco_engine.py
│   └── pybullet_engine.py
├── sensor/         # L3: 传感器管理
│   ├── base.py
│   ├── contact.py
│   ├── depth_camera.py
│   ├── imu.py
│   ├── manager.py
│   ├── rgb_camera.py
│   └── semantic_camera.py
├── agent/          # L4: 智能体 + 策略
│   ├── config.py
│   ├── joint_controller.py
│   ├── manager.py
│   ├── policy.py
│   ├── scheduler.py
│   └── task_templates.py
├── sim/            # 向后兼容环境包装器 + 场景格式
│   ├── env.py
│   ├── scene.py
│   └── scene_formats/
│       ├── json_loader.py
│       └── urdf_loader.py
├── train/          # L5: 训练
│   ├── gym_env.py
│   └── trainer.py
├── eval/           # L5: 评估
│   └── evaluator.py
├── dataset/        # 数据集管理 + 数据采集器
│   ├── collector.py
│   └── manager.py
├── service/        # L6: gRPC + HTTP + OpenAI API 服务
│   ├── config.py
│   ├── flatbuf/
│   │   └── serializer.py
│   ├── gateway_client.py
│   ├── metrics_server.py
│   ├── openai_api.py
│   ├── proto/
│   │   ├── simulation_pb2.py
│   │   └── simulation_pb2_grpc.py
│   └── server.py
├── render/         # 渲染引擎
│   ├── base.py
│   └── pybullet_render.py
├── gui/            # Web 仪表盘 (FastAPI + 静态 HTML)
│   ├── app.py
│   ├── ecs_compat.py
│   └── static/
│       └── index.html
├── bench.py
└── cli/            # 命令行接口
    └── __init__.py
```

## 许可证

[Apache License 2.0](LICENSE)
