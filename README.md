# Fusion-Simulation

Robot virtual simulation training and testing platform for **Apple Silicon**.

Uses PyBullet for physics simulation and delegates all neural network inference to [fusion-mlx](https://github.com/fusion-mlx) via HTTP API — **never imports torch, CUDA, or MLX directly**.

> 📖 [中文文档](README_CN.md)

## Architecture

Six-layer architecture inspired by NVIDIA Isaac Sim:

```
L1  Core        SimClock · ECS · EventBus · WorldState · SimulationKernel · FaultIsolation · Plugin · ResourceQuota
L2  Physics     PyBulletEngine (decoupled stepping) · MuJoCoEngine (stub)
L3  Sensor      SensorManager · SensorRegistry · RGB / Depth / IMU / Contact / Semantic
L4  Agent       AgentManager · PolicyClient (fusion-mlx HTTP) · PromptScheduler · Decimation · JointController · TaskTemplates
L5  Train/Eval  BCTrainer (data export) · SimulationEvaluator · FusionGymEnv
L6  Service     gRPC + JSON-RPC · OpenAI-compatible REST API · MetricsServer · GatewayClient
```

### SimulationKernel

Central scheduler integrating all subsystems:

```python
from fusion_simulation.core.kernel import SimulationKernel, KernelConfig

kernel = SimulationKernel(KernelConfig(headless=True))
kernel.init()
state = kernel.step(num_steps=100)
print(state.sim_time, state.frame_count)
kernel.close()
```

### ECS (Entity-Component-System)

All simulation entities are component-based:

- **Transform** — position, orientation
- **RigidBody** — mass, friction
- **Articulation** — joint positions/velocities
- **AgentBind** — bound agent name
- **CameraSensor / IMUSensor** — sensor metadata

```python
from fusion_simulation.core.ecs import EntityManager, Transform, RigidBody

mgr = EntityManager()
eid = mgr.create_entity()
mgr.add_component(eid, Transform(position=[0, 0, 1]))
mgr.add_component(eid, RigidBody(mass=1.0))
```

### SensorManager

Registry-based sensor management with update rate control:

```python
from fusion_simulation.sensor.manager import SensorManager
from fusion_simulation.sensor.base import SensorConfig, SensorType

sm = SensorManager()
sm.add_sensor(SensorConfig(sensor_type=SensorType.RGB_CAMERA, name="wrist_cam"))
observations = sm.get_observations()
```

### AgentManager + PolicyClient

Multi-agent management with decimation and fusion-mlx inference:

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

Manager-Based RL environment aligned with Isaac Lab architecture:

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

Four managers for clean MDP decomposition:
- **ObservationManager** — group observations (policy/critic), noise, history
- **ActionManager** — process + apply separation, decimation-aware, scale/clip
- **RewardManager** — composable reward functions, cumulative tracking
- **TerminationManager** — termination + timeout conditions

### SimulationServer

gRPC service with HTTP JSON-RPC fallback, MetricsServer, and Fusion-Gateway integration:

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

### OpenAI-Compatible REST API

Exposes `/v1/models`, `/v1/chat/completions` (SSE streaming), `/v1/health` on port 11434 by default. Supports Bearer token auth and dispatches simulation commands from chat messages:

```bash
# List models
curl http://localhost:11434/v1/models

# Chat completion (dispatches simulation command)
curl -X POST http://localhost:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"fusion-sim","messages":[{"role":"user","content":"{\"command\":\"status\"}"}]}'
```

### MetricsServer + GatewayClient

- **MetricsServer** exposes `/health` (JSON) and `/metrics` (Prometheus text format) on port 11456
- **GatewayClient** registers with Fusion-Gateway at :11432, sends heartbeats, deregisters on shutdown
- **MetricsCollector** provides thread-safe counters, gauges, and histograms

## Installation

### From Source

```bash
git clone <repo-url>
cd fusion-simulation
python -m venv .venv
source .venv/bin/activate
pip install -e ".[test,physics]"
```

> PyBullet (physics engine) is optional. Install with the `physics` extra (`pip install -e ".[physics]"`) or `pip install pybullet` separately. The package imports and runs without it; only physics simulation requires it.

### Via Homebrew (Apple Silicon)

```bash
# Published release (recommended) — pulls source tarball from GitHub releases
brew install dahai80/test/fusion-simulation

# Or install from the in-tree formula directly
brew install --formula Formula/fusion-simulation.rb
```

> The release formula (`Formula/fusion-simulation.rb`) points at a `git archive`-generated
> `fs-release.tar.gz` attached to the matching GitHub release, so its `sha256` is stable
> (no self-reference). The release asset excludes the `Formula/` directory from the tarball.

## CLI

```bash
# Version
fusion-sim version

# Environment
fusion-sim env init --engine=lerobot --headless

# Scenes
fusion-sim scene list
fusion-sim scene load --name=pick

# Agents
fusion-sim agent spawn --name=robot0 --role=robot --action-dim=6
fusion-sim agent list
fusion-sim agent destroy --name=robot0

# Sensors
fusion-sim sensor add --type=rgb_camera --name=wrist_cam
fusion-sim sensor add --type=imu --name=imu0
fusion-sim sensor list

# Snapshots
fusion-sim snapshot save --name=checkpoint_1
fusion-sim snapshot restore --name=checkpoint_1

# Dataset
fusion-sim dataset list
fusion-sim dataset import --name=mydata --path=/path/to/data

# Training
fusion-sim train --dataset=mydata --model-name=policy --epochs=10
fusion-sim train --dataset=mydata --use-kernel

# Evaluation
fusion-sim test --model=policy --engine=lerobot --episodes=5

# Benchmark
fusion-sim bench --model=policy --output=report.md

# Service
fusion-sim service start --port=11447 --metrics-port=11456 --openai-api-port=11434 --gui
fusion-sim service stop
fusion-sim service health

# Gateway
fusion-sim gateway register --gateway-url=http://localhost:11432

# Kernel direct
fusion-sim kernel run --steps=100 --headless
fusion-sim kernel status
```

## Web Dashboard GUI

5-page Web Dashboard per PRD Section 7, served by FastAPI on port 11455:

```bash
fusion-sim service start --gui --gui-port 11455
# Open http://localhost:11455
```

| Page | Features |
|------|----------|
| Welcome | Env auto-detect (PyBullet/gRPC/MLX/Service), quick templates |
| Workstation | 4-zone layout: sidebar + viewport + inspector + statusbar, transport controls, real-time metrics |
| Agent Orchestration | Agent CRUD, prompt editor, role config |
| Data & Recording | Snapshot save/restore, export buttons |
| Settings | Kernel/AI/Service config, env check |

API: REST (`/api/*`) + WebSocket (`/ws/events`) + static files.

## Testing

```bash
# All tests
pytest

# Single file
pytest tests/test_core.py
pytest tests/test_new_arch.py
pytest tests/test_e2e.py

# E2E tests only (reliability, stability, GUI)
pytest tests/test_e2e.py -v

# With coverage
pytest --cov=fusion_simulation --cov-report=term-missing

# Lint
ruff check .
ruff format --check .
```

### E2E Test Coverage

`tests/test_e2e.py` covers end-to-end reliability, stability, and GUI stability across 72 tests:

| Category | Tests | Coverage |
|----------|-------|----------|
| Kernel Lifecycle | 8 | Full lifecycle (init→step→pause→resume→stop→close), repeated cycles, reset, snapshots |
| Server RPC | 10 | All 12 RPC methods, unknown method error, health with/without kernel |
| Metrics HTTP | 10 | /health + /metrics endpoints, Prometheus format, thread safety, real HTTP |
| Gateway Client | 4 | Disabled gateway, unreachable gateway, health provider, close deregisters |
| SimulationEnv | 5 | Init/step/reset/close, repeated cycles, scenes, no-init error |
| FusionGymEnv | 6 | Reset/step/close, observation/action spaces, timeout, reward, termination |
| Sensor+Agent Pipeline | 5 | Multi-sensor, multi-agent, enable/disable, full pipeline with events |
| Stability/Stress | 7 | 10x cycles, 500 steps, concurrent RPC, 100 sensors, 50 agents, 1K entities, 10K events |
| GUI Stability | 7 | Health endpoint structure, Prometheus format, 404, event streaming, server start/stop |
| Gym Managers | 8 | ObservationManager, ActionManager, RewardManager, TerminationManager |
| CLI | 4 | version, scene list, agent spawn, sensor add |

## Key Constraints

- All model inference via fusion-mlx HTTP API (`http://localhost:11434/v1`, override with `FUSION_MLX_URL`)
- No direct torch/CUDA/MLX imports
- PyBullet optional — code handles `ImportError` gracefully
- Python ≥3.12
- Default model: `Qwen3.5-4B-bf16` via fusion-mlx (override with `FUSION_MLX_MODEL`)
- All fusion-mlx requests carry `X-Fusion-Route: mlx` + `Authorization: Bearer ${FUSION_MLX_API_KEY}`

## Environment Variables

| Variable | Purpose | Default |
|----------|---------|---------|
| `FUSION_MLX_URL` | fusion-mlx base URL (read by `KernelConfig`, `AgentConfig`, `GUIConfig`, CLI `--mlx-url`, `BCTrainer`) | `http://localhost:11434/v1` |
| `FUSION_MLX_API_KEY` | fusion-mlx Bearer auth key (read by `KernelConfig`, `GUIConfig`, CLI `--api-key`, `BCTrainer`) | empty — emits a WARNING on kernel start |
| `FUSION_MLX_MODEL` | default model name (read by `AgentConfig`, `PolicyClient`, `BCTrainer`, GUI) | `Qwen3.5-4B-bf16` |
| `FUSION_MLX_ROUTE` | `X-Fusion-Route` header value for fusion-mlx routing | `mlx` |

## Project Structure

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
├── physics/        # L2: Physics engine abstraction
│   ├── base.py
│   ├── mujoco_engine.py
│   └── pybullet_engine.py
├── sensor/         # L3: Sensor management
│   ├── base.py
│   ├── contact.py
│   ├── depth_camera.py
│   ├── imu.py
│   ├── manager.py
│   ├── rgb_camera.py
│   └── semantic_camera.py
├── agent/          # L4: Agent + Policy
│   ├── config.py
│   ├── joint_controller.py
│   ├── manager.py
│   ├── policy.py
│   ├── scheduler.py
│   └── task_templates.py
├── sim/            # Backward-compat env wrapper + scene formats
│   ├── env.py
│   ├── scene.py
│   └── scene_formats/
│       ├── json_loader.py
│       └── urdf_loader.py
├── train/          # L5: Training
│   ├── gym_env.py
│   └── trainer.py
├── eval/           # L5: Evaluation
│   └── evaluator.py
├── dataset/        # Dataset management + data collector
│   ├── collector.py
│   └── manager.py
├── service/        # L6: gRPC + HTTP + OpenAI API service
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
├── render/         # Render engine
│   ├── base.py
│   └── pybullet_render.py
├── gui/            # Web Dashboard (FastAPI + static HTML)
│   ├── app.py
│   ├── ecs_compat.py
│   └── static/
│       └── index.html
├── bench.py
└── cli/            # Command-line interface
    └── __init__.py
```

## License

[Apache License 2.0](LICENSE)
