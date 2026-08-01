# Fusion-Simulation

Robot virtual simulation training and testing platform for **Apple Silicon**.

Uses PyBullet for physics simulation and delegates all neural network inference to [fusion-mlx](https://github.com/fusion-mlx) via HTTP API — **never imports torch, CUDA, or MLX directly**.

## Architecture

Six-layer architecture inspired by NVIDIA Isaac Sim:

```
L1  Core        SimClock · ECS · EventBus · WorldState · SimulationKernel
L2  Physics     PyBulletEngine (decoupled stepping)
L3  Sensor      SensorManager · SensorRegistry · RGB/Depth/IMU/Contact
L4  Agent       AgentManager · PolicyClient (fusion-mlx HTTP) · PromptScheduler · Decimation
L5  Train/Eval  BCTrainer · SimulationEvaluator · FusionGymEnv
L6  Service     gRPC + JSON-RPC HTTP fallback · MetricsServer · GatewayClient
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
am.add_agent(AgentConfig(
    name="robot0",
    role=AgentRole.ROBOT,
    action_dim=6,
    decimation=4,
    policy_endpoint="http://localhost:11434/v1/chat/completions",
))
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
    metrics_config=MetricsConfig(port=8081),
)
server.handle_request("init", {})
server.handle_request("step", {"num_steps": 10})
server.handle_request("status", {})
server.handle_request("close", {})
```

### MetricsServer + GatewayClient

- **MetricsServer** exposes `/health` (JSON) and `/metrics` (Prometheus text format) on port 8081
- **GatewayClient** registers with Fusion-Gateway at :11432, sends heartbeats, deregisters on shutdown
- **MetricsCollector** provides thread-safe counters, gauges, and histograms

## Installation

### From Source

```bash
git clone <repo-url>
cd fusion-simulation
python -m venv .venv
source .venv/bin/activate
pip install -e ".[test]"
```

### Via Homebrew (Apple Silicon)

```bash
brew install --formula homebrew/fusion-simulation.rb
```

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
fusion-sim service start --port=50051 --metrics-port=8081 --gui
fusion-sim service stop
fusion-sim service health

# Gateway
fusion-sim gateway register --gateway-url=http://localhost:11432

# Kernel direct
fusion-sim kernel run --steps=100 --headless
fusion-sim kernel status
```

## Web Dashboard GUI

5-page Web Dashboard per PRD Section 7, served by FastAPI on port 8080:

```bash
fusion-sim service start --gui --gui-port 8080
# Open http://localhost:8080
```

| Page | Features |
|------|----------|
| Welcome | Env auto-detect (PyBullet/gRPC/MLX/Service), quick templates |
| Workstation | 4-zone layout: sidebar + viewport + inspector + statusbar, transport controls, real-time metrics |
| Agent Orchestration | Agent CRUD, prompt editor, role config |
| Data & Recording | Snapshot save/restore, export buttons |
| Settings | Kernel/AI/Service config, env check |

API: REST (`/api/*`) + WebSocket (`/ws/events`) + static files.

> **PRD Gap**: Current Web Dashboard is a V0.1 interim solution. PRD requires SwiftUI + Metal native client (V0.3). See `docs/gui-prd-gap-analysis.md` for full comparison.

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

- All model inference via fusion-mlx HTTP API (`http://localhost:11434/v1`)
- No direct torch/CUDA/MLX imports
- PyBullet optional — code handles `ImportError` gracefully
- Python ≥3.12
- Default model: `qwen3.5-9b` via fusion-mlx

## Project Structure

```
fusion_simulation/
├── core/           # L1: Kernel, Clock, ECS, EventBus, WorldState
│   ├── clock.py
│   ├── ecs.py
│   ├── event_bus.py
│   ├── kernel.py
│   └── world_state.py
├── physics/        # L2: Physics engine abstraction
│   ├── base.py
│   └── pybullet_engine.py
├── sensor/         # L3: Sensor management
│   ├── base.py
│   ├── manager.py
│   └── rgb_camera.py
├── agent/          # L4: Agent + Policy
│   ├── config.py
│   ├── manager.py
│   ├── policy.py
│   └── scheduler.py
├── sim/            # Backward-compat env wrapper
│   ├── env.py
│   └── scene.py
├── train/          # L5: Training
│   ├── trainer.py
│   └── gym_env.py
├── eval/           # L5: Evaluation
│   └── evaluator.py
├── dataset/        # Dataset management
│   └── manager.py
├── service/        # L6: gRPC + HTTP service
│   ├── config.py
│   ├── server.py
│   ├── gateway_client.py
│   ├── metrics_server.py
│   └── proto/
├── render/         # Render engine
│   ├── base.py
│   └── pybullet_render.py
├── gui/            # Web Dashboard (FastAPI + static HTML)
│   ├── __init__.py
│   ├── app.py
│   └── static/
│       └── index.html
├── api/            # REST API (placeholder)
└── cli/            # Command-line interface
    └── __init__.py
```

## License

MIT
