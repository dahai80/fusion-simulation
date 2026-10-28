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
        policy_endpoint="http://localhost:11432/v1/chat/completions",
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

Exposes `/v1/models`, `/v1/chat/completions` (SSE streaming), `/v1/health` on port 11434 by default. Supports Bearer token auth (constant-time `hmac.compare_digest`; `auth_enabled=True` by default, refuses to start with an empty token). Simulation command dispatch from chat messages is gated by a strict command allowlist (init/step/reset/pause/resume/load_scene/save_snapshot/restore_snapshot/add_sensor/add_agent/get_observations/status/close) — unlisted commands are rejected. Request body capped at 4 MB; oversized Content-Length refused.

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
# Simulated-loss fallback only with explicit opt-in (output tagged [SIMULATED]):
fusion-sim train --dataset=mydata --epochs=10 --allow-simulated

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

## Security

Defaults are now safe for local-first use (see audit `audit/fusion-simulation-fix-report-0824.md`):

- **Network bind**: gRPC (11447), Metrics (11456), GUI (11455), OpenAI API (11434) all default to `127.0.0.1`. Bind to `0.0.0.0` only via explicit `--host` and ensure `auth_enabled` + a non-empty token.
- **Auth**: `auth_enabled=True` by default; empty token with auth enabled refuses to start. Token comparison uses constant-time `hmac.compare_digest`.
- **WebSocket**: token auth (query `?token=` or `Authorization: Bearer`), Origin check (origin netloc must match Host header), and a command allowlist — unlisted commands return `{"error":"command not allowed","error_code":"forbidden"}`.
- **SSRF defense**: `add_agent` rejects user-supplied `policy_endpoint`/`api_key`; URL scheme allowlist (http/https only), link-local/loopback blocked; never falls back to the kernel key when the caller is unauthenticated.
- **Path safety**: dataset `--name` rejects `/`, `..`, leading dots; `import`/`delete` assert the resolved path stays under the data root. URDF parsing requires `defusedxml` (refuses vulnerable stdlib `xml.etree` — XXE defense). Bench `--output` must stay under the reports dir (`~/Library/Fusion/Simulation/reports`, override `FUSION_SIM_REPORTS_DIR`); absolute paths outside it or `..` traversal are rejected.
- **XSS**: dashboard dynamic text (agent/scene/snapshot names) is HTML-attribute escaped before `innerHTML`.
- **Training integrity**: empty dataset raises and exits non-zero; simulated-loss fallback is off by default and only with `--allow-simulated`, output tagged `[SIMULATED]`.

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

- All model inference goes through the Fusion-Gateway HTTP API (`http://localhost:11432/v1`, override with `FUSION_MLX_URL`), which proxies the OpenAI-compatible API to the fusion-mlx backend
- No direct torch/CUDA/MLX imports
- PyBullet optional — code handles `ImportError` gracefully
- Python ≥3.12
- Default model: `Qwen3.5-4B-bf16` via fusion-mlx (override with `FUSION_MLX_MODEL`)
- All inference requests carry `X-Fusion-Route: mlx` + `Authorization: Bearer ${FUSION_CLIENT_KEY}` (resolved via fusion-core `client_key()`)
- **Inference routes through the Fusion-Gateway (11432), not direct fusion-mlx (11434)** — per directive "建议全部使用gateway访问fusion-mlx" (issue #10 选项2 落地). The gateway forwards multimodal `image_url` content verbatim (live-verified), so issue #10's original premise (gateway cannot do multimodal) is obsolete. The sync `httpx.Client` is retained — only the base URL changed 11434→11432, no fusion-core migration. Note: the simulation's *own* OpenAI-compatible REST API (`openai_api.py`, `--openai-api-port`) is a separate service and still defaults to 11434. See [#10](https://github.com/dahai80/fusion-simulation/issues/10).

## Environment Variables

| Variable | Purpose | Default |
|----------|---------|---------|
| `FUSION_MLX_URL` | Fusion-Gateway base URL proxying fusion-mlx (read by `KernelConfig`, `AgentConfig`, `GUIConfig`, CLI `--mlx-url`, `BCTrainer`) | `http://localhost:11432/v1` |
| `FUSION_CLIENT_KEY` | fusion-mlx Bearer auth key, resolved single-point via `fusion_core.config.client_key()` (read by `KernelConfig`, `GUIConfig`, CLI `--api-key`, `BCTrainer`); legacy `FUSION_MLX_API_KEY` still honored as fallback with a deprecation warning | empty — emits a WARNING on kernel start; `FUSION_PROFILE=prod` raises |
| `FUSION_MLX_MODEL` | default model name (read by `AgentConfig`, `PolicyClient`, `BCTrainer`, GUI) | `Qwen3.5-4B-bf16` |
| `FUSION_MLX_ROUTE` | `X-Fusion-Route` header value for fusion-mlx routing | `mlx` |

## Reliability & Determinism (v0.1.9)

v0.1.9 hardens the platform against the v0.1.8 audit (22 P0–P2 blockers). Highlights:

**P0 — fakes & deployment blockers removed**
- `SimulationEvaluator._run_episode`: removed fake `time.sleep` "progress"; success metric is now normalized to `[0,1]` against an explicit target, not an absolute threshold.
- CLI `train` no longer fabricates data with `random.random()` — `BCTrainer` exports real dataset samples to fusion-mlx; the simulated-loss fallback is gated behind `allow_simulated=True` and logs a WARNING, so it cannot ship as "trained" by default.
- Headless camera rendering uses `ER_TINY_RENDERER` (CPU), not the unavailable OpenGL GUI renderer.
- Added `get_all_body_ids()` so IMU/Contact sensors iterate real physics bodies instead of an empty range.
- Camera sensor contract fixed (`view_matrix`/`projection_matrix`/`camera_orientation` replace the unused `camera_position`).
- Bare `except: pass` in sensors replaced with `logger.exception` + error reporting.
- `_step_once` wraps each subsystem in try/except wired to `FaultIsolationManager`; `ResourceQuotaManager` gates entity spawn and per-frame allocation.

**P1 — inference & lifecycle**
- LLM inference moved out of the blocking frame loop (async + circuit breaker / backoff / rate-limit); a failing fusion-mlx no longer stalls the simulation.
- Multi-node: unique instance ID, dynamic port allocation, graceful `SIGTERM` shutdown.
- Snapshots use real `pybullet.saveState`/`restoreState` with an LRU cap and eviction; restore re-syncs the ECS body map.

**P2 — long-run stability & determinism**
- Long-run leaks fixed: world entities cleaned on unload, reset unloads spawned bodies, collector enforces a frame cap, dataset `index.json` written atomically with `flock`.
- Determinism: `core/rng.py` seeds `random` + `numpy` globally (`seed_everything`) and isolates per-episode RNG via `np.random.default_rng(derived_seed)` — global `np.random` is no longer mutated mid-episode. Sensor/agent/trainer/collector/gym-env all consume a derived generator. Reproducibility verified by tests (same seed → identical sensor noise + RNG streams).
- Communication encoding: `SensorDataResponse`/`SimStateResponse` gained `bytes` payload fields; camera frames travel as raw RGB/depth bytes (≤4MB, under the gRPC max) instead of `json.dumps(default=str)` text. All `default=str` numpy-as-text encoding removed in favor of `numpy_to_native`. True zero-copy FlatBuffers needs the upstream `flatc` compiler (not in-tree); the proto bytes path is the production zero-copy route.

> The audit report lives at `audit/fusion-simulation-audit-report-0824.md`. 592 tests green, `ruff check` + `ruff format --check` clean.

## Project Structure

```
fusion_simulation/
├── core/           # L1: Kernel, Clock, ECS, EventBus, WorldState, FaultIsolation, Plugin, ResourceQuota, RNG
│   ├── clock.py
│   ├── ecs.py
│   ├── event_bus.py
│   ├── fault.py
│   ├── kernel.py
│   ├── plugin.py
│   ├── resource.py
│   ├── rng.py
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
│   ├── encoding.py
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
