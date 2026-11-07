---
doc_id: fusion-simulation/claude-guide
product: fusion-simulation
class: app
entry: service
doc_type: guide
title: { zh: "fusion-simulation Claude 工作指南", en: "fusion-simulation CLAUDE guide" }
audience: engineer
status: draft
version: 0.1.12
updated: 2026-11-07
owner: @dahai
source_of_truth:
  - CLAUDE.md
---

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Fusion-Simulation is a robot virtual simulation training and testing platform for Apple Silicon. It uses PyBullet for physics simulation and delegates all neural network inference to fusion-mlx via HTTP API — **never imports torch, CUDA, or MLX directly**.

## Build & Run Commands

```bash
source .venv/bin/activate   # shared repo-root venv, REQUIRED before any Python work

# Install (editable). physics extra pulls pybullet (optional runtime dep).
pip install -e ".[test,physics]"

# Run all tests
pytest

# Run a single test file / class / method
pytest tests/test_core.py
pytest tests/test_new_arch.py::TestSimulationKernel::test_step
pytest tests/test_e2e.py -v          # 72 E2E reliability/stability/GUI tests

# With coverage
pytest --cov=fusion_simulation --cov-report=term-missing

# Lint (config lives in pyproject.toml [tool.ruff]; excludes generated proto files)
ruff check .
ruff format --check .

# CLI entry point: fusion-sim (== python -m fusion_simulation.cli)
fusion-sim version
fusion-sim env init --engine=lerobot --headless
fusion-sim scene list
fusion-sim agent spawn --name=robot0 --role=robot --action-dim=6
fusion-sim sensor add --type=rgb_camera --name=cam0
fusion-sim snapshot save --name=checkpoint_1
fusion-sim dataset import --name=mydata --path=/path/to/data
fusion-sim train --dataset=mydata --model-name=policy --epochs=10
fusion-sim test --model=policy --engine=lerobot --episodes=5
fusion-sim bench --model=policy --output=report.md
fusion-sim service start --port=11447 --metrics-port=11456 --openai-api-port=11434 --gui
fusion-sim service health
fusion-sim gateway register --gateway-url=http://localhost:11432
fusion-sim kernel run --steps=100 --headless
fusion-sim kernel status
```

`cli/__init__.py` registers 14 argparse subcommands: `version`, `env`, `scene`, `agent`, `sensor`, `snapshot`, `train`, `test`, `bench`, `dataset`, `service`, `gateway`, `kernel`, plus `service start` flags for the OpenAI API + GUI ports.

## Architecture

Six-layer architecture (Isaac Sim-inspired) under `fusion_simulation/`. The central integration point is `SimulationKernel` (`core/kernel.py`), an async scheduler that owns the clock and steps every subsystem per frame. Understand the kernel's frame loop before editing any layer — it defines the call order and the data that flows between layers.

- **L1 `core/`** — `SimulationKernel` (`KernelState`, async `step()`, `FrameResult`), `ECS` (`EntityManager` + component types: `Transform`, `RigidBody`, `Articulation`, `AgentBind`, sensor binds), `EventBus`, `SimClock`, `WorldState`, `FaultIsolation`, `Plugin`, `ResourceQuota`, `rng.py` (`seed_everything` global seed + `make_rng`/`derive_seeds` per-episode RNG isolation + `numpy_to_native` JSON conversion).
- **L2 `physics/`** — `PyBulletEngine` (decoupled stepping) and `MuJoCoEngine` (stub). Engine abstraction in `base.py`.
- **L3 `sensor/`** — `SensorManager` + registry-based sensors (RGB / Depth / IMU / Contact / Semantic), each with update-rate control.
- **L4 `agent/`** — `AgentManager` (decimation, observe-act loop), `PolicyClient` (fusion-mlx vision+action over HTTP), `PromptScheduler`, `JointController`, `TaskTemplates`.
- **L5 `train/` + `eval/`** — `BCTrainer` (`train/trainer.py`) implements Behavior Cloning by exporting training data and importing a policy via fusion-mlx; falls back to simulated loss if fusion-mlx unreachable. `FusionGymEnv` (`train/gym_env.py`) is a manager-based RL env (Isaac Lab-style) decomposing the MDP into `ObservationManager` / `ActionManager` / `RewardManager` / `TerminationManager`. `SimulationEvaluator` (`eval/evaluator.py`) runs async eval episodes and emits markdown/JSON reports.
- **L6 `service/`** — `SimulationServer` (gRPC + JSON-RPC), `openai_api.py` (OpenAI-compatible REST: `/v1/models`, `/v1/chat/completions` SSE, `/v1/health` — dispatches sim commands from chat messages), `GatewayClient` (Fusion-Gateway registration/heartbeat), `MetricsServer` (`/health` + `/metrics` Prometheus), `MetricsCollector`, `encoding.py` (binary sensor/state payload encoding — raw RGB/depth bytes + json_native for small structs, replaces `default=str` numpy-as-text). `service/proto/` holds generated gRPC stubs (excluded from lint + coverage — regenerate from `simulation.proto` via `python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. fusion_simulation/service/proto/simulation.proto` run from repo root). `service/flatbuf/serializer.py` provides zero-copy serialization (true FlatBuffers needs upstream `flatc`; the proto bytes path is the live zero-copy route).

Other modules:
- **`sim/env.py`** — backward-compat `SimulationEnv` wrapping PyBullet; `sim/scene_formats/` has JSON + URDF loaders.
- **`dataset/`** — `DatasetManager` (local dataset import/list/delete/clean/synthetic gen, JSON index at `~/Library/Fusion/Simulation/datasets/index.json`) and `collector.py` (COCO/KITTI export).
- **`render/`** — render engine abstraction + PyBullet renderer.
- **`gui/app.py`** — Web Dashboard (FastAPI on port 11455, static `gui/static/index.html`): 5-page GUI (Welcome / Workstation / Agent Orchestration / Data & Recording / Settings). REST `/api/*` + WebSocket `/ws/events`. Enabled via `service start --gui --gui-port 11455`.
- **`bench.py`** — benchmark runner.

## Release & Homebrew

- Releases are GitHub releases (`vX.Y.Z`) with a `fs-release.tar.gz` source asset.
- `Formula/fusion-simulation.rb` is the Homebrew formula — its `url` + `sha256` must be pinned to the published release asset (not the git tag tarball). Every release = bump `pyproject.toml` version, cut release, update formula pin.
- Users install via `brew install dahai80/tap/fusion-simulation` or the in-tree formula (see README.md "Via Homebrew").

## Repo Workflow (AGENTS.md)

`AGENTS.md` holds the agent workflow contract — key points:
- Work is issue-driven: `gh issue view <ISSUE_NUM> --repo dahai80/fusion-simulation`; comment on the issue after local verification.
- Never modify other `fusion-*` repos. Upstream problems = file an issue on the upstream repo (documenting source repo, dependency contract, blocking relationship), then comment the new issue number on the parent issue.
- Ports: `~/fusion/architecture/port-registry.yaml` is the single source of truth — register before binding, `python scripts/check-ports.py` after changes.
- Long-running processes must set their title via `setproctitle` (`fusion-simulation <role> :<port>`), with try/except fallback.

## Key Constraints

- All model inference goes through the Fusion-Gateway HTTP API (default `http://localhost:11432/v1`, override via `FUSION_MLX_URL`), which proxies the OpenAI-compatible API to the fusion-mlx backend. No direct torch/CUDA/MLX imports.
- **Inference routes through the Fusion-Gateway (11432), not direct fusion-mlx (11434)** — per directive "建议全部使用gateway访问fusion-mlx" (issue #10 选项2 落地). The gateway forwards multimodal `image_url` content verbatim (live-verified: SSE + `/v1/models` + multimodal all work through 11432), so issue #10's original premise (gateway cannot do multimodal) is obsolete. The latency-sensitive robot control loop still uses a synchronous `httpx.Client` — the HTTP client is not migrated; only config/key resolution goes through fusion-core (`client_key()`, issue #14). Note: the simulation's *own* self-hosted OpenAI-compatible REST API (`openai_api.py`, CLI `--openai-api-port`) is a separate service and still defaults to 11434; it is not a fusion-mlx call.
- PyBullet is optional — code handles `ImportError` gracefully for test/headless environments.
- Depends on in-tree `fusion-core` (config single-point resolution `client_key()`, `set_process_title()`) — install it before this package. `fusion-sim service start` sets the process title to `fusion-sim service :<port>` via setproctitle (issue #13); API-key env resolution is `FUSION_CLIENT_KEY` via `fusion_core.config.client_key()` (issue #14).
- Default model: `Qwen3.5-4B-bf16` via fusion-mlx (override via `FUSION_MLX_MODEL`).
- Every fusion-mlx request carries `X-Fusion-Route: mlx` + `Authorization: Bearer ${FUSION_CLIENT_KEY}` (resolved via fusion-core `client_key()`).
- Dataset default path: `~/Library/Fusion/Simulation/datasets/`.
- Python ≥3.12. Tests use `pytest-asyncio` with `asyncio_mode = "auto"`, `testpaths = ["tests"]`.
- **Determinism**: `core/rng.py` is the single entry point for randomness. `seed_everything(seed)` seeds `random` + `numpy` globally at kernel init; per-episode RNG is isolated via `np.random.default_rng(derive_seeds(base_seed, count, offset))` — do NOT call `np.random.seed`/`np.random.normal` directly in subsystems, consume `kernel.rng`/`sensor_rng`/`agent_rng` or the injected `Generator` instead. PyBullet's `randomSeed` kwarg is build-dependent (this build rejects it); physics determinism relies on numpy seed + identical scene load order, documented in `pybullet_engine.py`.
- **Encoding**: never serialize numpy with `json.dumps(..., default=str)` (produces `str([[1 2 3]])` text). Use `numpy_to_native()` for JSON, or `service/encoding.py` `encode_sensor_payload`/`encode_state_payload` for gRPC binary payloads (raw bytes for cameras, ≤4MB gRPC max).
- Generated proto files (`service/proto/simulation_pb2*.py`) are excluded from ruff and coverage — do not hand-edit; regenerate from the `.proto` (run protoc from **repo root**, not the proto dir, so stubs emit `from fusion_simulation.service.proto import ...`).

## Environment Variables

| Variable | Purpose | Default |
|----------|---------|---------|
| `FUSION_MLX_URL` | Fusion-Gateway base URL proxying fusion-mlx (read by `KernelConfig`, `AgentConfig`, `GUIConfig`, CLI `--mlx-url`, `BCTrainer`) | `http://localhost:11432/v1` |
| `FUSION_CLIENT_KEY` | fusion-mlx Bearer auth key, resolved via `fusion_core.config.client_key()` (read by `KernelConfig`, `GUIConfig`, CLI `--api-key`, `BCTrainer`); legacy `FUSION_MLX_API_KEY` honored as deprecated fallback | empty — emits WARNING on kernel start; `FUSION_PROFILE=prod` raises |
| `FUSION_MLX_MODEL` | default model name (read by `AgentConfig`, `PolicyClient`, `BCTrainer`, GUI) | `Qwen3.5-4B-bf16` |
| `FUSION_MLX_ROUTE` | `X-Fusion-Route` header value for fusion-mlx routing | `mlx` |

## Service Ports

| Port | Service |
|------|---------|
| 11432 | Fusion-Gateway (external) — inference client default (`FUSION_MLX_URL`); proxies OpenAI-compatible API to fusion-mlx backend |
| 11434 | Simulation's own OpenAI-compatible REST API (`openai_api.py`, `--openai-api-port`); fusion-mlx backend also listens here externally |
| 11447 | gRPC simulation server (`--port`) |
| 11455 | Web Dashboard GUI (`--gui-port`) |
| 11456 | MetricsServer `/health` + `/metrics` (`--metrics-port`) |
| 11432 | Fusion-Gateway (external) |
