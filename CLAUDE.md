# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Fusion-Simulation is a robot virtual simulation training and testing platform for Apple Silicon. It uses PyBullet for physics simulation and delegates all neural network inference to fusion-mlx via HTTP API — **never imports torch, CUDA, or MLX directly**.

## Build & Run Commands

```bash
source .venv/bin/activate

# Install (editable)
pip install -e ".[test]"

# Run all tests
pytest

# Run a single test file
pytest tests/test_core.py

# Run a single test class/method
pytest tests/test_core.py::TestBCTrainer::test_train_step

# Run with coverage
pytest --cov=fusion_simulation --cov-report=term-missing

# CLI usage
python -m fusion_simulation.cli version
python -m fusion_simulation.cli env init --engine=lerobot --headless
python -m fusion_simulation.cli scene list
python -m fusion_simulation.cli agent spawn --name=robot0 --role=robot --action-dim=6
python -m fusion_simulation.cli sensor add --type=rgb_camera --name=cam0
python -m fusion_simulation.cli snapshot save --name=checkpoint_1
python -m fusion_simulation.cli dataset list
python -m fusion_simulation.cli dataset import --name=mydata --path=/path/to/data
python -m fusion_simulation.cli train --dataset=mydata --model-name=policy --epochs=10
python -m fusion_simulation.cli test --model=policy --engine=lerobot --episodes=5
python -m fusion_simulation.cli bench --model=policy --output=report.md
python -m fusion_simulation.cli service start --port=11447 --metrics-port=11456
python -m fusion_simulation.cli service health
python -m fusion_simulation.cli gateway register --gateway-url=http://localhost:11432
```

## Architecture

Six-layer architecture under `fusion_simulation/`:

- **`core/`** — `SimulationKernel` (KernelState, async run, FrameResult), `ECS`, `EventBus`, `SimClock`, `WorldState`
- **`sim/env.py`** — `SimulationEnv` wraps PyBullet physics. `EnvConfig` holds engine type (lerobot/xlerobot), scene, FPS, headless, gravity, seed.
- **`sensor/`** — `SensorManager` with registry-based sensor management (RGB/Depth/IMU/Contact)
- **`agent/`** — `AgentManager` (decimation, observe-act loop), `PolicyClient` (fusion-mlx vision+action), `PromptScheduler`
- **`dataset/manager.py`** — `DatasetManager` handles local dataset import, listing, deletion, cleaning, and synthetic sample generation. Persists a JSON index at `~/Library/Fusion/Simulation/datasets/index.json`.
- **`train/trainer.py`** — `BCTrainer` implements Behavior Cloning. Sends training steps to fusion-mlx at `http://localhost:11434/v1/chat/completions`. Falls back to simulated loss if fusion-mlx is unreachable.
- **`eval/evaluator.py`** — `SimulationEvaluator` runs async evaluation episodes, collects metrics (success rate, trajectory error, latency, FPS), generates markdown or JSON reports.
- **`service/`** — `SimulationServer` (gRPC + JSON-RPC), `GatewayClient` (Fusion-Gateway registration/heartbeat), `MetricsServer` (/health + /metrics Prometheus), `MetricsCollector`

`cli/__init__.py` provides argparse-based CLI routing with 13 subcommands: version, env, scene, agent, sensor, snapshot, train, test, bench, dataset, service, gateway, kernel.

## Key Constraints

- All model inference must go through fusion-mlx HTTP API (default `http://localhost:11434/v1`). No direct torch/CUDA/MLX imports.
- PyBullet is optional — code handles `ImportError` gracefully for test environments.
- Default model: `qwen3.5-9b` via fusion-mlx.
- Dataset default path: `~/Library/Fusion/Simulation/datasets/`.
- Python ≥3.12 required. Tests use `pytest-asyncio` with `asyncio_mode = "auto"`.
