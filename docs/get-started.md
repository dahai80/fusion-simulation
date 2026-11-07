---
doc_id: fusion-simulation/get-started
product: fusion-simulation
class: app
entry: service
doc_type: get-started
title: { zh: "fusion-simulation 快速上手", en: "fusion-simulation get started" }
audience: engineer
status: draft
version: 0.1.11
updated: 2026-11-07
owner: @dahai
source_of_truth:
  - README.md
  - fusion_simulation/cli/__init__.py
  - pyproject.toml
---

# 快速上手 / Get Started

## 1. 安装（源码，Apple Silicon，Python 3.12+）

```bash
git clone https://github.com/dahai80/fusion-simulation.git
cd fusion-simulation
python -m venv .venv
source .venv/bin/activate
pip install -e ".[test,physics]"
```

或使用 Homebrew 安装已发布版本：

```bash
brew install dahai80/test/fusion-simulation
```

说明：PyBullet 为可选依赖；不加 `physics` extra 时包可导入，仅物理仿真不可用。

## 2. 验证安装

```bash
fusion-sim version
```

预期输出包含 `v0.1.11`。

## 3. 初始化仿真环境

```bash
fusion-sim env init --engine=lerobot --headless
fusion-sim scene list
fusion-sim scene load --name=default --builtin
```

## 4. 生成智能体与传感器

```bash
fusion-sim agent spawn --name=robot0 --role=robot --action-dim=6
fusion-sim sensor add --type=rgb_camera --name=wrist_cam
fusion-sim sensor list
fusion-sim agent list
```

## 5. 训练与评测（推理走 Fusion-Gateway）

```bash
fusion-sim dataset list
fusion-sim dataset import --name=mydata --path=/path/to/data
fusion-sim train --dataset=mydata --model-name=policy --epochs=10
fusion-sim test --model=policy --engine=lerobot --episodes=5
```

约束：推理默认经 Fusion-Gateway `http://localhost:11432/v1`（可用 `FUSION_MLX_URL` 覆盖）；需先启动 fusion-gateway 与 fusion-mlx。空数据集会直接报错退出；模拟 loss 回退仅在显式 `--allow-simulated` 时可用，输出会标记 `[SIMULATED]`。

## 6. 启动服务

```bash
# gRPC (11447) + Metrics (11456)
fusion-sim service start --port=11447 --metrics-port=11456

# 追加 OpenAI 兼容 REST API (11434) 与 Web Dashboard (11455)
fusion-sim service start --port=11447 --metrics-port=11456 --openai-api-port=11434 --gui --gui-port 11455

# 健康检查与停止
fusion-sim service health
fusion-sim service stop
```

打开 `http://localhost:11455` 使用 Web Dashboard（欢迎页 / 工作台 / 智能体编排 / 数据录制 / 设置）。

## 7. 通过 REST API 派发仿真命令

```bash
# List models
curl http://localhost:11434/v1/models

# Chat completion (dispatches simulation command)
curl -X POST http://localhost:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"fusion-sim","messages":[{"role":"user","content":"{\"command\":\"status\"}"}]}'
```

约束：命令白名单为 init / step / reset / pause / resume / load_scene / save_snapshot / restore_snapshot / add_sensor / add_agent / get_observations / status / close；白名单之外返回 forbidden。认证默认开启，空 token 拒绝启动。

## 下一步

- API 明细见 [api-reference.md](api-reference.md)
- 环境变量与端口见 [config-reference.md](config-reference.md)
- 常见问题见 [troubleshooting.md](troubleshooting.md)
