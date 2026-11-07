---
doc_id: fusion-simulation/architecture
product: fusion-simulation
class: app
entry: service
doc_type: architecture
title: { zh: "fusion-simulation 架构", en: "fusion-simulation architecture" }
audience: engineer
status: draft
version: 0.1.12
updated: 2026-11-07
owner: "@dahai"
source_of_truth:
  - README.md
  - fusion_simulation/core/kernel.py
  - fusion_simulation/service/server.py
---

# 架构 / Architecture

## 六层结构（受 NVIDIA Isaac Sim 启发）

```
L1  Core        SimClock · ECS · EventBus · WorldState · SimulationKernel · FaultIsolation · Plugin · ResourceQuota
L2  Physics     PyBulletEngine (decoupled stepping) · MuJoCoEngine (stub)
L3  Sensor      SensorManager · SensorRegistry · RGB / Depth / IMU / Contact / Semantic
L4  Agent       AgentManager · PolicyClient (fusion-mlx HTTP) · PromptScheduler · Decimation · JointController · TaskTemplates
L5  Train/Eval  BCTrainer (data export) · SimulationEvaluator · FusionGymEnv
L6  Service     gRPC + JSON-RPC · OpenAI-compatible REST API · MetricsServer · GatewayClient
```

目录映射见 README「Project Structure」一节；各层与 `fusion_simulation/` 子目录一一对应（core / physics / sensor / agent / sim / train / eval / dataset / service / render / gui / cli）。

## 关键设计

- **SimulationKernel**（`core/kernel.py`）：中央调度器，`_step_once` 将每个子系统包进 try/except 并接入 `FaultIsolationManager`；`ResourceQuotaManager` 管控实体生成与逐帧分配。
- **ECS**：实体全部组件化（Transform / RigidBody / Articulation / AgentBind / CameraSensor / IMUSensor）。
- **物理引擎解耦**：PyBullet 为可选依赖，缺失时包正常导入（`ImportError` 优雅处理）；MuJoCo 为 stub。headless 相机渲染使用 CPU tiny renderer（`ER_TINY_RENDERER`），不依赖 OpenGL GUI 渲染器。
- **推理外置**：L4 的 PolicyClient 通过 HTTP 调 Fusion-Gateway（11432），LLM 推理移出阻塞帧循环（async + 断路器 / 退避 / 限速）；fusion-mlx 故障不会卡死仿真。
- **服务层**（`service/`）：gRPC 为主、HTTP JSON-RPC 回退；proto bytes 承载相机帧（≤4 MB）。真正的零编码 FlatBuffers 需上游 `flatc` 编译器，不在树内，当前以 proto bytes 为生产零拷贝路径。

## 数据流（一帧）

```
SimClock tick
  -> PhysicsEngine.step (decoupled stepping)
  -> SensorManager.update (registry-driven, rate-controlled)
  -> AgentManager.decide (PolicyClient via Fusion-Gateway, async)
  -> JointController / ActionManager.apply (decimation-aware)
  -> EventBus events -> WorldState commit -> MetricsCollector
```

## 确定性与生命周期

- `core/rng.py` 的 `seed_everything` 全局播种 `random` + `numpy`；每回合经 `np.random.default_rng(derived_seed)` 隔离，同种子得到相同传感器噪声与 RNG 流。
- 快照使用真实 `pybullet.saveState` / `restoreState`，带 LRU 上限与淘汰；恢复时重同步 ECS body map。
- 长跑泄漏防护：世界实体在 unload 时清理，reset 卸载生成的 body，collector 强制帧数上限，数据集 `index.json` 原子写入（`flock`）。

## 与其他 fusion 组件的关系

| 组件 | 关系 |
|------|------|
| fusion-gateway (:11432) | 推理出口；PolicyClient / BCTrainer / GUI 全部经网关访问 fusion-mlx |
| fusion-mlx (:11434) | 实际模型服务，由网关代理；本仓库不直接导入 MLX |
| fusion-core | 公共底座：HTTP 客户端、配置、`client_key()` 解析 |
| fusion-supervisor | 通过 start 脚本纳入编排（可选） |
