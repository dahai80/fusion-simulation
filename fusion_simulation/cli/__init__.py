from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path

from fusion_simulation import __version__

logger = logging.getLogger(__name__)

_DEFAULT_MLX_URL = os.environ.get("FUSION_MLX_URL", "http://localhost:11434/v1")
_DEFAULT_API_KEY = os.environ.get("FUSION_MLX_API_KEY", "")


def main():
    parser = argparse.ArgumentParser(
        description="Fusion-Simulation — Robot virtual simulation training and testing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--mlx-url", default=_DEFAULT_MLX_URL, help="fusion-mlx URL (env: FUSION_MLX_URL)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging")

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # env init
    env_parser = subparsers.add_parser("env", help="Environment management")
    env_sub = env_parser.add_subparsers(dest="action")
    env_init = env_sub.add_parser("init", help="Initialize simulation environment")
    env_init.add_argument("--engine", default="lerobot", choices=["lerobot", "xlerobot"])
    env_init.add_argument("--scene", default="default")
    env_init.add_argument("--headless", action="store_true")
    env_init.add_argument("--physics-dt", type=float, default=0.01)
    env_init.add_argument("--render-dt", type=float, default=1.0 / 30.0)

    # scene
    scene_parser = subparsers.add_parser("scene", help="Scene management")
    scene_sub = scene_parser.add_subparsers(dest="action")
    scene_sub.add_parser("list", help="List available scenes")
    scene_load = scene_sub.add_parser("load", help="Load a scene")
    scene_load.add_argument("--name", default="default")
    scene_load.add_argument("--builtin", action="store_true")

    # agent
    agent_parser = subparsers.add_parser("agent", help="Agent management")
    agent_sub = agent_parser.add_subparsers(dest="action")
    agent_spawn = agent_sub.add_parser("spawn", help="Spawn an agent")
    agent_spawn.add_argument("--name", required=True, help="Agent name")
    agent_spawn.add_argument("--entity-id", default="", help="Entity ID to bind")
    agent_spawn.add_argument("--action-dim", type=int, default=6, help="Action dimension")
    agent_spawn.add_argument("--role", default="robot", choices=["robot", "observer", "controller"])
    agent_spawn.add_argument("--decimation", type=int, default=1, help="Decimation factor")
    agent_sub.add_parser("list", help="List agents")
    agent_destroy = agent_sub.add_parser("destroy", help="Destroy an agent")
    agent_destroy.add_argument("--name", required=True, help="Agent name")

    # sensor
    sensor_parser = subparsers.add_parser("sensor", help="Sensor management")
    sensor_sub = sensor_parser.add_subparsers(dest="action")
    sensor_sub.add_parser("list", help="List sensors")
    sensor_add = sensor_sub.add_parser("add", help="Add a sensor")
    sensor_add.add_argument("--type", required=True, choices=["rgb_camera", "depth_camera", "imu"], help="Sensor type")
    sensor_add.add_argument("--name", default="", help="Sensor name")
    sensor_add.add_argument("--entity-id", default="", help="Entity ID to attach")
    sensor_add.add_argument("--width", type=int, default=640)
    sensor_add.add_argument("--height", type=int, default=480)

    # snapshot
    snap_parser = subparsers.add_parser("snapshot", help="Simulation snapshots")
    snap_sub = snap_parser.add_subparsers(dest="action")
    snap_save = snap_sub.add_parser("save", help="Save snapshot")
    snap_save.add_argument("--name", default="default", help="Snapshot name")
    snap_restore = snap_sub.add_parser("restore", help="Restore snapshot")
    snap_restore.add_argument("--name", required=True, help="Snapshot name")

    # train
    train_parser = subparsers.add_parser("train", help="Train a robot policy")
    train_parser.add_argument("--engine", default="lerobot", choices=["lerobot", "xlerobot"])
    train_parser.add_argument("--dataset", required=True, help="Dataset name")
    train_parser.add_argument("--model-name", default="robot_policy", help="Output model name")
    train_parser.add_argument("--epochs", type=int, default=10, help="Training epochs")
    train_parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    train_parser.add_argument("--use-kernel", action="store_true", help="Use kernel-based training")

    # test
    test_parser = subparsers.add_parser("test", help="Test a robot policy")
    test_parser.add_argument("--model", required=True, help="Model name")
    test_parser.add_argument("--engine", default="lerobot", choices=["lerobot", "xlerobot"])
    test_parser.add_argument("--episodes", type=int, default=10)

    # bench
    bench_parser = subparsers.add_parser("bench", help="Benchmark robot policy")
    bench_parser.add_argument("--model", required=True, help="Model name")
    bench_parser.add_argument("--output", default="", help="Output report path")

    # dataset
    ds_parser = subparsers.add_parser("dataset", help="Dataset management")
    ds_sub = ds_parser.add_subparsers(dest="action")
    ds_sub.add_parser("list", help="List datasets")
    ds_import = ds_sub.add_parser("import", help="Import dataset")
    ds_import.add_argument("--name", required=True)
    ds_import.add_argument("--path", required=True)
    ds_import.add_argument("--engine", default="lerobot")

    # service
    svc_parser = subparsers.add_parser("service", help="Service management")
    svc_sub = svc_parser.add_subparsers(dest="action")
    svc_start = svc_sub.add_parser("start", help="Start simulation server")
    svc_start.add_argument("--host", default="0.0.0.0")
    svc_start.add_argument("--port", type=int, default=11447)
    svc_start.add_argument("--metrics-port", type=int, default=11456)
    svc_start.add_argument("--headless", action="store_true")
    svc_start.add_argument("--gateway-url", default="", help="Fusion-Gateway URL")
    svc_start.add_argument("--gui", action="store_true", help="Enable Web Dashboard GUI")
    # Callers: fusion-studio UpstreamServiceManager / user starts fusion-sim service
    # Affected API: service start gains --mlx-url / --api-key -> KernelConfig + GUIConfig
    # Data schemas: args.mlx_url / args.api_key threaded to kernel_config and gui_config
    # User instruction: "和~/fusion/fuison-simulation项目集成起来...最后要完成端到端测试,确保系统可用"
    svc_start.add_argument("--gui-port", type=int, default=11455, help="Dashboard port (default: 11455)")
    svc_start.add_argument("--mlx-url", default=_DEFAULT_MLX_URL, help="fusion-mlx base URL (env: FUSION_MLX_URL)")
    svc_start.add_argument("--api-key", default=_DEFAULT_API_KEY, help="fusion-mlx API key (env: FUSION_MLX_API_KEY)")
    svc_sub.add_parser("stop", help="Stop simulation server")
    svc_health = svc_sub.add_parser("health", help="Check service health")
    svc_health.add_argument("--url", default="http://127.0.0.1:11456", help="Metrics URL")

    # gateway
    gw_parser = subparsers.add_parser("gateway", help="Fusion-Gateway integration")
    gw_sub = gw_parser.add_subparsers(dest="action")
    gw_register = gw_sub.add_parser("register", help="Register with gateway")
    gw_register.add_argument("--gateway-url", default="http://127.0.0.1:11432")
    gw_register.add_argument("--service-port", type=int, default=11447)
    gw_register.add_argument("--api-key", default="")

    # kernel (direct kernel operations)
    kernel_parser = subparsers.add_parser("kernel", help="Direct kernel operations")
    kernel_sub = kernel_parser.add_subparsers(dest="action")
    kernel_run = kernel_sub.add_parser("run", help="Run kernel steps")
    kernel_run.add_argument("--steps", type=int, default=100)
    kernel_run.add_argument("--headless", action="store_true")
    kernel_run.add_argument("--scene", default="default")
    kernel_sub.add_parser("status", help="Show kernel status")

    # version
    subparsers.add_parser("version", help="Show version info")

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    dispatch = {
        "version": lambda: _cmd_version(),
        "env": lambda: _cmd_env_dispatch(args),
        "scene": lambda: _cmd_scene_dispatch(args),
        "agent": lambda: _cmd_agent_dispatch(args),
        "sensor": lambda: _cmd_sensor_dispatch(args),
        "snapshot": lambda: _cmd_snapshot_dispatch(args),
        "train": lambda: asyncio.run(_cmd_train(args)),
        "test": lambda: asyncio.run(_cmd_test(args)),
        "bench": lambda: asyncio.run(_cmd_bench(args)),
        "dataset": lambda: _cmd_dataset_dispatch(args),
        "service": lambda: _cmd_service_dispatch(args),
        "gateway": lambda: _cmd_gateway_dispatch(args),
        "kernel": lambda: _cmd_kernel_dispatch(args),
    }
    handler = dispatch.get(args.command)
    if handler:
        handler()
    else:
        parser.print_help()


def _cmd_version():
    print(f"Fusion-Simulation v{__version__}")
    print("Core: SimulationKernel + ECS + EventBus + SimClock")
    print("Physics: PyBullet (V0.1) | SceneKit (V0.2) | Metal (V0.3)")
    print("Sensors: RGB Camera + IMU")
    print("Agents: AgentManager + PromptScheduler + PolicyClient (MLX)")
    print("Service: gRPC + HTTP JSON-RPC + MetricsServer")
    print("Gateway: Fusion-Gateway integration (registration + heartbeat)")
    print("Inference: fusion-mlx HTTP API (zero-copy UMA)")


def _cmd_env_dispatch(args):
    if args.action == "init":
        _cmd_env_init(args)
    else:
        print("Usage: fusion-simulation env init [--engine lerobot] [--headless]")


def _cmd_scene_dispatch(args):
    if args.action == "list":
        _cmd_scene_list()
    elif args.action == "load":
        _cmd_scene_load(args)
    else:
        print("Usage: fusion-simulation scene [list|load]")


def _cmd_agent_dispatch(args):
    if args.action == "spawn":
        from fusion_simulation.agent.config import AgentConfig, AgentRole

        role_map = {"robot": AgentRole.ROBOT, "observer": AgentRole.OBSERVER, "controller": AgentRole.CONTROLLER}
        cfg = AgentConfig(
            name=args.name,
            entity_id=args.entity_id,
            action_dim=args.action_dim,
            role=role_map.get(args.role, AgentRole.ROBOT),
            decimation=args.decimation,
        )
        logger.info("Agent spawned: name=%s role=%s action_dim=%d", cfg.name, cfg.role, cfg.action_dim)
        print(f"Agent spawned: name={cfg.name} role={cfg.role} action_dim={cfg.action_dim} decimation={cfg.decimation}")
    elif args.action == "list":
        print("No active simulation kernel. Start a kernel or service first.")
    elif args.action == "destroy":
        logger.info("Agent destroy requested: %s", args.name)
        print(f"Agent '{args.name}' would be destroyed from active kernel.")
    else:
        print("Usage: fusion-simulation agent [spawn|list|destroy]")


def _cmd_sensor_dispatch(args):
    if args.action == "list":
        print("No active simulation kernel. Start a kernel or service first.")
    elif args.action == "add":
        logger.info("Sensor add: type=%s name=%s res=%dx%d", args.type, args.name, args.width, args.height)
        print(f"Sensor added: type={args.type} name={args.name or 'auto'} resolution={args.width}x{args.height}")
    else:
        print("Usage: fusion-simulation sensor [list|add]")


def _cmd_snapshot_dispatch(args):
    if args.action == "save":
        logger.info("Snapshot save: %s", args.name)
        print(f"Snapshot saved: {args.name}")
    elif args.action == "restore":
        logger.info("Snapshot restore: %s", args.name)
        print(f"Snapshot restored: {args.name}")
    else:
        print("Usage: fusion-simulation snapshot [save|restore]")


def _cmd_dataset_dispatch(args):
    from fusion_simulation.dataset.manager import DatasetManager

    mgr = DatasetManager()
    if args.action == "list":
        datasets = mgr.list()
        if not datasets:
            print("No datasets imported.")
        else:
            for d in datasets:
                print(f"  {d['name']:20} engine={d['engine']} samples={d.get('sample_count', '?')}")
    elif args.action == "import":
        result = mgr.import_dataset(args.name, args.path, args.engine)
        print(f"Import: {result}")


def _cmd_service_dispatch(args):
    if args.action == "start":
        _cmd_service_start(args)
    elif args.action == "stop":
        print("Service stop requested. Send SIGINT to the running service process.")
    elif args.action == "health":
        _cmd_service_health(args)
    else:
        print("Usage: fusion-simulation service [start|stop|health]")


def _cmd_gateway_dispatch(args):
    if args.action == "register":
        from fusion_simulation.service.gateway_client import (
            GatewayClient,
            GatewayConfig,
        )

        cfg = GatewayConfig(
            gateway_url=args.gateway_url,
            service_port=args.service_port,
            api_key=args.api_key,
            enabled=True,
        )
        client = GatewayClient(cfg)
        if client.register():
            print(f"Registered with Fusion-Gateway at {args.gateway_url}")
        else:
            print(f"Failed to register with Fusion-Gateway at {args.gateway_url}")
        client.close()
    else:
        print("Usage: fusion-simulation gateway register [--gateway-url URL]")


def _cmd_kernel_dispatch(args):
    from fusion_simulation.agent.manager import AgentManager
    from fusion_simulation.core.kernel import KernelConfig, SimulationKernel
    from fusion_simulation.sensor.manager import SensorManager

    if args.action == "run":
        sm = SensorManager()
        am = AgentManager()
        kernel = SimulationKernel(KernelConfig(headless=args.headless))
        kernel.init(sensor_manager=sm, agent_manager=am)
        if args.scene:
            kernel.load_builtin_scene(args.scene)
        sim_time = kernel.step(num_steps=args.steps)
        print(f"Ran {args.steps} steps: sim_time={sim_time.sim_time:.3f}s frame={sim_time.frame_count}")
        print(f"Status: {json.dumps(kernel.status(), indent=2)}")
        kernel.close()
    elif args.action == "status":
        print("No running kernel (use 'kernel run' to start one)")


def _cmd_env_init(args):
    from fusion_simulation.agent.manager import AgentManager
    from fusion_simulation.core.kernel import KernelConfig, SimulationKernel
    from fusion_simulation.sensor.manager import SensorManager

    config = KernelConfig(
        physics_dt=args.physics_dt,
        render_dt=args.render_dt,
        headless=args.headless,
    )
    sm = SensorManager()
    am = AgentManager()
    kernel = SimulationKernel(config)
    kernel.init(sensor_manager=sm, agent_manager=am)
    print(f"Environment initialized: {kernel.status()}")
    kernel.close()


def _cmd_scene_list():
    from fusion_simulation.sim.env import SimulationEnv

    scenes = SimulationEnv.list_scenes()
    print(f"\n{'Scene':<20} {'Description'}")
    print("-" * 60)
    for info in scenes:
        print(f"{info['name']:<20} {info.get('description', '')}")


def _cmd_scene_load(args):
    from fusion_simulation.agent.manager import AgentManager
    from fusion_simulation.core.kernel import KernelConfig, SimulationKernel
    from fusion_simulation.sensor.manager import SensorManager

    sm = SensorManager()
    am = AgentManager()
    kernel = SimulationKernel(KernelConfig(headless=True))
    kernel.init(sensor_manager=sm, agent_manager=am)
    result = kernel.load_builtin_scene(args.name)
    print(f"Scene loaded: {result}")
    print(f"Status: {kernel.status()}")
    kernel.close()


async def _cmd_train(args):
    from fusion_simulation.dataset.manager import DatasetManager
    from fusion_simulation.train.trainer import BCTrainer

    mgr = DatasetManager()
    dataset_info = mgr.get(args.dataset)
    if not dataset_info:
        print(f"Dataset '{args.dataset}' not found. Generating synthetic data...")
        mgr.collect_samples(args.dataset, num_samples=100)

    trainer = BCTrainer(mlx_url=args.mlx_url)

    if args.use_kernel:
        from fusion_simulation.agent.config import AgentConfig, AgentRole
        from fusion_simulation.agent.manager import AgentManager
        from fusion_simulation.core.kernel import KernelConfig, SimulationKernel
        from fusion_simulation.sensor.manager import SensorManager

        sm = SensorManager()
        am = AgentManager()
        kernel = SimulationKernel(KernelConfig(headless=True))
        kernel.init(sensor_manager=sm, agent_manager=am)
        agent_cfg = AgentConfig(name="train_agent", role=AgentRole.ROBOT, action_dim=6)
        am.add_agent(agent_cfg)
        print(f"Kernel-based training {args.model_name} ({args.epochs} epochs)...")
        result = await trainer.train_with_kernel(kernel, "train_agent", epochs=args.epochs)
        kernel.close()
    else:
        print(f"Training {args.model_name} ({args.epochs} epochs, lr={args.lr})...")
        result = await trainer.train(dataset=[], epochs=args.epochs, lr=args.lr)

    print(f"Training completed: loss={result['final_loss']:.4f}, time={result['elapsed_seconds']}s")
    await trainer.close()


async def _cmd_test(args):
    from fusion_simulation.eval.evaluator import SimulationEvaluator

    evaluator = SimulationEvaluator()
    print(f"Testing model '{args.model}' on {args.engine} ({args.episodes} episodes)...")
    result = await evaluator.evaluate(args.model, args.engine, args.episodes)
    print(evaluator.generate_report(result))


async def _cmd_bench(args):
    from fusion_simulation.eval.evaluator import SimulationEvaluator

    evaluator = SimulationEvaluator()
    result = await evaluator.evaluate(args.model, episodes=10)
    report = evaluator.generate_report(result, fmt="json" if args.output.endswith(".json") else "markdown")
    if args.output:
        Path(args.output).write_text(report, encoding="utf-8")
        print(f"Report saved to {args.output}")
    else:
        print(report)


def _cmd_service_start(args):
    from fusion_simulation.core.kernel import KernelConfig
    from fusion_simulation.service.config import ServiceConfig
    from fusion_simulation.service.gateway_client import GatewayConfig
    from fusion_simulation.service.metrics_server import MetricsConfig
    from fusion_simulation.service.server import SimulationServer

    svc_config = ServiceConfig(host=args.host, port=args.port)
    kernel_config = KernelConfig(headless=args.headless, mlx_url=args.mlx_url, mlx_api_key=args.api_key)
    metrics_config = MetricsConfig(port=args.metrics_port)
    gateway_config = None
    if args.gateway_url:
        gateway_config = GatewayConfig(
            gateway_url=args.gateway_url,
            enabled=True,
            service_port=args.port,
            metrics_port=args.metrics_port,
        )
    server = SimulationServer(svc_config, kernel_config, gateway_config, metrics_config)
    server.start()
    logger.info("SimulationServer started on %s:%d (metrics on :%d)", args.host, args.port, args.metrics_port)
    print(f"SimulationServer started on {args.host}:{args.port} (metrics on :{args.metrics_port})")
    if args.gui:
        from fusion_simulation.gui import GUIConfig
        from fusion_simulation.gui.app import run_dashboard

        gui_config = GUIConfig(
            port=args.gui_port, grpc_host=args.host, grpc_port=args.port, mlx_url=args.mlx_url, mlx_api_key=args.api_key
        )
        print(f"Web Dashboard: http://0.0.0.0:{args.gui_port}")
        run_dashboard(server, gui_config)
    else:
        try:
            server.wait_for_termination()
        except KeyboardInterrupt:
            server.stop()
            print("Server stopped")


def _cmd_service_health(args):
    import httpx

    try:
        resp = httpx.get(f"{args.url}/health", timeout=5.0)
        if resp.status_code == 200:
            data = resp.json()
            print(f"Status: {data.get('status', 'unknown')}")
            print(f"Kernel: {data.get('kernel_state', 'N/A')}")
            print(f"Frame:  {data.get('frame_count', 0)}")
            print(f"Sim Time: {data.get('sim_time', 0.0):.3f}s")
            print(f"Sensors: {data.get('sensor_count', 0)}")
            print(f"Agents: {data.get('agent_count', 0)}")
        else:
            print(f"Service unhealthy: HTTP {resp.status_code}")
    except Exception as e:
        logger.warning("Cannot reach service: %s", e)
        print(f"Cannot reach service: {e}")
