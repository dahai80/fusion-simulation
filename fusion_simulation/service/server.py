from __future__ import annotations

import json
import logging
import queue
import threading
import time
from concurrent import futures
from typing import Any

from fusion_simulation.agent.manager import AgentManager
from fusion_simulation.core.kernel import KernelConfig, SimulationKernel
from fusion_simulation.sensor.manager import SensorManager
from fusion_simulation.service.config import ServiceConfig
from fusion_simulation.service.gateway_client import (
    GatewayClient,
    GatewayConfig,
    HealthPayload,
)
from fusion_simulation.service.metrics_server import (
    MetricsCollector,
    MetricsConfig,
    MetricsServer,
)

logger = logging.getLogger(__name__)


class SimulationServer:
    def __init__(
        self,
        config: ServiceConfig | None = None,
        kernel_config: KernelConfig | None = None,
        gateway_config: GatewayConfig | None = None,
        metrics_config: MetricsConfig | None = None,
    ) -> None:
        self._config = config or ServiceConfig()
        self._kernel_config = kernel_config or KernelConfig()
        self._gateway_config = gateway_config
        self._metrics_config = metrics_config
        self._kernel: SimulationKernel | None = None
        self._sensor_manager: SensorManager | None = None
        self._agent_manager: AgentManager | None = None
        self._server: Any = None
        self._running: bool = False
        self._lock = threading.Lock()
        self._event_subscribers: list[queue.Queue] = []
        self._gateway_client: GatewayClient | None = None
        self._metrics_server: MetricsServer | None = None
        self._metrics_collector: MetricsCollector | None = None

    @property
    def kernel(self) -> SimulationKernel | None:
        return self._kernel

    @property
    def is_running(self) -> bool:
        return self._running

    def _ensure_kernel(self) -> SimulationKernel:
        if self._kernel is None:
            self._sensor_manager = SensorManager()
            self._agent_manager = AgentManager()
            self._kernel = SimulationKernel(self._kernel_config)
            self._kernel.init(
                sensor_manager=self._sensor_manager,
                agent_manager=self._agent_manager,
            )
            self._kernel.events.subscribe_all(self._on_event)
        return self._kernel

    def _on_event(self, event: Any) -> None:
        dead = []
        for i, q in enumerate(self._event_subscribers):
            try:
                q.put_nowait(event)
            except Exception:
                dead.append(i)
        for i in reversed(dead):
            self._event_subscribers.pop(i)

    def start(self) -> None:
        if self._running:
            logger.warning("SimulationServer already running")
            return
        try:
            import importlib.util as _ilu

            from grpc import server as grpc_server

            if _ilu.find_spec("fusion_simulation.service.proto.simulation_pb2") is not None:
                from fusion_simulation.service.proto import (  # noqa: F401
                    simulation_pb2,
                    simulation_pb2_grpc,
                )
            self._server = grpc_server(
                futures.ThreadPoolExecutor(max_workers=self._config.max_workers),
                options=[
                    ("grpc.max_receive_message_length", self._config.max_message_size),
                    ("grpc.max_send_message_length", self._config.max_message_size),
                ],
            )
            from fusion_simulation.service.proto.simulation_pb2_grpc import (
                add_SimulationServiceServicer_to_server,
            )

            servicer = _SimulationServicer(self)
            add_SimulationServiceServicer_to_server(servicer, self._server)
            addr = f"{self._config.host}:{self._config.port}"
            self._server.add_insecure_port(addr)
            self._server.start()
            self._running = True
            logger.info("SimulationServer started on %s (gRPC)", addr)
        except ImportError:
            logger.info("gRPC dependencies not available, starting HTTP-only server")
            self._running = True
            logger.info("SimulationServer started (HTTP-only mode, no gRPC)")
        self._start_metrics()
        self._start_gateway()

    def stop(self) -> None:
        if not self._running:
            return
        self._stop_gateway()
        self._stop_metrics()
        if self._server is not None:
            self._server.stop(grace=self._config.graceful_timeout)
        if self._kernel is not None:
            self._kernel.close()
        self._running = False
        self._event_subscribers.clear()
        logger.info("SimulationServer stopped")

    def wait_for_termination(self) -> None:
        if self._server is not None:
            self._server.wait_for_termination()

    def handle_request(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        handler = {
            "init": self._rpc_init,
            "step": self._rpc_step,
            "reset": self._rpc_reset,
            "close": self._rpc_close,
            "status": self._rpc_status,
            "load_scene": self._rpc_load_scene,
            "save_snapshot": self._rpc_save_snapshot,
            "restore_snapshot": self._rpc_restore_snapshot,
            "add_sensor": self._rpc_add_sensor,
            "add_agent": self._rpc_add_agent,
            "get_observations": self._rpc_get_observations,
            "pause": self._rpc_pause,
            "resume": self._rpc_resume,
        }.get(method)
        if handler is None:
            return {"error": f"Unknown method: {method}"}
        try:
            return handler(params)
        except Exception as e:
            logger.exception("RPC error: %s", method)
            return {"error": str(e)}

    def _rpc_init(self, params: dict[str, Any]) -> dict[str, Any]:
        k = self._ensure_kernel()
        return {"status": "initialized", "kernel_status": k.status()}

    def _rpc_step(self, params: dict[str, Any]) -> dict[str, Any]:
        k = self._ensure_kernel()
        num_steps = params.get("num_steps", 1)
        sim_time = k.step(num_steps=num_steps)
        fr = k.frame_result
        if self._metrics_collector is not None:
            self._metrics_collector.inc_counter("fusion_sim_steps_total", value=num_steps)
            self._metrics_collector.observe_histogram("fusion_sim_step_duration_ms", fr.total_ms)
            self._metrics_collector.set_gauge("fusion_sim_frame_count", fr.frame_count)
            self._metrics_collector.set_gauge("fusion_sim_sim_time", fr.sim_time)
        return {
            "sim_time": sim_time.sim_time,
            "frame_count": sim_time.frame_count,
            "physics_step_ms": fr.physics_step_ms,
            "sensor_collect_ms": fr.sensor_collect_ms,
            "agent_decide_ms": fr.agent_decide_ms,
            "render_ms": fr.render_ms,
            "total_ms": fr.total_ms,
        }

    def _rpc_reset(self, params: dict[str, Any]) -> dict[str, Any]:
        k = self._ensure_kernel()
        k.reset()
        return {"status": "reset"}

    def _rpc_close(self, params: dict[str, Any]) -> dict[str, Any]:
        if self._kernel is not None:
            self._kernel.close()
        return {"status": "closed"}

    def _rpc_status(self, params: dict[str, Any]) -> dict[str, Any]:
        if self._kernel is None:
            return {"initialized": False}
        return self._kernel.status()

    def _rpc_load_scene(self, params: dict[str, Any]) -> dict[str, Any]:
        k = self._ensure_kernel()
        scene_name = params.get("name", "default")
        result = k.load_builtin_scene(scene_name)
        return {"status": "loaded", "result": result}

    def _rpc_save_snapshot(self, params: dict[str, Any]) -> dict[str, Any]:
        k = self._ensure_kernel()
        name = params.get("name", "")
        snap_id = k.save_snapshot(name)
        return {"snapshot_id": snap_id}

    def _rpc_restore_snapshot(self, params: dict[str, Any]) -> dict[str, Any]:
        k = self._ensure_kernel()
        snap_id = params.get("snapshot_id", "")
        ok = k.restore_snapshot(snap_id)
        return {"restored": ok}

    def _rpc_add_sensor(self, params: dict[str, Any]) -> dict[str, Any]:
        self._ensure_kernel()
        if self._sensor_manager is None:
            return {"error": "No sensor manager"}
        from fusion_simulation.sensor.base import SensorConfig, SensorType

        cfg = SensorConfig(
            sensor_type=SensorType(params.get("type", "rgb_camera")),
            name=params.get("name", ""),
            entity_id=params.get("entity_id", ""),
            params=params.get("params", {}),
        )
        self._sensor_manager.add_sensor(cfg)
        return {"status": "added", "name": cfg.name}

    def _rpc_add_agent(self, params: dict[str, Any]) -> dict[str, Any]:
        self._ensure_kernel()
        if self._agent_manager is None:
            return {"error": "No agent manager"}
        from fusion_simulation.agent.config import AgentConfig, AgentRole

        # Callers: gui.api_add_agent -> handle_request("add_agent") -> _rpc_add_agent
        # Affected API: add_agent params gain optional api_key; falls back to kernel_config.mlx_api_key
        # Data schemas: AgentConfig(api_key=...) -> PolicyClient Authorization header
        # User instruction: "和~/fusion/fuison-simulation项目集成起来...最后要完成端到端测试,确保系统可用"
        cfg = AgentConfig(
            name=params.get("name", ""),
            role=AgentRole(params.get("role", "robot")),
            entity_id=params.get("entity_id", ""),
            action_dim=params.get("action_dim", 0),
            policy_endpoint=params.get("policy_endpoint", "")
            or (self._kernel_config.mlx_url.rstrip("/") + "/chat/completions"),
            model_name=params.get("model_name", "qwen3.5-9b"),
            api_key=params.get("api_key", "") or self._kernel_config.mlx_api_key,
        )
        self._agent_manager.add_agent(cfg)
        return {"status": "added", "name": cfg.name}

    def _rpc_get_observations(self, params: dict[str, Any]) -> dict[str, Any]:
        if self._sensor_manager is None:
            return {"error": "No sensor manager"}
        return self._sensor_manager.get_observations()

    def _rpc_pause(self, params: dict[str, Any]) -> dict[str, Any]:
        k = self._ensure_kernel()
        k.pause()
        return {"status": "paused"}

    def _rpc_resume(self, params: dict[str, Any]) -> dict[str, Any]:
        k = self._ensure_kernel()
        k.resume()
        return {"status": "resumed"}

    def subscribe_events(self) -> queue.Queue:
        q: queue.Queue = queue.Queue(maxsize=1000)
        self._event_subscribers.append(q)
        return q

    def unsubscribe_events(self, q: queue.Queue) -> None:
        try:
            self._event_subscribers.remove(q)
        except ValueError:
            pass

    @property
    def metrics_collector(self) -> MetricsCollector | None:
        return self._metrics_collector

    @property
    def gateway_client(self) -> GatewayClient | None:
        return self._gateway_client

    def get_health(self) -> HealthPayload:
        status = "healthy" if self._running else "stopped"
        kernel_state = ""
        frame_count = 0
        sim_time = 0.0
        sensor_count = 0
        agent_count = 0
        uptime = 0.0
        if self._kernel is not None:
            kernel_state = self._kernel.kernel_state.name
            frame_count = self._kernel.clock.frame_count
            sim_time = self._kernel.clock.sim_time
            if self._kernel.sensor_manager:
                sensor_count = len(self._kernel.sensor_manager._sensors)
            if self._kernel.agent_manager:
                agent_count = len(self._kernel.agent_manager._agents)
        return HealthPayload(
            status=status,
            kernel_state=kernel_state,
            frame_count=frame_count,
            sim_time=sim_time,
            sensor_count=sensor_count,
            agent_count=agent_count,
            uptime_seconds=uptime,
        )

    def _start_metrics(self) -> None:
        self._metrics_collector = MetricsCollector()
        self._metrics_server = MetricsServer(
            config=self._metrics_config,
            collector=self._metrics_collector,
        )
        self._metrics_server.set_health_provider(self.get_health)
        try:
            self._metrics_server.start()
        except Exception as e:
            logger.warning("Failed to start MetricsServer: %s", e)

    def _stop_metrics(self) -> None:
        if self._metrics_server is not None:
            self._metrics_server.stop()
            self._metrics_server = None

    def _start_gateway(self) -> None:
        cfg = self._gateway_config or GatewayConfig()
        self._gateway_client = GatewayClient(cfg)
        self._gateway_client.set_health_provider(self.get_health)
        try:
            self._gateway_client.register()
        except Exception as e:
            logger.warning("Gateway registration failed: %s", e)

    def _stop_gateway(self) -> None:
        if self._gateway_client is not None:
            try:
                self._gateway_client.deregister()
            except Exception as e:
                logger.debug("Gateway deregistration error: %s", e)
            self._gateway_client.close()
            self._gateway_client = None


try:
    from fusion_simulation.service.proto import simulation_pb2, simulation_pb2_grpc

    class _SimulationServicer(simulation_pb2_grpc.SimulationServiceServicer):
        def __init__(self, server: SimulationServer) -> None:
            self._server = server

        def Init(self, request, context):
            resp = self._server.handle_request(
                "init",
                {
                    "physics_dt": request.physics_dt,
                    "render_dt": request.render_dt,
                    "headless": request.headless,
                    "seed": request.seed,
                },
            )
            return simulation_pb2.InitResponse(
                initialized=resp.get("status") == "initialized",
                sim_time=0.0,
            )

        def Step(self, request, context):
            resp = self._server.handle_request("step", {"num_steps": request.num_steps})
            return simulation_pb2.StepResponse(
                sim_time=resp.get("sim_time", 0.0),
                frame_count=resp.get("frame_count", 0),
                physics_step_ms=resp.get("physics_step_ms", 0.0),
                sensor_collect_ms=resp.get("sensor_collect_ms", 0.0),
                agent_decide_ms=resp.get("agent_decide_ms", 0.0),
                render_ms=resp.get("render_ms", 0.0),
                total_ms=resp.get("total_ms", 0.0),
            )

        def Reset(self, request, context):
            self._server.handle_request("reset", {})
            return simulation_pb2.ResetResponse(reset=True)

        def GetStatus(self, request, context):
            resp = self._server.handle_request("status", {})
            return simulation_pb2.StatusResponse(
                initialized=resp.get("initialized", False),
                running=resp.get("running", False),
                sim_time=resp.get("sim_time", 0.0),
                frame_count=resp.get("frame_count", 0),
                entity_count=resp.get("entity_count", 0),
                real_time_factor=resp.get("real_time_factor", 0.0),
                state=resp.get("state", ""),
                paused=resp.get("paused", False),
            )

        def StreamEvents(self, request, context):
            event_kinds = set(request.event_kinds) if request.event_kinds else set()
            q = self._server.subscribe_events()
            try:
                while context.is_active():
                    try:
                        event = q.get(timeout=0.5)
                    except queue.Empty:
                        continue
                    kind_val = event.kind.value if hasattr(event.kind, "value") else str(event.kind)
                    if event_kinds and kind_val not in event_kinds:
                        continue
                    yield simulation_pb2.EventResponse(
                        kind=kind_val,
                        sim_time=event.data.get("sim_time", 0.0),
                        data_json=json.dumps(event.data, default=str),
                    )
            finally:
                self._server.unsubscribe_events(q)

        def LoadScene(self, request, context):
            if request.scene_json:
                params = json.loads(request.scene_json)
                params["name"] = request.name or params.get("name", "unnamed")
                resp = self._server._rpc_load_scene(params)
            else:
                resp = self._server._rpc_load_scene({"name": request.name})
            return simulation_pb2.LoadSceneResponse(
                loaded=resp.get("status") == "loaded",
                result_json=json.dumps(resp.get("result", {}), default=str),
            )

        def SaveSnapshot(self, request, context):
            resp = self._server.handle_request("save_snapshot", {"name": request.name})
            return simulation_pb2.SaveSnapshotResponse(snapshot_id=resp.get("snapshot_id", ""))

        def RestoreSnapshot(self, request, context):
            resp = self._server.handle_request("restore_snapshot", {"snapshot_id": request.snapshot_id})
            return simulation_pb2.RestoreSnapshotResponse(restored=resp.get("restored", False))

        def StreamSensorData(self, request, context):
            sensor_names = list(request.sensor_names) if request.sensor_names else []
            interval = request.interval_sec if request.interval_sec > 0 else 0.1
            max_frames = max(0, request.max_frames)
            frame_count = 0
            try:
                while context.is_active():
                    if max_frames > 0 and frame_count >= max_frames:
                        break
                    k = self._server.kernel
                    if k is None:
                        time.sleep(0.1)
                        continue
                    sm = k.sensor_manager
                    if sm is None:
                        time.sleep(0.1)
                        continue
                    all_obs = sm.get_observations()
                    for sname, sdata in all_obs.items():
                        if sensor_names and sname not in sensor_names:
                            continue
                        yield simulation_pb2.SensorDataResponse(
                            sim_time=k.clock.sim_time,
                            frame_count=k.clock.frame_count,
                            sensor_name=sname,
                            data_json=json.dumps(sdata, default=str),
                        )
                    frame_count += 1
                    time.sleep(interval)
            except Exception:
                logger.exception("StreamSensorData error")

        def StreamSimState(self, request_iterator, context):
            for req in request_iterator:
                if not context.is_active():
                    break
                command = req.command
                params = {}
                if req.params_json:
                    try:
                        params = json.loads(req.params_json)
                    except json.JSONDecodeError:
                        params = {}
                if command == "step":
                    resp = self._server.handle_request("step", params)
                elif command == "status":
                    resp = self._server.handle_request("status", params)
                elif command == "pause":
                    resp = self._server.handle_request("pause", params)
                elif command == "resume":
                    resp = self._server.handle_request("resume", params)
                elif command == "reset":
                    resp = self._server.handle_request("reset", params)
                else:
                    resp = {"error": f"Unknown command: {command}"}
                k = self._server.kernel
                yield simulation_pb2.SimStateResponse(
                    sim_time=k.clock.sim_time if k else 0.0,
                    frame_count=k.clock.frame_count if k else 0,
                    state_json=json.dumps(resp, default=str),
                )

except ImportError:
    logger.debug("gRPC proto modules not available, servicer not registered")
