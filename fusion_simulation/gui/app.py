# Web Dashboard FastAPI application — serves static HTML/CSS/JS GUI
# Callers: gui.run_dashboard(), cli service start --gui
# API: create_app(server) returns FastAPI, endpoints: /api/*, /ws/events, static files
# Data schemas: JSON REST API mirroring SimulationServer.handle_request()
# User instruction: implement Web Dashboard per PRD Section 7 GUI specs
from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from fusion_simulation.gui import GUIConfig

logger = logging.getLogger(__name__)

_STATIC_DIR = Path(__file__).parent / "static"


def create_app(server=None, config: GUIConfig | None = None) -> FastAPI:
    cfg = config or GUIConfig()
    app = FastAPI(title="Fusion-Simulation Dashboard", version="0.1.1")

    app.state.sim_server = server
    app.state.gui_config = cfg
    app.state.ws_clients: list[WebSocket] = []

    # ---- REST API ----

    @app.get("/api/health")
    async def api_health():
        srv = app.state.sim_server
        if srv is None:
            return JSONResponse({"status": "no_server", "initialized": False})
        return JSONResponse(srv.handle_request("status", {}))

    @app.get("/api/status")
    async def api_status():
        srv = app.state.sim_server
        if srv is None:
            return JSONResponse({"initialized": False})
        return JSONResponse(srv.handle_request("status", {}))

    @app.post("/api/init")
    async def api_init():
        srv = app.state.sim_server
        if srv is None:
            return JSONResponse({"error": "no_server"}, status_code=503)
        return JSONResponse(srv.handle_request("init", {}))

    @app.post("/api/step")
    async def api_step(num_steps: int = 1):
        srv = app.state.sim_server
        if srv is None:
            return JSONResponse({"error": "no_server"}, status_code=503)
        result = srv.handle_request("step", {"num_steps": num_steps})
        await _broadcast(app, result)
        return JSONResponse(result)

    @app.post("/api/reset")
    async def api_reset():
        srv = app.state.sim_server
        if srv is None:
            return JSONResponse({"error": "no_server"}, status_code=503)
        result = srv.handle_request("reset", {})
        await _broadcast(app, result)
        return JSONResponse(result)

    @app.post("/api/pause")
    async def api_pause():
        srv = app.state.sim_server
        if srv is None:
            return JSONResponse({"error": "no_server"}, status_code=503)
        result = srv.handle_request("pause", {})
        await _broadcast(app, result)
        return JSONResponse(result)

    @app.post("/api/resume")
    async def api_resume():
        srv = app.state.sim_server
        if srv is None:
            return JSONResponse({"error": "no_server"}, status_code=503)
        result = srv.handle_request("resume", {})
        await _broadcast(app, result)
        return JSONResponse(result)

    @app.post("/api/load_scene")
    async def api_load_scene(name: str = "default"):
        srv = app.state.sim_server
        if srv is None:
            return JSONResponse({"error": "no_server"}, status_code=503)
        result = srv.handle_request("load_scene", {"name": name})
        await _broadcast(app, result)
        return JSONResponse(result)

    @app.post("/api/save_snapshot")
    async def api_save_snapshot(name: str = ""):
        srv = app.state.sim_server
        if srv is None:
            return JSONResponse({"error": "no_server"}, status_code=503)
        result = srv.handle_request("save_snapshot", {"name": name})
        return JSONResponse(result)

    @app.post("/api/restore_snapshot")
    async def api_restore_snapshot(snapshot_id: str = ""):
        srv = app.state.sim_server
        if srv is None:
            return JSONResponse({"error": "no_server"}, status_code=503)
        result = srv.handle_request("restore_snapshot", {"snapshot_id": snapshot_id})
        return JSONResponse(result)

    @app.post("/api/add_sensor")
    async def api_add_sensor(type: str = "rgb_camera", name: str = "", entity_id: str = ""):
        srv = app.state.sim_server
        if srv is None:
            return JSONResponse({"error": "no_server"}, status_code=503)
        result = srv.handle_request(
            "add_sensor",
            {
                "type": type,
                "name": name,
                "entity_id": entity_id,
            },
        )
        await _broadcast(app, result)
        return JSONResponse(result)

    @app.post("/api/add_agent")
    async def api_add_agent(
        name: str = "agent0",
        role: str = "robot",
        action_dim: int = 6,
        entity_id: str = "",
        model_name: str = os.environ.get("FUSION_MLX_MODEL", "Qwen3.5-4B-bf16"),
    ):
        srv = app.state.sim_server
        if srv is None:
            return JSONResponse({"error": "no_server"}, status_code=503)
        result = srv.handle_request(
            "add_agent",
            {
                "name": name,
                "role": role,
                "action_dim": action_dim,
                "entity_id": entity_id,
                "model_name": model_name,
            },
        )
        await _broadcast(app, result)
        return JSONResponse(result)

    @app.get("/api/observations")
    async def api_observations():
        srv = app.state.sim_server
        if srv is None:
            return JSONResponse({"error": "no_server"}, status_code=503)
        return JSONResponse(srv.handle_request("get_observations", {}))

    @app.get("/api/metrics")
    async def api_metrics():
        import httpx

        url = cfg.metrics_url.rstrip("/") + "/metrics"
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                resp = await client.get(url)
                if resp.status_code == 200:
                    return JSONResponse({"prometheus": resp.text})
        except Exception as e:
            logger.debug("Metrics fetch failed: %s", e)
        return JSONResponse({"prometheus": ""})

    @app.get("/api/env_check")
    async def api_env_check():
        checks = {}
        try:
            import pybullet

            checks["pybullet"] = {"available": True, "version": getattr(pybullet, "__version__", "unknown")}
        except ImportError:
            checks["pybullet"] = {"available": False}
        try:
            import grpc

            checks["grpc"] = {"available": True, "version": getattr(grpc, "__version__", "unknown")}
        except ImportError:
            checks["grpc"] = {"available": False}
        # Callers: GET /api/env_check (fusion-studio SimulationBridge.envCheckRequest)
        # Affected API: env_check fusion_mlx probe now sends Authorization + uses cfg.mlx_url
        # Data schemas: fusion_mlx probe uses GUIConfig.mlx_url / mlx_api_key
        # User instruction: "和~/fusion/fuison-simulation项目集成起来...最后要完成端到端测试,确保系统可用"
        try:
            import httpx

            mlx_headers = {"X-Fusion-Route": os.environ.get("FUSION_MLX_ROUTE", "mlx")}
            if cfg.mlx_api_key:
                mlx_headers["Authorization"] = f"Bearer {cfg.mlx_api_key}"
            async with httpx.AsyncClient(timeout=2.0) as client:
                resp = await client.get(f"{cfg.mlx_url}/models", headers=mlx_headers)
                checks["fusion_mlx"] = {"available": resp.status_code == 200}
        except Exception:
            checks["fusion_mlx"] = {"available": False}
        try:
            import httpx

            metrics_port = cfg.metrics_url.split(":")[-1]
            async with httpx.AsyncClient(timeout=2.0) as client:
                resp = await client.get(f"http://127.0.0.1:{metrics_port}/health")
                checks["simulation_service"] = {"available": resp.status_code == 200}
        except Exception:
            checks["simulation_service"] = {"available": False}
        return JSONResponse(checks)

    # ---- WebSocket ----

    @app.websocket("/ws/events")
    async def ws_events(websocket: WebSocket):
        await websocket.accept()
        app.state.ws_clients.append(websocket)
        try:
            while True:
                data = await websocket.receive_text()
                try:
                    msg = json.loads(data)
                    cmd = msg.get("command", "")
                    params = msg.get("params", {})
                    srv = app.state.sim_server
                    if srv and cmd:
                        result = srv.handle_request(cmd, params)
                        await websocket.send_json(result)
                except json.JSONDecodeError:
                    await websocket.send_json({"error": "invalid json"})
        except WebSocketDisconnect:
            pass
        finally:
            if websocket in app.state.ws_clients:
                app.state.ws_clients.remove(websocket)

    # ---- Static files (must be last mount) ----

    if _STATIC_DIR.exists():
        app.mount("/", StaticFiles(directory=str(_STATIC_DIR), html=True), name="static")

    return app


async def _broadcast(app: FastAPI, data: dict) -> None:
    dead = []
    for i, ws in enumerate(app.state.ws_clients):
        try:
            await ws.send_json(data)
        except Exception:
            dead.append(i)
    for i in reversed(dead):
        app.state.ws_clients.pop(i)


def run_dashboard(server=None, config: GUIConfig | None = None) -> None:
    import uvicorn

    cfg = config or GUIConfig()
    app = create_app(server=server, config=cfg)

    def _event_broadcaster():
        if server is None:
            return
        event_q = server.subscribe_events()
        while server.is_running:
            try:
                event = event_q.get(timeout=1.0)
                loop = asyncio.new_event_loop()
                try:
                    clients = list(app.state.ws_clients)
                    for ws in clients:
                        try:
                            data = {
                                "type": "event",
                                "kind": event.kind.value if hasattr(event.kind, "value") else str(event.kind),
                                "sim_time": event.data.get("sim_time", 0.0),
                                "data": event.data,
                            }
                            loop.run_until_complete(ws.send_json(data))
                        except Exception:
                            pass
                finally:
                    loop.close()
            except Exception:
                pass
        server.unsubscribe_events(event_q)

    t = threading.Thread(target=_event_broadcaster, daemon=True)
    t.start()

    logger.info("Starting dashboard on %s:%d", cfg.host, cfg.port)
    uvicorn.run(app, host=cfg.host, port=cfg.port, log_level="info")
