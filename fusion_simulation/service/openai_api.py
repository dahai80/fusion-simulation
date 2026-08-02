from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

logger = logging.getLogger(__name__)

_SIM_COMMAND_MAP = {
    "init": "init",
    "step": "step",
    "reset": "reset",
    "close": "close",
    "status": "status",
    "load_scene": "load_scene",
    "save_snapshot": "save_snapshot",
    "restore_snapshot": "restore_snapshot",
    "add_sensor": "add_sensor",
    "add_agent": "add_agent",
    "get_observations": "get_observations",
    "pause": "pause",
    "resume": "resume",
    "create_simulation": "create_simulation",
    "start_simulation": "start_simulation",
    "pause_simulation": "pause_simulation",
    "reset_simulation": "reset_simulation",
    "destroy_simulation": "destroy_simulation",
    "spawn_agent": "spawn_agent",
    "destroy_agent": "destroy_agent",
    "apply_action": "apply_action",
    "list_scenes": "list_scenes",
}


@dataclass
class OpenAI_API_Config:
    host: str = "0.0.0.0"
    port: int = 11434
    auth_enabled: bool = False
    auth_token: str = ""


class _OpenAI_API_Handler(BaseHTTPRequestHandler):
    _server_ref: Any = None
    _auth_token: str = ""
    _auth_enabled: bool = False

    def do_GET(self) -> None:
        if self._check_auth() is False:
            return
        if self.path == "/v1/models":
            self._handle_list_models()
        elif self.path == "/v1/health":
            self._handle_health()
        else:
            self._send_json({"error": {"message": f"Not found: {self.path}", "type": "not_found"}}, status=404)

    def do_POST(self) -> None:
        if self._check_auth() is False:
            return
        if self.path == "/v1/chat/completions":
            self._handle_chat_completions()
        else:
            self._send_json({"error": {"message": f"Not found: {self.path}", "type": "not_found"}}, status=404)

    def _check_auth(self) -> bool | None:
        if not self._auth_enabled:
            return True
        auth_header = self.headers.get("Authorization", "")
        if auth_header == f"Bearer {self._auth_token}":
            return True
        self._send_json(
            {"error": {"message": "Invalid or missing API key", "type": "authentication_error"}},
            status=401,
        )
        return False

    def _handle_list_models(self) -> None:
        models = [
            {
                "id": "fusion-simulation",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "fusion-simulation",
            },
            {
                "id": "fusion-simulation-policy",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "fusion-simulation",
            },
        ]
        if self._server_ref is not None:
            kernel = self._server_ref.kernel
            if kernel is not None and kernel.agent_manager is not None:
                for agent_id, agent in kernel.agent_manager._agents.items():
                    model_name = getattr(agent, "model_name", "unknown")
                    models.append(
                        {
                            "id": model_name,
                            "object": "model",
                            "created": int(time.time()),
                            "owned_by": "fusion-simulation",
                        }
                    )
        self._send_json(
            {
                "object": "list",
                "data": models,
            }
        )

    def _handle_chat_completions(self) -> None:
        content_length = int(self.headers.get("Content-Length", 0))
        if content_length == 0:
            self._send_json(
                {"error": {"message": "Empty request body", "type": "invalid_request_error"}},
                status=400,
            )
            return
        body = self.rfile.read(content_length)
        try:
            data = json.loads(body)
        except json.JSONDecodeError:
            self._send_json(
                {"error": {"message": "Invalid JSON", "type": "invalid_request_error"}},
                status=400,
            )
            return
        model = data.get("model", "fusion-simulation")
        messages = data.get("messages", [])
        stream = data.get("stream", False)
        if not messages:
            self._send_json(
                {"error": {"message": "messages is required", "type": "invalid_request_error"}},
                status=400,
            )
            return
        last_message = messages[-1].get("content", "")
        if isinstance(last_message, list):
            text_parts = [p.get("text", "") for p in last_message if p.get("type") == "text"]
            last_message = " ".join(text_parts)
        sim_result = self._dispatch_sim_command(last_message)
        response_text = json.dumps(sim_result, default=str, ensure_ascii=False)
        request_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        created = int(time.time())
        if stream:
            self._send_streaming_response(request_id, created, model, response_text)
        else:
            self._send_json(
                {
                    "id": request_id,
                    "object": "chat.completion",
                    "created": created,
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": response_text,
                            },
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": len(last_message),
                        "completion_tokens": len(response_text),
                        "total_tokens": len(last_message) + len(response_text),
                    },
                }
            )

    def _dispatch_sim_command(self, text: str) -> dict[str, Any]:
        if self._server_ref is None:
            return {"error": "server not initialized"}
        try:
            parsed = json.loads(text)
        except (json.JSONDecodeError, TypeError):
            return {"result": text}
        if not isinstance(parsed, dict):
            return {"result": text}
        command = parsed.pop("command", "")
        method = _SIM_COMMAND_MAP.get(command)
        if method is None:
            return {"error": f"Unknown simulation command: {command}"}
        try:
            result = self._server_ref.handle_request(method, parsed)
            logger.info("OpenAI API dispatched command=%s", command)
            return result
        except Exception as e:
            logger.exception("OpenAI API dispatch error: %s", command)
            return {"error": str(e)}

    def _handle_health(self) -> None:
        if self._server_ref is not None:
            try:
                health = self._server_ref.get_health()
                from fusion_simulation.service.gateway_client import HealthPayload

                if isinstance(health, HealthPayload):
                    payload = {
                        "status": health.status,
                        "kernel_state": health.kernel_state,
                        "frame_count": health.frame_count,
                        "sim_time": health.sim_time,
                        "sensor_count": health.sensor_count,
                        "agent_count": health.agent_count,
                        "uptime_seconds": health.uptime_seconds,
                    }
                else:
                    payload = health if isinstance(health, dict) else {"status": "unknown"}
                self._send_json(payload)
                return
            except Exception as e:
                logger.debug("OpenAI health provider error: %s", e)
        self._send_json({"status": "unknown"}, status=503)

    def _send_streaming_response(self, request_id: str, created: int, model: str, text: str) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        chunk_data = {
            "id": request_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": text},
                    "finish_reason": None,
                }
            ],
        }
        self.wfile.write(f"data: {json.dumps(chunk_data)}\n\n".encode())
        done_data = {
            "id": request_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop",
                }
            ],
        }
        self.wfile.write(f"data: {json.dumps(done_data)}\n\n".encode())
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()

    def _send_json(self, data: dict, status: int = 200) -> None:
        body = json.dumps(data, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args) -> None:
        logger.debug("OpenAI_API: %s", format % args)


class OpenAI_API_Server:
    def __init__(self, config: OpenAI_API_Config | None = None) -> None:
        self._config = config or OpenAI_API_Config()
        self._server: HTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._sim_server: Any = None

    def set_simulation_server(self, server: Any) -> None:
        self._sim_server = server

    def start(self) -> None:
        sim_server = self._sim_server
        auth_token = self._config.auth_token
        auth_enabled = self._config.auth_enabled

        class Handler(_OpenAI_API_Handler):
            def __init__(self, *args, **kwargs):
                self._server_ref = sim_server
                self._auth_token = auth_token
                self._auth_enabled = auth_enabled
                super().__init__(*args, **kwargs)

        self._server = HTTPServer(
            (self._config.host, self._config.port),
            Handler,
        )
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        logger.info(
            "OpenAI_API_Server started on %s:%d (auth=%s)",
            self._config.host,
            self._config.port,
            auth_enabled,
        )

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server = None
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        logger.info("OpenAI_API_Server stopped")
