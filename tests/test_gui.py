from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def mock_server():
    server = MagicMock()
    server.handle_request.return_value = {"state": "IDLE", "frame_count": 0, "sim_time": 0.0}
    return server


@pytest.fixture
def app(mock_server):
    from fusion_simulation.gui import GUIConfig
    from fusion_simulation.gui.app import create_app

    config = GUIConfig()
    return create_app(mock_server, config)


@pytest.fixture
def client(app):
    from starlette.testclient import TestClient

    return TestClient(app)


class TestGUIConfig:
    def test_defaults(self):
        from fusion_simulation.gui import GUIConfig

        cfg = GUIConfig()
        assert cfg.host == "0.0.0.0"
        assert cfg.port == 8080
        assert cfg.metrics_url == "http://127.0.0.1:8081"
        assert cfg.grpc_host == "0.0.0.0"
        assert cfg.grpc_port == 50051


class TestRESTEndpoints:
    def test_health(self, client, mock_server):
        mock_server.handle_request.return_value = {
            "status": "healthy",
            "kernel_state": "IDLE",
            "frame_count": 0,
            "sim_time": 0.0,
            "sensor_count": 0,
            "agent_count": 0,
        }
        resp = client.get("/api/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"

    def test_status(self, client, mock_server):
        mock_server.handle_request.return_value = {
            "state": "IDLE",
            "frame_count": 0,
            "sim_time": 0.0,
        }
        resp = client.get("/api/status")
        assert resp.status_code == 200

    def test_init(self, client, mock_server):
        mock_server.handle_request.return_value = {"state": "IDLE", "frame_count": 0}
        resp = client.post("/api/init", json={})
        assert resp.status_code == 200
        mock_server.handle_request.assert_called()

    def test_step(self, client, mock_server):
        mock_server.handle_request.return_value = {
            "sim_time": 0.01,
            "frame_count": 1,
            "total_ms": 5.0,
        }
        resp = client.post("/api/step", json={"num_steps": 1})
        assert resp.status_code == 200
        data = resp.json()
        assert data["frame_count"] == 1

    def test_reset(self, client, mock_server):
        mock_server.handle_request.return_value = {"state": "IDLE"}
        resp = client.post("/api/reset", json={})
        assert resp.status_code == 200

    def test_pause(self, client, mock_server):
        mock_server.handle_request.return_value = {"state": "PAUSED"}
        resp = client.post("/api/pause", json={})
        assert resp.status_code == 200

    def test_resume(self, client, mock_server):
        mock_server.handle_request.return_value = {"state": "RUNNING"}
        resp = client.post("/api/resume", json={})
        assert resp.status_code == 200

    def test_load_scene(self, client, mock_server):
        mock_server.handle_request.return_value = {"result": "ok"}
        resp = client.post("/api/load_scene", json={"name": "default"})
        assert resp.status_code == 200

    def test_save_snapshot(self, client, mock_server):
        mock_server.handle_request.return_value = {"snapshot_id": "snap_1"}
        resp = client.post("/api/save_snapshot", json={"name": "test"})
        assert resp.status_code == 200

    def test_restore_snapshot(self, client, mock_server):
        mock_server.handle_request.return_value = {"result": "ok"}
        resp = client.post("/api/restore_snapshot", json={"snapshot_id": "snap_1"})
        assert resp.status_code == 200

    def test_add_sensor(self, client, mock_server):
        mock_server.handle_request.return_value = {"result": "ok"}
        resp = client.post("/api/add_sensor", json={"type": "rgb_camera", "name": "cam0"})
        assert resp.status_code == 200

    def test_add_agent(self, client, mock_server):
        mock_server.handle_request.return_value = {"result": "ok"}
        resp = client.post("/api/add_agent", json={"name": "robot0", "role": "robot", "action_dim": 6})
        assert resp.status_code == 200

    def test_observations(self, client, mock_server):
        mock_server.handle_request.return_value = {"observations": {}}
        resp = client.get("/api/observations")
        assert resp.status_code == 200

    def test_env_check(self, client, mock_server):
        with patch("httpx.get") as mock_get:
            mock_get.return_value = MagicMock(
                status_code=200,
                json=lambda: {"status": "healthy"},
            )
            resp = client.get("/api/env_check")
            assert resp.status_code == 200


class TestWebSocket:
    def test_ws_connect(self, client):
        with client.websocket_connect("/ws/events") as ws:
            pass


class TestStaticFiles:
    def test_index_html_served(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        assert b"Fusion-Simulation" in resp.content
