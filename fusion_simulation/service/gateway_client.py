# GatewayClient: service registration, heartbeat, deregistration with Fusion-Gateway.
# Called by: SimulationServer.start() registers, SimulationServer.stop() deregisters.
# Affects API: GatewayClient, GatewayConfig.
# Data schemas: GatewayConfig(gateway_url, service_name, service_port, heartbeat_interval, api_key),
# RegistrationPayload, HealthPayload.
# User instruction: refactor fusion-simulation per PRD, target NVIDIA Isaac Sim competitiveness.
from __future__ import annotations

import logging
import os
import threading
import time
import uuid
from dataclasses import dataclass, field

import httpx

logger = logging.getLogger(__name__)


def _default_instance_id() -> str:
    return os.environ.get("FUSION_SIM_INSTANCE_ID") or str(uuid.uuid4())


@dataclass
class GatewayConfig:
    gateway_url: str = "http://127.0.0.1:11432"
    service_name: str = "fusion-simulation"
    service_host: str = "127.0.0.1"
    service_port: int = 11447
    metrics_port: int = 11456
    heartbeat_interval: float = 10.0
    api_key: str = ""
    enabled: bool = False
    instance_id: str = field(default_factory=_default_instance_id)


@dataclass
class RegistrationPayload:
    name: str = ""
    host: str = ""
    port: int = 0
    metrics_port: int = 0
    protocol: str = "grpc"
    routes: list[str] = field(default_factory=list)
    health_endpoint: str = ""
    metadata: dict[str, str] = field(default_factory=dict)
    instance_id: str = ""


@dataclass
class HealthPayload:
    status: str = "unknown"
    kernel_state: str = ""
    frame_count: int = 0
    sim_time: float = 0.0
    sensor_count: int = 0
    agent_count: int = 0
    uptime_seconds: float = 0.0


class GatewayClient:
    """Registers fusion-simulation as a service with Fusion-Gateway.

    Callers: SimulationServer.start() → register(), SimulationServer.stop() → deregister().
    API: GatewayClient(config), register(routes), deregister(), send_heartbeat(), close().
    """

    def __init__(self, config: GatewayConfig | None = None):
        self._config = config or GatewayConfig()
        self._client = httpx.Client(timeout=5.0)
        self._registered = False
        self._heartbeat_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._start_time = 0.0
        self._health_provider = None
        self._heartbeat_failures = 0
        self._reregister_threshold = 3

    @property
    def is_registered(self) -> bool:
        return self._registered

    def set_health_provider(self, provider) -> None:
        self._health_provider = provider

    def register(self, routes: list[str] | None = None) -> bool:
        if not self._config.enabled:
            logger.info("Gateway integration disabled, skipping registration")
            self._registered = True
            self._start_time = time.time()
            return True
        default_routes = [
            "/v1/simulation/init",
            "/v1/simulation/step",
            "/v1/simulation/reset",
            "/v1/simulation/status",
            "/v1/simulation/pause",
            "/v1/simulation/resume",
            "/v1/simulation/scene/*",
            "/v1/simulation/agent/*",
            "/v1/simulation/sensor/*",
            "/v1/simulation/snapshot/*",
            "/v1/simulation/stream/*",
        ]
        payload = RegistrationPayload(
            name=self._config.service_name,
            host=self._config.service_host,
            port=self._config.service_port,
            metrics_port=self._config.metrics_port,
            routes=routes or default_routes,
            health_endpoint=f"http://{self._config.service_host}:{self._config.metrics_port}/health",
            metadata={
                "protocol": "grpc",
                "version": "0.1.0",
                "description": "Fusion-Simulation gRPC service",
                "instance_id": self._config.instance_id,
            },
            instance_id=self._config.instance_id,
        )
        url = f"{self._config.gateway_url}/v1/services/register"
        headers = self._build_headers()
        try:
            resp = self._client.post(
                url,
                json=self._payload_to_dict(payload),
                headers=headers,
            )
            if resp.status_code in (200, 201):
                self._registered = True
                self._start_time = time.time()
                self._start_heartbeat()
                logger.info(
                    "Registered with Fusion-Gateway at %s",
                    self._config.gateway_url,
                )
                return True
            logger.warning(
                "Gateway registration failed: %d %s",
                resp.status_code,
                resp.text[:200],
            )
            return False
        except Exception as e:
            logger.warning("Gateway registration error: %s", e)
            return False

    def deregister(self) -> bool:
        if not self._registered:
            return True
        if not self._config.enabled:
            self._registered = False
            self._stop_heartbeat()
            logger.debug("Gateway disabled, marking deregistered without contacting gateway")
            return True
        self._stop_heartbeat()
        url = f"{self._config.gateway_url}/v1/services/deregister"
        headers = self._build_headers()
        payload = {
            "name": self._config.service_name,
            "host": self._config.service_host,
            "port": self._config.service_port,
            "instance_id": self._config.instance_id,
        }
        try:
            resp = self._client.post(url, json=payload, headers=headers)
            self._registered = False
            if resp.status_code in (200, 204):
                logger.info("Deregistered from Fusion-Gateway")
                return True
            logger.warning("Gateway deregistration failed: %d", resp.status_code)
            return False
        except Exception as e:
            self._registered = False
            logger.warning("Gateway deregistration error: %s", e)
            return False

    def send_heartbeat(self) -> bool:
        if not self._registered:
            return False
        health = self._get_health()
        url = f"{self._config.gateway_url}/v1/services/heartbeat"
        headers = self._build_headers()
        payload = {
            "name": self._config.service_name,
            "host": self._config.service_host,
            "port": self._config.service_port,
            "instance_id": self._config.instance_id,
            "health": self._health_to_dict(health),
        }
        try:
            resp = self._client.post(url, json=payload, headers=headers)
            return resp.status_code == 200
        except Exception as e:
            logger.debug("Heartbeat failed: %s", e)
            return False

    def _get_health(self) -> HealthPayload:
        if self._health_provider is not None:
            try:
                return self._health_provider()
            except Exception as e:
                logger.debug("Health provider error: %s", e)
        return HealthPayload(
            status="degraded",
            uptime_seconds=time.time() - self._start_time if self._start_time else 0.0,
        )

    def _start_heartbeat(self) -> None:
        if self._heartbeat_thread is not None and self._heartbeat_thread.is_alive():
            return
        self._stop_event.clear()
        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop,
            daemon=True,
        )
        self._heartbeat_thread.start()

    def _stop_heartbeat(self) -> None:
        self._stop_event.set()
        if self._heartbeat_thread is not None:
            self._heartbeat_thread.join(timeout=3.0)
            self._heartbeat_thread = None

    def _heartbeat_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                ok = self.send_heartbeat()
                if ok:
                    self._heartbeat_failures = 0
                else:
                    self._heartbeat_failures += 1
                    if self._heartbeat_failures >= self._reregister_threshold:
                        logger.warning(
                            "Heartbeat failed %d times, attempting re-registration with gateway",
                            self._heartbeat_failures,
                        )
                        self._heartbeat_failures = 0
                        try:
                            self.register()
                        except Exception as e:
                            logger.warning("Re-registration attempt failed: %s", e)
            except Exception as e:
                logger.debug("Heartbeat loop error: %s", e)
            self._stop_event.wait(timeout=self._config.heartbeat_interval)

    def _build_headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self._config.api_key:
            headers["Authorization"] = f"Bearer {self._config.api_key}"
        return headers

    @staticmethod
    def _payload_to_dict(payload: RegistrationPayload) -> dict:
        return {
            "name": payload.name,
            "host": payload.host,
            "port": payload.port,
            "metrics_port": payload.metrics_port,
            "protocol": payload.protocol,
            "routes": payload.routes,
            "health_endpoint": payload.health_endpoint,
            "metadata": payload.metadata,
            "instance_id": payload.instance_id,
        }

    @staticmethod
    def _health_to_dict(health: HealthPayload) -> dict:
        return {
            "status": health.status,
            "kernel_state": health.kernel_state,
            "frame_count": health.frame_count,
            "sim_time": health.sim_time,
            "sensor_count": health.sensor_count,
            "agent_count": health.agent_count,
            "uptime_seconds": health.uptime_seconds,
        }

    def close(self) -> None:
        self.deregister()
        self._client.close()
