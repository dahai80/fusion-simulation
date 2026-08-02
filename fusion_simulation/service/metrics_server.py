# MetricsServer: HTTP server exposing /health and /metrics (Prometheus format) on port 11456.
# Called by: SimulationServer.start() starts metrics server.
# Affects API: MetricsServer, MetricsCollector.
# Data schemas: MetricsConfig(host, port), metric counters/gauges/histograms.
# User instruction: refactor fusion-simulation per PRD, target NVIDIA Isaac Sim competitiveness.
from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, HTTPServer

logger = logging.getLogger(__name__)


@dataclass
class MetricsConfig:
    host: str = "0.0.0.0"
    port: int = 11456


class MetricsCollector:
    def __init__(self):
        self._counters: dict[str, float] = {}
        self._gauges: dict[str, float] = {}
        self._histograms: dict[str, list[float]] = {}
        self._start_time = time.time()
        self._lock = threading.Lock()

    def inc_counter(self, name: str, value: float = 1.0, labels: dict[str, str] | None = None) -> None:
        key = self._make_key(name, labels)
        with self._lock:
            self._counters[key] = self._counters.get(key, 0.0) + value

    def set_gauge(self, name: str, value: float, labels: dict[str, str] | None = None) -> None:
        key = self._make_key(name, labels)
        with self._lock:
            self._gauges[key] = value

    def observe_histogram(self, name: str, value: float, labels: dict[str, str] | None = None) -> None:
        key = self._make_key(name, labels)
        with self._lock:
            if key not in self._histograms:
                self._histograms[key] = []
            self._histograms[key].append(value)
            if len(self._histograms[key]) > 1000:
                self._histograms[key] = self._histograms[key][-500:]

    def get_counter(self, name: str, labels: dict[str, str] | None = None) -> float:
        key = self._make_key(name, labels)
        with self._lock:
            return self._counters.get(key, 0.0)

    def get_gauge(self, name: str, labels: dict[str, str] | None = None) -> float:
        key = self._make_key(name, labels)
        with self._lock:
            return self._gauges.get(key, 0.0)

    def to_prometheus(self) -> str:
        lines = []
        with self._lock:
            lines.append("# TYPE fusion_sim_uptime_seconds gauge")
            lines.append(f"fusion_sim_uptime_seconds {time.time() - self._start_time:.2f}")
            for key, val in sorted(self._counters.items()):
                name, labels = self._parse_key(key)
                lines.append(f"# TYPE {name} counter")
                lines.append(f"{name}{labels} {val:.6g}")
            for key, val in sorted(self._gauges.items()):
                name, labels = self._parse_key(key)
                lines.append(f"# TYPE {name} gauge")
                lines.append(f"{name}{labels} {val:.6g}")
            for key, vals in sorted(self._histograms.items()):
                name, labels = self._parse_key(key)
                if not vals:
                    continue
                sorted_vals = sorted(vals)
                count = len(sorted_vals)
                total = sum(sorted_vals)
                lines.append(f"# TYPE {name} summary")
                lines.append(f"{name}_count{labels} {count}")
                lines.append(f"{name}_sum{labels} {total:.6g}")
                for q in [0.5, 0.9, 0.95, 0.99]:
                    idx = min(int(count * q), count - 1)
                    label_inner = labels[1:-1] if labels else ""
                    prefix = f"{label_inner}," if label_inner else ""
                    lines.append(
                        f'{name}{{{prefix}quantile="{q}"}} {sorted_vals[idx]:.6g}'
                    )
        return "\n".join(lines) + "\n"

    @staticmethod
    def _make_key(name: str, labels: dict[str, str] | None = None) -> str:
        if not labels:
            return name
        label_str = ",".join(f'{k}="{v}"' for k, v in sorted(labels.items()))
        return f"{name}{{{label_str}}}"

    @staticmethod
    def _parse_key(key: str) -> tuple[str, str]:
        if "{" not in key:
            return key, ""
        name, rest = key.split("{", 1)
        labels = "{" + rest
        return name, labels

    def reset(self) -> None:
        with self._lock:
            self._counters.clear()
            self._gauges.clear()
            self._histograms.clear()
            self._start_time = time.time()


class _MetricsHandler(BaseHTTPRequestHandler):
    _collector: MetricsCollector | None = None
    _health_provider = None

    def do_GET(self) -> None:
        if self.path == "/metrics":
            self._send_prometheus()
        elif self.path == "/health":
            self._send_health()
        else:
            self.send_response(404)
            self.end_headers()

    def _send_prometheus(self) -> None:
        if self._collector is None:
            self.send_response(503)
            self.end_headers()
            return
        body = self._collector.to_prometheus().encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_health(self) -> None:
        if self._health_provider is not None:
            try:
                health = self._health_provider()
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
                body = json.dumps(payload).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            except Exception as e:
                logger.debug("Health provider error: %s", e)
        body = b'{"status":"unknown"}'
        self.send_response(503)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args) -> None:
        logger.debug("MetricsServer: %s", format % args)


class MetricsServer:
    """HTTP server exposing /health and /metrics endpoints.

    Callers: SimulationServer.start() → start(), SimulationServer.stop() → stop().
    API: MetricsServer(config, collector), start(), stop(), set_health_provider(provider).
    """

    def __init__(self, config: MetricsConfig | None = None, collector: MetricsCollector | None = None):
        self._config = config or MetricsConfig()
        self._collector = collector or MetricsCollector()
        self._server: HTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._health_provider = None

    @property
    def collector(self) -> MetricsCollector:
        return self._collector

    def set_health_provider(self, provider) -> None:
        self._health_provider = provider

    def start(self) -> None:
        collector = self._collector
        health_provider = self._health_provider

        class Handler(_MetricsHandler):
            def __init__(self, *args, **kwargs):
                self._collector = collector
                self._health_provider = health_provider
                super().__init__(*args, **kwargs)

        self._server = HTTPServer(
            (self._config.host, self._config.port), Handler,
        )
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        logger.info("MetricsServer started on %s:%d", self._config.host, self._config.port)

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server = None
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        logger.info("MetricsServer stopped")
