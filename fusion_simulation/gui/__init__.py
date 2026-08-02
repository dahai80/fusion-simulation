# GUI module — Web Dashboard for Fusion-Simulation
# Callers: cli service start --gui, SimulationServer.start()
# API: GUIConfig, create_app(), run_dashboard()
# Data schemas: GUIConfig(host, port, metrics_url, grpc_host, grpc_port)
# User instruction: implement Web Dashboard per PRD Section 7 GUI specs
from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class GUIConfig:
    host: str = "0.0.0.0"
    port: int = 11455
    metrics_url: str = "http://127.0.0.1:11456"
    grpc_host: str = "0.0.0.0"
    grpc_port: int = 11447
