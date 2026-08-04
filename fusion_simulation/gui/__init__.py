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
# Callers: cli._cmd_service_start -> run_dashboard(server, GUIConfig); app.api_env_check
# Affected API: GUIConfig gains mlx_url / mlx_api_key for authenticated env_check probe
# Data schemas: GUIConfig(host, port, metrics_url, grpc_host, grpc_port, mlx_url, mlx_api_key)
# User instruction: "和~/fusion/fuison-simulation项目集成起来...最后要完成端到端测试，确保系统可用"
class GUIConfig:
    host: str = "0.0.0.0"
    port: int = 11455
    metrics_url: str = "http://127.0.0.1:11456"
    grpc_host: str = "0.0.0.0"
    grpc_port: int = 11447
    mlx_url: str = "http://127.0.0.1:11434/v1"
    mlx_api_key: str = ""
