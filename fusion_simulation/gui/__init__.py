# GUI module — Web Dashboard for Fusion-Simulation
# Callers: cli service start --gui, SimulationServer.start()
# API: GUIConfig, create_app(), run_dashboard()
# Data schemas: GUIConfig(host, port, metrics_url, grpc_host, grpc_port)
# User instruction: implement Web Dashboard per PRD Section 7 GUI specs
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


def _default_gui_mlx_url() -> str:
    return os.environ.get("FUSION_MLX_URL", "http://127.0.0.1:11434/v1")


def _default_gui_mlx_api_key() -> str:
    return os.environ.get("FUSION_MLX_API_KEY", "")


@dataclass
# Callers: cli._cmd_service_start -> run_dashboard(server, GUIConfig); app.api_env_check
# Affected API: GUIConfig gains mlx_url / mlx_api_key for authenticated env_check probe
# Data schemas: GUIConfig(host, port, metrics_url, grpc_host, grpc_port, mlx_url, mlx_api_key)
# User instruction: "处理issue和pr...发布补丁版本" + issue #5 (11434->11432) + issue #8 (env FUSION_MLX_API_KEY)
class GUIConfig:
    host: str = "0.0.0.0"
    port: int = 11455
    metrics_url: str = "http://127.0.0.1:11456"
    grpc_host: str = "0.0.0.0"
    grpc_port: int = 11447
    mlx_url: str = field(default_factory=_default_gui_mlx_url)
    mlx_api_key: str = field(default_factory=_default_gui_mlx_api_key)
