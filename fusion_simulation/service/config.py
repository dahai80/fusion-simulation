from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ServiceConfig:
    host: str = "0.0.0.0"
    port: int = 11447
    max_workers: int = 4
    use_ssl: bool = False
    cert_path: str = ""
    key_path: str = ""
    max_message_size: int = 4 * 1024 * 1024
    graceful_timeout: float = 5.0
