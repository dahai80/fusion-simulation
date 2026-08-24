from __future__ import annotations

import base64
import io
import json
import logging
import os
import time
from typing import Any

import httpx

from fusion_simulation.core.rng import numpy_to_native

logger = logging.getLogger(__name__)


def _default_endpoint() -> str:
    base = os.environ.get("FUSION_MLX_URL", "http://localhost:11432/v1")
    return base.rstrip("/") + "/chat/completions"


_DEFAULT_ENDPOINT = _default_endpoint()
_DEFAULT_MODEL = os.environ.get("FUSION_MLX_MODEL", "Qwen3.5-4B-bf16")
_DEFAULT_ROUTE = os.environ.get("FUSION_MLX_ROUTE", "mlx")
_REQUEST_TIMEOUT = 30.0
_CIRCUIT_FAIL_THRESHOLD = 5
_CIRCUIT_COOLDOWN_SEC = 10.0
_CIRCUIT_BACKOFF_BASE = 0.5
_MIN_INFER_INTERVAL_SEC = 0.0


# Callers: AgentManager.add_agent, server._rpc_add_agent
# Affected API: PolicyClient.__init__(api_key), check_available/predict/infer_from_image* send Authorization
# Data schemas: PolicyClient._headers = {"Authorization": "Bearer <key>"} when api_key set
# User instruction: "和~/fusion/fuison-simulation项目集成起来...最后要完成端到端测试,确保系统可用"
class PolicyClient:
    def __init__(self, endpoint: str = _DEFAULT_ENDPOINT, model_name: str = _DEFAULT_MODEL, api_key: str = "") -> None:
        self._endpoint = endpoint
        self._model_name = model_name
        self._api_key = api_key
        self._headers = {"X-Fusion-Route": _DEFAULT_ROUTE}
        if api_key:
            self._headers["Authorization"] = f"Bearer {api_key}"
        self._client = httpx.Client(timeout=_REQUEST_TIMEOUT)
        self._async_client: httpx.AsyncClient | None = None
        self._request_count: int = 0
        self._total_latency: float = 0.0
        self._last_action: list[float] = []
        self._available: bool = False
        self._failure_count: int = 0
        self._last_failure: str = ""
        self._consecutive_failures: int = 0
        self._circuit_open_until: float = 0.0
        self._last_infer_time: float = 0.0

    @property
    def endpoint(self) -> str:
        return self._endpoint

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def is_available(self) -> bool:
        return self._available

    @property
    def avg_latency(self) -> float:
        return self._total_latency / self._request_count if self._request_count > 0 else 0.0

    def check_available(self) -> bool:
        try:
            resp = self._client.get(
                self._endpoint.replace("/v1/chat/completions", "/v1/models"), headers=self._headers, timeout=5.0
            )
            self._available = resp.status_code == 200
        except Exception:
            self._available = False
        logger.info("PolicyClient availability check: %s", self._available)
        return self._available

    def _circuit_open(self) -> bool:
        now = time.monotonic()
        if now < self._circuit_open_until:
            return True
        return False

    def _record_success(self) -> None:
        if self._consecutive_failures > 0:
            logger.info("PolicyClient circuit CLOSED after %d consecutive failures", self._consecutive_failures)
        self._consecutive_failures = 0
        self._circuit_open_until = 0.0

    def _record_failure(self, context: str, exc: BaseException | None = None) -> None:
        self._failure_count += 1
        self._consecutive_failures += 1
        self._last_failure = str(exc) if exc is not None else context
        self._available = False
        if self._consecutive_failures >= _CIRCUIT_FAIL_THRESHOLD:
            backoff = min(
                _CIRCUIT_COOLDOWN_SEC,
                _CIRCUIT_BACKOFF_BASE * (2 ** (self._consecutive_failures - _CIRCUIT_FAIL_THRESHOLD)),
            )
            self._circuit_open_until = time.monotonic() + backoff
            logger.error(
                "PolicyClient circuit OPEN for %.1fs after %d consecutive failures (context=%s): %s",
                backoff,
                self._consecutive_failures,
                context,
                self._last_failure,
            )
        else:
            logger.error(
                "PolicyClient %s FAILED (#%d consec=%d): %s — emitting ZERO action (safe default), agent uncontrolled",
                context,
                self._failure_count,
                self._consecutive_failures,
                self._last_failure,
            )

    def _rate_limit_wait(self) -> None:
        if _MIN_INFER_INTERVAL_SEC <= 0:
            return
        now = time.monotonic()
        delta = _MIN_INFER_INTERVAL_SEC - (now - self._last_infer_time)
        if delta > 0:
            time.sleep(delta)
        self._last_infer_time = time.monotonic()

    def predict(self, observation: dict[str, Any], action_dim: int = 0) -> list[float]:
        if self._circuit_open():
            logger.debug("PolicyClient circuit open, short-circuiting predict -> zeros")
            return [0.0] * max(action_dim, 1)
        self._rate_limit_wait()
        self._request_count += 1
        t0 = time.monotonic()
        try:
            obs_str = json.dumps(numpy_to_native(observation), ensure_ascii=False)
            payload = {
                "model": self._model_name,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a robot control policy. Given observations, output a JSON array of action values.",
                    },
                    {
                        "role": "user",
                        "content": f"Observations: {obs_str}\nOutput action array of {action_dim} values as JSON.",
                    },
                ],
                "temperature": 0.1,
                "max_tokens": 256,
            }
            resp = self._client.post(self._endpoint, json=payload, headers=self._headers)
            resp.raise_for_status()
            data = resp.json()
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            action, parsed_ok = self._parse_action(content, action_dim)
            if not parsed_ok:
                self._record_failure("predict_parse", None)
            else:
                self._last_action = action
                self._available = True
                self._record_success()
        except Exception as e:
            self._record_failure("predict", e)
            action = [0.0] * max(action_dim, 1)
        latency = time.monotonic() - t0
        self._total_latency += latency
        logger.debug("Policy predict: action=%s latency=%.3fs", action[:4], latency)
        return action

    def infer_from_image(
        self,
        image: Any,
        prompt: str = "",
        action_dim: int = 0,
    ) -> list[float]:
        if self._circuit_open():
            logger.debug("PolicyClient circuit open, short-circuiting infer_from_image -> zeros")
            return [0.0] * max(action_dim, 1)
        self._rate_limit_wait()
        self._request_count += 1
        t0 = time.monotonic()
        try:
            image_b64 = self._encode_image(image)
            content_parts = []
            if prompt:
                content_parts.append({"type": "text", "text": prompt})
            else:
                content_parts.append(
                    {
                        "type": "text",
                        "text": f"You are a robot control policy. Analyze this camera image and output a JSON array of {action_dim} action values.",
                    }
                )
            content_parts.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                }
            )
            payload = {
                "model": self._model_name,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a vision-based robot control policy. Analyze the image and output action values.",
                    },
                    {"role": "user", "content": content_parts},
                ],
                "temperature": 0.1,
                "max_tokens": 256,
            }
            resp = self._client.post(self._endpoint, json=payload, headers=self._headers)
            resp.raise_for_status()
            data = resp.json()
            text = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            action, parsed_ok = self._parse_action(text, action_dim)
            if not parsed_ok:
                self._record_failure("infer_from_image_parse", None)
            else:
                self._last_action = action
                self._available = True
                self._record_success()
        except Exception as e:
            self._record_failure("infer_from_image", e)
            action = [0.0] * max(action_dim, 1)
        latency = time.monotonic() - t0
        self._total_latency += latency
        logger.debug("Policy infer_from_image: action=%s latency=%.3fs", action[:4], latency)
        return action

    async def infer_from_image_async(
        self,
        image: Any,
        prompt: str = "",
        action_dim: int = 0,
    ) -> list[float]:
        if self._circuit_open():
            logger.debug("PolicyClient circuit open, short-circuiting infer_from_image_async -> zeros")
            return [0.0] * max(action_dim, 1)
        if self._async_client is None:
            self._async_client = httpx.AsyncClient(timeout=_REQUEST_TIMEOUT)
        self._request_count += 1
        t0 = time.monotonic()
        try:
            image_b64 = self._encode_image(image)
            content_parts = []
            if prompt:
                content_parts.append({"type": "text", "text": prompt})
            else:
                content_parts.append(
                    {
                        "type": "text",
                        "text": f"You are a robot control policy. Analyze this camera image and output a JSON array of {action_dim} action values.",
                    }
                )
            content_parts.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                }
            )
            payload = {
                "model": self._model_name,
                "messages": [
                    {"role": "system", "content": "You are a vision-based robot control policy."},
                    {"role": "user", "content": content_parts},
                ],
                "temperature": 0.1,
                "max_tokens": 256,
            }
            resp = await self._async_client.post(self._endpoint, json=payload, headers=self._headers)
            resp.raise_for_status()
            data = resp.json()
            text = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            action, parsed_ok = self._parse_action(text, action_dim)
            if not parsed_ok:
                self._record_failure("infer_from_image_async_parse", None)
            else:
                self._last_action = action
                self._available = True
                self._record_success()
        except Exception as e:
            self._record_failure("infer_from_image_async", e)
            action = [0.0] * max(action_dim, 1)
        latency = time.monotonic() - t0
        self._total_latency += latency
        logger.debug("Policy infer_from_image_async: action=%s latency=%.3fs", action[:4], latency)
        return action

    def _encode_image(self, image: Any) -> str:
        if isinstance(image, str):
            if image.startswith("data:"):
                return image.split(",", 1)[1]
            return image
        if isinstance(image, bytes):
            return base64.b64encode(image).decode("ascii")
        try:
            import numpy as np

            if isinstance(image, np.ndarray):
                from PIL import Image as PILImage

                pil_img = PILImage.fromarray(image)
                buf = io.BytesIO()
                pil_img.save(buf, format="PNG")
                return base64.b64encode(buf.getvalue()).decode("ascii")
        except ImportError:
            pass
        raise TypeError(f"Unsupported image type: {type(image)}. Expected str (base64), bytes, or numpy array.")

    def _parse_action(self, content: str | None, action_dim: int) -> tuple[list[float], bool]:
        if not content:
            logger.warning("Empty model output, using zeros (parse failure)")
            return [0.0] * max(action_dim, 1), False
        try:
            fenced = content
            if "```" in fenced:
                parts = fenced.split("```")
                if len(parts) >= 2:
                    fenced = parts[1]
                    if fenced.lstrip().startswith("json"):
                        fenced = fenced.lstrip()[4:]
            start = fenced.index("[")
            end = fenced.rindex("]") + 1
            arr = json.loads(fenced[start:end])
            if isinstance(arr, list):
                action: list[float] = []
                for v in arr:
                    try:
                        action.append(float(v))
                    except (TypeError, ValueError):
                        action.append(0.0)
                if action_dim > 0 and len(action) != action_dim:
                    if len(action) < action_dim:
                        action.extend([0.0] * (action_dim - len(action)))
                    else:
                        action = action[:action_dim]
                return action, True
        except (ValueError, json.JSONDecodeError) as e:
            logger.warning(
                "Failed to parse action from model output (parse failure counted): %s | content=%r", e, content[:120]
            )
        return [0.0] * max(action_dim, 1), False

    def close(self) -> None:
        self._client.close()
        if self._async_client is not None:
            import asyncio

            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.ensure_future(self._async_client.aclose())
                else:
                    loop.run_until_complete(self._async_client.aclose())
            except Exception:
                logger.debug("Error closing async client", exc=True)
            self._async_client = None
        logger.info("PolicyClient closed, requests=%d avg_latency=%.3fs", self._request_count, self.avg_latency)

    def stats(self) -> dict[str, Any]:
        return {
            "endpoint": self._endpoint,
            "model": self._model_name,
            "available": self._available,
            "request_count": self._request_count,
            "failure_count": self._failure_count,
            "last_failure": self._last_failure,
            "avg_latency": self.avg_latency,
            "last_action": self._last_action[:8],
            "consecutive_failures": self._consecutive_failures,
            "circuit_open": self._circuit_open(),
        }
