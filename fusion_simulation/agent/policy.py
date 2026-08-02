from __future__ import annotations

import base64
import io
import json
import logging
import time
from typing import Any

import httpx

logger = logging.getLogger(__name__)

_DEFAULT_ENDPOINT = "http://localhost:11434/v1/chat/completions"
_REQUEST_TIMEOUT = 30.0


class PolicyClient:
    def __init__(self, endpoint: str = _DEFAULT_ENDPOINT, model_name: str = "qwen3.5-9b") -> None:
        self._endpoint = endpoint
        self._model_name = model_name
        self._client = httpx.Client(timeout=_REQUEST_TIMEOUT)
        self._async_client: httpx.AsyncClient | None = None
        self._request_count: int = 0
        self._total_latency: float = 0.0
        self._last_action: list[float] = []
        self._available: bool = False

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
            resp = self._client.get(self._endpoint.replace("/v1/chat/completions", "/v1/models"), timeout=5.0)
            self._available = resp.status_code == 200
        except Exception:
            self._available = False
        logger.info("PolicyClient availability check: %s", self._available)
        return self._available

    def predict(self, observation: dict[str, Any], action_dim: int = 0) -> list[float]:
        self._request_count += 1
        t0 = time.monotonic()
        try:
            obs_str = json.dumps(observation, default=str, ensure_ascii=False)
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
            resp = self._client.post(self._endpoint, json=payload)
            resp.raise_for_status()
            data = resp.json()
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            action = self._parse_action(content, action_dim)
            self._last_action = action
        except Exception:
            logger.exception("PolicyClient predict failed, using zero action")
            action = [0.0] * max(action_dim, 1)
            self._available = False
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
            resp = self._client.post(self._endpoint, json=payload)
            resp.raise_for_status()
            data = resp.json()
            text = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            action = self._parse_action(text, action_dim)
            self._last_action = action
        except Exception:
            logger.exception("PolicyClient infer_from_image failed, using zero action")
            action = [0.0] * max(action_dim, 1)
            self._available = False
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
            resp = await self._async_client.post(self._endpoint, json=payload)
            resp.raise_for_status()
            data = resp.json()
            text = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            action = self._parse_action(text, action_dim)
            self._last_action = action
        except Exception:
            logger.exception("PolicyClient infer_from_image_async failed, using zero action")
            action = [0.0] * max(action_dim, 1)
            self._available = False
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

    def _parse_action(self, content: str, action_dim: int) -> list[float]:
        try:
            start = content.index("[")
            end = content.rindex("]") + 1
            arr = json.loads(content[start:end])
            if isinstance(arr, list):
                action = [float(v) for v in arr]
                if action_dim > 0 and len(action) != action_dim:
                    if len(action) < action_dim:
                        action.extend([0.0] * (action_dim - len(action)))
                    else:
                        action = action[:action_dim]
                return action
        except (ValueError, json.JSONDecodeError):
            pass
        logger.warning("Failed to parse action from model output, using zeros")
        return [0.0] * max(action_dim, 1)

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
            "avg_latency": self.avg_latency,
            "last_action": self._last_action[:8],
        }
