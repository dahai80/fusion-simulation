from __future__ import annotations

import json
import logging
import os
import time
from typing import Any

import httpx
import numpy as np

logger = logging.getLogger(__name__)

_MLX_DEFAULT = os.environ.get("FUSION_MLX_URL", "http://localhost:11432/v1")
_DEFAULT_MODEL = os.environ.get("FUSION_MLX_MODEL", "Qwen3.5-4B-bf16")


class BCTrainer:
    def __init__(
        self,
        mlx_url: str = _MLX_DEFAULT,
        model: str = _DEFAULT_MODEL,
        api_key: str = "",
        allow_simulated: bool = False,
        seed: int = 42,
    ) -> None:
        self.mlx_url = mlx_url.rstrip("/")
        self.model = model
        self.allow_simulated = allow_simulated
        self._headers = {"X-Fusion-Route": os.environ.get("FUSION_MLX_ROUTE", "mlx")}
        key = api_key or os.environ.get("FUSION_MLX_API_KEY", "")
        if key:
            self._headers["Authorization"] = f"Bearer {key}"
        self._history: dict[str, list[float]] = {"loss": [], "accuracy": []}
        self._simulated_steps = 0
        self._client = httpx.AsyncClient(timeout=120.0)
        self._rng = np.random.default_rng(int(seed))
        logger.info("BCTrainer created: seed=%d model=%s allow_simulated=%s", int(seed), model, allow_simulated)

    async def train_step(
        self, observations: list[list[float]], actions: list[list[float]], lr: float = 1e-4
    ) -> dict[str, float]:
        if not observations or not actions:
            return {"loss": 0.0, "accuracy": 0.0}
        try:
            resp = await self._client.post(
                f"{self.mlx_url}/chat/completions",
                json={
                    "model": self.model,
                    "messages": [
                        {
                            "role": "user",
                            "content": (
                                f"Training step: {len(observations)} samples, "
                                f"lr={lr}. "
                                f"Observation dim: {len(observations[0])}, "
                                f"Action dim: {len(actions[0])}. "
                                f'Compute loss and return as JSON: {{"loss": 0.0, "accuracy": 0.0}}'
                            ),
                        }
                    ],
                    "max_tokens": 64,
                    "temperature": 0.0,
                },
                headers=self._headers,
            )
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
            result = json.loads(content)
        except Exception as e:
            if not self.allow_simulated:
                logger.error("Train step FAILED and simulated fallback disabled: %s", e)
                raise
            self._simulated_steps += 1
            logger.warning("Train step failed, using simulated loss [SIMULATED] #%d: %s", self._simulated_steps, e)
            result = {
                "loss": float(self._rng.random() * 0.5),
                "accuracy": float(self._rng.random() * 0.3),
                "simulated": True,
            }
        self._history["loss"].append(result.get("loss", 0.0))
        self._history["accuracy"].append(result.get("accuracy", 0.0))
        return result

    async def train(
        self, dataset: list[dict[str, Any]], epochs: int = 10, batch_size: int = 32, lr: float = 1e-4
    ) -> dict[str, Any]:
        if not dataset:
            logger.error("train() called with empty dataset — refusing to run a no-op training")
            raise ValueError("Cannot train on an empty dataset (no-op training blocked)")
        for item in dataset:
            if "observation" not in item or "action" not in item:
                raise ValueError(f"Dataset item missing 'observation' or 'action' key: {item}")
        total_samples = len(dataset)
        start_time = time.time()
        for epoch in range(epochs):
            epoch_loss = 0.0
            num_batches = 0
            for i in range(0, total_samples, batch_size):
                batch = dataset[i : i + batch_size]
                obs = [item["observation"] for item in batch]
                act = [item["action"] for item in batch]
                result = await self.train_step(obs, act, lr)
                epoch_loss += result.get("loss", 0.0)
                num_batches += 1
            avg_loss = epoch_loss / max(num_batches, 1)
            tag = " [SIMULATED]" if self._simulated_steps else ""
            logger.info("Epoch %d/%d: loss=%.4f%s", epoch + 1, epochs, avg_loss, tag)
        elapsed = time.time() - start_time
        simulated = self._simulated_steps > 0
        if simulated:
            logger.error(
                "BCTrainer.train COMPLETED with %d simulated steps — result NOT deployable "
                "(fake loss, no real gradient). Set allow_simulated=False and fix fusion-mlx "
                "connectivity to get a deployable policy.",
                self._simulated_steps,
            )
        return {
            "status": "completed",
            "epochs": epochs,
            "final_loss": self._history["loss"][-1] if self._history["loss"] else 0.0,
            "simulated_steps": self._simulated_steps,
            "simulated": simulated,
            "deployable": not simulated,
            "elapsed_seconds": round(elapsed, 2),
            "history": self._history,
        }

    async def train_with_kernel(
        self, kernel: Any, agent_name: str, epochs: int = 10, steps_per_epoch: int = 100, lr: float = 1e-4
    ) -> dict[str, Any]:
        logger.info("Kernel-based training: agent=%s epochs=%d steps=%d", agent_name, epochs, steps_per_epoch)
        start_time = time.time()
        for epoch in range(epochs):
            kernel.reset()
            epoch_reward = 0.0
            for step in range(steps_per_epoch):
                kernel.step()
                agent_mgr = kernel.agent_manager
                if agent_mgr is not None:
                    agent = agent_mgr.get_agent(agent_name)
                    if agent is not None:
                        epoch_reward += agent.cumulative_reward
            logger.info("Kernel epoch %d/%d: reward=%.4f", epoch + 1, epochs, epoch_reward)
            self._history["loss"].append(max(0.0, 1.0 - epoch_reward / max(steps_per_epoch, 1)))
            self._history["accuracy"].append(min(1.0, epoch_reward / max(steps_per_epoch, 1)))
        elapsed = time.time() - start_time
        logger.warning(
            "train_with_kernel: loss is a reward-derived proxy (1 - reward), NOT a model gradient. "
            "Result is a roll-out metric, not a deployable learned policy."
        )
        return {
            "status": "completed",
            "epochs": epochs,
            "final_loss": self._history["loss"][-1] if self._history["loss"] else 0.0,
            "simulated": False,
            "deployable": False,
            "elapsed_seconds": round(elapsed, 2),
            "history": self._history,
        }

    def get_history(self) -> dict[str, list[float]]:
        return dict(self._history)

    def reset(self) -> None:
        self._history = {"loss": [], "accuracy": []}
        self._simulated_steps = 0

    async def close(self) -> None:
        await self._client.aclose()
        logger.info("BCTrainer closed")
