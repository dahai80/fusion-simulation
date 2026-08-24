from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from fusion_simulation.core.kernel import SimulationKernel

logger = logging.getLogger(__name__)

_MAX_DETAILS_KEPT = 200


def _current_rss_mb() -> float:
    try:
        import resource

        rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if rss_kb > 0:
            divisor = 1024.0 if hasattr(resource, "RUSAGE_SELF") else 1.0
            return rss_kb / divisor
    except (ImportError, OSError):
        pass
    return 0.0


@dataclass
class EvalResult:
    task_success_rate: float = 0.0
    trajectory_error: float = 0.0
    joint_stability: float = 0.0
    inference_latency_ms: float = 0.0
    fps: float = 0.0
    memory_mb: float = 0.0
    total_episodes: int = 0
    details: list[dict] = field(default_factory=list)


class SimulationEvaluator:
    def __init__(self, max_steps: int = 500, success_threshold: float = 0.5) -> None:
        self._max_steps = max_steps
        self._success_threshold = success_threshold
        logger.info(
            "SimulationEvaluator created: max_steps=%d success_threshold=%.2f",
            max_steps,
            success_threshold,
        )

    async def evaluate(
        self,
        model_name: str,
        engine: str = "lerobot",
        episodes: int = 10,
        max_steps: int | None = None,
        success_threshold: float | None = None,
    ) -> EvalResult:
        steps = max_steps if max_steps is not None else self._max_steps
        threshold = success_threshold if success_threshold is not None else self._success_threshold
        result = EvalResult(total_episodes=episodes)
        start = time.time()
        successes = 0
        latencies = []
        for _ in range(episodes):
            ep_result = await self._run_episode(model_name, engine, steps, threshold)
            if ep_result.get("success"):
                successes += 1
            latencies.append(ep_result.get("latency_ms", 0))
            result.details.append(ep_result)
        elapsed = time.time() - start
        result.task_success_rate = successes / max(episodes, 1)
        result.inference_latency_ms = sum(latencies) / max(len(latencies), 1)
        result.fps = episodes / max(elapsed, 0.001)
        result.trajectory_error = sum(d.get("trajectory_error", 0) for d in result.details) / max(episodes, 1)
        result.memory_mb = _current_rss_mb()
        if len(result.details) > _MAX_DETAILS_KEPT:
            logger.info("Eval details trimmed: kept last %d of %d", _MAX_DETAILS_KEPT, len(result.details))
            result.details = result.details[-_MAX_DETAILS_KEPT:]
        return result

    async def evaluate_with_kernel(
        self,
        kernel: SimulationKernel,
        agent_name: str = "",
        episodes: int = 10,
        max_steps: int = 500,
        success_threshold: float | None = None,
    ) -> EvalResult:
        threshold = success_threshold if success_threshold is not None else self._success_threshold
        result = EvalResult(total_episodes=episodes)
        start = time.time()
        successes = 0
        latencies = []
        agent_mgr = getattr(kernel, "agent_manager", None)
        for _ in range(episodes):
            kernel.reset()
            if agent_name:
                if agent_mgr is not None and hasattr(agent_mgr, "reset_agent"):
                    agent_mgr.reset_agent(agent_name)
            elif agent_mgr is not None and hasattr(agent_mgr, "reset_all"):
                agent_mgr.reset_all()
            ep_result = await self._run_kernel_episode(kernel, agent_name, max_steps, threshold)
            if ep_result.get("success"):
                successes += 1
            latencies.append(ep_result.get("latency_ms", 0))
            result.details.append(ep_result)
        elapsed = time.time() - start
        result.task_success_rate = successes / max(episodes, 1)
        result.inference_latency_ms = sum(latencies) / max(len(latencies), 1)
        result.fps = sum(d.get("steps", 0) for d in result.details) / max(elapsed, 0.001)
        result.trajectory_error = sum(d.get("trajectory_error", 0) for d in result.details) / max(episodes, 1)
        result.memory_mb = _current_rss_mb()
        if len(result.details) > _MAX_DETAILS_KEPT:
            logger.info("Eval details trimmed: kept last %d of %d", _MAX_DETAILS_KEPT, len(result.details))
            result.details = result.details[-_MAX_DETAILS_KEPT:]
        return result

    async def _run_episode(
        self, model_name: str, engine: str, max_steps: int, success_threshold: float
    ) -> dict[str, Any]:
        from fusion_simulation.agent.config import AgentConfig, AgentRole
        from fusion_simulation.agent.manager import AgentManager
        from fusion_simulation.core.kernel import KernelConfig
        from fusion_simulation.sensor.manager import SensorManager

        start = time.time()
        kernel: SimulationKernel | None = None
        try:
            sm = SensorManager()
            am = AgentManager()
            kernel = SimulationKernel(KernelConfig(headless=True))
            kernel.init(sensor_manager=sm, agent_manager=am)
            agent_cfg = AgentConfig(
                name="eval_agent",
                role=AgentRole.ROBOT,
                action_dim=6,
                model_name=model_name,
            )
            am.add_agent(agent_cfg)
            for _ in range(max_steps):
                kernel.step(num_steps=1)
            agent = am.get_agent("eval_agent")
            cumulative_reward = agent.cumulative_reward if agent else 0.0
            normalized_reward = self._normalize_reward(cumulative_reward, max_steps)
            success = normalized_reward >= success_threshold
            elapsed_ms = (time.time() - start) * 1000.0
            return {
                "success": success,
                "latency_ms": elapsed_ms / max(max_steps, 1),
                "trajectory_error": max(0.0, 1.0 - normalized_reward),
                "steps": max_steps,
                "cumulative_reward": cumulative_reward,
                "normalized_reward": normalized_reward,
            }
        except Exception as e:
            logger.exception(
                "Evaluator _run_episode FAILED for model=%s engine=%s — reporting real failure, not fake success",
                model_name,
                engine,
            )
            return {
                "success": False,
                "error": str(e),
                "steps": 0,
                "latency_ms": 0.0,
                "trajectory_error": 1.0,
            }
        finally:
            if kernel is not None:
                try:
                    kernel.close()
                except Exception:
                    logger.debug("kernel.close failed during eval cleanup", exc_info=True)

    async def _run_kernel_episode(
        self, kernel: SimulationKernel, agent_name: str, max_steps: int, success_threshold: float
    ) -> dict[str, Any]:
        start = time.time()
        try:
            for _ in range(max_steps):
                kernel.step(num_steps=1)
            elapsed_ms = (time.time() - start) * 1000.0
            agent_mgr = kernel.agent_manager
            agent = agent_mgr.get_agent(agent_name) if agent_mgr and agent_name else None
            cumulative_reward = agent.cumulative_reward if agent else 0.0
            normalized_reward = self._normalize_reward(cumulative_reward, max_steps)
            success = normalized_reward >= success_threshold
            return {
                "success": success,
                "latency_ms": elapsed_ms / max(max_steps, 1),
                "trajectory_error": max(0.0, 1.0 - normalized_reward),
                "steps": max_steps,
                "cumulative_reward": cumulative_reward,
                "normalized_reward": normalized_reward,
            }
        except Exception as e:
            logger.exception("Kernel episode failed")
            return {"success": False, "error": str(e), "steps": 0, "trajectory_error": 1.0}

    @staticmethod
    def _normalize_reward(cumulative_reward: float, max_steps: int) -> float:
        return cumulative_reward / max(max_steps, 1)

    def generate_report(self, result: EvalResult, fmt: str = "markdown") -> str:
        if fmt == "json":
            return json.dumps(
                {
                    "task_success_rate": round(result.task_success_rate, 2),
                    "trajectory_error": round(result.trajectory_error, 4),
                    "inference_latency_ms": round(result.inference_latency_ms, 2),
                    "fps": round(result.fps, 1),
                    "memory_mb": round(result.memory_mb, 2),
                    "total_episodes": result.total_episodes,
                },
                indent=2,
            )
        return (
            f"## Simulation Evaluation Report\n\n"
            f"| Metric | Value |\n"
            f"|--------|-------|\n"
            f"| Task Success Rate | {result.task_success_rate:.1%} |\n"
            f"| Trajectory Error | {result.trajectory_error:.4f} |\n"
            f"| Inference Latency | {result.inference_latency_ms:.1f} ms |\n"
            f"| FPS | {result.fps:.1f} |\n"
            f"| Episodes | {result.total_episodes} |\n"
            f"| Memory | {result.memory_mb:.1f} MB |\n"
        )
