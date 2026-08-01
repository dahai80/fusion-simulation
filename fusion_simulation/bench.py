from __future__ import annotations

import json
import logging
import sys
import time

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def bench_physics_step_frequency(physics_dt: float = 0.01, num_steps: int = 10000) -> dict:
    from fusion_simulation.core.kernel import KernelConfig, SimulationKernel
    cfg = KernelConfig(headless=True, physics_dt=physics_dt)
    kernel = SimulationKernel(cfg)
    kernel.init()
    t0 = time.perf_counter()
    kernel.step(num_steps=num_steps)
    elapsed = time.perf_counter() - t0
    kernel.close()
    hz = num_steps / elapsed
    result = {
        "metric": "physics_step_frequency",
        "num_steps": num_steps,
        "elapsed_s": round(elapsed, 4),
        "frequency_hz": round(hz, 2),
        "target_hz": 100,
        "pass": hz >= 100,
    }
    logger.info("Physics step: %.2f Hz (%s)", hz, "PASS" if result["pass"] else "FAIL")
    return result


def bench_rgb_sensor_latency(num_captures: int = 100) -> dict:
    from fusion_simulation.core.kernel import KernelConfig, SimulationKernel
    from fusion_simulation.sensor.base import SensorConfig, SensorType
    from fusion_simulation.sensor.manager import SensorManager
    kernel = SimulationKernel(KernelConfig(headless=True))
    sm = SensorManager()
    kernel.init(sensor_manager=sm)
    sm.add_sensor(SensorConfig(sensor_type=SensorType.RGB_CAMERA, name="bench_cam"))
    latencies = []
    for i in range(num_captures):
        sm.set_sim_time(i * 0.01)
        sm.set_physics_engine(kernel._physics)
        t0 = time.perf_counter()
        sm.update()
        latencies.append((time.perf_counter() - t0) * 1000.0)
    kernel.close()
    avg_ms = sum(latencies) / len(latencies)
    result = {
        "metric": "rgb_sensor_latency",
        "num_captures": num_captures,
        "avg_latency_ms": round(avg_ms, 3),
        "max_latency_ms": round(max(latencies), 3),
        "min_latency_ms": round(min(latencies), 3),
        "target_ms": 33,
        "pass": avg_ms < 33,
    }
    logger.info("RGB sensor latency: %.3f ms avg (%s)", avg_ms, "PASS" if result["pass"] else "FAIL")
    return result


def bench_grpc_call_latency(num_calls: int = 1000) -> dict:
    from fusion_simulation.service.server import SimulationServer
    server = SimulationServer()
    server.handle_request("init", {})
    latencies = []
    for _ in range(num_calls):
        t0 = time.perf_counter()
        server.handle_request("status", {})
        latencies.append((time.perf_counter() - t0) * 1000.0)
    server.handle_request("close", {})
    avg_ms = sum(latencies) / len(latencies)
    result = {
        "metric": "grpc_call_latency",
        "num_calls": num_calls,
        "avg_latency_ms": round(avg_ms, 4),
        "p99_latency_ms": round(sorted(latencies)[int(0.99 * len(latencies))], 4),
        "target_ms": 5,
        "pass": avg_ms < 5,
    }
    logger.info("gRPC call latency: %.4f ms avg (%s)", avg_ms, "PASS" if result["pass"] else "FAIL")
    return result


def bench_multi_agent_memory(num_agents: int = 5, num_steps: int = 1000) -> dict:
    from fusion_simulation.agent.config import AgentConfig, AgentRole
    from fusion_simulation.agent.manager import AgentManager
    from fusion_simulation.core.kernel import KernelConfig, SimulationKernel
    kernel = SimulationKernel(KernelConfig(headless=True))
    am = AgentManager()
    kernel.init(agent_manager=am)
    for i in range(num_agents):
        am.add_agent(AgentConfig(
            name=f"bot_{i}",
            role=AgentRole.ROBOT,
            action_dim=6,
            decimation=4,
        ))
    kernel.step(num_steps=num_steps)
    import resource
    rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024
    kernel.close()
    result = {
        "metric": "multi_agent_memory",
        "num_agents": num_agents,
        "num_steps": num_steps,
        "rss_mb": round(rss_mb, 2),
        "target_mb": 1024,
        "pass": rss_mb < 1024,
    }
    logger.info("5-agent memory: %.2f MB (%s)", rss_mb, "PASS" if result["pass"] else "FAIL")
    return result


def bench_cold_start() -> dict:
    t0 = time.perf_counter()
    from fusion_simulation.core.kernel import KernelConfig, SimulationKernel
    kernel = SimulationKernel(KernelConfig(headless=True))
    kernel.init()
    elapsed = time.perf_counter() - t0
    kernel.close()
    result = {
        "metric": "cold_start",
        "elapsed_s": round(elapsed, 4),
        "target_s": 5,
        "pass": elapsed < 5,
    }
    logger.info("Cold start: %.4f s (%s)", elapsed, "PASS" if result["pass"] else "FAIL")
    return result


def main():
    logger.info("=== Fusion-Simulation Performance Benchmarks ===")
    results = []
    results.append(bench_cold_start())
    results.append(bench_physics_step_frequency())
    results.append(bench_rgb_sensor_latency())
    results.append(bench_grpc_call_latency())
    results.append(bench_multi_agent_memory())

    passed = sum(1 for r in results if r["pass"])
    total = len(results)
    summary = {
        "benchmarks": results,
        "summary": {
            "total": total,
            "passed": passed,
            "failed": total - passed,
            "pass_rate": f"{passed}/{total}",
        },
    }

    output_path = sys.argv[1] if len(sys.argv) > 1 else "bench_report.json"
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("=== Results: %d/%d passed ===", passed, total)
    logger.info("Report saved to: %s", output_path)

    for r in results:
        status = "PASS" if r["pass"] else "FAIL"
        logger.info("  %s: %s", r["metric"], status)

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
