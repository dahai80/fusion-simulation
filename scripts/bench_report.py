#!/usr/bin/env python3
from __future__ import annotations

import json
import time
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fusion_simulation.core.kernel import KernelConfig, SimulationKernel
from fusion_simulation.core.ecs import EntityManager, Transform
from fusion_simulation.core.event_bus import EventBus, EventKind
from fusion_simulation.sensor.manager import SensorManager
from fusion_simulation.sensor.base import SensorConfig, SensorType
from fusion_simulation.agent.manager import AgentManager
from fusion_simulation.agent.config import AgentConfig, AgentRole
from fusion_simulation.service.metrics_server import MetricsCollector


def bench_kernel_step(n_steps=500):
    sm = SensorManager()
    am = AgentManager()
    kernel = SimulationKernel(KernelConfig(headless=True))
    kernel.init(sensor_manager=sm, agent_manager=am)
    kernel.step(num_steps=10)
    start = time.perf_counter()
    kernel.step(num_steps=n_steps)
    elapsed = time.perf_counter() - start
    kernel.close()
    return {"name": "kernel_step", "steps": n_steps, "elapsed_s": round(elapsed, 4),
            "per_step_ms": round((elapsed / n_steps) * 1000, 3)}


def bench_sensor_obs(n_sensors=5, n_obs=2000):
    sm = SensorManager()
    for i in range(n_sensors):
        sm.add_sensor(SensorConfig(sensor_type=SensorType.RGB_CAMERA, name=f"cam{i}"))
    start = time.perf_counter()
    for _ in range(n_obs):
        sm.get_observations()
    elapsed = time.perf_counter() - start
    return {"name": "sensor_observation", "sensors": n_sensors, "observations": n_obs,
            "elapsed_s": round(elapsed, 4), "per_obs_us": round((elapsed / n_obs) * 1e6, 1)}


def bench_agent_decision(n_agents=3, n_decisions=200):
    am = AgentManager()
    for i in range(n_agents):
        am.add_agent(AgentConfig(name=f"bot{i}", role=AgentRole.ROBOT, action_dim=6))
    start = time.perf_counter()
    for _ in range(n_decisions):
        am.step_all()
    elapsed = time.perf_counter() - start
    am.close()
    return {"name": "agent_decision", "agents": n_agents, "decisions": n_decisions,
            "elapsed_s": round(elapsed, 4), "per_decision_ms": round((elapsed / n_decisions) * 1000, 3)}


def bench_ecs_create(n_entities=2000):
    mgr = EntityManager()
    start = time.perf_counter()
    for i in range(n_entities):
        eid = mgr.create_entity()
        mgr.add_component(eid, Transform(position=[float(i), 0.0, 0.0]))
    elapsed = time.perf_counter() - start
    return {"name": "ecs_create", "entities": n_entities, "elapsed_s": round(elapsed, 4),
            "per_entity_us": round((elapsed / n_entities) * 1e6, 1)}


def bench_event_bus(n_events=50000):
    bus = EventBus()
    counter = [0]
    bus.subscribe(EventKind.PHYSICS_POST_STEP, lambda e: counter.__setitem__(0, counter[0] + 1))
    start = time.perf_counter()
    for _ in range(n_events):
        bus.emit(EventKind.PHYSICS_POST_STEP, {})
    elapsed = time.perf_counter() - start
    return {"name": "event_bus", "events": n_events, "elapsed_s": round(elapsed, 4),
            "per_event_us": round((elapsed / n_events) * 1e6, 2), "delivered": counter[0]}


def bench_metrics_collector(n_ops=50000):
    mc = MetricsCollector()
    start = time.perf_counter()
    for i in range(n_ops):
        mc.inc_counter("bench_counter")
        mc.set_gauge("bench_gauge", float(i))
    elapsed = time.perf_counter() - start
    return {"name": "metrics_collector", "ops": n_ops * 2, "elapsed_s": round(elapsed, 4),
            "per_op_us": round((elapsed / (n_ops * 2)) * 1e6, 2)}


def main():
    print("Fusion-Simulation Performance Benchmark Report")
    print("=" * 60)

    benchmarks = [
        bench_kernel_step,
        bench_sensor_obs,
        bench_agent_decision,
        bench_ecs_create,
        bench_event_bus,
        bench_metrics_collector,
    ]

    results = []
    for bench_fn in benchmarks:
        print(f"\nRunning {bench_fn.__name__}...")
        try:
            result = bench_fn()
            results.append(result)
            metric = (result.get("per_step_ms"), result.get("per_obs_us"),
                      result.get("per_decision_ms"), result.get("per_entity_us"),
                      result.get("per_event_us"), result.get("per_op_us"))
            metric = [m for m in metric if m is not None]
            if metric:
                print(f"  OK {result['name']}: {metric[0]}")
        except Exception as e:
            print(f"  FAIL {bench_fn.__name__}: {e}")
            results.append({"name": bench_fn.__name__, "error": str(e)})

    report = {
        "title": "Fusion-Simulation Performance Benchmark Report",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "platform": "Apple Silicon",
        "results": results,
    }

    report_path = os.path.join(os.path.dirname(__file__), "..", "benchmark_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved to {report_path}")


if __name__ == "__main__":
    main()
