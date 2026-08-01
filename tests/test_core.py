"""Tests for Fusion-Simulation core modules."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from fusion_simulation.sim.env import SimulationEnv, EnvConfig, EngineType, SimulationState
from fusion_simulation.train.trainer import BCTrainer
from fusion_simulation.eval.evaluator import SimulationEvaluator, EvalResult
from fusion_simulation.dataset.manager import DatasetManager


# ── EnvConfig ──

class TestEnvConfig:
    def test_defaults(self):
        cfg = EnvConfig()
        assert cfg.engine == EngineType.LEROBOT
        assert cfg.render_fps == 30
        assert cfg.seed == 42

    def test_headless(self):
        cfg = EnvConfig(headless=True)
        assert cfg.headless is True


# ── SimulationState ──

class TestSimulationState:
    def test_defaults(self):
        s = SimulationState()
        assert s.step == 0
        assert s.task_completed is False
        assert s.error == ""


# ── SimulationEnv ──

class TestSimulationEnv:
    def test_init_no_pybullet(self):
        env = SimulationEnv()
        result = env.init()
        # PyBullet not installed in test env
        assert "status" in result

    def test_list_scenes(self):
        scenes = SimulationEnv.list_scenes()
        assert len(scenes) >= 5
        assert scenes[0]["name"] == "pick"

    def test_reset_without_init(self):
        env = SimulationEnv()
        env.reset()  # Should not crash

    def test_close_without_init(self):
        env = SimulationEnv()
        env.close()  # Should not crash

    def test_step_without_init(self):
        env = SimulationEnv()
        state = env.step()
        assert state.step == 0
        assert state.error != ""

    def test_capture_without_init(self):
        env = SimulationEnv()
        img = env.capture_camera()
        assert img == b""


# ── BCTrainer ──

class TestBCTrainer:
    @pytest.mark.asyncio
    async def test_train_step(self):
        trainer = BCTrainer()
        with patch("httpx.AsyncClient.post", new=AsyncMock()) as mock_post:
            mock_resp = MagicMock()
            mock_resp.json.return_value = {
                "choices": [{"message": {"content": '{"loss": 0.5, "accuracy": 0.3}'}}]
            }
            mock_post.return_value = mock_resp
            result = await trainer.train_step([[1.0, 0.0]], [[0.5, 0.0]], lr=1e-4)
            assert "loss" in result
            assert "accuracy" in result

    @pytest.mark.asyncio
    async def test_train_step_fallback(self):
        trainer = BCTrainer(mlx_url="http://localhost:19999")
        result = await trainer.train_step([[1.0]], [[0.0]], lr=1e-4)
        assert "loss" in result  # Falls back to simulated loss

    @pytest.mark.asyncio
    async def test_train_step_empty(self):
        trainer = BCTrainer()
        result = await trainer.train_step([], [])
        assert result["loss"] == 0.0

    @pytest.mark.asyncio
    async def test_train(self):
        trainer = BCTrainer()
        dataset = [{"observation": [1.0], "action": [0.5]} for _ in range(5)]
        with patch.object(trainer, "train_step", AsyncMock(return_value={"loss": 0.1, "accuracy": 0.9})):
            result = await trainer.train(dataset, epochs=2, batch_size=10)
            assert result["status"] == "completed"
            assert result["epochs"] == 2

    @pytest.mark.asyncio
    async def test_get_history(self):
        trainer = BCTrainer()
        trainer._history["loss"].append(0.5)
        trainer._history["loss"].append(0.3)
        history = trainer.get_history()
        assert len(history["loss"]) == 2

    @pytest.mark.asyncio
    async def test_reset(self):
        trainer = BCTrainer()
        trainer._history["loss"].append(0.5)
        trainer.reset()
        assert len(trainer._history["loss"]) == 0


# ── SimulationEvaluator ──

class TestSimulationEvaluator:
    @pytest.mark.asyncio
    async def test_evaluate(self):
        evaluator = SimulationEvaluator()
        result = await evaluator.evaluate("test_model", episodes=3)
        assert result.total_episodes == 3
        assert result.task_success_rate >= 0
        assert len(result.details) == 3

    @pytest.mark.asyncio
    async def test_generate_report_markdown(self):
        result = EvalResult(task_success_rate=0.8, inference_latency_ms=15.0, fps=30.0)
        evaluator = SimulationEvaluator()
        report = evaluator.generate_report(result, fmt="markdown")
        assert "80.0%" in report
        assert "15.0" in report

    @pytest.mark.asyncio
    async def test_generate_report_json(self):
        result = EvalResult(task_success_rate=0.9, inference_latency_ms=10.0, fps=60.0)
        evaluator = SimulationEvaluator()
        report = evaluator.generate_report(result, fmt="json")
        data = json.loads(report)
        assert data["task_success_rate"] == 0.9


# ── DatasetManager ──

class TestDatasetManager:
    def test_list_empty(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = DatasetManager(data_path=tmpdir)
            assert mgr.list() == []

    def test_import_nonexistent(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = DatasetManager(data_path=tmpdir)
            result = mgr.import_dataset("test", "/nonexistent/path")
            assert result["status"] == "error"

    def test_import_and_list(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            src = Path(tmpdir) / "source"
            src.mkdir()
            (src / "data.json").write_text('[{"obs": [1.0]}]')
            mgr = DatasetManager(data_path=str(Path(tmpdir) / "datasets"))
            result = mgr.import_dataset("test_ds", str(src), engine="lerobot")
            assert result["status"] == "imported"
            datasets = mgr.list()
            assert len(datasets) == 1
            assert datasets[0]["name"] == "test_ds"

    def test_get(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            src = Path(tmpdir) / "source"
            src.mkdir()
            mgr = DatasetManager(data_path=str(Path(tmpdir) / "datasets"))
            mgr.import_dataset("ds1", str(src))
            assert mgr.get("ds1") is not None
            assert mgr.get("nonexistent") is None

    def test_delete(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            src = Path(tmpdir) / "source"
            src.mkdir()
            mgr = DatasetManager(data_path=str(Path(tmpdir) / "datasets"))
            mgr.import_dataset("ds1", str(src))
            assert mgr.delete("ds1") is True
            assert mgr.delete("nonexistent") is False

    def test_clean(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = DatasetManager(data_path=tmpdir)
            result = mgr.clean("test")
            assert result["status"] == "cleaned"

    def test_collect_samples(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = DatasetManager(data_path=str(Path(tmpdir) / "datasets"))
            samples = mgr.collect_samples("test", num_samples=5)
            assert len(samples) == 5
            for s in samples:
                assert "observation" in s
                assert "action" in s
                assert "reward" in s