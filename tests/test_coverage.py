"""Coverage tests for Fusion-Simulation — targets uncovered lines in CLI, env, dataset."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from fusion_simulation.sim.env import SimulationEnv, EnvConfig, EngineType
from fusion_simulation.dataset.manager import DatasetManager
from fusion_simulation.eval.evaluator import SimulationEvaluator, EvalResult


# ── CLI Coverage ──

class TestCLICoverage:
    def test_version(self):
        from fusion_simulation.cli import main
        with patch.object(sys, "argv", ["fusion", "version"]):
            main()  # Just prints version, doesn't raise SystemExit

    def test_env_init(self):
        from fusion_simulation.cli import main
        with patch.object(sys, "argv", ["fusion", "env", "init", "--engine=lerobot"]):
            with patch("fusion_simulation.core.kernel.SimulationKernel.init", return_value=None):
                with patch("fusion_simulation.core.kernel.SimulationKernel.close", return_value=None):
                    main()

    def test_env_init_xlerobot(self):
        from fusion_simulation.cli import main
        with patch.object(sys, "argv", ["fusion", "env", "init", "--engine=xlerobot", "--headless"]):
            with patch("fusion_simulation.core.kernel.SimulationKernel.init", return_value=None):
                with patch("fusion_simulation.core.kernel.SimulationKernel.close", return_value=None):
                    main()

    def test_scene_list(self):
        from fusion_simulation.cli import main
        with patch.object(sys, "argv", ["fusion", "scene", "list"]):
            with patch("fusion_simulation.sim.env.SimulationEnv.list_scenes", return_value=[
                {"name": "pick", "description": "Object picking task"}
            ]):
                main()

    def test_dataset_list(self):
        from fusion_simulation.cli import main
        with patch.object(sys, "argv", ["fusion", "dataset", "list"]):
            with patch("fusion_simulation.dataset.manager.DatasetManager.list", return_value=[]):
                main()

    def test_dataset_list_with_data(self):
        from fusion_simulation.cli import main
        with patch.object(sys, "argv", ["fusion", "dataset", "list"]):
            with patch("fusion_simulation.dataset.manager.DatasetManager.list", return_value=[
                {"name": "ds1", "engine": "lerobot", "sample_count": 100}
            ]):
                main()

    def test_dataset_import(self):
        from fusion_simulation.cli import main
        with patch.object(sys, "argv", ["fusion", "dataset", "import", "--name=test", "--path=/tmp", "--engine=lerobot"]):
            with patch("fusion_simulation.dataset.manager.DatasetManager.import_dataset",
                       return_value={"status": "imported"}):
                main()

    def test_train(self):
        from fusion_simulation.cli import main
        with patch.object(sys, "argv", ["fusion", "train", "--dataset=test", "--model-name=m1", "--epochs=2"]):
            with patch("fusion_simulation.train.trainer.BCTrainer.train",
                       AsyncMock(return_value={"status": "completed", "final_loss": 0.5, "elapsed_seconds": 1.0})):
                with patch("fusion_simulation.dataset.manager.DatasetManager.get", return_value={"name": "test"}):
                    main()

    def test_train_no_dataset(self):
        from fusion_simulation.cli import main
        with patch.object(sys, "argv", ["fusion", "train", "--dataset=unknown", "--model-name=m1"]):
            with patch("fusion_simulation.dataset.manager.DatasetManager.get", return_value=None):
                with patch("fusion_simulation.dataset.manager.DatasetManager.collect_samples",
                           return_value=[{"observation": [1.0], "action": [0.5]}]):
                    with patch("fusion_simulation.train.trainer.BCTrainer.train",
                               AsyncMock(return_value={"status": "completed", "final_loss": 0.5, "elapsed_seconds": 1.0})):
                        main()

    def test_test(self):
        from fusion_simulation.cli import main
        with patch.object(sys, "argv", ["fusion", "test", "--model=test", "--engine=lerobot", "--episodes=3"]):
            with patch("fusion_simulation.cli._cmd_test"):
                main()

    def test_bench(self):
        from fusion_simulation.cli import main
        with patch.object(sys, "argv", ["fusion", "bench", "--model=test"]):
            with patch("fusion_simulation.cli._cmd_bench"):
                main()

    def test_bench_with_output(self):
        from fusion_simulation.cli import main
        with tempfile.TemporaryDirectory() as tmpdir:
            out = str(Path(tmpdir) / "report.json")
            with patch.object(sys, "argv", ["fusion", "bench", "--model=test", "--output=" + out]):
                with patch("fusion_simulation.cli._cmd_bench"):
                    main()


# ── SimulationEnv Coverage ──

class TestEnvCoverage:
    def test_init_with_pybullet(self):
        env = SimulationEnv()
        with patch("pybullet.connect", return_value=0) as mock_connect:
            with patch("pybullet.setGravity"):
                with patch("pybullet.setTimeStep"):
                    with patch("pybullet.setRealTimeSimulation"):
                        result = env.init()
                        assert result["status"] == "initialized"

    def test_step_with_pybullet(self):
        env = SimulationEnv()
        with patch("pybullet.stepSimulation"):
            state = env.step()
            assert state.step == 0

    def test_reset_with_pybullet(self):
        env = SimulationEnv()
        with patch("pybullet.resetSimulation"):
            with patch("pybullet.setGravity"):
                env.reset()
                assert env._state.step == 0

    def test_close_with_pybullet(self):
        env = SimulationEnv()
        env._physics_client = 0
        with patch("pybullet.disconnect"):
            env.close()
            assert env._physics_client == 0

    def test_capture_camera(self):
        env = SimulationEnv()
        img = env.capture_camera()
        assert img == b""

    def test_config_headless(self):
        cfg = EnvConfig(engine=EngineType.XLEROBOT, headless=True, render_fps=60)
        assert cfg.engine == EngineType.XLEROBOT
        assert cfg.headless is True
        assert cfg.render_fps == 60

    def test_engine_type_values(self):
        assert EngineType.LEROBOT.value == "lerobot"
        assert EngineType.XLEROBOT.value == "xlerobot"


# ── DatasetManager Coverage ──

class TestDatasetCoverage:
    def test_import_with_samples(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            src = Path(tmpdir) / "source"
            src.mkdir()
            # Create sample files
            (src / "samples.json").write_text('[{"obs": [1.0, 2.0], "action": [0.5]}]')
            (src / "extra.json").write_text('{"single": "data"}')
            mgr = DatasetManager(data_path=str(Path(tmpdir) / "datasets"))
            result = mgr.import_dataset("test_ds", str(src), engine="xlerobot")
            assert result["status"] == "imported"
            assert result["samples"] >= 1

    def test_clean_all(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = DatasetManager(data_path=tmpdir)
            result = mgr.clean()
            assert result["all"] is True

    def test_collect_and_save(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = DatasetManager(data_path=str(Path(tmpdir) / "datasets"))
            # First register dataset
            ds_path = Path(tmpdir) / "source"
            ds_path.mkdir()
            mgr.import_dataset("test", str(ds_path))
            # Then collect samples
            samples = mgr.collect_samples("test", num_samples=10)
            assert len(samples) == 10
            # Check that samples were saved to disk
            samples_file = Path(mgr._datasets["test"]["storage_path"]) / "samples" / "samples.json"
            assert samples_file.exists()

    def test_persistence(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr1 = DatasetManager(data_path=str(Path(tmpdir) / "datasets"))
            src = Path(tmpdir) / "src"
            src.mkdir()
            mgr1.import_dataset("persistent", str(src))
            mgr2 = DatasetManager(data_path=str(Path(tmpdir) / "datasets"))
            assert len(mgr2.list()) == 1


# ── Evaluator Coverage ──

class TestEvalCoverage:
    @pytest.mark.asyncio
    async def test_evaluate_zero_episodes(self):
        evaluator = SimulationEvaluator()
        result = await evaluator.evaluate("test", episodes=0)
        assert result.total_episodes == 0
        assert result.task_success_rate == 0.0

    @pytest.mark.asyncio
    async def test_episode_failure(self):
        evaluator = SimulationEvaluator()
        with patch.object(evaluator, "_run_episode", AsyncMock(return_value={"success": False, "error": "fail"})):
            result = await evaluator.evaluate("test", episodes=1)
            assert result.task_success_rate == 0.0

    @pytest.mark.asyncio
    async def test_eval_result_defaults(self):
        r = EvalResult()
        assert r.task_success_rate == 0.0
        assert r.total_episodes == 0
        assert r.details == []


import sys