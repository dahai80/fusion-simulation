"""Final coverage push — targets remaining uncovered lines in CLI."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from fusion_simulation.eval.evaluator import EvalResult

# ── CLI _cmd_test and _cmd_bench ──


class TestCLIFinal:
    @pytest.mark.asyncio
    async def test_cmd_test_direct(self):
        from fusion_simulation.cli import _cmd_test

        args = MagicMock()
        args.model = "test"
        args.engine = "lerobot"
        args.episodes = 3
        args.mlx_url = "http://localhost:11434/v1"
        with patch(
            "fusion_simulation.eval.evaluator.SimulationEvaluator.evaluate",
            AsyncMock(return_value=EvalResult(task_success_rate=0.8)),
        ):
            with patch(
                "fusion_simulation.eval.evaluator.SimulationEvaluator.generate_report", return_value="report content"
            ):
                await _cmd_test(args)

    @pytest.mark.asyncio
    async def test_cmd_bench_direct(self):
        from fusion_simulation.cli import _cmd_bench

        with tempfile.TemporaryDirectory() as tmpdir:
            out = str(Path(tmpdir) / "report.md")
            args = MagicMock()
            args.model = "test"
            args.output = out
            args.mlx_url = "http://localhost:11434/v1"
            with patch(
                "fusion_simulation.eval.evaluator.SimulationEvaluator.evaluate",
                AsyncMock(return_value=EvalResult(task_success_rate=0.9)),
            ):
                with patch(
                    "fusion_simulation.eval.evaluator.SimulationEvaluator.generate_report", return_value="# Report"
                ):
                    await _cmd_bench(args)
            assert Path(out).exists()

    @pytest.mark.asyncio
    async def test_cmd_bench_no_output(self):
        from fusion_simulation.cli import _cmd_bench

        args = MagicMock()
        args.model = "test"
        args.output = ""
        args.mlx_url = "http://localhost:11434/v1"
        with patch(
            "fusion_simulation.eval.evaluator.SimulationEvaluator.evaluate",
            AsyncMock(return_value=EvalResult(task_success_rate=0.9)),
        ):
            with patch("fusion_simulation.eval.evaluator.SimulationEvaluator.generate_report", return_value="report"):
                await _cmd_bench(args)
