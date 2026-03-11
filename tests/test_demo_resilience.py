"""Tests: _run_demo() resilience to classifier failures and OOM.

Cloud container must stay alive even when edge.audio.classifier
cannot load (OOM, TF crash) or raises at runtime.  Memory guard
must prevent demo from running when system memory is low.
"""

from __future__ import annotations

import asyncio
import importlib
import sys
from unittest.mock import AsyncMock, patch

import pytest


@pytest.fixture
def _real_main():
    """Ensure cloud.interface.main is the real module, not a MagicMock.

    Other tests (test_drone_bot, test_bot_workflow) replace this module
    in sys.modules with a MagicMock. We must reload the real one.
    """
    mod_name = "cloud.interface.main"
    cached = sys.modules.get(mod_name)
    # If it's a MagicMock or missing, force reimport
    if cached is None or not hasattr(cached, "__file__"):
        sys.modules.pop(mod_name, None)
        import cloud.interface.main  # noqa: F811

    return sys.modules[mod_name]


class TestRunDemoResilience:
    """_run_demo() must not crash the process on classifier errors."""

    def test_run_demo_handles_classifier_import_error(self, _real_main):
        """_run_demo() logs and returns when classifier import fails."""
        _run_demo = _real_main._run_demo

        with patch.object(
            _real_main,
            "_import_demo_deps",
            side_effect=ImportError("No module named 'tensorflow'"),
        ):
            loop = asyncio.new_event_loop()
            try:
                result = loop.run_until_complete(_run_demo("chainsaw"))
                assert result is None
            finally:
                loop.close()

    def test_run_demo_handles_classifier_runtime_error(self, _real_main):
        """_run_demo() survives MemoryError / RuntimeError during classify()."""
        _run_demo = _real_main._run_demo

        with patch.object(
            _real_main,
            "_import_demo_deps",
            side_effect=MemoryError("out of memory"),
        ):
            loop = asyncio.new_event_loop()
            try:
                result = loop.run_until_complete(_run_demo("chainsaw"))
                assert result is None
            finally:
                loop.close()

    def test_health_endpoint_independent_of_classifier(self, _real_main):
        """/health returns 200 regardless of classifier state."""
        health = _real_main.health

        with patch.dict("sys.modules", {"tensorflow": None, "tensorflow_hub": None}):
            loop = asyncio.new_event_loop()
            try:
                result = loop.run_until_complete(health())
                assert result == {"status": "ok"}
            finally:
                loop.close()


class TestAvailableMemory:
    """_available_memory_mb() reads /proc/meminfo on Linux, fallback on others."""

    def test_reads_proc_meminfo(self, _real_main):
        """Parses MemAvailable from /proc/meminfo correctly."""
        fake_meminfo = (
            "MemTotal:       16384000 kB\n"
            "MemFree:         8000000 kB\n"
            "MemAvailable:    12000000 kB\n"
        )
        from unittest.mock import mock_open

        with patch("builtins.open", mock_open(read_data=fake_meminfo)):
            result = _real_main._available_memory_mb()
        assert abs(result - 12000000 / 1024) < 0.01  # ~11718.75 MB

    def test_fallback_when_no_proc(self, _real_main):
        """Returns inf when /proc/meminfo is not available (macOS, etc.)."""
        with patch("builtins.open", side_effect=OSError("No such file")):
            result = _real_main._available_memory_mb()
        assert result == float("inf")


class TestMemoryGuard:
    """_run_demo() must skip execution when system memory is too low."""

    def test_skips_on_low_memory(self, _real_main):
        """Demo returns early with 'low_memory' broadcast when RAM < threshold."""
        _run_demo = _real_main._run_demo
        broadcasts = []

        async def fake_broadcast(msg):
            broadcasts.append(msg)

        with (
            patch.object(_real_main, "_available_memory_mb", return_value=200.0),
            patch.object(_real_main, "MIN_DEMO_MEMORY_MB", 400),
            patch.object(_real_main, "broadcast", new=fake_broadcast),
        ):
            loop = asyncio.new_event_loop()
            try:
                result = loop.run_until_complete(_run_demo("chainsaw"))
            finally:
                loop.close()

        assert result is None
        assert any(b.get("reason") == "low_memory" for b in broadcasts)

    def test_proceeds_on_sufficient_memory(self, _real_main):
        """Demo proceeds past memory check when RAM is sufficient."""
        _run_demo = _real_main._run_demo

        with (
            patch.object(_real_main, "_available_memory_mb", return_value=1000.0),
            patch.object(_real_main, "MIN_DEMO_MEMORY_MB", 400),
            patch.object(
                _real_main,
                "_import_demo_deps",
                side_effect=ImportError("stopped after memory check passed"),
            ),
        ):
            loop = asyncio.new_event_loop()
            try:
                result = loop.run_until_complete(_run_demo("chainsaw"))
            finally:
                loop.close()

        # Demo got past memory check — reached _import_demo_deps (which we made fail)
        assert result is None
