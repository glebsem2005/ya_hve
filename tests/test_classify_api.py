"""Tests: Edge classify HTTP API — cloud → edge without TF in cloud container.

Edge already holds TF loaded. Instead of importing classify() directly
(which pulls ~800 MB of TF), cloud calls edge via HTTP.
"""

from __future__ import annotations

import asyncio
import io
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from edge.audio.classifier import AudioResult


@pytest.fixture
def _real_main():
    """Ensure cloud.interface.main is the real module, not a MagicMock."""
    mod_name = "cloud.interface.main"
    cached = sys.modules.get(mod_name)
    if cached is None or not hasattr(cached, "__file__"):
        sys.modules.pop(mod_name, None)
        import cloud.interface.main  # noqa: F811
    return sys.modules[mod_name]


# ---- 1. POST /api/v1/classify returns JSON ----


class TestClassifyEndpoint:
    def test_classify_endpoint_returns_json(self):
        """POST audio file → JSON {label, confidence, raw_scores}."""
        fake_result = AudioResult(
            label="chainsaw", confidence=0.92, raw_scores={"chainsaw": 0.92}
        )

        with patch("edge.audio.classifier.classify", return_value=fake_result):
            from edge.classify_api import app
            from starlette.testclient import TestClient

            client = TestClient(app)
            # Send a minimal valid WAV file (44-byte header + silence)
            import struct

            wav = _make_wav_bytes(num_samples=16000)
            resp = client.post(
                "/api/v1/classify",
                files={"file": ("test.wav", io.BytesIO(wav), "audio/wav")},
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["label"] == "chainsaw"
        assert data["confidence"] == pytest.approx(0.92)
        assert "raw_scores" in data

    def test_classify_endpoint_invalid_audio(self):
        """Empty/broken file → label='unknown', confidence=0."""
        from edge.classify_api import app
        from starlette.testclient import TestClient

        with patch(
            "edge.audio.classifier.classify",
            return_value=AudioResult(label="unknown", confidence=0.0, raw_scores={}),
        ):
            client = TestClient(app)
            resp = client.post(
                "/api/v1/classify",
                files={"file": ("bad.wav", io.BytesIO(b""), "audio/wav")},
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["label"] == "unknown"
        assert data["confidence"] == 0.0


# ---- 2. Cloud wrapper → AudioResult ----


class TestClassifyViaEdge:
    def test_classify_via_edge_returns_audio_result(self, _real_main, tmp_path):
        """_classify_via_edge() returns AudioResult from edge HTTP response."""
        wav_file = tmp_path / "test.wav"
        wav_file.write_bytes(_make_wav_bytes(num_samples=16000))

        fake_json = {
            "label": "gunshot",
            "confidence": 0.88,
            "raw_scores": {"gunshot": 0.88, "background": 0.12},
        }
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = fake_json

        with patch.object(_real_main, "httpx") as mock_httpx:
            mock_httpx.post.return_value = mock_response
            result = _real_main._classify_via_edge(str(wav_file))

        assert isinstance(result, AudioResult)
        assert result.label == "gunshot"
        assert result.confidence == pytest.approx(0.88)
        assert result.raw_scores == {"gunshot": 0.88, "background": 0.12}

    def test_classify_via_edge_handles_connection_error(self, _real_main, tmp_path):
        """Edge unavailable → AudioResult(label='unknown', confidence=0)."""
        wav_file = tmp_path / "test.wav"
        wav_file.write_bytes(_make_wav_bytes(num_samples=16000))

        with patch.object(
            _real_main, "httpx", **{"post.side_effect": Exception("connection refused")}
        ):
            result = _real_main._classify_via_edge(str(wav_file))

        assert isinstance(result, AudioResult)
        assert result.label == "unknown"
        assert result.confidence == 0.0


# ---- 3. _import_demo_deps does NOT import TF ----


class TestDemoDepsNoTF:
    def test_demo_deps_uses_http_classify(self, _real_main):
        """_import_demo_deps() returns _classify_via_edge, not edge.audio.classifier.classify."""
        deps = _real_main._import_demo_deps()
        classify_fn = deps["classify"]

        # Must be _classify_via_edge, NOT edge.audio.classifier.classify
        assert classify_fn is _real_main._classify_via_edge


# ---- helpers ----


def _make_wav_bytes(num_samples: int = 16000, sr: int = 16000) -> bytes:
    """Create a minimal valid WAV file (PCM 16-bit mono silence)."""
    import struct

    data_size = num_samples * 2  # 16-bit = 2 bytes per sample
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        36 + data_size,
        b"WAVE",
        b"fmt ",
        16,  # chunk size
        1,  # PCM
        1,  # mono
        sr,  # sample rate
        sr * 2,  # byte rate
        2,  # block align
        16,  # bits per sample
        b"data",
        data_size,
    )
    return header + b"\x00" * data_size
