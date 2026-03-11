"""TDD tests for GET /api/v1/incidents/{incident_id}/protocol.pdf

Behaviors:
1. 404 for unknown incident
2. Returns cached PDF (incident.protocol_pdf already set)
3. Generates PDF on the fly with RAG legal articles
4. Generates without RAG on RAG failure (timeout/error)
5. Caches generated PDF in incident after generation
"""

from __future__ import annotations

import asyncio
import importlib
import sys
from unittest.mock import AsyncMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PDF_BYTES = b"%PDF-fake-content"


def _get_real_main():
    """Ensure cloud.interface.main is the real module (not a MagicMock)."""
    mod_name = "cloud.interface.main"
    cached = sys.modules.get(mod_name)
    if cached is None or not hasattr(cached, "__file__"):
        sys.modules.pop(mod_name, None)
        importlib.import_module(mod_name)
    return sys.modules[mod_name]


@pytest.fixture
def _fresh_incidents():
    """Reset in-memory incident store between tests."""
    from cloud.db.incidents import _incidents

    _incidents.clear()
    yield
    _incidents.clear()


@pytest.fixture
def client(_fresh_incidents):
    """httpx AsyncClient bound to the FastAPI app."""
    import httpx

    main_mod = _get_real_main()
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=main_mod.app),
        base_url="http://test",
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestProtocolPdfEndpoint:
    @pytest.mark.asyncio
    async def test_protocol_pdf_returns_404_for_unknown_incident(self, client):
        """GET with non-existent incident_id → 404."""
        resp = await client.get("/api/v1/incidents/nonexistent-id/protocol.pdf")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_protocol_pdf_returns_cached_pdf(self, client):
        """If incident already has protocol_pdf bytes, return them directly."""
        from cloud.db.incidents import create_incident, update_incident

        inc = create_incident("chainsaw", 57.3, 45.0, 0.92, "alert")
        update_incident(inc.id, protocol_pdf=PDF_BYTES)

        resp = await client.get(f"/api/v1/incidents/{inc.id}/protocol.pdf")

        assert resp.status_code == 200
        assert resp.content == PDF_BYTES
        assert resp.headers["content-type"] == "application/pdf"
        assert "attachment" in resp.headers["content-disposition"]

    @pytest.mark.asyncio
    async def test_protocol_pdf_generates_on_the_fly(self, client):
        """No cached PDF → call query_legal_articles + generate_protocol."""
        from cloud.db.incidents import create_incident

        inc = create_incident("gunshot", 57.2, 44.8, 0.85, "alert")

        with (
            patch(
                "cloud.interface.main.query_legal_articles",
                new_callable=AsyncMock,
                return_value="Ст. 258 УК РФ",
            ) as mock_rag,
            patch(
                "cloud.interface.main.generate_protocol",
                return_value=PDF_BYTES,
            ) as mock_gen,
        ):
            resp = await client.get(f"/api/v1/incidents/{inc.id}/protocol.pdf")

        assert resp.status_code == 200
        assert resp.content == PDF_BYTES
        mock_rag.assert_awaited_once_with(inc.audio_class, inc.lat, inc.lon)
        mock_gen.assert_called_once_with(inc, "Ст. 258 УК РФ")

    @pytest.mark.asyncio
    async def test_protocol_pdf_generates_without_rag_on_failure(self, client):
        """RAG failure (timeout/error) → generate PDF with empty legal_articles."""
        from cloud.db.incidents import create_incident

        inc = create_incident("engine", 57.4, 45.1, 0.78, "verify")

        with (
            patch(
                "cloud.interface.main.query_legal_articles",
                new_callable=AsyncMock,
                side_effect=asyncio.TimeoutError,
            ),
            patch(
                "cloud.interface.main.generate_protocol",
                return_value=PDF_BYTES,
            ) as mock_gen,
        ):
            resp = await client.get(f"/api/v1/incidents/{inc.id}/protocol.pdf")

        assert resp.status_code == 200
        assert resp.content == PDF_BYTES
        mock_gen.assert_called_once_with(inc, "")

    @pytest.mark.asyncio
    async def test_protocol_pdf_caches_after_generation(self, client):
        """After generating, the PDF bytes are saved to incident.protocol_pdf."""
        from cloud.db.incidents import create_incident, get_incident

        inc = create_incident("chainsaw", 57.1, 44.9, 0.95, "alert")

        with (
            patch(
                "cloud.interface.main.query_legal_articles",
                new_callable=AsyncMock,
                return_value="",
            ),
            patch(
                "cloud.interface.main.generate_protocol",
                return_value=PDF_BYTES,
            ),
        ):
            await client.get(f"/api/v1/incidents/{inc.id}/protocol.pdf")

        updated = get_incident(inc.id)
        assert updated.protocol_pdf == PDF_BYTES
