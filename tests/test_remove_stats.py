"""Tests: СТАТ tab and KPI block removal.

Verifies that:
1. No "СТАТ" tab button in navigation
2. No KPI block in analytics page
3. DataLens iframe is preserved
4. /api/v1/datalens/stats endpoint is removed
"""

from pathlib import Path

import pytest

INDEX_HTML = (
    Path(__file__).resolve().parent.parent / "cloud" / "interface" / "index.html"
)


@pytest.fixture
def html():
    return INDEX_HTML.read_text()


def test_no_stats_tab_in_navigation(html):
    """Navigation must not contain СТАТ tab button."""
    assert "tab-stats" not in html
    assert "switchTab('stats')" not in html
    assert ">СТАТ<" not in html


def test_no_stats_page_element(html):
    """There should be no page-stats div."""
    assert 'id="page-stats"' not in html


def test_no_kpi_block_in_analytics(html):
    """Analytics page must not contain the KPI block."""
    assert 'id="analytics-kpi"' not in html
    assert "analytics-kpi" not in html


def test_datalens_iframe_preserved(html):
    """DataLens iframe must still be present."""
    assert 'id="datalens-iframe"' in html
    assert "datalens.yandex" in html


def test_no_load_stats_function(html):
    """loadStats() function must be removed."""
    assert "loadStats()" not in html
    assert "renderStats" not in html


def test_no_stats_css(html):
    """Stats-specific CSS classes must be removed."""
    assert ".stats-page" not in html
    assert ".stats-kpi" not in html
    assert ".stats-grid" not in html
    assert ".stats-panel" not in html


def test_stats_endpoint_removed():
    """Endpoint /api/v1/datalens/stats must not exist in main.py."""
    main_py = (
        Path(__file__).resolve().parent.parent / "cloud" / "interface" / "main.py"
    ).read_text()
    assert "/api/v1/datalens/stats" not in main_py
    assert "get_datalens_stats" not in main_py


def test_datalens_incidents_endpoint_preserved():
    """Endpoint /api/v1/datalens/incidents must still exist."""
    main_py = (
        Path(__file__).resolve().parent.parent / "cloud" / "interface" / "main.py"
    ).read_text()
    assert "/api/v1/datalens/incidents" in main_py
    assert "get_datalens_incidents" in main_py


def test_get_datalens_stats_removed():
    """get_datalens_stats function must be removed from datalens.py."""
    datalens_py = (
        Path(__file__).resolve().parent.parent / "cloud" / "analytics" / "datalens.py"
    ).read_text()
    assert "get_datalens_stats" not in datalens_py


def test_get_datalens_incidents_preserved():
    """get_datalens_incidents function must still exist."""
    datalens_py = (
        Path(__file__).resolve().parent.parent / "cloud" / "analytics" / "datalens.py"
    ).read_text()
    assert "def get_datalens_incidents" in datalens_py
