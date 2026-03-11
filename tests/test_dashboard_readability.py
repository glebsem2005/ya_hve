"""Tests: Dashboard readability — regression guard for CSS sizing.

Verifies minimum thresholds for font sizes, panel widths, and element
dimensions to prevent accidental regressions to hard-to-read values.
"""

import re
from pathlib import Path

import pytest

INDEX_HTML = (
    Path(__file__).resolve().parent.parent / "cloud" / "interface" / "index.html"
)


@pytest.fixture
def html():
    return INDEX_HTML.read_text()


def _css_px(html: str, pattern: str) -> int:
    """Extract a pixel value from CSS using a regex pattern."""
    m = re.search(pattern, html)
    assert m, f"Pattern not found: {pattern}"
    return int(m.group(1))


def test_sidebar_width(html):
    """Sidebar must be at least 460px wide for readability."""
    width = _css_px(html, r"grid-template-columns:\s*1fr\s+(\d+)px")
    assert width >= 460, f"Sidebar width {width}px is too narrow (min 460px)"
