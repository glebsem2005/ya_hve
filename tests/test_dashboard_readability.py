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


def test_events_log_size(html):
    """Events log must have max-width and be tall enough to read comfortably."""
    max_w = _css_px(html, r"\.events-log\s*\{[^}]*max-width:\s*(\d+)px")
    max_h = _css_px(html, r"\.events-log\s*\{[^}]*max-height:\s*(\d+)px")
    assert max_w >= 520, f"Events-log max-width {max_w}px too narrow (min 520px)"
    assert max_h >= 260, f"Events-log max-height {max_h}px too short (min 260px)"


def test_events_log_font(html):
    """Events log text must be at least 14px for readability."""
    font = _css_px(html, r"\.ev-line\s*\{[^}]*font-size:\s*(\d+)px")
    assert font >= 14, f"Events-log font {font}px too small (min 14px)"


def test_sidebar_fonts(html):
    """Key sidebar fonts must meet minimum readability thresholds."""
    detect = _css_px(html, r"\.detect-class\s*\{[^}]*font-size:\s*(\d+)px")
    label = _css_px(html, r"\.sb-label\s*\{[^}]*font-size:\s*(\d+)px")
    gpt = _css_px(html, r"\.gpt-text\s*\{[^}]*font-size:\s*(\d+)px")
    assert detect >= 20, f"detect-class font {detect}px too small (min 20px)"
    assert label >= 12, f"sb-label font {label}px too small (min 12px)"
    assert gpt >= 15, f"gpt-text font {gpt}px too small (min 15px)"


def test_header_size(html):
    """Header must be tall enough and logo readable."""
    height = _css_px(html, r"\.header\s*\{[^}]*height:\s*(\d+)px")
    logo = _css_px(html, r"\.logo\s*\{[^}]*font-size:\s*(\d+)px")
    assert height >= 48, f"Header height {height}px too short (min 48px)"
    assert logo >= 18, f"Logo font {logo}px too small (min 18px)"


# --- P0/P1 regression guards ---


def _script_block(html: str) -> str:
    """Extract the main <script> block content."""
    m = re.search(r"<script>(.*)</script>", html, re.DOTALL)
    assert m, "No <script> block found"
    return m.group(1)


def test_ws_onmessage_has_try_catch(html):
    """ws.onmessage must wrap JSON.parse in try/catch to prevent crashes."""
    script = _script_block(html)
    m = re.search(r"ws\.onmessage\s*=\s*function.*?\{(.*?)\};", script, re.DOTALL)
    assert m, "ws.onmessage handler not found"
    body = m.group(1)
    assert "try" in body and "catch" in body, (
        "ws.onmessage must wrap JSON.parse in try/catch"
    )


def test_handle_event_validates_source_point_coords(html):
    """case 'source_point' must validate d.lat/d.lon before use."""
    script = _script_block(html)
    m = re.search(r"case\s+['\"]source_point['\"].*?break;", script, re.DOTALL)
    assert m, "case 'source_point' not found"
    block = m.group(0)
    assert "typeof d.lat" in block, "source_point must validate d.lat type before use"


def test_handle_event_validates_location_coords(html):
    """case 'location_found' must validate d.lat/d.lon before use."""
    script = _script_block(html)
    m = re.search(r"case\s+['\"]location_found['\"].*?break;", script, re.DOTALL)
    assert m, "case 'location_found' not found"
    block = m.group(0)
    assert "typeof d.lat" in block, "location_found must validate d.lat type before use"


def test_apply_bold_uses_function_replacer(html):
    """applyBold must use a function replacer, not a string with $1."""
    script = _script_block(html)
    m = re.search(r"function\s+applyBold.*?\}", script, re.DOTALL)
    assert m, "applyBold function not found"
    body = m.group(0)
    assert "function(" in body or "function (" in body, (
        "applyBold .replace() must use a function replacer"
    )
    assert "'$1'" not in body and '"$1"' not in body, (
        "applyBold must not use string '$1' — use function replacer"
    )


def test_inline_js_font_sizes_minimum(html):
    """All font-size values inside <script> must be >= 13px."""
    script = _script_block(html)
    sizes = re.findall(r"font-size[:\s]*['\"]?(\d+)px", script)
    for s in sizes:
        assert int(s) >= 13, f"Inline JS font-size {s}px is below minimum 13px"


def test_events_log_responsive_width(html):
    """Events log must use calc() width and max-width for responsiveness."""
    m = re.search(r"\.events-log\s*\{([^}]*)\}", html)
    assert m, ".events-log CSS block not found"
    block = m.group(1)
    assert "max-width" in block, ".events-log must have max-width"
    assert "calc(" in block, ".events-log width must use calc() for responsiveness"


# --- Protocol PDF link removal ---


def test_no_protocol_pdf_link_in_html(html):
    """HTML must not contain a protocol.pdf download link."""
    assert "protocol.pdf" not in html, (
        "protocol.pdf link must be removed from dashboard (download via Telegram bot)"
    )


def test_no_build_protocol_function(html):
    """HTML must not contain the buildProtocol JS function."""
    assert "buildProtocol" not in html, (
        "buildProtocol function must be removed from dashboard"
    )


def test_no_proto_link_css(html):
    """HTML must not contain .proto-link CSS class."""
    assert ".proto-link" not in html, ".proto-link CSS must be removed from dashboard"
