"""PDF protocol generator — Акт патрулирования лесов.

Generates a Forest Patrol Act (Приказ Минприроды N 955, Приложение 3)
using LuaLaTeX with Jinja2 templating. Falls back to fpdf2 if lualatex
is not available.

Technical stack:
  Jinja2 (custom delimiters \\VAR{}, \\BLOCK{}) → .tex → lualatex → PDF
  Font: PT Serif (ГОСТ Р 7.0.97-2016 compliant)
  Margins: 30/10/20/20 mm per ГОСТ
"""

import base64
import io
import logging
import os
import shutil
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path

import jinja2

from cloud.db.incidents import Incident
from cloud.db.rangers import get_ranger_by_chat_id

logger = logging.getLogger(__name__)

_TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"
_TEMPLATE_NAME = "act_patrol.tex"

# Russian names for audio classes
CLASS_NAME_RU = {
    "chainsaw": "Незаконная рубка леса (бензопила)",
    "gunshot": "Незаконная охота / браконьерство (выстрел)",
    "engine": "Несанкционированный заезд техники (двигатель)",
    "axe": "Незаконная рубка леса (топор)",
    "fire": "Лесной пожар (огонь)",
}

# Audio class → applicable legal article
CLASS_ARTICLE = {
    "chainsaw": "ст. 8.28 КоАП РФ / ст. 260 УК РФ (незаконная рубка)",
    "axe": "ст. 8.28 КоАП РФ / ст. 260 УК РФ (незаконная рубка)",
    "gunshot": "ст. 8.37 КоАП РФ / ст. 258 УК РФ (незаконная охота)",
    "engine": "ст. 8.25 КоАП РФ (нарушение правил лесопользования)",
    "fire": "ст. 8.32 КоАП РФ / ст. 261 УК РФ (лесной пожар)",
}

_MONTHS_RU = [
    "",
    "января",
    "февраля",
    "марта",
    "апреля",
    "мая",
    "июня",
    "июля",
    "августа",
    "сентября",
    "октября",
    "ноября",
    "декабря",
]


# ---------------------------------------------------------------------------
# Jinja2 environment with LaTeX-safe delimiters
# ---------------------------------------------------------------------------


def _make_jinja_env(template_dir: str) -> jinja2.Environment:
    """Create Jinja2 env with delimiters that don't conflict with LaTeX.

    Delimiters: ((*  *)) for variables, ((%  %)) for blocks, ((#  #)) for comments.
    """
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(template_dir),
        block_start_string="((%",
        block_end_string="%))",
        variable_start_string="((*",
        variable_end_string="*))",
        comment_start_string="((#",
        comment_end_string="#))",
        trim_blocks=True,
        lstrip_blocks=True,
        autoescape=False,
    )
    env.filters["e"] = _latex_escape
    env.filters["default"] = _jinja_default
    return env


def _jinja_default(value: str, default_value: str = "", boolean: bool = False) -> str:
    """Jinja2 default filter."""
    if boolean:
        return value if value else default_value
    return value if value is not None else default_value


def _latex_escape(value: str) -> str:
    """Escape LaTeX special characters in user-supplied text."""
    if not isinstance(value, str):
        value = str(value)
    replacements = [
        ("\\", r"\textbackslash{}"),
        ("&", r"\&"),
        ("%", r"\%"),
        ("$", r"\$"),
        ("#", r"\#"),
        ("_", r"\_"),
        ("{", r"\{"),
        ("}", r"\}"),
        ("~", r"\textasciitilde{}"),
        ("^", r"\textasciicircum{}"),
        ('"', "''"),
    ]
    for char, escaped in replacements:
        value = value.replace(char, escaped)
    return value


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------


def _save_b64_image(b64_data: str, tmpdir: str, name: str = "photo") -> str | None:
    """Decode base64 image and save to tmpdir. Returns file path or None."""
    try:
        img_bytes = base64.b64decode(b64_data)
        path = os.path.join(tmpdir, f"{name}.jpg")
        with open(path, "wb") as f:
            f.write(img_bytes)
        return path
    except Exception as exc:
        logger.warning("Failed to decode %s image: %s", name, exc)
        return None


# ---------------------------------------------------------------------------
# LuaLaTeX compilation
# ---------------------------------------------------------------------------


def _compile_latex(tex_path: str, tmpdir: str, runs: int = 2) -> bytes:
    """Compile .tex file with lualatex. Returns PDF bytes."""
    for i in range(runs):
        result = subprocess.run(
            [
                "lualatex",
                "-interaction=nonstopmode",
                "-halt-on-error",
                f"-output-directory={tmpdir}",
                tex_path,
            ],
            capture_output=True,
            text=True,
            cwd=tmpdir,
            timeout=120,
        )
        if result.returncode != 0:
            log_tail = result.stdout[-3000:] if result.stdout else ""
            logger.error(
                "lualatex run %d/%d failed (rc=%d):\n%s",
                i + 1,
                runs,
                result.returncode,
                log_tail,
            )
            raise RuntimeError(
                f"lualatex failed (run {i + 1}/{runs}): {log_tail[-500:]}"
            )

    pdf_path = Path(tex_path).with_suffix(".pdf")
    if not pdf_path.exists():
        raise RuntimeError(f"PDF not found after compilation: {pdf_path}")
    return pdf_path.read_bytes()


# ---------------------------------------------------------------------------
# Context builder
# ---------------------------------------------------------------------------


def _build_context(incident: Incident, legal_articles: str = "") -> dict:
    """Build Jinja2 template context from incident data."""
    now = (
        datetime.fromtimestamp(incident.created_at)
        if incident.created_at
        else datetime.now()
    )

    ranger_name = incident.accepted_by_name or ""
    badge_number = ""
    if incident.accepted_by_chat_id:
        ranger = get_ranger_by_chat_id(incident.accepted_by_chat_id)
        if ranger:
            ranger_name = ranger_name or ranger.name
            badge_number = ranger.badge_number or ""

    return {
        # Act metadata
        "act_number": incident.id[:8].upper(),
        "act_day": f"{now.day:02d}",
        "act_month": _MONTHS_RU[now.month],
        "act_year": str(now.year),
        # Patrol info
        "patrol_date": now.strftime("%d.%m.%Y"),
        "patrol_time_start": now.strftime("%H:%M"),
        "patrol_time_end": now.strftime("%H:%M"),
        # Location
        "lat": f"{incident.lat:.4f}",
        "lon": f"{incident.lon:.4f}",
        "sub_district": getattr(incident, "district", "") or "",
        "quarter": "",
        "compartment": "",
        "forest_purpose": "",
        # Ranger
        "ranger_name": ranger_name,
        "badge_number": badge_number,
        # Detection
        "violation_type": CLASS_NAME_RU.get(incident.audio_class, incident.audio_class),
        "confidence": f"{incident.confidence * 100:.0f}",
        "gating_level": incident.gating_level,
        "detected_at": now.strftime("%d.%m.%Y %H:%M:%S"),
        "incident_id": incident.id,
        "article": CLASS_ARTICLE.get(incident.audio_class, "требует квалификации"),
        # Legal
        "legal_articles": legal_articles,
        # Reports
        "drone_comment": incident.drone_comment or "",
        "ranger_report": incident.ranger_report_legal or "",
        # Image paths (set later in generate_protocol)
        "drone_photo_path": "",
        "ranger_photo_path": "",
        # Font path (set later if bundled fonts exist)
        "font_path": "",
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def generate_protocol(incident: Incident, legal_articles: str = "") -> bytes:
    """Generate PDF patrol act for an incident. Returns PDF as bytes.

    Uses LuaLaTeX when available, falls back to fpdf2 otherwise.
    """
    if not shutil.which("lualatex"):
        logger.warning("lualatex not found, falling back to fpdf2")
        return _generate_fpdf2_fallback(incident, legal_articles)

    context = _build_context(incident, legal_articles)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Copy template to tmpdir
        src_template = _TEMPLATE_DIR / _TEMPLATE_NAME
        dst_template = os.path.join(tmpdir, _TEMPLATE_NAME)
        shutil.copy2(src_template, dst_template)

        # Check for bundled fonts
        fonts_dir = _TEMPLATE_DIR / "fonts"
        if fonts_dir.is_dir() and any(fonts_dir.glob("*.ttf")):
            dst_fonts = os.path.join(tmpdir, "fonts")
            shutil.copytree(fonts_dir, dst_fonts)
            context["font_path"] = "./fonts/"

        # Save images to tmpdir
        if incident.drone_photo_b64:
            path = _save_b64_image(incident.drone_photo_b64, tmpdir, "drone")
            if path:
                context["drone_photo_path"] = path

        if incident.ranger_photo_b64:
            path = _save_b64_image(incident.ranger_photo_b64, tmpdir, "ranger")
            if path:
                context["ranger_photo_path"] = path

        # Render template
        env = _make_jinja_env(tmpdir)
        template = env.get_template(_TEMPLATE_NAME)
        tex_source = template.render(**context)

        rendered_path = os.path.join(tmpdir, "act.tex")
        with open(rendered_path, "w", encoding="utf-8") as f:
            f.write(tex_source)

        # Compile
        return _compile_latex(rendered_path, tmpdir)


# ---------------------------------------------------------------------------
# fpdf2 fallback (simplified, for environments without lualatex)
# ---------------------------------------------------------------------------


def _generate_fpdf2_fallback(incident: Incident, legal_articles: str = "") -> bytes:
    """Minimal PDF using fpdf2 when lualatex is unavailable."""
    from fpdf import FPDF

    _FONT_PATHS = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/app/fonts/DejaVuSans.ttf",
    ]

    pdf = FPDF()
    pdf.add_page()

    font_name = "Helvetica"
    for fpath in _FONT_PATHS:
        if os.path.exists(fpath):
            pdf.add_font("DejaVu", "", fpath, uni=True)
            bold_path = fpath.replace("DejaVuSans", "DejaVuSans-Bold")
            if os.path.exists(bold_path):
                pdf.add_font("DejaVu", "B", bold_path, uni=True)
            font_name = "DejaVu"
            break

    now = datetime.now()
    violation_type = CLASS_NAME_RU.get(incident.audio_class, incident.audio_class)
    article = CLASS_ARTICLE.get(incident.audio_class, "")

    pdf.set_font(font_name, "B", 14)
    pdf.cell(
        0, 10, text="АКТ ПАТРУЛИРОВАНИЯ ЛЕСОВ", new_x="LMARGIN", new_y="NEXT", align="C"
    )
    pdf.set_font(font_name, "", 10)
    pdf.cell(
        0,
        7,
        text=f"№ {incident.id[:8].upper()} от {now.strftime('%d.%m.%Y')}",
        new_x="LMARGIN",
        new_y="NEXT",
        align="C",
    )
    pdf.ln(4)

    pdf.set_font(font_name, "", 10)
    pdf.cell(
        0,
        7,
        text=f"Координаты: {incident.lat:.4f}°N, {incident.lon:.4f}°E",
        new_x="LMARGIN",
        new_y="NEXT",
    )
    pdf.cell(
        0, 7, text=f"Тип нарушения: {violation_type}", new_x="LMARGIN", new_y="NEXT"
    )
    pdf.cell(
        0,
        7,
        text=f"Уверенность: {incident.confidence:.0%}",
        new_x="LMARGIN",
        new_y="NEXT",
    )
    pdf.cell(
        0, 7, text=f"Уровень: {incident.gating_level}", new_x="LMARGIN", new_y="NEXT"
    )
    if article:
        pdf.cell(0, 7, text=f"Статья: {article}", new_x="LMARGIN", new_y="NEXT")

    if incident.accepted_by_name:
        pdf.ln(3)
        pdf.cell(
            0,
            7,
            text=f"Инспектор: {incident.accepted_by_name}",
            new_x="LMARGIN",
            new_y="NEXT",
        )

    if incident.ranger_report_legal:
        pdf.ln(3)
        pdf.set_font(font_name, "B", 10)
        pdf.cell(0, 7, text="Описание:", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font(font_name, "", 10)
        pdf.multi_cell(0, 6, text=incident.ranger_report_legal)

    if legal_articles:
        pdf.ln(3)
        pdf.set_font(font_name, "B", 10)
        pdf.cell(0, 7, text="Правовая база:", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font(font_name, "", 10)
        pdf.multi_cell(0, 6, text=legal_articles)

    pdf.ln(10)
    pdf.cell(
        0, 7, text="Подпись инспектора: _______________", new_x="LMARGIN", new_y="NEXT"
    )

    pdf.set_font(font_name, "", 8)
    pdf.ln(6)
    pdf.cell(
        0,
        5,
        text="Сгенерировано системой Faun (ForestGuard)",
        new_x="LMARGIN",
        new_y="NEXT",
        align="C",
    )
    pdf.cell(
        0,
        5,
        text="Приказ Минприроды N 955 от 15.12.2021, Приложение 3",
        new_x="LMARGIN",
        new_y="NEXT",
        align="C",
    )

    buf = io.BytesIO()
    pdf.output(buf)
    return buf.getvalue()
