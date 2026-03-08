"""PDF protocol generator using fpdf2.

Generates an administrative violation protocol with:
- System monitoring data (class, confidence, coordinates)
- Drone photo + AI comment
- Ranger photo + legal description
- Applicable legal articles from RAG
"""

import io
import base64
import logging
import tempfile
from datetime import datetime

from fpdf import FPDF

from cloud.db.incidents import Incident

logger = logging.getLogger(__name__)

# DejaVu Sans supports Cyrillic; bundled path or system fallback
_FONT_PATHS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/app/fonts/DejaVuSans.ttf",
]

CLASS_NAME_RU = {
    "chainsaw": "Незаконная рубка леса (бензопила)",
    "gunshot": "Незаконная охота / браконьерство (выстрел)",
    "engine": "Несанкционированный заезд техники (двигатель)",
    "axe": "Незаконная рубка леса (топор)",
    "fire": "Лесной пожар (огонь)",
}


def _find_font() -> str | None:
    import os

    for path in _FONT_PATHS:
        if os.path.exists(path):
            return path
    return None


def _add_image_from_b64(pdf: FPDF, b64_data: str, w: int = 80) -> None:
    """Decode base64 image and add to PDF."""
    try:
        img_bytes = base64.b64decode(b64_data)
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            f.write(img_bytes)
            f.flush()
            pdf.image(f.name, w=w)
    except Exception as e:
        logger.warning("Failed to add image to PDF: %s", e)
        pdf.cell(0, 8, text="[Фото недоступно]", new_x="LMARGIN", new_y="NEXT")


def generate_protocol(incident: Incident, legal_articles: str = "") -> bytes:
    """Generate PDF protocol for an incident. Returns PDF as bytes."""
    pdf = FPDF()
    pdf.add_page()

    font_path = _find_font()
    if font_path:
        pdf.add_font("DejaVu", "", font_path, uni=True)
        pdf.add_font(
            "DejaVu", "B", font_path.replace("DejaVuSans", "DejaVuSans-Bold"), uni=True
        )
        font_name = "DejaVu"
    else:
        logger.warning("DejaVu font not found, PDF may not render Cyrillic")
        font_name = "Helvetica"

    now = datetime.now()

    # Title
    pdf.set_font(font_name, "B", 16)
    pdf.cell(0, 12, text="ПРОТОКОЛ", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.set_font(font_name, "", 11)
    pdf.cell(
        0,
        8,
        text="об административном правонарушении",
        new_x="LMARGIN",
        new_y="NEXT",
        align="C",
    )
    pdf.ln(6)

    # Date and coordinates
    pdf.set_font(font_name, "", 10)
    pdf.cell(
        0,
        7,
        text=f"Дата: {now.strftime('%d.%m.%Y %H:%M')}",
        new_x="LMARGIN",
        new_y="NEXT",
    )
    pdf.cell(
        0,
        7,
        text=f"Координаты: {incident.lat:.4f} N, {incident.lon:.4f} E",
        new_x="LMARGIN",
        new_y="NEXT",
    )
    pdf.ln(4)

    # System data
    pdf.set_font(font_name, "B", 11)
    pdf.cell(0, 8, text="ДАННЫЕ СИСТЕМЫ МОНИТОРИНГА", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font(font_name, "", 10)

    violation_type = CLASS_NAME_RU.get(incident.audio_class, incident.audio_class)
    pdf.cell(
        0, 7, text=f"Тип нарушения: {violation_type}", new_x="LMARGIN", new_y="NEXT"
    )
    pdf.cell(
        0,
        7,
        text=f"Уверенность системы: {incident.confidence:.0%}",
        new_x="LMARGIN",
        new_y="NEXT",
    )
    pdf.cell(
        0,
        7,
        text=f"Уровень: {incident.gating_level}",
        new_x="LMARGIN",
        new_y="NEXT",
    )
    pdf.ln(3)

    # Drone photo
    if incident.drone_photo_b64:
        pdf.set_font(font_name, "B", 10)
        pdf.cell(0, 7, text="Снимок с дрона:", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font(font_name, "", 10)
        _add_image_from_b64(pdf, incident.drone_photo_b64, w=90)
        if incident.drone_comment:
            pdf.ln(2)
            pdf.multi_cell(0, 6, text=f"Комментарий системы: {incident.drone_comment}")
        pdf.ln(3)

    # Ranger data
    pdf.set_font(font_name, "B", 11)
    pdf.cell(0, 8, text="ДАННЫЕ ИНСПЕКТОРА", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font(font_name, "", 10)

    if incident.accepted_by_name:
        pdf.cell(
            0,
            7,
            text=f"Инспектор: {incident.accepted_by_name}",
            new_x="LMARGIN",
            new_y="NEXT",
        )

    if incident.ranger_photo_b64:
        pdf.set_font(font_name, "B", 10)
        pdf.cell(0, 7, text="Фото нарушения:", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font(font_name, "", 10)
        _add_image_from_b64(pdf, incident.ranger_photo_b64, w=90)
        pdf.ln(3)

    if incident.ranger_report_legal:
        pdf.set_font(font_name, "B", 10)
        pdf.cell(0, 7, text="Описание:", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font(font_name, "", 10)
        pdf.multi_cell(0, 6, text=incident.ranger_report_legal)
        pdf.ln(3)

    # Legal articles
    if legal_articles:
        pdf.set_font(font_name, "B", 11)
        pdf.cell(0, 8, text="ПРАВОВАЯ БАЗА", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font(font_name, "", 10)
        pdf.multi_cell(0, 6, text=legal_articles)
        pdf.ln(6)

    # Signature
    pdf.ln(10)
    pdf.cell(
        0, 7, text="Подпись инспектора: _______________", new_x="LMARGIN", new_y="NEXT"
    )

    # Footer
    pdf.set_font(font_name, "", 8)
    pdf.ln(8)
    pdf.cell(
        0,
        5,
        text="Сгенерировано системой ForestGuard",
        new_x="LMARGIN",
        new_y="NEXT",
        align="C",
    )

    buf = io.BytesIO()
    pdf.output(buf)
    return buf.getvalue()
