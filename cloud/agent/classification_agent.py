"""Real-time classification verification agent (AI Studio).

Receives edge classification results and enriches them with:
1. Context analysis (nearby permits, history, zone type)
2. YandexGPT domain expertise verification
3. Priority assessment and recommended action

Pattern: same SDK + fallback chain as rag_agent.py.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

logger = logging.getLogger(__name__)

YANDEX_API_KEY = os.getenv("YANDEX_API_KEY", "")
YANDEX_FOLDER_ID = os.getenv("YANDEX_FOLDER_ID", "")


@dataclass
class ClassificationVerification:
    """Result of AI agent verification."""

    verified_class: str
    original_class: str
    confidence: float
    priority: str  # critical, high, medium, low
    context_analysis: str
    recommended_action: str
    permit_status: str  # none, valid, expired


PRIORITY_MAP = {
    "chainsaw": "critical",
    "gunshot": "critical",
    "fire": "critical",
    "axe": "high",
    "engine": "medium",
    "background": "low",
}


async def verify_classification(
    audio_class: str,
    confidence: float,
    lat: float,
    lon: float,
    zone_type: str = "exploitation",
    ndsi: float | None = None,
) -> ClassificationVerification:
    """Verify edge classification via AI Studio agent.

    Enriches the raw classification with domain context, permit check,
    and generates recommended action.
    """
    # Check for active permits
    permit_status = "none"
    try:
        from cloud.db.permits import has_valid_permit

        if has_valid_permit(lat, lon):
            permit_status = "valid"
    except Exception:
        pass

    # Build context for YandexGPT
    context = (
        f"Детекция: {audio_class} (уверенность {confidence:.0%})\n"
        f"Координаты: {lat:.4f}N, {lon:.4f}E\n"
        f"Зона: {zone_type}\n"
        f"Разрешение на рубку: {permit_status}"
    )
    if ndsi is not None:
        context += f"\nNDSI: {ndsi:.3f}"

    # Try AI Studio SDK verification
    analysis = await _verify_with_gpt(audio_class, context)

    priority = PRIORITY_MAP.get(audio_class, "medium")
    if permit_status == "valid" and audio_class in ("chainsaw", "axe", "engine"):
        priority = "low"

    # Generate recommended action
    action = _recommend_action(audio_class, confidence, permit_status, priority)

    return ClassificationVerification(
        verified_class=audio_class,
        original_class=audio_class,
        confidence=confidence,
        priority=priority,
        context_analysis=analysis,
        recommended_action=action,
        permit_status=permit_status,
    )


async def _verify_with_gpt(audio_class: str, context: str) -> str:
    """Call YandexGPT for domain-expert verification."""
    if not YANDEX_API_KEY:
        return _fallback_analysis(audio_class, context)

    try:
        from cloud.agent.rag_agent import _call_yandex_plain

        prompt = (
            "Ты -- эксперт по лесному мониторингу. "
            "Проанализируй результат акустической детекции и дай краткую оценку "
            "(2-3 предложения): насколько вероятно нарушение, какие факторы учесть.\n\n"
            f"{context}"
        )
        return await _call_yandex_plain(prompt)
    except Exception as e:
        logger.warning("GPT verification failed: %s", e)
        return _fallback_analysis(audio_class, context)


def _fallback_analysis(audio_class: str, context: str) -> str:
    """Static fallback when API is unavailable."""
    analyses = {
        "chainsaw": "Детектирована бензопила. Необходимо проверить наличие лесной декларации. При отсутствии -- вероятная незаконная рубка (ст. 260 УК РФ).",
        "gunshot": "Детектирован выстрел. Возможно браконьерство. Требуется срочная проверка -- угроза жизни и здоровью.",
        "engine": "Детектирован двигатель техники. Проверить наличие разрешения на проезд по лесу. Возможен несанкционированный заезд.",
        "axe": "Детектирован топор. Малый масштаб, но требует проверки при отсутствии разрешения.",
        "fire": "Детектирован огонь/треск. Возможный лесной пожар. Требуется немедленная реакция.",
    }
    return analyses.get(
        audio_class,
        f"Детектирован звук класса '{audio_class}'. Требуется дополнительная проверка.",
    )


def _recommend_action(
    audio_class: str,
    confidence: float,
    permit_status: str,
    priority: str,
) -> str:
    """Generate recommended action based on classification context."""
    if permit_status == "valid" and audio_class in ("chainsaw", "axe", "engine"):
        return "Активность в зоне действующей декларации. Мониторинг без вмешательства."

    if priority == "critical":
        if confidence >= 0.70:
            return "Немедленно направить дрон и инспектора. Зафиксировать координаты и время."
        return "Направить дрон для визуальной верификации. Уведомить ближайшего инспектора."

    if priority == "high":
        return "Уведомить инспектора зоны. Подготовить дрон к вылету."

    return "Записать в журнал. Повторная проверка при следующей детекции."
