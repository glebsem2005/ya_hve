"""Declarative pipeline definition for incident processing workflow.

JSON-serializable workflow that can be:
1. Rendered on the dashboard (workflow visualization)
2. Executed step-by-step by WorkflowExecutor
3. Exported to Yandex Workflows API
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PipelineStep:
    """A single step in the processing pipeline."""

    id: str
    name: str
    description: str
    service: str  # Which YC service handles this step
    depends_on: list[str] = field(default_factory=list)
    timeout_seconds: int = 30
    optional: bool = False

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "service": self.service,
            "depends_on": self.depends_on,
            "timeout_seconds": self.timeout_seconds,
            "optional": self.optional,
        }


# Full incident processing pipeline
PIPELINE_STEPS = [
    PipelineStep(
        id="onset",
        name="Onset Detection",
        description="Energy-ratio threshold detection of sharp sounds",
        service="Edge (ESP32)",
    ),
    PipelineStep(
        id="classification",
        name="Audio Classification",
        description="YAMNet v7 fine-tuned head (6 classes, 2048-dim)",
        service="Edge / DataSphere",
        depends_on=["onset"],
    ),
    PipelineStep(
        id="ndsi",
        name="NDSI Analysis",
        description="Normalized Difference Soundscape Index for anthropogenic corroboration",
        service="Edge",
        depends_on=["onset"],
    ),
    PipelineStep(
        id="agent_verification",
        name="AI Agent Verification",
        description="YandexGPT domain expert verifies classification with context",
        service="AI Studio (YandexGPT)",
        depends_on=["classification"],
        optional=True,
    ),
    PipelineStep(
        id="tdoa",
        name="TDOA Triangulation",
        description="GCC-PHAT + energy-based distance fusion (3-mic array)",
        service="Edge",
        depends_on=["onset"],
    ),
    PipelineStep(
        id="permit_check",
        name="Permit Verification",
        description="Check FGIS-LK / local DB for valid felling permits",
        service="Cloud DB / FGIS-LK",
        depends_on=["tdoa", "classification"],
    ),
    PipelineStep(
        id="decision",
        name="Gating Decision",
        description="3-level confidence gating: alert (>70%), verify (40-70%), log (<40%)",
        service="Edge",
        depends_on=["classification", "tdoa", "permit_check"],
    ),
    PipelineStep(
        id="drone",
        name="Drone Dispatch",
        description="Autonomous drone flight to incident location + photo capture",
        service="Edge (ArduPilot)",
        depends_on=["decision"],
        optional=True,
    ),
    PipelineStep(
        id="vision",
        name="Vision Classification",
        description="Gemma 3 27B / YandexGPT Vision analysis of drone photo",
        service="AI Studio (Gemma 3)",
        depends_on=["drone"],
        optional=True,
    ),
    PipelineStep(
        id="alert",
        name="Alert Composition",
        description="YandexGPT generates ranger alert with context",
        service="AI Studio (YandexGPT)",
        depends_on=["classification", "vision"],
    ),
    PipelineStep(
        id="rag",
        name="RAG Legal Advisor",
        description="File Search over 9 normative docs + Web Search (consultant.ru)",
        service="AI Studio (Assistants API + File Search)",
        depends_on=["classification", "tdoa"],
        optional=True,
    ),
    PipelineStep(
        id="notification",
        name="Ranger Notification",
        description="Zone-based Telegram alert with inline actions",
        service="Telegram Bot",
        depends_on=["alert"],
    ),
]


def get_pipeline_definition() -> dict:
    """Return the full pipeline definition as JSON-serializable dict."""
    return {
        "name": "ForestGuard Incident Pipeline",
        "version": "2.0",
        "steps": [s.to_dict() for s in PIPELINE_STEPS],
        "total_steps": len(PIPELINE_STEPS),
        "services_used": sorted(set(s.service for s in PIPELINE_STEPS)),
    }
