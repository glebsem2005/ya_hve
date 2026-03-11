"""Edge classify HTTP API — exposes YAMNet classifier over HTTP.

Cloud container calls this instead of importing TF directly,
saving ~800 MB RAM on the VPS.
"""

import asyncio
import logging
import tempfile
import os

from fastapi import FastAPI, UploadFile
from edge.audio.classifier import classify, AudioResult

logger = logging.getLogger(__name__)

app = FastAPI(title="Edge Classify API")


@app.post("/api/v1/classify")
async def classify_audio(file: UploadFile):
    """Classify uploaded audio file via YAMNet.

    Returns JSON {label, confidence, raw_scores}.
    On invalid/empty audio returns label='unknown'.
    """
    tmp_path = None
    try:
        content = await file.read()
        if not content:
            return {"label": "unknown", "confidence": 0.0, "raw_scores": {}}

        # Write to temp file for classifier
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        result: AudioResult = await asyncio.to_thread(classify, tmp_path)
        return {
            "label": result.label,
            "confidence": result.confidence,
            "raw_scores": result.raw_scores,
        }
    except Exception:
        logger.exception("classify_audio failed")
        return {"label": "unknown", "confidence": 0.0, "raw_scores": {}}
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
