"""DataSphere Node client -- cloud inference for YAMNet classifier.

Two-tier classification:
- Edge: fast inference on device
- Cloud: verification via DataSphere Node (fresher model, higher accuracy)

Environment variables:
  DATASPHERE_NODE_ID  -- DataSphere Node ID for REST inference
  YANDEX_FOLDER_ID    -- Yandex Cloud folder ID
  YANDEX_API_KEY      -- API key for Yandex Cloud
"""

import os
import logging

import httpx

logger = logging.getLogger(__name__)

DATASPHERE_NODE_ID = os.getenv("DATASPHERE_NODE_ID", "")
DATASPHERE_FOLDER_ID = os.getenv("YANDEX_FOLDER_ID", "")
YANDEX_API_KEY = os.getenv("YANDEX_API_KEY", "")
DATASPHERE_URL = f"https://node-api.datasphere.yandexcloud.net/datasphere/user-node/{DATASPHERE_NODE_ID}"

CLASSES = ["chainsaw", "gunshot", "engine", "axe", "fire", "background"]


async def classify_embeddings(embeddings: list[float]) -> dict | None:
    """Send YAMNet embeddings to DataSphere Node for cloud classification."""
    if not DATASPHERE_NODE_ID:
        return None
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(
                DATASPHERE_URL,
                headers={
                    "x-node-id": DATASPHERE_NODE_ID,
                    "x-folder-id": DATASPHERE_FOLDER_ID,
                    "Authorization": f"Api-Key {YANDEX_API_KEY}",
                },
                json={"embeddings": embeddings},
            )
            resp.raise_for_status()
            data = resp.json()
            predictions = data.get("predictions", data.get("output", []))
            if predictions:
                idx = max(range(len(predictions)), key=lambda i: predictions[i])
                return {"label": CLASSES[idx], "confidence": predictions[idx]}
    except Exception as e:
        logger.warning("DataSphere Node call failed: %s", e)
    return None
