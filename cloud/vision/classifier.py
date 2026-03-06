import httpx
import os
from dataclasses import dataclass

YANDEX_API_KEY   = os.getenv("YANDEX_API_KEY")
YANDEX_FOLDER_ID = os.getenv("YANDEX_FOLDER_ID")
API_URL = ""

@dataclass
class VisionResult:
    description: str
    has_human: bool
    has_fire: bool
    has_felling: bool


async def classify_photo(photo_b64: str) -> VisionResult:

    prompt = """
Проанализируй снимок с дрона в лесу.
Ответь только JSON без markdown, формат:
{
  "description": "краткое описание что на снимке (1-2 предложения)",
  "has_human": true/false,
  "has_fire": true/false,
  "has_felling": true/false
}
"""

    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            API_URL,
            headers={"Authorization": f"Api-Key {YANDEX_API_KEY}"},
            json={
                "modelUri": f"gpt://{YANDEX_FOLDER_ID}/yandexgpt-vision-lite",
                "completionOptions": {"temperature": 0.1},
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{photo_b64}"
                                },
                            },
                            {"type": "text", "text": prompt},
                        ],
                    }
                ],
            },
        )
    resp.raise_for_status()

    raw = resp.json()["result"]["alternatives"][0]["message"]["text"]

    raw = raw.strip().removeprefix("```json").removesuffix("```").strip()

    import json
    data = json.loads(raw)

    return VisionResult(
        description=data.get("description", ""),
        has_human=data.get("has_human", False),
        has_fire=data.get("has_fire", False),
        has_felling=data.get("has_felling", False),
    )
