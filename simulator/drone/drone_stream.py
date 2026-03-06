import asyncio
import os
from pathlib import Path
from dataclasses import dataclass

DEMO_PHOTOS_DIR = Path(__file__).parent.parent.parent / "demo" / "photos"

DEFAULT_FLIGHT_PATH = [
    (55.7510, 37.6130),
    (55.7511, 37.6131),
    (55.7512, 37.6132),
    (55.7513, 37.6133),
    (55.7514, 37.6134),
]

@dataclass
class DroneFrame:
    lat: float
    lon: float
    alt: float
    photo_path: str | None
  
class DroneSimulator:

    def __init__(self, target_lat: float, target_lon: float, scenario: str = "chainsaw"):
        self.target_lat = target_lat
        self.target_lon = target_lon
        self.scenario = scenario

    async def stream(self):
        start_lat = DEFAULT_FLIGHT_PATH[0][0]
        start_lon = DEFAULT_FLIGHT_PATH[0][1]

        steps = 8
        for i in range(steps):
            t = i / (steps - 1)
            lat = start_lat + (self.target_lat - start_lat) * t
            lon = start_lon + (self.target_lon - start_lon) * t
            alt = 50.0 + 10.0 * (1 - abs(t - 0.5) * 2)  
          
            if i == steps - 1:
                photo_path = self._get_photo()
                yield DroneFrame(lat=lat, lon=lon, alt=alt, photo_path=photo_path)
            else:
                yield DroneFrame(lat=lat, lon=lon, alt=alt, photo_path=None)

            await asyncio.sleep(0.5)

    def _get_photo(self) -> str | None:
        specific = DEMO_PHOTOS_DIR / f"{self.scenario}.jpg"
        if specific.exists():
            return str(specific)

        photos = list(DEMO_PHOTOS_DIR.glob("*.jpg")) + \
                 list(DEMO_PHOTOS_DIR.glob("*.jpeg")) + \
                 list(DEMO_PHOTOS_DIR.glob("*.png"))
        if photos:
            return str(photos[0])

        return None
