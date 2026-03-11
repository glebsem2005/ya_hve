import asyncio
import random
from pathlib import Path

from edge.drone.base import DroneInterface, GpsPosition, Photo

DEMO_PHOTOS_DIR = Path(__file__).parent.parent.parent / "demo" / "photos"

# Minimal valid JPEG (1x1 white pixel) — used when no demo photos available
_STUB_JPEG = bytes.fromhex(
    "ffd8ffe000104a46494600010100000100010000"
    "ffdb004300080606070605080707070909080a0c"
    "140d0c0b0b0c1912130f141d1a1f1e1d1a1c1c"
    "20242e2720222c231c1c2837292c30313434341f"
    "27393d38323c2e333432ffc00011080001000103"
    "012200021101031101ffc4001f00000105010101"
    "01010100000000000000000102030405060708090a"
    "0bffc400b5100002010303020403050504040000"
    "017d01020300041105122131410613516107227114"
    "328191a1082342b1c11552d1f02433627282090a16"
    "1718191a25262728292a3435363738393a43444546"
    "4748494a535455565758595a636465666768696a73"
    "7475767778797a838485868788898a929394959697"
    "98999aa2a3a4a5a6a7a8a9aab2b3b4b5b6b7b8b9"
    "bac2c3c4c5c6c7c8c9cad2d3d4d5d6d7d8d9dae1"
    "e2e3e4e5e6e7e8e9eaf1f2f3f4f5f6f7f8f9faffc4"
    "001f0100030101010101010101010000000000"
    "000102030405060708090a0bffda000c03010002"
    "110311003f00fdfca4a4a0028a2800a2928a00ffda"
    "000c0301000211003f00fdfca4028a0028a28a00ffd9"
)


class SimulatedDrone(DroneInterface):
    def __init__(
        self,
        home_lat: float = 55.750,
        home_lon: float = 37.610,
        scenario: str | None = None,
        photos_dir: Path | None = None,
    ):
        self.home_lat = home_lat
        self.home_lon = home_lon
        self.current_lat = home_lat
        self.current_lon = home_lon
        self.scenario = scenario
        self._photos_dir = photos_dir or DEMO_PHOTOS_DIR

    async def takeoff(self) -> None:
        await asyncio.sleep(1.0)

    async def fly_to(self, lat: float, lon: float):
        steps = 8
        for i in range(1, steps + 1):
            self.current_lat = self.home_lat + (lat - self.home_lat) * i / steps
            self.current_lon = self.home_lon + (lon - self.home_lon) * i / steps
            yield GpsPosition(lat=self.current_lat, lon=self.current_lon)
            await asyncio.sleep(0.4)

    async def capture_photo(self) -> Photo:
        await asyncio.sleep(0.5)

        photos: list[Path] = []

        # 1. Try scenario-specific directory
        if self.scenario:
            scenario_dir = self._photos_dir / self.scenario
            if scenario_dir.is_dir():
                photos = list(scenario_dir.glob("*.jpg")) + list(
                    scenario_dir.glob("*.jpeg")
                )

        # 2. Fall back to root photos directory
        if not photos:
            photos = list(self._photos_dir.glob("*.jpg")) + list(
                self._photos_dir.glob("*.jpeg")
            )

        # 3. Pick random or use stub
        if photos:
            data = random.choice(photos).read_bytes()
        else:
            data = _STUB_JPEG

        return Photo(data=data, lat=self.current_lat, lon=self.current_lon)

    async def return_home(self) -> None:
        await asyncio.sleep(1.0)
        self.current_lat = self.home_lat
        self.current_lon = self.home_lon
