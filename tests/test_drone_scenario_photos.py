"""Tests for scenario-specific drone photo selection.

SimulatedDrone should pick a photo from demo/photos/{scenario}/
when a scenario is set, fall back to any photo in demo/photos/,
or return a valid stub JPEG when no photos exist.
"""

from __future__ import annotations

import asyncio
import struct
from pathlib import Path

import pytest

from edge.drone.simulated import SimulatedDrone

# Minimal valid JPEG (1x1 white pixel) for test fixtures
_TINY_JPEG = bytes.fromhex(
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


def _make_jpeg(path: Path, content: bytes | None = None) -> Path:
    """Create a JPEG file at path with given or default content."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content or _TINY_JPEG)
    return path


@pytest.fixture
def photos_dir(tmp_path):
    """Empty photos directory."""
    d = tmp_path / "photos"
    d.mkdir()
    return d


class TestScenarioPhotoSelection:
    """SimulatedDrone picks photos based on scenario."""

    def test_picks_photo_from_scenario_dir(self, photos_dir):
        """When scenario dir has photos, pick from there."""
        chainsaw_dir = photos_dir / "chainsaw"
        chainsaw_dir.mkdir()
        jpeg_path = _make_jpeg(chainsaw_dir / "1.jpg")
        expected = jpeg_path.read_bytes()

        drone = SimulatedDrone(scenario="chainsaw", photos_dir=photos_dir)
        photo = asyncio.get_event_loop().run_until_complete(drone.capture_photo())

        assert photo.data == expected

    def test_picks_random_from_multiple(self, photos_dir):
        """When scenario dir has multiple photos, picks one of them."""
        fire_dir = photos_dir / "fire"
        fire_dir.mkdir()
        contents = set()
        for i in range(3):
            c = _TINY_JPEG + bytes([i])  # unique content
            _make_jpeg(fire_dir / f"{i}.jpg", c)
            contents.add(c)

        drone = SimulatedDrone(scenario="fire", photos_dir=photos_dir)
        photo = asyncio.get_event_loop().run_until_complete(drone.capture_photo())

        assert photo.data in contents

    def test_fallback_to_root_photos(self, photos_dir):
        """When scenario dir doesn't exist, fall back to root photos dir."""
        _make_jpeg(photos_dir / "random.jpg")
        expected = (photos_dir / "random.jpg").read_bytes()

        drone = SimulatedDrone(scenario="gunshot", photos_dir=photos_dir)
        photo = asyncio.get_event_loop().run_until_complete(drone.capture_photo())

        assert photo.data == expected

    def test_stub_when_no_photos(self, photos_dir):
        """When no photos at all, return a valid JPEG stub."""
        drone = SimulatedDrone(scenario="chainsaw", photos_dir=photos_dir)
        photo = asyncio.get_event_loop().run_until_complete(drone.capture_photo())

        # Must start with JPEG SOI marker
        assert photo.data[:2] == b"\xff\xd8"
        # Must end with JPEG EOI marker
        assert photo.data[-2:] == b"\xff\xd9"
        # Must be non-trivial (not the broken 28-byte stub)
        assert len(photo.data) > 100

    def test_no_scenario_picks_any(self, photos_dir):
        """When scenario is None, pick any photo from root dir."""
        _make_jpeg(photos_dir / "any.jpg")
        expected = (photos_dir / "any.jpg").read_bytes()

        drone = SimulatedDrone(photos_dir=photos_dir)
        photo = asyncio.get_event_loop().run_until_complete(drone.capture_photo())

        assert photo.data == expected

    def test_jpeg_extension_variants(self, photos_dir):
        """Picks both .jpg and .jpeg files."""
        engine_dir = photos_dir / "engine"
        engine_dir.mkdir()
        _make_jpeg(engine_dir / "test.jpeg")
        expected = (engine_dir / "test.jpeg").read_bytes()

        drone = SimulatedDrone(scenario="engine", photos_dir=photos_dir)
        photo = asyncio.get_event_loop().run_until_complete(drone.capture_photo())

        assert photo.data == expected

    def test_photo_has_drone_coordinates(self, photos_dir):
        """Photo should have drone's current coordinates."""
        _make_jpeg(photos_dir / "x.jpg")

        drone = SimulatedDrone(home_lat=57.30, home_lon=45.00, photos_dir=photos_dir)
        photo = asyncio.get_event_loop().run_until_complete(drone.capture_photo())

        assert photo.lat == 57.30
        assert photo.lon == 45.00
