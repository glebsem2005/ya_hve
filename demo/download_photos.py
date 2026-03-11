#!/usr/bin/env python3
"""Download demo drone photos from Pexels for each scenario.

Usage:
    python demo/download_photos.py

Downloads 15 aerial photos (3 per scenario) into demo/photos/{scenario}/.
Photos are resized to max 800px wide for WebSocket efficiency (~50-100 KB each).
Idempotent: skips already-downloaded files.
"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

PHOTOS_DIR = Path(__file__).parent / "photos"

# scenario → [(url, description), ...]
PHOTO_MAP: dict[str, list[tuple[str, str]]] = {
    "chainsaw": [
        (
            "https://images.pexels.com/photos/3374069/pexels-photo-3374069.jpeg?auto=compress&cs=tinysrgb&w=800",
            "Aerial view of a logging site",
        ),
        (
            "https://images.pexels.com/photos/34434517/pexels-photo-34434517.jpeg?auto=compress&cs=tinysrgb&w=800",
            "Logging trucks on forest road",
        ),
        (
            "https://images.pexels.com/photos/9575021/pexels-photo-9575021.jpeg?auto=compress&cs=tinysrgb&w=800",
            "Stacks of wood pallets",
        ),
    ],
    "gunshot": [
        (
            "https://images.pexels.com/photos/35632340/pexels-photo-35632340.jpeg?auto=compress&cs=tinysrgb&w=800",
            "Forest camping site with vehicle",
        ),
        (
            "https://images.pexels.com/photos/19866099/pexels-photo-19866099.jpeg?auto=compress&cs=tinysrgb&w=800",
            "Riverside campsite",
        ),
        (
            "https://images.pexels.com/photos/16726156/pexels-photo-16726156.jpeg?auto=compress&cs=tinysrgb&w=800",
            "Large rural campsite",
        ),
    ],
    "engine": [
        (
            "https://images.pexels.com/photos/3374066/pexels-photo-3374066.jpeg?auto=compress&cs=tinysrgb&w=800",
            "Truck carrying timber, view 1",
        ),
        (
            "https://images.pexels.com/photos/3374065/pexels-photo-3374065.jpeg?auto=compress&cs=tinysrgb&w=800",
            "Truck carrying timber, view 2",
        ),
        (
            "https://images.pexels.com/photos/5851532/pexels-photo-5851532.jpeg?auto=compress&cs=tinysrgb&w=800",
            "Tree harvester in forest",
        ),
    ],
    "fire": [
        (
            "https://images.pexels.com/photos/11733087/pexels-photo-11733087.jpeg?auto=compress&cs=tinysrgb&w=800",
            "Smoke over forest",
        ),
        (
            "https://images.pexels.com/photos/29693642/pexels-photo-29693642.jpeg?auto=compress&cs=tinysrgb&w=800",
            "Smoke rising from a forest fire",
        ),
        (
            "https://images.pexels.com/photos/4552900/pexels-photo-4552900.jpeg?auto=compress&cs=tinysrgb&w=800",
            "Night bonfire in forest",
        ),
    ],
    "background": [
        (
            "https://images.pexels.com/photos/14541416/pexels-photo-14541416.jpeg?auto=compress&cs=tinysrgb&w=800",
            "Green forest canopy",
        ),
        (
            "https://images.pexels.com/photos/19301034/pexels-photo-19301034.jpeg?auto=compress&cs=tinysrgb&w=800",
            "Winding road through forest",
        ),
        (
            "https://images.pexels.com/photos/7689026/pexels-photo-7689026.jpeg?auto=compress&cs=tinysrgb&w=800",
            "Winter coniferous forest",
        ),
    ],
}


def download_all() -> None:
    total = sum(len(v) for v in PHOTO_MAP.values())
    downloaded = 0
    skipped = 0

    for scenario, photos in PHOTO_MAP.items():
        scenario_dir = PHOTOS_DIR / scenario
        scenario_dir.mkdir(parents=True, exist_ok=True)

        for i, (url, desc) in enumerate(photos, start=1):
            dest = scenario_dir / f"{i}.jpg"
            if dest.exists() and dest.stat().st_size > 1000:
                print(f"  SKIP  {scenario}/{i}.jpg ({desc})")
                skipped += 1
                continue

            print(f"  GET   {scenario}/{i}.jpg ... ", end="", flush=True)
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=30) as resp:
                    dest.write_bytes(resp.read())
                size_kb = dest.stat().st_size / 1024
                print(f"{size_kb:.0f} KB")
                downloaded += 1
            except Exception as e:
                print(f"FAILED: {e}")

    print(f"\nDone: {downloaded} downloaded, {skipped} skipped, {total} total")


if __name__ == "__main__":
    download_all()
