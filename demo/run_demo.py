import argparse
import subprocess
import webbrowser
import time
import httpx
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent

SCENARIOS = {"chainsaw": "Бензопила", "gunshot": "Выстрел", "normal": "Птицы"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scenario",
        choices=list(SCENARIOS.keys()),
        default="chainsaw",
        help="Demo scenario to run",
    )
    args = parser.parse_args()

    env_file = ROOT / ".env"
    if not env_file.exists():
        sys.exit(1)

    server = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "cloud.interface.main:app",
            "--host",
            "0.0.0.0",
            "--port",
            "8000",
            "--reload",
        ],
        cwd=str(ROOT),
    )

    time.sleep(2)
    webbrowser.open("http://localhost:8000")

    time.sleep(1)

    try:
        r = httpx.post(
            "http://localhost:8000/api/v1/demo",
            json={"scenario": args.scenario},
            timeout=5,
        )
    except Exception as e:
        print(f"Ошибка {e}")

    print("Press Ctrl+C to stop the process")

    try:
        server.wait()
    except KeyboardInterrupt:
        server.terminate()
        print("Stopped")


if __name__ == "__main__":
    main()
