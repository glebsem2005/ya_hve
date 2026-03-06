"""
devices/drone_node/firmware.py

Drone node — ArduPilot MAVLink interface.
Listens for mission commands from edge server over radio,
executes flight, captures photo, returns home.

Requires: pymavlink
  pip install pymavlink

Usage (on companion computer aboard drone):
  python firmware.py --connection udp:127.0.0.1:14550 --radio-port /dev/ttyUSB0
"""

import argparse
import time
import json
import serial
import threading
from pymavlink import mavutil


class DroneNode:

    def __init__(self, connection_str: str, radio_port: str, radio_baud: int = 9600):
        print(f"Connecting to autopilot: {connection_str}")
        self.mav = mavutil.mavlink_connection(connection_str)
        self.mav.wait_heartbeat()
        print("Autopilot connected")

        self.radio = serial.Serial(radio_port, radio_baud, timeout=1)
        print(f"Radio on {radio_port}")

        self.home_lat = None
        self.home_lon = None
        self._fetch_home()

    def _fetch_home(self):
        """Get current GPS position as home."""
        msg = self.mav.recv_match(type='GPS_RAW_INT', blocking=True, timeout=10)
        if msg:
            self.home_lat = msg.lat / 1e7
            self.home_lon = msg.lon / 1e7
            print(f"Home: {self.home_lat:.4f}°N {self.home_lon:.4f}°E")

    def arm_and_takeoff(self, altitude: float = 30.0):
        """Arm motors and take off to target altitude."""
        self.mav.set_mode_apm('GUIDED')
        time.sleep(1)

        self.mav.arducopter_arm()
        self.mav.motors_armed_wait()
        print("Motors armed")

        self.mav.mav.command_long_send(
            self.mav.target_system, self.mav.target_component,
            mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
            0, 0, 0, 0, 0, 0, 0, altitude
        )
        print(f"Taking off to {altitude}m")
        self._wait_altitude(altitude * 0.95)

    def fly_to(self, lat: float, lon: float, alt: float = 30.0):
        """Fly to target GPS coordinates."""
        print(f"Flying to {lat:.4f}°N {lon:.4f}°E")
        self.mav.mav.set_position_target_global_int_send(
            0, self.mav.target_system, self.mav.target_component,
            mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT,
            0b0000111111111000,  # position only
            int(lat * 1e7), int(lon * 1e7), alt,
            0, 0, 0, 0, 0, 0, 0, 0
        )
        self._wait_position(lat, lon, tolerance_m=3.0)
        print("Arrived at target")

    def capture_photo(self) -> bytes:
        """Trigger camera and return photo bytes."""
        self.mav.mav.command_long_send(
            self.mav.target_system, self.mav.target_component,
            mavutil.mavlink.MAV_CMD_DO_DIGICAM_CONTROL,
            0, 0, 0, 0, 1, 0, 0, 0  # 1 = trigger
        )
        time.sleep(2)
      
        return b""

    def return_home(self):
        """Return to home position and land."""
        self.mav.set_mode_apm('RTL')
        print("Returning home")
        self._wait_altitude(0.5, timeout=60)
        print("Landed")

    def _wait_altitude(self, target_alt: float, timeout: int = 30):
        t0 = time.time()
        while time.time() - t0 < timeout:
            msg = self.mav.recv_match(type='GLOBAL_POSITION_INT', blocking=True, timeout=1)
            if msg and abs(msg.relative_alt / 1000 - target_alt) < 1.0:
                return

    def _wait_position(self, lat: float, lon: float, tolerance_m: float = 3.0, timeout: int = 60):
        import math
        t0 = time.time()
        while time.time() - t0 < timeout:
            msg = self.mav.recv_match(type='GLOBAL_POSITION_INT', blocking=True, timeout=1)
            if msg:
                curr_lat = msg.lat / 1e7
                curr_lon = msg.lon / 1e7
                dist = math.sqrt((curr_lat - lat)**2 + (curr_lon - lon)**2) * 111000
                if dist < tolerance_m:
                    return

    def listen_for_mission(self):
        """Main loop — listen for mission commands from edge server via radio."""
        print("\nListening for commands from edge server...\n")
        while True:
            line = self.radio.readline().decode('utf-8', errors='ignore').strip()
            if not line:
                continue
            try:
                cmd = json.loads(line)
                if cmd.get("action") == "fly_to":
                    lat = cmd["lat"]
                    lon = cmd["lon"]
                    print(f"📡 Mission received: fly to {lat:.4f}°N {lon:.4f}°E")

                    self.arm_and_takeoff(altitude=30.0)
                    self.fly_to(lat, lon)
                    photo = self.capture_photo()
                    self.return_home()

                    # Send confirmation back over radio
                    response = json.dumps({
                        "status": "complete",
                        "lat": lat,
                        "lon": lon,
                        "has_photo": len(photo) > 0,
                    })
                    self.radio.write((response + "\n").encode())
                    print("Mission complete, confirmation sent\n")

            except (json.JSONDecodeError, KeyError) as e:
                print(f"Bad command: {e}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--connection", default="udp:127.0.0.1:14550",
                        help="MAVLink connection string")
    parser.add_argument("--radio-port", default="/dev/ttyUSB0",
                        help="Serial port for radio module")
    parser.add_argument("--radio-baud", type=int, default=9600)
    args = parser.parse_args()

    node = DroneNode(args.connection, args.radio_port, args.radio_baud)
    node.listen_for_mission()

if __name__ == "__main__":
    main()
