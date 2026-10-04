"""Fake ESP32: registers a zone, posts telemetry, obeys commands. Stdlib only.
Usage: python scripts/simulate_device.py [base_url] [api_key]"""
import json, random, sys, time, urllib.request, urllib.error

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
KEY = sys.argv[2] if len(sys.argv) > 2 else "dev-key-change-me"


def call(method, path, body=None):
    req = urllib.request.Request(BASE + path, method=method, data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Content-Type": "application/json", "X-API-Key": KEY})
    try:
        with urllib.request.urlopen(req) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        return {"error": e.code, **json.loads(e.read() or b"{}")}


print(call("POST", "/api/zones", {"id": "north", "name": "North Field", "crop": "wheat", "stage": "mid", "area_m2": 100,
                                  "lat": 19.17, "lon": 73.24, "device_id": "esp32-north", "pump_flow_lpm": 40, "auto_mode": True}))
moisture, tank, pump_until = 24.0, 90.0, 0.0
while True:
    pumping = time.time() < pump_until
    moisture += 0.8 if pumping else -random.uniform(0.15, 0.35)
    res = call("POST", "/api/telemetry", {"device_id": "esp32-north", "soil_moisture": round(moisture, 1),
              "air_temp": round(random.uniform(31, 37), 1), "humidity": round(random.uniform(30, 50), 1),
              "tank_pct": round(tank, 1), "flow_lpm": 40 if pumping else 0, "pump_on": pumping})
    cmd = res.get("command")
    print(f"moisture={moisture:5.1f}%  tank={tank:4.1f}%  ->", cmd or "no command")
    if cmd and cmd["kind"] == "IRRIGATE":
        secs = min(cmd["duration_s"], 20)          # compress time for the demo
        pump_until = time.time() + secs
        time.sleep(secs)
        tank -= cmd["volume_l"] / 40
        delivered = cmd["volume_l"] * random.uniform(0.92, 1.0)
        moisture += delivered / 100 / 300 * 100      # litres / area(m2) / root depth(mm) -> vol% gain
        print(call("POST", f"/api/commands/{cmd['id']}/complete", {"actual_volume_l": delivered}))
    time.sleep(5)
