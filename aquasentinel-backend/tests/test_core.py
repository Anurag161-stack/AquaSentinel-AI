"""Run with:  python -m tests.test_core   (or pytest)"""
import os
import tempfile
import time

from app.config import Settings
from app.ml.model import StressModel
from app.services.budget import allocate
from app.services.decision import decide
from app.services.engine import Conflict, Engine

S = Settings()
_MODEL = None


class FakeWeather:
    def __init__(self, rain6=0.0, rain24=0.0):
        self.rain6, self.rain24 = rain6, rain24

    def get(self, lat, lon):
        return {"rain_6h_mm": self.rain6, "rain_24h_mm": self.rain24, "rain_prob_6h": 1.0,
                "temp_max_24h": 36.0, "et0_mm_day": None, "source": "test"}


def make_engine(rain6=0.0, rain24=0.0, private_model=False):
    global _MODEL
    _MODEL = _MODEL or StressModel.train_synthetic(n=4000)
    d = tempfile.mkdtemp()
    s = Settings(db_path=os.path.join(d, "t.db"), model_path=os.path.join(d, "m.joblib"))
    model = StressModel.train_synthetic(n=2000) if private_model else _MODEL
    return Engine(s, model=model, weather=FakeWeather(rain6, rain24))


def add_zone(e, zid="z1", moisture=17.0, temp=36.0, hum=30.0, tank=80.0, auto=False, weight=1.0, area=100.0, ts=None):
    e.create_zone({"id": zid, "name": zid, "crop": "wheat", "stage": "mid", "area_m2": area, "lat": 19.0, "lon": 73.0,
                   "device_id": f"dev-{zid}", "pump_flow_lpm": 20.0, "weight": weight, "auto_mode": auto})
    return e.ingest({"device_id": f"dev-{zid}", "soil_moisture": moisture, "air_temp": temp, "humidity": hum,
                     "tank_pct": tank, **({"ts": ts} if ts else {})})


def test_decision_rules():
    kw = dict(need_mm=20, rain6_mm=0, rain24_mm=0, tank_pct=80, moisture=18, critical_eff=23,
              area_m2=100, pump_lpm=20, settings=S)
    assert decide(risk=0.9, risk_dry=0.9, **kw).action == "IRRIGATE"
    assert decide(risk=0.4, risk_dry=0.4, **dict(kw, moisture=21)).action == "REDUCE"
    assert decide(risk=0.4, risk_dry=0.4, **kw).action == "IRRIGATE"   # severe (<= threshold-4) overrides
    assert decide(risk=0.1, risk_dry=0.1, **kw).action == "WAIT"
    d = decide(risk=0.1, risk_dry=0.8, **dict(kw, rain6_mm=10))
    assert d.action == "WAIT" and "Rain expected" in d.reasons[0]
    d = decide(risk=0.9, risk_dry=0.9, **dict(kw, tank_pct=2))
    assert d.pump_inhibited
    d = decide(risk=0.9, risk_dry=0.9, **dict(kw, need_mm=40, area_m2=10000))
    assert d.duration_min <= S.max_duration_min + 0.1


def test_budget_prioritises_high_risk():
    items = [{"zone_id": "a", "risk": 0.9, "volume_l": 500, "weight": 1, "pump_inhibited": False},
             {"zone_id": "b", "risk": 0.6, "volume_l": 500, "weight": 1, "pump_inhibited": False},
             {"zone_id": "c", "risk": 0.7, "volume_l": 500, "weight": 1, "pump_inhibited": False}]
    r = allocate(items, 700)
    st = {a["zone_id"]: a for a in r["allocations"]}
    assert st["a"]["status"] == "FULL" and st["c"]["status"] == "PARTIAL" and st["b"]["status"] == "DEFERRED"
    assert r["allocated_l"] <= 700


def test_dry_hot_irrigates_and_explains():
    e = make_engine()
    add_zone(e, moisture=16.0)
    r = e.recommend("z1")
    assert r["action"] == "IRRIGATE" and r["volume_l"] > 0 and r["id"]
    assert r["factors"] and r["factors"][0]["factor"]
    assert r["reasons"]


def test_wet_soil_waits():
    e = make_engine()
    add_zone(e, moisture=36.0, temp=22.0, hum=70.0)
    r = e.recommend("z1")
    assert r["action"] == "WAIT" and r["volume_l"] == 0


def test_rain_aware_simulation():
    e = make_engine()
    add_zone(e, moisture=21.5)
    sim = e.simulate("z1")
    sc = {s["scenario"]: s for s in sim["scenarios"]}
    assert sc["rain"]["stress_risk"] < sc["baseline"]["stress_risk"] - 0.2
    assert sc["rain"]["action"] in ("WAIT", "REDUCE")
    assert sc["heatwave"]["stress_risk"] >= sc["baseline"]["stress_risk"] - 0.02
    comp = {c["action"]: c for c in sim["action_comparison"]}
    assert comp["IRRIGATE"]["projected_moisture_6h"] > comp["WAIT"]["projected_moisture_6h"]
    assert comp["IRRIGATE"]["projected_stress_risk"] < comp["WAIT"]["projected_stress_risk"]


def test_command_lifecycle_and_feedback_loop():
    e = make_engine()
    t0 = time.time()
    add_zone(e, moisture=16.0, area=100.0, ts=t0)
    cmd = e.actuate("z1")
    assert cmd["status"] == "pending"
    try:
        e.actuate("z1")
        assert False, "second open command should conflict"
    except Conflict:
        pass
    out = e.ingest({"device_id": "dev-z1", "soil_moisture": 16.0, "air_temp": 36, "humidity": 30, "tank_pct": 80})
    assert out["command"]["id"] == cmd["id"] and out["command"]["duration_s"] > 0
    assert e.ingest({"device_id": "dev-z1", "soil_moisture": 16, "air_temp": 36, "humidity": 30,
                     "tank_pct": 80})["command"] is None            # already claimed
    e.complete_command(cmd["id"], actual_volume_l=cmd["volume_l"] * 0.9)
    # a reading 40 min later closes the loop
    e.ingest({"device_id": "dev-z1", "soil_moisture": 21.0, "air_temp": 34, "humidity": 35,
              "tank_pct": 78, "ts": time.time() + 2400})
    fb = e.feedback("z1")["records"][0]
    assert fb["moisture_after"] == 21.0 and abs(fb["volume_error_pct"] + 10) < 1
    m = e.metrics()
    assert m["water_used"]["total_l"] > 0 and "No water savings" in m["note"]


def test_tank_interlock_blocks_pump():
    e = make_engine()
    add_zone(e, moisture=16.0, tank=2.0)
    r = e.recommend("z1")
    assert r["pump_inhibited"]
    try:
        e.actuate("z1", volume_l=100)
        assert False
    except Conflict:
        pass


def test_auto_mode_creates_single_command():
    e = make_engine()
    add_zone(e, moisture=16.0, auto=True)
    # first ingest already triggered auto-evaluation; command is claimed on that same response
    with e.db.conn() as c:
        n = c.execute("SELECT COUNT(*) n FROM commands WHERE source='auto'").fetchone()["n"]
    assert n == 1
    e.ingest({"device_id": "dev-z1", "soil_moisture": 16, "air_temp": 36, "humidity": 30, "tank_pct": 80})
    with e.db.conn() as c:
        assert c.execute("SELECT COUNT(*) n FROM commands").fetchone()["n"] == 1


def test_budget_mode_in_engine():
    e = make_engine()
    add_zone(e, "a", moisture=15.0, area=200.0)
    add_zone(e, "b", moisture=19.0, area=200.0)
    full = e.allocate_budget(10_000)["allocated_l"]
    r = e.allocate_budget(full * 0.5)
    first = r["allocations"][0]
    assert first["zone_id"] == "a" and first["status"] in ("FULL", "PARTIAL")
    assert r["allocated_l"] <= full * 0.5 + 0.1


def test_fingerprint_and_dashboard():
    e = make_engine()
    now = time.time()
    add_zone(e, moisture=30.0, ts=now - 20 * 1800)
    for i in range(19, 0, -1):
        e.ingest({"device_id": "dev-z1", "soil_moisture": 30.0 - 0.2 * (20 - i), "air_temp": 30, "humidity": 50,
                  "tank_pct": 90, "ts": now - i * 1800})
    fp = e.fingerprint("z1")
    assert fp["ready"] and fp["dry_down_pct_per_h"]["day_median"] or fp["dry_down_pct_per_h"]["night_median"]
    d = e.dashboard()
    assert d["summary"]["zones"] == 1 and d["zones"][0]["recommendation"]


def test_no_telemetry_conflict():
    e = make_engine()
    e.create_zone({"id": "q", "name": "q", "crop": "maize", "area_m2": 50, "lat": 1, "lon": 1, "device_id": "d"})
    try:
        e.recommend("q")
        assert False
    except Conflict:
        pass


def test_prediction_outcomes_and_retrain():
    e = make_engine(private_model=True)
    add_zone(e, moisture=24.0, ts=time.time() - 7 * 3600)
    r = e.recommend("z1")
    with e.db.conn() as c:
        c.execute("UPDATE recommendations SET ts=? WHERE id=?", (time.time() - 7 * 3600, r["id"]))
        rec_ts = c.execute("SELECT ts FROM recommendations WHERE id=?", (r["id"],)).fetchone()["ts"]
    e.ingest({"device_id": "dev-z1", "soil_moisture": 22.0, "air_temp": 33, "humidity": 40,
              "tank_pct": 80, "ts": rec_ts + 6 * 3600})
    m = e.metrics()["prediction_quality"]
    assert m["samples_with_outcome"] == 1 and m["stress_risk_mae"] is not None
    assert e.retrain(min_samples=100)["retrained"] is False
    assert e.retrain(min_samples=1)["retrained"] is True


def test_dashboard_not_stale_after_new_reading():
    e = make_engine()
    add_zone(e, moisture=26.0)
    assert e.dashboard()["zones"][0]["recommendation"]["soil_moisture"] == 26.0
    e.ingest({"device_id": "dev-z1", "soil_moisture": 16.0, "air_temp": 36, "humidity": 30, "tank_pct": 80})
    z = e.dashboard()["zones"][0]
    assert z["recommendation"]["soil_moisture"] == 16.0 and z["reading"]["soil_moisture"] == 16.0


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print("PASS", t.__name__)
    print(f"{len(tests)} tests passed")
