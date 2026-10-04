"""Orchestration layer. Framework-independent so it can be tested without FastAPI."""
import json
import sqlite3
import statistics
import time
import uuid
from datetime import datetime, timezone

import numpy as np

from app.config import Settings
from app.db import Database
from app.ml import features as F
from app.ml.model import StressModel
from app.services import simulator
from app.services.budget import allocate as budget_allocate
from app.services.decision import decide
from app.services.weather import WeatherService

ZONE_COLS = ("id", "name", "crop", "stage", "area_m2", "lat", "lon", "device_id",
             "pump_flow_lpm", "weight", "tz_offset_min", "auto_mode")
PATCHABLE = ("name", "crop", "stage", "area_m2", "lat", "lon", "pump_flow_lpm", "weight",
             "tz_offset_min", "auto_mode")


class NotFound(LookupError):
    pass


class Conflict(Exception):
    pass


def iso(ts):
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat() if ts else None


class Engine:
    def __init__(self, settings: Settings, model: StressModel = None, weather: WeatherService = None):
        self.s = settings
        self.db = Database(settings.db_path)
        self.model = model or StressModel.load_or_train(settings.model_path)
        self.weather = weather or WeatherService(settings.weather_provider)

    # ------------------------------------------------------------------ zones
    @staticmethod
    def _check_crop(crop, stage):
        if crop not in F.CROPS:
            raise ValueError(f"Unknown crop '{crop}'. Supported: {sorted(F.CROPS)}")
        if stage not in F.STAGES:
            raise ValueError(f"Unknown stage '{stage}'. Supported: {list(F.STAGES)}")

    def create_zone(self, d: dict) -> dict:
        d = dict(d)
        self._check_crop(d["crop"], d.get("stage", "initial"))
        d.setdefault("stage", "initial")
        d["id"] = d.get("id") or uuid.uuid4().hex[:8]
        row = {k: d.get(k) for k in ZONE_COLS}
        row.update(pump_flow_lpm=d.get("pump_flow_lpm", 20.0), weight=d.get("weight", 1.0),
                   tz_offset_min=d.get("tz_offset_min", 330), auto_mode=int(bool(d.get("auto_mode", False))),
                   created_at=time.time())
        try:
            with self.db.conn() as c:
                c.execute(f"INSERT INTO zones({','.join(row)}) VALUES({','.join(':' + k for k in row)})", row)
        except sqlite3.IntegrityError:
            raise Conflict("A zone with this id or device_id already exists")
        return self.get_zone(d["id"])

    def get_zone(self, zid) -> dict:
        with self.db.conn() as c:
            r = c.execute("SELECT * FROM zones WHERE id=?", (zid,)).fetchone()
        if not r:
            raise NotFound(f"Zone '{zid}' not found")
        z = dict(r)
        z["auto_mode"] = bool(z["auto_mode"])
        return z

    def list_zones(self):
        with self.db.conn() as c:
            ids = [r["id"] for r in c.execute("SELECT id FROM zones ORDER BY created_at")]
        return [self.get_zone(i) for i in ids]

    def update_zone(self, zid, patch: dict) -> dict:
        z = self.get_zone(zid)
        patch = {k: v for k, v in patch.items() if k in PATCHABLE and v is not None}
        self._check_crop(patch.get("crop", z["crop"]), patch.get("stage", z["stage"]))
        if "auto_mode" in patch:
            patch["auto_mode"] = int(bool(patch["auto_mode"]))
        if patch:
            with self.db.conn() as c:
                c.execute(f"UPDATE zones SET {','.join(k + '=:' + k for k in patch)} WHERE id=:id", dict(patch, id=zid))
        return self.get_zone(zid)

    # --------------------------------------------------------------- readings
    def latest_reading(self, zid):
        with self.db.conn() as c:
            r = c.execute("SELECT * FROM readings WHERE zone_id=? ORDER BY ts DESC LIMIT 1", (zid,)).fetchone()
        return dict(r) if r else None

    def readings(self, zid, hours=24, limit=5000):
        self.get_zone(zid)
        with self.db.conn() as c:
            rows = c.execute("SELECT * FROM readings WHERE zone_id=? AND ts>=? ORDER BY ts LIMIT ?",
                             (zid, time.time() - hours * 3600, limit)).fetchall()
        return [dict(r, time=iso(r["ts"])) for r in rows]

    def ingest(self, p: dict) -> dict:
        """ESP32 telemetry. Returns the next pending command (if any) in the same round trip."""
        with self.db.conn() as c:
            z = c.execute("SELECT id FROM zones WHERE device_id=?", (p["device_id"],)).fetchone()
        if not z:
            raise NotFound(f"No zone registered for device '{p['device_id']}'")
        zid, ts = z["id"], p.get("ts") or time.time()
        with self.db.conn() as c:
            c.execute("INSERT INTO readings(zone_id,ts,soil_moisture,air_temp,humidity,tank_pct,flow_lpm,pump_on)"
                      " VALUES(?,?,?,?,?,?,?,?)",
                      (zid, ts, p["soil_moisture"], p["air_temp"], p["humidity"], p.get("tank_pct"),
                       p.get("flow_lpm", 0.0), int(bool(p.get("pump_on", False)))))
        self._close_feedback(zid, ts, p["soil_moisture"])
        zone = self.get_zone(zid)
        if zone["auto_mode"]:
            self._auto(zone)
        return {"stored": True, "zone_id": zid, "command": self._claim_command(zid)}

    def _trend(self, zid, now):
        with self.db.conn() as c:
            rows = c.execute("SELECT ts, soil_moisture FROM readings WHERE zone_id=? AND ts>=? ORDER BY ts",
                             (zid, now - 3 * 3600)).fetchall()
        if len(rows) < 2:
            return 0.0
        t = np.array([r["ts"] for r in rows]) / 3600.0
        t -= t[0]
        if t[-1] < 0.1:
            return 0.0
        return float(np.clip(np.polyfit(t, [r["soil_moisture"] for r in rows], 1)[0], -1.5, 1.0))

    # --------------------------------------------------------- recommendation
    def _context(self, zone, overrides=None):
        ov = overrides or {}
        r = self.latest_reading(zone["id"])
        if not r:
            raise Conflict("No telemetry received yet for this zone")
        now = time.time()
        fc = self.weather.get(zone["lat"], zone["lon"])
        temp = ov.get("air_temp", r["air_temp"])
        hum = ov.get("humidity", r["humidity"])
        et0 = ov.get("et0_mm_day") or fc.get("et0_mm_day") or float(F.proxy_et0(temp, hum))
        hour = int(((now + zone["tz_offset_min"] * 60) // 3600) % 24)
        feats = F.build_features(
            soil_moisture=ov.get("soil_moisture", r["soil_moisture"]), air_temp=temp, humidity=hum, hour=hour,
            crop=zone["crop"], stage=zone["stage"],
            rain_6h_mm=ov.get("rain_6h_mm", fc["rain_6h_mm"]), rain_24h_mm=ov.get("rain_24h_mm", fc["rain_24h_mm"]),
            temp_max_24h=ov.get("temp_max_24h", max(fc["temp_max_24h"], temp)),
            trend=ov.get("moisture_trend_per_h", self._trend(zone["id"], now)), et0=et0)
        return {"reading": r, "forecast": fc, "features": feats, "hour": hour,
                "stale": now - r["ts"] > self.s.stale_after_s}

    def _last_rec(self, zid):
        with self.db.conn() as c:
            r = c.execute("SELECT * FROM recommendations WHERE zone_id=? ORDER BY ts DESC LIMIT 1", (zid,)).fetchone()
        return r

    @staticmethod
    def _rec_out(r) -> dict:
        feats = json.loads(r["features"])
        return {"id": r["id"], "zone_id": r["zone_id"], "created_at": iso(r["ts"]), "action": r["action"],
                "stress_risk": r["risk"], "stress_risk_without_rain": r["risk_dry"],
                "gross_need_mm": r["need_mm"], "net_need_mm": r["net_need_mm"], "volume_l": r["volume_l"],
                "duration_min": r["duration_min"], "pump_inhibited": bool(r["pump_inhibited"]),
                "data_stale": bool(r["data_stale"]), "soil_moisture": r["moisture"],
                "reasons": json.loads(r["reasons"]), "factors": json.loads(r["factors"]), "inputs": feats}

    def recommend(self, zid, persist=True, overrides=None, cache_s=0):
        zone = self.get_zone(zid)
        if cache_s and not overrides:
            last, rd = self._last_rec(zid), self.latest_reading(zid)
            if last and rd and last["ts"] >= rd["ts"] and time.time() - last["ts"] < cache_s:
                return self._rec_out(last)   # reuse only if no newer telemetry arrived since
        ctx = self._context(zone, overrides)
        f, rd = ctx["features"], ctx["reading"]
        risk, need = self.model.predict(f)
        risk_dry = self.model.predict_dry(f)
        d = decide(risk=risk, risk_dry=risk_dry, need_mm=need, rain6_mm=f["rain_6h_mm"], rain24_mm=f["rain_24h_mm"],
                   tank_pct=rd["tank_pct"], moisture=f["soil_moisture"], critical_eff=f["critical_eff"],
                   area_m2=zone["area_m2"], pump_lpm=zone["pump_flow_lpm"], settings=self.s,
                   volume_factor=self.calibration_factor(zid))
        reasons = list(d.reasons)
        if ctx["stale"]:
            reasons.append("Warning: latest sensor reading is older than 3 h; check the node.")
        rec = {"id": None, "zone_id": zid, "created_at": iso(time.time()), "action": d.action,
               "stress_risk": round(risk, 3), "stress_risk_without_rain": round(risk_dry, 3),
               "gross_need_mm": round(need, 2), "net_need_mm": d.net_need_mm, "volume_l": d.volume_l,
               "duration_min": d.duration_min, "pump_inhibited": d.pump_inhibited, "data_stale": ctx["stale"],
               "soil_moisture": f["soil_moisture"], "reasons": reasons, "factors": self.model.explain(f),
               "inputs": dict(f, forecast_source=ctx["forecast"].get("source"))}
        if persist and not overrides:
            with self.db.conn() as c:
                cur = c.execute(
                    "INSERT INTO recommendations(zone_id,ts,action,risk,risk_dry,need_mm,net_need_mm,volume_l,"
                    "duration_min,pump_inhibited,data_stale,moisture,features,reasons,factors)"
                    " VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (zid, time.time(), d.action, rec["stress_risk"], rec["stress_risk_without_rain"], rec["gross_need_mm"],
                     d.net_need_mm, d.volume_l, d.duration_min, int(d.pump_inhibited), int(ctx["stale"]),
                     f["soil_moisture"], json.dumps(f), json.dumps(reasons), json.dumps(rec["factors"])))
                rec["id"] = cur.lastrowid
        return rec

    def dashboard(self):
        zones, counts, litres, tanks, at_risk = [], {}, 0.0, [], 0
        for z in self.list_zones():
            rd = self.latest_reading(z["id"])
            rec = None
            if rd:
                rec = self.recommend(z["id"], cache_s=300)
                counts[rec["action"]] = counts.get(rec["action"], 0) + 1
                litres += rec["volume_l"]
                at_risk += rec["stress_risk"] >= self.s.irrigate_above
                if rd["tank_pct"] is not None:
                    tanks.append(rd["tank_pct"])
            zones.append({"zone": z, "reading": dict(rd, time=iso(rd["ts"]), age_min=round((time.time() - rd["ts"]) / 60, 1))
                          if rd else None, "recommendation": rec})
        return {"generated_at": iso(time.time()), "zones": zones,
                "summary": {"zones": len(zones), "actions": counts, "zones_at_risk": int(at_risk),
                            "recommended_water_l": round(litres, 1),
                            "avg_tank_pct": round(sum(tanks) / len(tanks), 1) if tanks else None}}

    # ----------------------------------------------------------- budget mode
    def allocate_budget(self, available_l, apply=False):
        items, recs = [], {}
        for z in self.list_zones():
            if not self.latest_reading(z["id"]):
                continue
            rec = self.recommend(z["id"])
            recs[z["id"]] = rec
            items.append({"zone_id": z["id"], "risk": rec["stress_risk"], "volume_l": rec["volume_l"],
                          "weight": z["weight"], "pump_inhibited": rec["pump_inhibited"]})
        res = budget_allocate(items, available_l)
        for a in res["allocations"]:
            rec = recs[a["zone_id"]]
            a["recommended_action"] = rec["action"]
            a["final_action"] = "DEFERRED" if a["status"] == "DEFERRED" else (
                rec["action"] if a["allocated_l"] > 0 or rec["action"] == "WAIT" else "WAIT")
            a["command_id"] = None
            if apply and a["allocated_l"] > 0:
                zone = self.get_zone(a["zone_id"])
                a["command_id"] = self._create_command(zone, a["allocated_l"], rec["id"], "budget")["id"]
        return res

    # ------------------------------------------------------------- simulator
    def simulate(self, zid):
        zone = self.get_zone(zid)
        ctx = self._context(zone)
        f = ctx["features"]
        out = []
        for name, ov in simulator.scenario_overrides(f).items():
            r = self.recommend(zid, persist=False, overrides=ov)
            out.append({"scenario": name, "action": r["action"], "stress_risk": r["stress_risk"],
                        "stress_risk_without_rain": r["stress_risk_without_rain"], "volume_l": r["volume_l"],
                        "duration_min": r["duration_min"], "reasons": r["reasons"][:2],
                        "overrides": {k: round(v, 2) for k, v in ov.items()}})
        net_need = max(0.0, self.model.predict(f)[1] - f["rain_6h_mm"])
        comparison = simulator.action_projections(f, ctx["hour"], net_need, zone["area_m2"], zone["pump_flow_lpm"])
        for row in comparison:
            row["recommended"] = row["action"] == out[0]["action"]
        return {"zone_id": zid, "horizon_h": F.HORIZON_H, "scenarios": out, "action_comparison": comparison,
                "note": "Scenario rows use the ML model; action_comparison uses the crop-water balance projection."}

    # ----------------------------------------------------- commands / actuation
    def _open_command(self, zid):
        with self.db.conn() as c:
            return c.execute("SELECT * FROM commands WHERE zone_id=? AND status IN ('pending','sent') "
                             "ORDER BY ts DESC LIMIT 1", (zid,)).fetchone()

    def _create_command(self, zone, volume_l, rec_id=None, source="manual", kind="IRRIGATE"):
        rd = self.latest_reading(zone["id"])
        if kind == "IRRIGATE":
            if rd and rd["tank_pct"] is not None and rd["tank_pct"] <= self.s.min_tank_pct:
                raise Conflict(f"Tank at {rd['tank_pct']:.0f}%: pump inhibited")
            if self._open_command(zone["id"]):
                raise Conflict("This zone already has an open command")
            volume_l = min(volume_l, zone["pump_flow_lpm"] * self.s.max_duration_min)
        dur = volume_l / zone["pump_flow_lpm"] if kind == "IRRIGATE" else 0.0
        with self.db.conn() as c:
            cur = c.execute("INSERT INTO commands(zone_id,rec_id,kind,source,ts,volume_l,duration_min,moisture_before)"
                            " VALUES(?,?,?,?,?,?,?,?)",
                            (zone["id"], rec_id, kind, source, time.time(), round(volume_l, 1), round(dur, 1),
                             rd["soil_moisture"] if rd else None))
        return {"id": cur.lastrowid, "zone_id": zone["id"], "kind": kind, "volume_l": round(volume_l, 1),
                "duration_min": round(dur, 1), "status": "pending", "source": source}

    def actuate(self, zid, volume_l=None, duration_min=None):
        zone = self.get_zone(zid)
        rec_id = None
        if volume_l is None and duration_min is not None:
            volume_l = duration_min * zone["pump_flow_lpm"]
        if volume_l is None:
            rec = self.recommend(zid)
            if rec["action"] == "WAIT" or rec["volume_l"] <= 0:
                raise Conflict("Current recommendation is WAIT; pass volume_l or duration_min to override")
            volume_l, rec_id = rec["volume_l"], rec["id"]
        return self._create_command(zone, volume_l, rec_id, "manual")

    def stop(self, zid):
        zone = self.get_zone(zid)
        with self.db.conn() as c:
            c.execute("UPDATE commands SET status='cancelled' WHERE zone_id=? AND status='pending'", (zid,))
        return self._create_command(zone, 0.0, None, "manual", kind="STOP")

    def _claim_command(self, zid):
        now = time.time()
        with self.db.conn() as c:
            c.execute("UPDATE commands SET status='expired' WHERE zone_id=? AND status='pending' AND ts<?",
                      (zid, now - self.s.command_ttl_s))
            c.execute("UPDATE commands SET status='failed' WHERE zone_id=? AND status='sent' AND kind='IRRIGATE' "
                      "AND sent_ts < ?-(duration_min*60+600)", (zid, now))
            busy = c.execute("SELECT 1 FROM commands WHERE zone_id=? AND status='sent' AND kind='IRRIGATE'", (zid,)).fetchone()
            cmd = c.execute("SELECT * FROM commands WHERE zone_id=? AND status='pending' "
                            "ORDER BY (kind='STOP') DESC, ts LIMIT 1", (zid,)).fetchone()
            if not cmd or (busy and cmd["kind"] != "STOP"):
                return None
            c.execute("UPDATE commands SET status='sent', sent_ts=? WHERE id=?", (now, cmd["id"]))
        return {"id": cmd["id"], "kind": cmd["kind"], "volume_l": cmd["volume_l"],
                "duration_s": int(cmd["duration_min"] * 60)}

    def _auto(self, zone):
        last = self._last_rec(zone["id"])
        if last and time.time() - last["ts"] < self.s.auto_interval_s:
            return
        rec = self.recommend(zone["id"])
        if rec["action"] == "WAIT" or rec["pump_inhibited"] or rec["volume_l"] <= 0 or self._open_command(zone["id"]):
            return
        with self.db.conn() as c:
            prev = c.execute("SELECT MAX(ts) t FROM commands WHERE zone_id=? AND kind='IRRIGATE' "
                             "AND status IN ('sent','done')", (zone["id"],)).fetchone()["t"]
        if prev and time.time() - prev < self.s.min_gap_min * 60:
            return
        self._create_command(zone, rec["volume_l"], rec["id"], "auto")

    def complete_command(self, cmd_id, actual_volume_l, duration_s=None, status="done"):
        with self.db.conn() as c:
            cmd = c.execute("SELECT * FROM commands WHERE id=?", (cmd_id,)).fetchone()
        if not cmd:
            raise NotFound(f"Command {cmd_id} not found")
        if cmd["status"] not in ("sent", "pending"):
            raise Conflict(f"Command already {cmd['status']}")
        now = time.time()
        with self.db.conn() as c:
            c.execute("UPDATE commands SET status=?, completed_ts=?, actual_volume_l=? WHERE id=?",
                      (status, now, actual_volume_l, cmd_id))
            if cmd["kind"] == "IRRIGATE" and status == "done":
                pred = cmd["volume_l"] or 0.0
                err = round(100 * (actual_volume_l - pred) / pred, 1) if pred else None
                c.execute("INSERT INTO feedback(command_id,zone_id,ts_completed,predicted_l,actual_l,volume_error_pct,"
                          "moisture_before) VALUES(?,?,?,?,?,?,?)",
                          (cmd_id, cmd["zone_id"], now, pred, actual_volume_l, err, cmd["moisture_before"]))
        return {"command_id": cmd_id, "status": status, "actual_volume_l": actual_volume_l}

    # -------------------------------------------------------------- closed loop
    def _close_feedback(self, zid, ts, moisture):
        """Once a reading arrives >=30 min after irrigation ended, record how the soil actually responded."""
        with self.db.conn() as c:
            rows = c.execute("SELECT f.*, z.area_m2 FROM feedback f JOIN zones z ON z.id=f.zone_id "
                             "WHERE f.zone_id=? AND f.moisture_after IS NULL AND f.ts_completed+1800<=?",
                             (zid, ts)).fetchall()
            for r in rows:
                expected = r["actual_l"] / r["area_m2"] / F.ROOT_DEPTH_MM * 100.0
                actual = None if r["moisture_before"] is None else moisture - r["moisture_before"]
                ratio = actual / expected if (actual is not None and expected >= 0.5) else None
                c.execute("UPDATE feedback SET moisture_after=?, expected_gain=?, actual_gain=?, response_ratio=? WHERE id=?",
                          (moisture, round(expected, 2), None if actual is None else round(actual, 2),
                           None if ratio is None else round(ratio, 3), r["id"]))

    def calibration_factor(self, zid):
        """Median soil-response ratio -> volume multiplier (needs >=3 closed-loop samples)."""
        with self.db.conn() as c:
            ratios = [r["response_ratio"] for r in c.execute(
                "SELECT response_ratio FROM feedback WHERE zone_id=? AND response_ratio IS NOT NULL "
                "ORDER BY id DESC LIMIT 10", (zid,))]
        if len(ratios) < 3:
            return 1.0
        med = min(max(statistics.median(ratios), 0.5), 1.5)
        return float(min(max(1.0 / med, 0.7), 1.5))

    def feedback(self, zid, limit=50):
        self.get_zone(zid)
        with self.db.conn() as c:
            rows = [dict(r) for r in c.execute("SELECT * FROM feedback WHERE zone_id=? ORDER BY id DESC LIMIT ?", (zid, limit))]
        return {"zone_id": zid, "calibration_factor": self.calibration_factor(zid), "records": rows}

    # -------------------------------------------------------------- fingerprint
    def fingerprint(self, zid):
        zone = self.get_zone(zid)
        with self.db.conn() as c:
            rows = c.execute("SELECT * FROM readings WHERE zone_id=? AND ts>=? ORDER BY ts LIMIT 20000",
                             (zid, time.time() - 30 * 86400)).fetchall()
        if len(rows) < 10:
            return {"zone_id": zid, "ready": False, "n_readings": len(rows),
                    "message": "Needs at least 10 readings to build a baseline."}
        m = np.array([r["soil_moisture"] for r in rows]); t = np.array([r["air_temp"] for r in rows])
        h = np.array([r["humidity"] for r in rows])
        day, night = [], []
        for a, b in zip(rows, rows[1:]):
            dt = (b["ts"] - a["ts"]) / 3600
            if 1 / 6 <= dt <= 3 and not a["pump_on"] and not b["pump_on"] and b["soil_moisture"] <= a["soil_moisture"]:
                rate = (a["soil_moisture"] - b["soil_moisture"]) / dt
                hr = ((a["ts"] + zone["tz_offset_min"] * 60) // 3600) % 24
                (day if 6 <= hr < 18 else night).append(rate)
        allr = day + night
        recent = self._trend(zid, time.time())
        med = statistics.median(allr) if allr else None
        return {"zone_id": zid, "ready": True, "crop": zone["crop"], "stage": zone["stage"], "n_readings": len(rows),
                "moisture": {"mean": round(float(m.mean()), 1), "std": round(float(m.std()), 2),
                             "p10": round(float(np.percentile(m, 10)), 1), "p90": round(float(np.percentile(m, 90)), 1)},
                "air_temp": {"mean": round(float(t.mean()), 1), "p90": round(float(np.percentile(t, 90)), 1)},
                "humidity_mean": round(float(h.mean()), 1),
                "dry_down_pct_per_h": {"day_median": round(statistics.median(day), 3) if day else None,
                                       "night_median": round(statistics.median(night), 3) if night else None},
                "recent_trend_pct_per_h": round(recent, 3),
                "drying_faster_than_usual": bool(med and -recent > 1.5 * med)}

    # ------------------------------------------------------------------ metrics
    def _outcomes(self):
        """Pairs each past prediction with the stress actually observed ~6 h later (no irrigation in between)."""
        now, out = time.time(), []
        with self.db.conn() as c:
            recs = c.execute("SELECT * FROM recommendations WHERE ts<?", (now - 6 * 3600 - 1800,)).fetchall()
            for r in recs:
                tgt = r["ts"] + 6 * 3600
                if c.execute("SELECT 1 FROM commands WHERE zone_id=? AND kind='IRRIGATE' AND ts BETWEEN ? AND ?",
                             (r["zone_id"], r["ts"] - 600, tgt)).fetchone():
                    continue
                rd = c.execute("SELECT soil_moisture FROM readings WHERE zone_id=? AND ts BETWEEN ? AND ? "
                               "ORDER BY ABS(ts-?) LIMIT 1", (r["zone_id"], tgt - 1800, tgt + 1800, tgt)).fetchone()
                if rd:
                    f = json.loads(r["features"])
                    out.append((r, f, float(F.stress_from_moisture(rd["soil_moisture"], f["critical_eff"]))))
        return out

    def metrics(self):
        now = time.time()
        with self.db.conn() as c:
            used = c.execute("SELECT zone_id, SUM(actual_volume_l) l, COUNT(*) n FROM commands "
                             "WHERE kind='IRRIGATE' AND status='done' GROUP BY zone_id").fetchall()
            used7 = c.execute("SELECT COALESCE(SUM(actual_volume_l),0) l FROM commands WHERE kind='IRRIGATE' "
                              "AND status='done' AND completed_ts>=?", (now - 7 * 86400,)).fetchone()["l"]
            errs = [abs(r["volume_error_pct"]) for r in c.execute(
                "SELECT volume_error_pct FROM feedback WHERE volume_error_pct IS NOT NULL")]
            recs = c.execute("SELECT zone_id, ts, action FROM recommendations ORDER BY zone_id, ts").fetchall()
        pairs = flips = 0
        for a, b in zip(recs, recs[1:]):
            if a["zone_id"] == b["zone_id"] and b["ts"] - a["ts"] <= 3600:
                pairs += 1
                flips += a["action"] != b["action"]
        outs = self._outcomes()
        mae = float(np.mean([abs(r["risk"] - y) for r, _, y in outs])) if outs else None
        return {
            "water_used": {"total_l": round(sum(r["l"] or 0 for r in used), 1), "last_7_days_l": round(used7, 1),
                           "by_zone": {r["zone_id"]: {"litres": round(r["l"] or 0, 1), "irrigations": r["n"]} for r in used}},
            "prediction_quality": {"samples_with_outcome": len(outs),
                                   "stress_risk_mae": None if mae is None else round(mae, 3),
                                   "model": self.model.metrics, "model_meta": self.model.meta},
            "decision_consistency": {"consecutive_pairs_within_1h": pairs,
                                     "action_change_rate": round(flips / pairs, 3) if pairs else None},
            "actuation_accuracy": {"commands_measured": len(errs),
                                   "mean_abs_volume_error_pct": round(sum(errs) / len(errs), 1) if errs else None},
            "note": "Reports measured values only. No water savings are assumed: compare water_used against your "
                    "previous schedule or a control plot.",
        }

    def retrain(self, min_samples=100):
        outs = self._outcomes()
        if len(outs) < min_samples:
            return {"retrained": False, "samples_with_outcome": len(outs),
                    "message": f"Need at least {min_samples} predictions with observed outcomes."}
        X = F.to_matrix([f for _, f, _ in outs])
        y = np.array([o for _, _, o in outs])
        meta = self.model.retrain_risk(X, y)
        try:
            self.model.save(self.s.model_path)
        except OSError:
            pass
        return {"retrained": True, **meta}
