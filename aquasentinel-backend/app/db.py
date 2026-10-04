import sqlite3
from contextlib import contextmanager

SCHEMA = """
CREATE TABLE IF NOT EXISTS zones(
  id TEXT PRIMARY KEY, name TEXT NOT NULL, crop TEXT NOT NULL, stage TEXT NOT NULL,
  area_m2 REAL NOT NULL, lat REAL NOT NULL, lon REAL NOT NULL,
  device_id TEXT NOT NULL UNIQUE, pump_flow_lpm REAL NOT NULL DEFAULT 20,
  weight REAL NOT NULL DEFAULT 1.0, tz_offset_min INTEGER NOT NULL DEFAULT 330,
  auto_mode INTEGER NOT NULL DEFAULT 0, created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS readings(
  id INTEGER PRIMARY KEY AUTOINCREMENT, zone_id TEXT NOT NULL, ts REAL NOT NULL,
  soil_moisture REAL, air_temp REAL, humidity REAL, tank_pct REAL,
  flow_lpm REAL DEFAULT 0, pump_on INTEGER DEFAULT 0
);
CREATE INDEX IF NOT EXISTS ix_readings ON readings(zone_id, ts);
CREATE TABLE IF NOT EXISTS recommendations(
  id INTEGER PRIMARY KEY AUTOINCREMENT, zone_id TEXT NOT NULL, ts REAL NOT NULL,
  action TEXT NOT NULL, risk REAL, risk_dry REAL, need_mm REAL, net_need_mm REAL,
  volume_l REAL, duration_min REAL, pump_inhibited INTEGER, data_stale INTEGER,
  moisture REAL, features TEXT, reasons TEXT, factors TEXT
);
CREATE INDEX IF NOT EXISTS ix_recs ON recommendations(zone_id, ts);
CREATE TABLE IF NOT EXISTS commands(
  id INTEGER PRIMARY KEY AUTOINCREMENT, zone_id TEXT NOT NULL, rec_id INTEGER,
  kind TEXT NOT NULL DEFAULT 'IRRIGATE', source TEXT, ts REAL NOT NULL,
  volume_l REAL, duration_min REAL, status TEXT NOT NULL DEFAULT 'pending',
  sent_ts REAL, completed_ts REAL, actual_volume_l REAL, moisture_before REAL
);
CREATE INDEX IF NOT EXISTS ix_cmds ON commands(zone_id, status);
CREATE TABLE IF NOT EXISTS feedback(
  id INTEGER PRIMARY KEY AUTOINCREMENT, command_id INTEGER UNIQUE, zone_id TEXT NOT NULL,
  ts_completed REAL, predicted_l REAL, actual_l REAL, volume_error_pct REAL,
  moisture_before REAL, moisture_after REAL, expected_gain REAL, actual_gain REAL,
  response_ratio REAL
);
"""


class Database:
    def __init__(self, path: str):
        self.path = path
        with self.conn() as c:
            c.executescript(SCHEMA)

    @contextmanager
    def conn(self):
        c = sqlite3.connect(self.path, timeout=15)
        c.row_factory = sqlite3.Row
        try:
            c.execute("PRAGMA journal_mode=WAL")
            yield c
            c.commit()
        except Exception:
            c.rollback()
            raise
        finally:
            c.close()
