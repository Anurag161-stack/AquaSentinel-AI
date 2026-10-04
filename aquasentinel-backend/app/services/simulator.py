"""What-If Climate Simulator helpers (scenario definitions + action projections)."""
import numpy as np

from app.ml import features as F


def scenario_overrides(base: dict) -> dict:
    """Absolute override dicts, built from the zone's current features."""
    return {
        "baseline": {},
        "heatwave": {"air_temp": base["air_temp"] + 6, "humidity": max(10.0, base["humidity"] - 15),
                     "temp_max_24h": base["temp_max_24h"] + 6, "et0_mm_day": base["et0_mm_day"] * 1.4},
        "rain": {"rain_6h_mm": 12.0, "rain_24h_mm": 20.0},
    }


def action_projections(feats: dict, hour: int, net_need_mm: float, area_m2: float, hr_pump_lpm: float):
    """Physics projection of soil moisture / stress after 6 h for WAIT vs REDUCE vs IRRIGATE."""
    et_mm = float(F.et_window_mm(feats["et0_mm_day"], feats["kc"], hour))
    rows = []
    for name, mm in (("WAIT", 0.0), ("REDUCE", 0.5 * net_need_mm), ("IRRIGATE", net_need_mm)):
        m6 = float(F.project_moisture(feats["soil_moisture"], et_mm, feats["rain_6h_mm"], mm))
        risk = float(F.stress_from_moisture(m6, feats["critical_eff"]))
        litres = mm * area_m2
        rows.append({"action": name, "water_l": round(litres, 1),
                     "duration_min": round(litres / hr_pump_lpm, 1) if hr_pump_lpm else 0.0,
                     "projected_moisture_6h": round(m6, 1), "projected_stress_risk": round(risk, 3)})
    return rows
