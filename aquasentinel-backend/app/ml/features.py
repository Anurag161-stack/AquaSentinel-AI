"""Crop-water physics + feature construction.

The same physics generates synthetic training labels and the what-if projections,
so the simulator and the model agree with each other.
Soil moisture is volumetric water content (%).
"""
import math

import numpy as np

# critical = volumetric moisture (%) below which the crop starts to suffer
CROPS = {
    "wheat":     {"critical": 20.0, "kc": {"initial": 0.40, "development": 0.80, "mid": 1.15, "late": 0.40}},
    "rice":      {"critical": 32.0, "kc": {"initial": 1.05, "development": 1.10, "mid": 1.20, "late": 0.90}},
    "cotton":    {"critical": 22.0, "kc": {"initial": 0.35, "development": 0.75, "mid": 1.15, "late": 0.70}},
    "maize":     {"critical": 22.0, "kc": {"initial": 0.40, "development": 0.80, "mid": 1.20, "late": 0.60}},
    "tomato":    {"critical": 26.0, "kc": {"initial": 0.60, "development": 0.90, "mid": 1.15, "late": 0.80}},
    "sugarcane": {"critical": 24.0, "kc": {"initial": 0.40, "development": 0.90, "mid": 1.25, "late": 0.75}},
}
STAGES = ("initial", "development", "mid", "late")
STAGE_SENSITIVITY = {"initial": 0.0, "development": 1.0, "mid": 3.0, "late": 0.0}  # raises threshold

ROOT_DEPTH_MM = 300.0    # 1 mm of water = 1/3 volumetric %
FIELD_CAPACITY = 42.0
MAX_NEED_MM = 40.0
HORIZON_H = 6

FEATURES = [
    "soil_moisture", "air_temp", "humidity", "hour_sin", "hour_cos", "kc", "critical_eff",
    "et0_mm_day", "rain_6h_mm", "rain_24h_mm", "temp_max_24h", "moisture_trend_per_h",
]

# diurnal ET weighting: share of daily ET falling in each hour; WINDOW_FRAC[h] = share in h..h+5
_raw = np.full(24, 0.02)
for _h in range(6, 19):
    _raw[_h] = max(0.02, math.sin(math.pi * (_h - 6) / 12))
_w = _raw / _raw.sum()
WINDOW_FRAC = np.array([sum(_w[(h + i) % 24] for i in range(HORIZON_H)) for h in range(24)])


def proxy_et0(temp, hum):
    """Cheap reference-ET proxy (mm/day) from temperature & humidity when no forecast ET is available."""
    return np.maximum(0.5, 0.18 * np.asarray(temp) * (1.3 - np.asarray(hum) / 100.0))


def et_window_mm(et0_mm_day, kc, hour):
    return np.asarray(et0_mm_day) * np.asarray(kc) * WINDOW_FRAC[np.asarray(hour, dtype=int) % 24]


def project_moisture(m, et_mm, rain_mm, irrigation_mm=0.0):
    delta = (np.asarray(rain_mm) + np.asarray(irrigation_mm) - np.asarray(et_mm)) / ROOT_DEPTH_MM * 100.0
    return np.clip(np.asarray(m) + delta, 0.0, FIELD_CAPACITY)


def stress_from_moisture(m, critical_eff):
    return 1.0 / (1.0 + np.exp(-(np.asarray(critical_eff) - np.asarray(m)) / 2.0))


def gross_need_mm(m, et_mm, critical_eff):
    """Water (mm, rain excluded) to bring moisture to a comfortable level by end of horizon."""
    target = np.asarray(critical_eff) + 8.0
    need = (target - project_moisture(m, et_mm, 0.0)) * ROOT_DEPTH_MM / 100.0
    return np.clip(need, 0.0, MAX_NEED_MM)


def crop_params(crop: str, stage: str):
    c = CROPS[crop]
    return c["kc"][stage], c["critical"] + STAGE_SENSITIVITY[stage]


def build_features(*, soil_moisture, air_temp, humidity, hour, crop, stage,
                   rain_6h_mm, rain_24h_mm, temp_max_24h, trend, et0=None) -> dict:
    kc, crit = crop_params(crop, stage)
    if et0 is None:
        et0 = float(proxy_et0(air_temp, humidity))
    ang = 2 * math.pi * (hour % 24) / 24
    return {
        "soil_moisture": float(soil_moisture), "air_temp": float(air_temp), "humidity": float(humidity),
        "hour_sin": math.sin(ang), "hour_cos": math.cos(ang), "kc": float(kc), "critical_eff": float(crit),
        "et0_mm_day": float(et0), "rain_6h_mm": float(rain_6h_mm), "rain_24h_mm": float(rain_24h_mm),
        "temp_max_24h": float(temp_max_24h), "moisture_trend_per_h": float(trend),
    }


def to_matrix(feats) -> np.ndarray:
    if isinstance(feats, dict):
        feats = [feats]
    return np.array([[f[k] for k in FEATURES] for f in feats], dtype=float)
