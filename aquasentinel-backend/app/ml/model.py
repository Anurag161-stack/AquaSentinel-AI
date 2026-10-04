"""Random-forest stress + water-need model with local explanations.

Bootstraps on physics-based synthetic data (so the system runs on day one), then
can be re-trained by blending in real outcomes recorded by the closed loop.
"""
import os
import time

import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

from . import features as F

GROUPS = {
    "Current soil moisture": ["soil_moisture"],
    "Soil drying trend": ["moisture_trend_per_h"],
    "Heat & evaporative demand": ["air_temp", "humidity", "temp_max_24h", "et0_mm_day"],
    "Rain forecast": ["rain_6h_mm", "rain_24h_mm"],
    "Crop type & growth stage": ["kc", "critical_eff"],
    "Time of day": ["hour_sin", "hour_cos"],
}


def synth_dataset(n: int = 8000, seed: int = 42):
    rng = np.random.default_rng(seed)
    crops = list(F.CROPS)
    ci = rng.integers(0, len(crops), n)
    si = rng.integers(0, len(F.STAGES), n)
    crit = np.array([F.CROPS[crops[i]]["critical"] + F.STAGE_SENSITIVITY[F.STAGES[j]] for i, j in zip(ci, si)])
    kc = np.array([F.CROPS[crops[i]]["kc"][F.STAGES[j]] for i, j in zip(ci, si)])

    m_true = np.clip(rng.uniform(crit - 12, crit + 16), 3, F.FIELD_CAPACITY)
    temp = rng.uniform(14, 43, n)
    hum = rng.uniform(18, 95, n)
    hour = rng.integers(0, 24, n)
    rain6 = np.where(rng.random(n) < 0.3, rng.exponential(4.0, n), 0.0)
    rain24 = rain6 + np.where(rng.random(n) < 0.3, rng.exponential(5.0, n), 0.0)
    tmax = temp + rng.uniform(0, 7, n)
    et0 = F.proxy_et0(temp, hum) * rng.uniform(0.85, 1.15, n)

    et_mm = F.et_window_mm(et0, kc, hour)
    m_future = F.project_moisture(m_true, et_mm, rain6)
    y_risk = np.clip(F.stress_from_moisture(m_future, crit) + rng.normal(0, 0.02, n), 0, 1)
    y_need = F.gross_need_mm(m_true, et_mm, crit)

    m_obs = np.clip(m_true + rng.normal(0, 1.0, n), 0, 60)           # sensor noise
    trend = -(et_mm / F.HORIZON_H) / F.ROOT_DEPTH_MM * 100 + rng.normal(0, 0.05, n)
    ang = 2 * np.pi * hour / 24
    X = np.column_stack([m_obs, temp, hum, np.sin(ang), np.cos(ang), kc, crit,
                         et0, rain6, rain24, tmax, trend])
    return X, y_risk, y_need


def _rf(trees, seed=0):
    return RandomForestRegressor(n_estimators=trees, min_samples_leaf=2, max_features=0.7,
                                 n_jobs=-1, random_state=seed)


class StressModel:
    def __init__(self, risk, need, baseline, metrics, meta):
        self.risk, self.need, self.baseline, self.metrics, self.meta = risk, need, baseline, metrics, meta

    # ---- training ----
    @classmethod
    def train_synthetic(cls, n=8000, seed=42):
        X, yr, yn = synth_dataset(n, seed)
        cut = int(n * 0.85)
        risk, need = _rf(150, seed).fit(X[:cut], yr[:cut]), _rf(100, seed).fit(X[:cut], yn[:cut])
        metrics = {
            "risk_mae": float(mean_absolute_error(yr[cut:], risk.predict(X[cut:]))),
            "risk_r2": float(r2_score(yr[cut:], risk.predict(X[cut:]))),
            "need_mae_mm": float(mean_absolute_error(yn[cut:], need.predict(X[cut:]))),
            "holdout": "synthetic",
        }
        meta = {"source": "synthetic", "trained_at": time.time(), "n_synthetic": n, "n_real": 0}
        return cls(risk, need, X.mean(axis=0), metrics, meta)

    def retrain_risk(self, X_real, y_real, weight=3.0, n_synth=4000, seed=7):
        Xs, yrs, _ = synth_dataset(n_synth, seed)
        X = np.vstack([Xs, X_real])
        y = np.concatenate([yrs, y_real])
        w = np.concatenate([np.ones(len(Xs)), np.full(len(X_real), weight)])
        self.risk = _rf(150, seed).fit(X, y, sample_weight=w)
        self.meta.update(source="synthetic+field", trained_at=time.time(), n_real=int(len(X_real)))
        return self.meta

    # ---- persistence ----
    def save(self, path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        joblib.dump({"risk": self.risk, "need": self.need, "baseline": self.baseline,
                     "metrics": self.metrics, "meta": self.meta, "features": F.FEATURES}, path)

    @classmethod
    def load_or_train(cls, path):
        if os.path.exists(path):
            try:
                b = joblib.load(path)
                if b.get("features") == F.FEATURES:
                    return cls(b["risk"], b["need"], b["baseline"], b["metrics"], b["meta"])
            except Exception:
                pass
        m = cls.train_synthetic()
        try:
            m.save(path)
        except OSError:
            pass
        return m

    # ---- inference ----
    def predict(self, feats: dict):
        X = F.to_matrix(feats)
        return (float(np.clip(self.risk.predict(X)[0], 0, 1)),
                float(np.clip(self.need.predict(X)[0], 0, F.MAX_NEED_MM)))

    def predict_dry(self, feats: dict) -> float:
        """Stress risk if no rain arrives (used to tell 'WAIT because moist' from 'WAIT because rain')."""
        return self.predict(dict(feats, rain_6h_mm=0.0, rain_24h_mm=0.0))[0]

    def explain(self, feats: dict, top=4):
        """Local attribution: swap each factor group for its typical value and measure the change in risk."""
        x = F.to_matrix(feats)[0]
        idx = {k: i for i, k in enumerate(F.FEATURES)}
        rows = [x]
        for cols in GROUPS.values():
            z = x.copy()
            for c in cols:
                z[idx[c]] = self.baseline[idx[c]]
            rows.append(z)
        preds = self.risk.predict(np.array(rows))
        out = []
        for (name, _), p in zip(GROUPS.items(), preds[1:]):
            impact = float(preds[0] - p)
            if abs(impact) >= 0.01:
                out.append({"factor": name, "impact": round(impact, 3),
                            "effect": "raises risk" if impact > 0 else "lowers risk",
                            "detail": _detail(name, feats)})
        out.sort(key=lambda r: abs(r["impact"]), reverse=True)
        return out[:top]


def _detail(name, f):
    if name == "Current soil moisture":
        return f"{f['soil_moisture']:.1f}% vs stress threshold {f['critical_eff']:.0f}%"
    if name == "Soil drying trend":
        return f"{f['moisture_trend_per_h']:+.2f} %/h"
    if name == "Heat & evaporative demand":
        return f"{f['air_temp']:.0f}°C (max {f['temp_max_24h']:.0f}°C), RH {f['humidity']:.0f}%, ET0 {f['et0_mm_day']:.1f} mm/day"
    if name == "Rain forecast":
        return f"{f['rain_6h_mm']:.1f} mm expected in 6 h, {f['rain_24h_mm']:.1f} mm in 24 h"
    if name == "Crop type & growth stage":
        return f"Kc {f['kc']:.2f}, stress threshold {f['critical_eff']:.0f}%"
    return ""
