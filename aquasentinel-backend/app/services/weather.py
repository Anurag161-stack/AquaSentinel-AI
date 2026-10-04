"""Forecast provider: Open-Meteo (free, no key) with cache and deterministic mock fallback."""
import json
import math
import time
import urllib.request

_URL = ("https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
        "&hourly=precipitation,precipitation_probability,temperature_2m,et0_fao_evapotranspiration"
        "&forecast_hours=24&timezone=auto")


class WeatherService:
    def __init__(self, provider="open-meteo", ttl_s=1800):
        self.provider, self.ttl, self._cache = provider, ttl_s, {}

    def get(self, lat, lon) -> dict:
        if self.provider == "mock":
            return self.mock(lat, lon)
        key = (round(lat, 2), round(lon, 2))
        hit = self._cache.get(key)
        if hit and time.time() - hit[0] < self.ttl:
            return hit[1]
        try:
            data = self._fetch(lat, lon)
        except Exception:
            data = dict(self.mock(lat, lon), source="fallback-mock (weather API unreachable)")
        self._cache[key] = (time.time(), data)
        return data

    def _fetch(self, lat, lon):
        with urllib.request.urlopen(_URL.format(lat=lat, lon=lon), timeout=8) as r:
            h = json.load(r)["hourly"]
        n = len(h["time"])
        z = lambda v: 0.0 if v is None else float(v)
        precip = [z(v) for v in h["precipitation"]]
        prob = [z(v) / 100.0 for v in h.get("precipitation_probability", [100] * n)]
        exp = [p * q for p, q in zip(precip, prob)]  # probability-weighted expected rain
        temps = [z(v) for v in h["temperature_2m"]]
        et = [z(v) for v in h["et0_fao_evapotranspiration"]]
        return {"rain_6h_mm": sum(exp[:6]), "rain_24h_mm": sum(exp), "rain_prob_6h": max(prob[:6]) if prob else 0.0,
                "temp_max_24h": max(temps), "et0_mm_day": sum(et) or None, "source": "open-meteo"}

    @staticmethod
    def mock(lat, lon):
        day = int(time.time() // 86400)
        rainy = (day + int(abs(lat) * 10)) % 5 == 0
        return {"rain_6h_mm": 3.0 if rainy else 0.0, "rain_24h_mm": 8.0 if rainy else 0.0,
                "rain_prob_6h": 0.7 if rainy else 0.0, "temp_max_24h": 33.0 + 3 * math.sin(day),
                "et0_mm_day": None, "source": "mock"}
