
"""Weather provider: Open-Meteo with cached current conditions and mock fallback."""
import json
import math
import time
import urllib.request

_URL = (
    "https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
    "&current=temperature_2m,relative_humidity_2m"
    "&hourly=precipitation,precipitation_probability,temperature_2m,et0_fao_evapotranspiration"
    "&forecast_hours=24&timezone=auto"
)


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
        except Exception as exc:
            data = dict(
                self.mock(lat, lon),
                source=f"fallback-mock (weather API unavailable: {type(exc).__name__})"
            )

        self._cache[key] = (time.time(), data)
        return data

    def _fetch(self, lat, lon):
        url = _URL.format(lat=lat, lon=lon)

        with urllib.request.urlopen(url, timeout=8) as response:
            payload = json.load(response)

        current = payload.get("current", {})
        hourly = payload["hourly"]

        current_temp = current.get("temperature_2m")
        current_humidity = current.get("relative_humidity_2m")

        if current_temp is None or current_humidity is None:
            raise ValueError("Current temperature or humidity missing from weather response")

        n = len(hourly["time"])

        def number(value):
            return 0.0 if value is None else float(value)

        precipitation = [number(v) for v in hourly["precipitation"]]
        probability = [
            number(v) / 100.0
            for v in hourly.get("precipitation_probability", [100] * n)
        ]

        expected_rain = [
            p * q for p, q in zip(precipitation, probability)
        ]

        temperatures = [number(v) for v in hourly["temperature_2m"]]
        et0_values = [
            number(v) for v in hourly["et0_fao_evapotranspiration"]
        ]

        return {
            "current_temp_c": float(current_temp),
            "current_humidity_pct": float(current_humidity),
            "rain_6h_mm": sum(expected_rain[:6]),
            "rain_24h_mm": sum(expected_rain),
            "rain_prob_6h": max(probability[:6]) if probability else 0.0,
            "temp_max_24h": max(temperatures),
            "et0_mm_day": sum(et0_values) or None,
            "source": "open-meteo"
        }

    @staticmethod
    def mock(lat, lon):
        day = int(time.time() // 86400)

        rainy = (day + int(abs(lat) * 10)) % 5 == 0

        current_temp = 29.0 + 2.5 * math.sin(day * 0.4)
        current_humidity = 65.0 + 8.0 * math.cos(day * 0.3)

        return {
            "current_temp_c": round(current_temp, 1),
            "current_humidity_pct": round(
                max(35.0, min(85.0, current_humidity)), 1
            ),
            "rain_6h_mm": 3.0 if rainy else 0.0,
            "rain_24h_mm": 8.0 if rainy else 0.0,
            "rain_prob_6h": 0.7 if rainy else 0.0,
            "temp_max_24h": 33.0 + 3.0 * math.sin(day),
            "et0_mm_day": None,
            "source": "mock"
        }
