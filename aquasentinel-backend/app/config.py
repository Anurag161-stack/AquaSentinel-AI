import os
from dataclasses import dataclass


def _env(name: str, default: str) -> str:
    return os.getenv(name, default)


@dataclass(frozen=True)
class Settings:
    db_path: str = _env("AQUA_DB_PATH", "aquasentinel.db")
    model_path: str = _env("AQUA_MODEL_PATH", "models/aquasentinel.joblib")
    api_key: str = _env("AQUA_API_KEY", "dev-key-change-me")
    weather_provider: str = _env("AQUA_WEATHER", "open-meteo")  # "open-meteo" | "mock"
    cors_origins: tuple = tuple(_env("AQUA_CORS", "http://localhost:5173,http://localhost:3000").split(","))

    # decision thresholds (predicted stress risk, 0-1)
    wait_below: float = 0.25
    irrigate_above: float = 0.55

    # safety interlocks
    min_tank_pct: float = 5.0          # pump inhibited at/below this tank level
    max_duration_min: float = 60.0     # hard cap on a single irrigation command
    min_gap_min: float = float(_env("AQUA_MIN_GAP_MIN", "180"))        # auto-mode: minimum time between irrigations per zone
    command_ttl_s: float = 1800.0      # pending commands expire if the device never picks them up
    stale_after_s: float = 3 * 3600.0  # sensor data older than this is flagged stale
    auto_interval_s: float = float(_env("AQUA_AUTO_INTERVAL_S", "900"))    # auto-mode re-evaluation interval per zone
