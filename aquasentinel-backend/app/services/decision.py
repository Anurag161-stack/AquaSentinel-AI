"""Decision engine: WAIT / REDUCE / IRRIGATE with human-readable reasons and safety interlocks."""
from dataclasses import dataclass, field


@dataclass
class Decision:
    action: str
    volume_l: float
    duration_min: float
    net_need_mm: float
    pump_inhibited: bool = False
    reasons: list = field(default_factory=list)


def decide(*, risk, risk_dry, need_mm, rain6_mm, rain24_mm, tank_pct, moisture, critical_eff,
           area_m2, pump_lpm, settings, volume_factor=1.0) -> Decision:
    net = max(0.0, need_mm - rain6_mm)
    severe = moisture <= critical_eff - 4.0
    reasons = []

    if risk_dry < settings.wait_below:
        action = "WAIT"
        reasons.append(f"Predicted 6 h stress risk is low ({risk_dry:.0%}) even with no rain; "
                       f"soil at {moisture:.1f}% vs threshold {critical_eff:.0f}%.")
    elif risk < settings.wait_below:
        action = "WAIT"
        reasons.append(f"Rain expected (~{rain6_mm:.1f} mm in 6 h) cuts predicted stress risk "
                       f"from {risk_dry:.0%} to {risk:.0%}; irrigating now would waste water.")
    elif not severe and rain24_mm >= need_mm and risk < settings.irrigate_above:
        action = "WAIT"
        reasons.append(f"~{rain24_mm:.0f} mm of rain expected within 24 h covers the estimated need "
                       f"({need_mm:.0f} mm) and stress is not yet critical ({risk:.0%}).")
    elif risk >= settings.irrigate_above or severe:
        action = "IRRIGATE"
        reasons.append(f"Predicted stress risk {risk:.0%} within 6 h; soil {moisture:.1f}% "
                       f"vs threshold {critical_eff:.0f}%.")
    else:
        action = "REDUCE"
        reasons.append(f"Moderate predicted stress risk ({risk:.0%}); a partial irrigation is enough.")

    litres = 0.0
    if action in ("IRRIGATE", "REDUCE"):
        mm = net * (1.0 if action == "IRRIGATE" else 0.5) * volume_factor
        litres = mm * area_m2  # 1 mm over 1 m² = 1 L
        if rain6_mm >= 1.0:
            reasons.append(f"Volume is net of ~{rain6_mm:.1f} mm expected rain.")
        if abs(volume_factor - 1.0) > 0.02:
            reasons.append(f"Volume scaled x{volume_factor:.2f} from this zone's measured soil response.")
        cap = pump_lpm * settings.max_duration_min
        if litres > cap:
            litres = cap
            reasons.append(f"Capped at {settings.max_duration_min:.0f} min of pumping per run.")
        if litres < 1.0:
            action, litres = "WAIT", 0.0
            reasons.append("Net water need is negligible after expected rain.")

    inhibited = False
    if action != "WAIT" and tank_pct is not None and tank_pct <= settings.min_tank_pct:
        inhibited = True
        reasons.append(f"Tank at {tank_pct:.0f}%: pump inhibited until refilled.")

    duration = litres / pump_lpm if pump_lpm > 0 else 0.0
    return Decision(action, round(litres, 1), round(duration, 1), round(net, 2), inhibited, reasons)
