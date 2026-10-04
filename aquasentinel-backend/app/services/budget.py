"""Water Budget Mode: when water is limited, serve the highest-risk zones first."""


def allocate(items, available_l, min_fraction=0.25):
    """items: dicts with zone_id, risk, volume_l, weight, pump_inhibited.
    Zones are ranked by risk*weight. A zone gets a partial share only if it is
    at least `min_fraction` of its request (otherwise water is better kept for others)."""
    ranked = sorted(items, key=lambda i: (i["volume_l"] > 0, i["risk"] * i["weight"]), reverse=True)
    remaining, out = float(available_l), []
    for rank, i in enumerate(ranked, 1):
        req, alloc = i["volume_l"], 0.0
        if i.get("pump_inhibited"):
            status = "INHIBITED"
        elif req <= 0:
            status = "NOT_NEEDED"
        elif remaining >= req:
            alloc, status = req, "FULL"
        elif remaining >= min_fraction * req:
            alloc, status = remaining, "PARTIAL"
        else:
            status = "DEFERRED"
        remaining -= alloc
        out.append({"zone_id": i["zone_id"], "rank": rank, "priority_score": round(i["risk"] * i["weight"], 3),
                    "requested_l": round(req, 1), "allocated_l": round(alloc, 1),
                    "fulfilment_pct": round(100 * alloc / req, 1) if req > 0 else None, "status": status})
    total = sum(o["allocated_l"] for o in out)
    return {"available_l": float(available_l), "allocated_l": round(total, 1),
            "unmet_l": round(sum(o["requested_l"] for o in out) - total, 1), "allocations": out}
