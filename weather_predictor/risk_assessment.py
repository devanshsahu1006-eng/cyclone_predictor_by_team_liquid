"""
Cyclone Impact & Risk Assessment Helpers
-----------------------------------------
Ported from server3(1).py and adapted to plug into the existing
CycloneLens FastAPI backend (app.py / model.py).

These estimators are deliberately simple, threshold-based heuristics —
same as in the source prototype — intended to complement (not replace)
the existing intensity/category/pressure/radius outputs already
produced by model.py. They take the already-computed fused wind speed
(vmax, in knots) as input so no duplicate inference or duplicate IMD
category logic is introduced.
"""

import math


# ---------------------------------------------------------------------------
# Property damage risk
# ---------------------------------------------------------------------------
def estimate_property_damage(wind_speed: float) -> dict:
    if wind_speed < 34:
        risk, score = "Low", 10
    elif wind_speed < 48:
        risk, score = "Moderate", 30
    elif wind_speed < 64:
        risk, score = "High", 50
    elif wind_speed < 90:
        risk, score = "Very High", 70
    elif wind_speed < 120:
        risk, score = "Extreme", 85
    else:
        risk, score = "Catastrophic", 100

    return {
        "risk_level": risk,
        "damage_score": score,
        "note": "Estimated from cyclone intensity",
    }


# ---------------------------------------------------------------------------
# Dissipation / calming time
# ---------------------------------------------------------------------------
def estimate_dissipation_time(wind_speed: float) -> dict:
    if wind_speed < 34:
        hours = 12
    elif wind_speed < 48:
        hours = 24
    elif wind_speed < 64:
        hours = 36
    elif wind_speed < 90:
        hours = 48
    elif wind_speed < 120:
        hours = 72
    else:
        hours = 96

    return {
        "estimated_hours": hours,
        "estimated_days": round(hours / 24, 1),
        "note": "Approximate estimate based primarily on intensity",
    }


# ---------------------------------------------------------------------------
# Mortality risk
# ---------------------------------------------------------------------------
def estimate_mortality_risk(wind_speed: float) -> dict:
    if wind_speed < 34:
        risk, score = "Low", 5
    elif wind_speed < 48:
        risk, score = "Moderate", 20
    elif wind_speed < 64:
        risk, score = "High", 40
    elif wind_speed < 90:
        risk, score = "Very High", 60
    elif wind_speed < 120:
        risk, score = "Extreme", 80
    else:
        risk, score = "Catastrophic", 100

    return {
        "risk_level": risk,
        "risk_score": score,
        "note": "Risk estimate, not predicted number of fatalities",
    }


# ---------------------------------------------------------------------------
# Death rate (estimated fatality rate, as a percentage of the exposed
# population in the affected area) — a literal rate, complementing the
# qualitative mortality risk score above.
# ---------------------------------------------------------------------------
def estimate_death_rate(wind_speed: float) -> dict:
    if wind_speed < 34:
        risk, rate_pct = "Low", 0.001
    elif wind_speed < 48:
        risk, rate_pct = "Moderate", 0.01
    elif wind_speed < 64:
        risk, rate_pct = "High", 0.05
    elif wind_speed < 90:
        risk, rate_pct = "Very High", 0.2
    elif wind_speed < 120:
        risk, rate_pct = "Extreme", 0.5
    else:
        risk, rate_pct = "Catastrophic", 1.0

    return {
        "risk_level": risk,
        "estimated_death_rate_percent": rate_pct,
        "note": (
            "Illustrative heuristic estimate of fatality rate among the exposed "
            "population, derived only from wind intensity. Actual outcomes depend "
            "heavily on evacuation, preparedness, and local vulnerability, and can "
            "be far lower. Not an actuarial or epidemiological prediction."
        ),
    }


# ---------------------------------------------------------------------------
# Combined overall risk score
# ---------------------------------------------------------------------------
def calculate_overall_risk(wind_speed: float, damage_score: float, mortality_score: float) -> dict:
    intensity_score = min(max((wind_speed / 120) * 100, 0), 100)
    overall_score = (
        intensity_score * 0.4
        + damage_score * 0.35
        + mortality_score * 0.25
    )

    if overall_score < 25:
        level = "Low"
    elif overall_score < 50:
        level = "Moderate"
    elif overall_score < 75:
        level = "High"
    else:
        level = "Critical"

    return {
        "overall_risk_score": round(overall_score, 2),
        "risk_level": level,
    }


# ---------------------------------------------------------------------------
# Affected area (geospatial)
# ---------------------------------------------------------------------------
def estimate_affected_area(wind_speed: float, lat: float | None = None, lon: float | None = None) -> dict:
    if wind_speed < 34:
        radius_km = 80.0
    elif wind_speed < 48:
        radius_km = 150.0
    elif wind_speed < 64:
        radius_km = 220.0
    elif wind_speed < 90:
        radius_km = 300.0
    elif wind_speed < 120:
        radius_km = 400.0
    else:
        radius_km = 500.0

    area_sq_km = math.pi * (radius_km ** 2)

    result: dict = {
        "impact_radius_km": round(radius_km, 2),
        "total_affected_area_sq_km": round(area_sq_km, 2),
    }

    if lat is not None and lon is not None:
        # 1 degree latitude ~ 111 km
        lat_delta = radius_km / 111.0
        cos_lat = math.cos(math.radians(lat))
        lon_delta = radius_km / (111.0 * cos_lat) if cos_lat != 0 else lat_delta

        result["eye_coordinates"] = {
            "latitude": round(lat, 4),
            "longitude": round(lon, 4),
        }
        result["affected_bounding_box"] = {
            "min_latitude": round(lat - lat_delta, 4),
            "max_latitude": round(lat + lat_delta, 4),
            "min_longitude": round(lon - lon_delta, 4),
            "max_longitude": round(lon + lon_delta, 4),
        }
    else:
        result["note"] = "Eye coordinates not provided. Provide 'latitude' and 'longitude' form fields to get bounding box coordinates."

    return result
