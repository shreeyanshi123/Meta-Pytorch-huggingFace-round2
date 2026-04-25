"""
co2_calculator.py – CO2 savings estimator for green-code refactoring.

Converts CPU-time savings into energy consumption, then into CO2 equivalents
using real-world constants (global grid intensity, tree absorption, car
emissions).  No external dependencies required.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from environment.track_c import GreenScore

# ── Physical / environmental constants ────────────────────────────────────────
CARBON_INTENSITY_G_PER_KWH: float = 475.0   # gCO2 / kWh  (global average grid)
CPU_TDP_WATTS: float = 15.0                  # Watts per CPU core (typical server)

_TREE_KG_PER_YEAR: float = 21.0             # kg CO2 absorbed by one tree per year
_CAR_G_PER_KM: float = 120.0               # gCO2 emitted per km by average car
_MS_PER_HOUR: float = 3_600_000.0           # milliseconds in one hour
_G_PER_KG: float = 1_000.0
_KG_PER_TONNE: float = 1_000.0
_DAYS_PER_YEAR: float = 365.0


def estimate_co2_saved(
    cpu_time_saved_ms: float,
    runs_per_day: int = 10_000,
) -> dict[str, float]:
    """Estimate CO2 saved by reducing CPU execution time.

    Conversion chain:
        ms saved/run → hours saved/day → kWh saved/day → gCO2 saved/day

    Parameters
    ----------
    cpu_time_saved_ms:
        CPU time saved per invocation in milliseconds (can be negative if
        the refactored version is slower, in which case savings are zero).
    runs_per_day:
        Number of times the code is executed per day in production.

    Returns
    -------
    dict with keys:
        ``grams_per_day``    – CO2 saved per day (grams)
        ``kg_per_year``      – CO2 saved per year (kilograms)
        ``equivalent_trees`` – trees needed to absorb the same CO2/year
        ``equivalent_car_km``– car kilometres equivalent to the CO2 saved/year
    """
    saved_ms = max(0.0, cpu_time_saved_ms)
    hours_saved_per_day = (saved_ms * runs_per_day) / _MS_PER_HOUR
    kwh_saved_per_day = hours_saved_per_day * CPU_TDP_WATTS / 1_000.0
    grams_per_day = kwh_saved_per_day * CARBON_INTENSITY_G_PER_KWH
    kg_per_year = grams_per_day * _DAYS_PER_YEAR / _G_PER_KG
    equivalent_trees = kg_per_year / _TREE_KG_PER_YEAR
    equivalent_car_km = (kg_per_year * _G_PER_KG) / _CAR_G_PER_KM

    return {
        "grams_per_day": round(grams_per_day, 4),
        "kg_per_year": round(kg_per_year, 4),
        "equivalent_trees": round(equivalent_trees, 4),
        "equivalent_car_km": round(equivalent_car_km, 4),
    }


def generate_dashboard_data(
    green_score: "GreenScore",
    orig_files: dict[str, str],
    updated_files: dict[str, str],
) -> dict[str, Any]:
    """Build a full dashboard payload for the /dashboard/co2 endpoint.

    Parameters
    ----------
    green_score:
        A :class:`~environment.track_c.GreenScore` instance produced by
        :meth:`~environment.track_c.GreenCodeEvaluator.evaluate`.
    orig_files:
        Original source files (pre-refactoring).
    updated_files:
        Refactored source files.

    Returns
    -------
    A JSON-serialisable dict with all green metrics and CO2 estimates.
    """
    # ── Rough CPU-time saved estimate (proxy: score * 100 ms baseline) ────────
    # In production the caller should pass real measured values; here we derive
    # a conservative proxy from the cpu_improvement ratio.
    _baseline_cpu_ms: float = 100.0
    cpu_saved_ms = green_score.cpu_improvement * _baseline_cpu_ms

    co2_data = estimate_co2_saved(cpu_saved_ms)

    orig_lines = sum(len(c.splitlines()) for c in orig_files.values())
    updated_lines = sum(len(c.splitlines()) for c in updated_files.values())

    return {
        "summary": {
            "graphlet_score": green_score.graphlet_score,
            "cpu_improvement": green_score.cpu_improvement,
            "memory_improvement": green_score.memory_improvement,
            "total_green_score": green_score.total,
        },
        "co2_estimates": co2_data,
        "code_delta": {
            "original_lines": orig_lines,
            "updated_lines": updated_lines,
            "lines_reduced": orig_lines - updated_lines,
            "file_count_orig": len(orig_files),
            "file_count_updated": len(updated_files),
        },
        "constants_used": {
            "carbon_intensity_g_per_kwh": CARBON_INTENSITY_G_PER_KWH,
            "cpu_tdp_watts": CPU_TDP_WATTS,
            "tree_absorption_kg_per_year": _TREE_KG_PER_YEAR,
            "car_emissions_g_per_km": _CAR_G_PER_KM,
        },
    }
