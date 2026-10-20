"""Transparent educational arithmetic, separate from the learned models."""

import math
import pandas as pd
from src.ml import predict

ACRES_PER_HECTARE = 2.47105
NUTRIENT_TARGETS = {
    "Rice": [40, 30, 30], "Wheat": [45, 35, 25], "Maize": [45, 35, 20],
    "Cotton": [50, 25, 25], "Sugarcane": [60, 35, 40], "Tomato": [35, 40, 45],
    "Potato": [40, 45, 45], "Groundnut": [20, 40, 30], "Pulses": [15, 40, 20],
    "Barley": [35, 25, 20],
}


def nonnegative(**values):
    for name, value in values.items():
        if not math.isfinite(float(value)) or float(value) < 0:
            raise ValueError(f"{name} must be a finite, non-negative number.")


def nutrient_comparison(crop, nitrogen, phosphorus, potassium):
    """Targets are illustrative dataset values, not measured nutrient requirements."""
    if crop not in NUTRIENT_TARGETS:
        raise ValueError("No reference targets for this crop.")
    nonnegative(nitrogen=nitrogen, phosphorus=phosphorus, potassium=potassium)
    current = [nitrogen, phosphorus, potassium]
    targets = NUTRIENT_TARGETS[crop]
    return pd.DataFrame({
        "Nutrient": ["Nitrogen", "Phosphorus", "Potassium"],
        "Input": current, "Illustrative target": targets,
        "Gap": [max(target - value, 0) for value, target in zip(current, targets)],
        "Status": ["Below reference" if value < target else "At/above reference"
                   for value, target in zip(current, targets)],
    })


def fertilizer_quantity(rate_kg_ha, area_ha, bag_kg=50):
    """A user-specified rate illustrates unit arithmetic; this does not prescribe a dose."""
    nonnegative(rate_kg_ha=rate_kg_ha, area_ha=area_ha, bag_kg=bag_kg)
    if area_ha == 0 or bag_kg == 0:
        raise ValueError("Area and bag size must be positive.")
    total = rate_kg_ha * area_ha
    return {"total_kg": total, "bags_equivalent": total / bag_kg, "whole_bags": math.ceil(total / bag_kg)}


def farm_economics(yield_tons_ha, area_ha, price_per_quintal, cost_per_ha):
    nonnegative(yield_tons_ha=yield_tons_ha, area_ha=area_ha,
                price_per_quintal=price_per_quintal, cost_per_ha=cost_per_ha)
    if area_ha == 0:
        raise ValueError("Area must be positive.")
    production_tons = yield_tons_ha * area_ha
    revenue = production_tons * 10 * price_per_quintal
    cost = cost_per_ha * area_ha
    profit = revenue - cost
    return {
        "production_tons": production_tons,
        "yield_quintals_acre": yield_tons_ha * 10 / ACRES_PER_HECTARE,
        "revenue": revenue, "cost": cost, "profit": profit,
        "margin_pct": profit / revenue * 100 if revenue else None,
    }


def rainfall_scenarios(values, bundle=None):
    rows = []
    for factor, label in [(0.75, "25% less rain"), (1.0, "Entered rainfall"), (1.25, "25% more rain")]:
        scenario = {**values, "Rainfall_mm": values["Rainfall_mm"] * factor}
        result = predict("yield", scenario, bundle)
        rows.append({
            "Scenario": label, "Rainfall (mm)": scenario["Rainfall_mm"],
            "Yield (tons/ha)": result["prediction"],
            "Outside training range": bool(result["warnings"]),
        })
    return pd.DataFrame(rows)
