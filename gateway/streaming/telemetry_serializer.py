"""
TelemetrySerializer: Normalizes and enriches building state telemetry for 3D Three.js
digital twin heatmaps, real-time analytics charts, and high-frequency WebSocket streams.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


def temp_to_hex_color(temp_c: float, min_temp: float = 19.0, max_temp: float = 26.0) -> str:
    """
    Interpolate temperature into high-contrast thermal RGB hex colors for Three.js:
    < 20.5°C: Cool Cyan/Blue (#06b6d4)
    20.5 - 23.0°C: Optimal Emerald (#10b981)
    23.0 - 24.5°C: Warm Amber (#f59e0b)
    > 24.5°C: Overheated Rose/Red (#ef4444)
    """
    t = max(min_temp, min(max_temp, temp_c))
    norm = (t - min_temp) / (max_temp - min_temp)

    if norm < 0.35:  # Cool
        r = int(6 + (norm / 0.35) * (16 - 6))
        g = int(182 + (norm / 0.35) * (185 - 182))
        b = int(212 + (norm / 0.35) * (129 - 212))
    elif norm < 0.65:  # Optimal
        frac = (norm - 0.35) / 0.30
        r = int(16 + frac * (245 - 16))
        g = int(185 + frac * (158 - 185))
        b = int(129 + frac * (11 - 129))
    else:  # Warm to Hot
        frac = (norm - 0.65) / 0.35
        r = int(245 + frac * (239 - 245))
        g = int(158 + frac * (68 - 158))
        b = int(11 + frac * (68 - 11))

    return f"#{r:02x}{g:02x}{b:02x}"


class ZoneTelemetryModel(BaseModel):
    name: str
    temp_c: float
    mass_temp_c: float
    setpoint_c: float
    pmv: float
    ppd: float
    comfort_compliant: bool
    occupancy: int
    cooling_load_kw: float
    thermal_color: str
    hex_color: Optional[str] = None
    heat_intensity: float


class PowerTelemetryModel(BaseModel):
    chiller_kw: float
    fans_kw: float
    supply_fan_kw: Optional[float] = None
    total_hvac_kw: float
    baseline_hvac_kw: float
    demand_shaved_kw: float



class SafetyTelemetryModel(BaseModel):
    intervention_active: bool
    shield_status: str
    dwell_time_remaining_sec: int
    solve_time_ms: float = 1.15
    active_constraints: List[str] = []
    t_min_bound: float = 20.0
    t_max_bound: float = 24.5
    max_slew_per_step: float = 0.75



class MetricsTelemetryModel(BaseModel):
    cumulative_cost_actual: float
    cumulative_cost_baseline: float
    cumulative_savings_usd: float
    cumulative_cost_actual_inr: float = 0.0
    cumulative_cost_baseline_inr: float = 0.0
    cumulative_savings_inr: float = 0.0
    cost_savings_pct: float = 0.0
    carbon_avoided_kg: float = 0.0
    comfort_compliance_pct: float = 100.0
    cumulative_energy_actual_kwh: float = 0.0
    cumulative_energy_baseline_kwh: float = 0.0
    peak_demand_reduction_pct: float


class TelemetryFrame(BaseModel):
    step: int
    timestamp_hour: float
    time_display: str
    ambient_temp_c: float
    solar_irradiance_wm2: float
    dynamic_lmp_price: float
    dynamic_lmp_price_mwh: float
    grid_dr_event_active: bool
    dr_event_id: Optional[str] = None
    zones: Dict[str, ZoneTelemetryModel]
    power: PowerTelemetryModel
    safety: SafetyTelemetryModel
    metrics: MetricsTelemetryModel


class TelemetrySerializer:
    """
    Serializes and normalizes raw simulation step dictionaries into verified
    TelemetryFrame payloads ready for WebSocket broadcasting and Three.js rendering.
    """

    def __init__(self):
        pass

    def serialize_state(
        self,
        raw_state: Dict[str, Any],
        dr_event_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Convert raw simulator state dictionary into an enriched, normalized telemetry dictionary."""
        hr = float(raw_state.get("timestamp_hour", 0.0))
        hours_int = int(hr) % 24
        minutes_int = int((hr % 1.0) * 60)
        time_display = f"{hours_int:02d}:{minutes_int:02d}"

        price_kwh = float(raw_state.get("dynamic_lmp_price", 0.15))
        price_mwh = round(price_kwh * 1000.0, 2)

        # Enrich zone metrics with 3D color codes and intensities
        zones_enriched: Dict[str, Dict[str, Any]] = {}
        raw_zones = raw_state.get("zones", {})
        compliant_zones_count = 0
        total_occupied_zones = 0

        for zid, zval in raw_zones.items():
            t_c = float(zval.get("temp_c", 22.0))
            color_hex = temp_to_hex_color(t_c)
            intensity = max(0.0, min(1.0, (t_c - 20.0) / 6.0))
            is_comp = bool(zval.get("comfort_compliant", True))
            occ = int(zval.get("occupancy", 0))
            if occ > 0:
                total_occupied_zones += 1
                if is_comp:
                    compliant_zones_count += 1

            zones_enriched[zid] = {
                "name": zval.get("name", zid),
                "temp_c": round(t_c, 2),
                "mass_temp_c": round(float(zval.get("mass_temp_c", t_c)), 2),
                "setpoint_c": round(float(zval.get("setpoint_c", 22.0)), 2),
                "pmv": round(float(zval.get("pmv", 0.0)), 3),
                "ppd": round(float(zval.get("ppd", 5.0)), 2),
                "comfort_compliant": is_comp,
                "occupancy": occ,
                "cooling_load_kw": round(float(zval.get("cooling_load_kw", 0.0)), 2),
                "thermal_color": color_hex,
                "hex_color": color_hex,
                "heat_intensity": round(intensity, 3),
            }

        power_data = raw_state.get("power", {})
        safety_data = raw_state.get("safety", {})
        metrics_data = raw_state.get("metrics", {})
        fans_kw_val = round(float(power_data.get("fans_kw", 0.0)), 2)

        cost_act_usd = float(metrics_data.get("cumulative_cost_actual", 0.0))
        cost_base_usd = float(metrics_data.get("cumulative_cost_baseline", 0.0))
        savings_usd = max(0.0, cost_base_usd - cost_act_usd)
        savings_pct = (savings_usd / cost_base_usd * 100.0) if cost_base_usd > 0 else 0.0

        energy_act_kwh = float(metrics_data.get("cumulative_energy_actual_kwh", 0.0))
        energy_base_kwh = float(metrics_data.get("cumulative_energy_baseline_kwh", 0.0))
        energy_saved_kwh = max(0.0, energy_base_kwh - energy_act_kwh)
        carbon_avoided_kg = round(energy_saved_kwh * 0.385, 2)

        compliance_rate = (compliant_zones_count / max(1, total_occupied_zones) * 100.0) if total_occupied_zones > 0 else 100.0

        frame = TelemetryFrame(
            step=int(raw_state.get("step", 0)),
            timestamp_hour=round(hr, 3),
            time_display=time_display,
            ambient_temp_c=round(float(raw_state.get("ambient_temp_c", 25.0)), 2),
            solar_irradiance_wm2=round(float(raw_state.get("solar_irradiance_wm2", 0.0)), 1),
            dynamic_lmp_price=round(price_kwh, 4),
            dynamic_lmp_price_mwh=price_mwh,
            grid_dr_event_active=bool(raw_state.get("grid_dr_event_active", False)),
            dr_event_id=dr_event_id,
            zones={k: ZoneTelemetryModel(**v) for k, v in zones_enriched.items()},
            power=PowerTelemetryModel(
                chiller_kw=round(float(power_data.get("chiller_kw", 0.0)), 2),
                fans_kw=fans_kw_val,
                supply_fan_kw=fans_kw_val,
                total_hvac_kw=round(float(power_data.get("total_hvac_kw", 0.0)), 2),
                baseline_hvac_kw=round(float(power_data.get("baseline_hvac_kw", 0.0)), 2),
                demand_shaved_kw=round(float(power_data.get("demand_shaved_kw", 0.0)), 2),
            ),

            safety=SafetyTelemetryModel(
                intervention_active=bool(safety_data.get("intervention_active", False)),
                shield_status=str(safety_data.get("shield_status", "OPTIMAL")),
                dwell_time_remaining_sec=int(safety_data.get("dwell_time_remaining_sec", 0)),
                solve_time_ms=round(float(safety_data.get("solve_time_ms", 1.15)), 2),
                active_constraints=list(safety_data.get("active_constraints", [])),
                t_min_bound=round(float(safety_data.get("t_min_bound", 20.0)), 2),
                t_max_bound=round(float(safety_data.get("t_max_bound", 24.5)), 2),
                max_slew_per_step=round(float(safety_data.get("max_slew_per_step", 0.75)), 2),
            ),

            metrics=MetricsTelemetryModel(
                cumulative_cost_actual=round(cost_act_usd, 2),
                cumulative_cost_baseline=round(cost_base_usd, 2),
                cumulative_savings_usd=round(savings_usd, 2),
                cumulative_cost_actual_inr=round(cost_act_usd * 83.0, 2),
                cumulative_cost_baseline_inr=round(cost_base_usd * 83.0, 2),
                cumulative_savings_inr=round(savings_usd * 83.0, 2),
                cost_savings_pct=round(savings_pct, 1),
                carbon_avoided_kg=carbon_avoided_kg,
                comfort_compliance_pct=round(compliance_rate, 1),
                cumulative_energy_actual_kwh=round(energy_act_kwh, 2),
                cumulative_energy_baseline_kwh=round(energy_base_kwh, 2),
                peak_demand_reduction_pct=round(float(metrics_data.get("peak_demand_reduction_pct", 0.0)), 1),
            ),
        )


        return frame.model_dump()

    def to_json(self, raw_state: Dict[str, Any], dr_event_id: Optional[str] = None) -> str:
        """Serialize state directly to JSON string."""
        serialized = self.serialize_state(raw_state, dr_event_id)
        return json.dumps(serialized)
