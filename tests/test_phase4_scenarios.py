import pytest
from core.scenarios.fault_injection import (
    run_arbitrage_scenario,
    inject_malicious_setpoint,
    trigger_chiller_short_cycle,
    generate_comparative_metrics
)

def test_scenario1_dynamic_arbitrage():
    result = run_arbitrage_scenario(total_steps=96)  # 8-hour test run
    
    assert result["scenario_name"] == "Dynamic_CAISO_Tariff_Arbitrage"
    assert "summary" in result
    assert "time_series" in result
    assert "metrics" in result

    summary = result["summary"]
    assert summary["cost_savings_usd"] >= 0.0
    assert summary["cumulative_cost_actual_usd"] > 0.0

    ts = result["time_series"]
    assert len(ts["timestamps_hour"]) == 96
    assert len(ts["prices_usd_per_kwh"]) == 96
    assert len(ts["power_aetheris_kw"]) == 96
    assert len(ts["power_baseline_kw"]) == 96


def test_scenario2_malicious_setpoint_override():
    # 1. Hot Override Attack (38.0 C)
    res_hot = inject_malicious_setpoint(zone_id="zone_1", malicious_temp=38.0)
    assert res_hot["intervention_active"] is True
    assert res_hot["injected_temperature_c"] == 38.0
    assert res_hot["shielded_safe_temperature_c"] <= 24.5
    assert res_hot["verdict"] == "ATTACK_INTERCEPTED_AND_BOUNDED"
    assert "zone_1_comfort_upper_barrier" in res_hot["active_constraints"]

    # 2. Cold Override Attack (10.0 C)
    res_cold = inject_malicious_setpoint(zone_id="zone_3", malicious_temp=10.0)
    assert res_cold["intervention_active"] is True
    assert res_cold["injected_temperature_c"] == 10.0
    assert res_cold["shielded_safe_temperature_c"] >= 20.0
    assert res_cold["verdict"] == "ATTACK_INTERCEPTED_AND_BOUNDED"


def test_scenario3_chiller_short_cycling_attack():
    res = trigger_chiller_short_cycle(toggle_steps=6, min_dwell_steps=3)
    
    assert res["scenario_name"] == "Compressor_Short_Cycling_Attack"
    assert res["total_toggle_attempts"] == 6
    assert res["blocked_toggle_count"] > 0
    assert res["dwell_enforcement_rate_pct"] > 0.0
    assert res["verdict"] == "SHORT_CYCLING_ELIMINATED"

    # Verify toggle log sequence
    log = res["toggle_log"]
    assert len(log) == 6
    # Step 0 allowed (initial set), Step 1 blocked (dwell in effect)
    assert log[0]["dwell_time_blocked"] is False
    assert log[1]["dwell_time_blocked"] is True


def test_generate_comparative_metrics_calculation():
    metrics = generate_comparative_metrics(
        baseline_cost=200.0,
        aetheris_cost=130.0,
        peak_baseline_kw=100.0,
        peak_aetheris_kw=65.0,
        energy_saved_kwh=150.0,
        total_interventions=5
    )

    assert metrics["baseline_cost_usd"] == 200.0
    assert metrics["aetheris_cost_usd"] == 130.0
    assert metrics["cost_savings_usd"] == 70.0
    assert metrics["cost_savings_pct"] == 35.0
    assert metrics["peak_demand_shaved_kw"] == 35.0
    assert metrics["peak_demand_shaved_pct"] == 35.0
    assert metrics["energy_saved_kwh"] == 150.0
    assert metrics["carbon_avoided_kg"] == round(150.0 * 0.385, 2)
    assert metrics["comfort_compliance_rate_pct"] == 100.0
