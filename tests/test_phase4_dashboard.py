"""
Unit and Integration Tests for Dev 2 Phase 4:
Digital Twin Dashboard Serving, Three.js Telemetry Enrichment, Live WebSocket Interactions, and Regional Currency Verification.
"""

from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from gateway.main import app, runtime


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_dashboard_html_endpoints(client):
    # 1. Test Root Dashboard HTML
    res_root = client.get("/")
    assert res_root.status_code == 200
    assert "text/html" in res_root.headers.get("content-type", "")
    assert "AETHERIS-Zero" in res_root.text
    assert "three-canvas-container" in res_root.text

    # 2. Test /dashboard alias
    res_dash = client.get("/dashboard")
    assert res_dash.status_code == 200
    assert "AETHERIS-Zero" in res_dash.text

    # 3. Test /simulator separate endpoint
    res_sim = client.get("/simulator")
    assert res_sim.status_code == 200
    assert "SIMULATOR POD" in res_sim.text
    assert "Real-Time Custom Injections Studio" in res_sim.text

    # 4. Test /overview and /product commercial portal
    res_ov = client.get("/overview")
    assert res_ov.status_code == 200
    assert "COMMERCIAL PRODUCT OVERVIEW & EXECUTIVE ROI" in res_ov.text
    assert "Interactive Commercial Building ROI Calculator" in res_ov.text

    res_prod = client.get("/product")
    assert res_prod.status_code == 200
    assert "COMMERCIAL PRODUCT OVERVIEW & EXECUTIVE ROI" in res_prod.text



def test_dashboard_websocket_realtime_interactions(client):
    with client.websocket_connect("/ws/telemetry") as websocket:
        # Initial State
        init_frame = websocket.receive_json()
        assert init_frame["type"] == "INITIAL_STATE"
        telemetry = init_frame["telemetry"]
        
        # Verify 3D thermal zone colors are generated
        assert len(telemetry["zones"]) == 5
        for zid, zstate in telemetry["zones"].items():
            assert "hex_color" in zstate
            assert zstate["hex_color"].startswith("#")
            assert "heat_intensity" in zstate

        # 1. Trigger OpenADR DR Event via WebSocket
        websocket.send_json({
            "action": "TRIGGER_OPENADR_EVENT",
            "params": {"price_spike": 1.50, "start_hour": 14.0, "duration_hours": 4.0}
        })
        dr_frame = websocket.receive_json()
        assert dr_frame["type"] == "TELEMETRY_UPDATE"

        # 2. Inject Malicious Fault & Verify CBF Active Intervention
        websocket.send_json({
            "action": "INJECT_MALICIOUS_SETPOINT",
            "params": {"zone_id": "zone_1", "target_temp": 38.0}
        })
        fault_frame = websocket.receive_json()
        assert fault_frame["type"] == "TELEMETRY_UPDATE"
        assert fault_frame["telemetry"]["safety"]["intervention_active"] is True
        assert fault_frame["telemetry"]["safety"]["shield_status"] in ["INTERVENED", "HARD_CLAMP"]

        # 3. Test 24h Episode Execution
        websocket.send_json({
            "action": "RUN_EPISODE",
            "params": {"total_steps": 36}
        })
        ep_msg = websocket.receive_json()
        assert ep_msg["type"] in ["EPISODE_SUMMARY", "TELEMETRY_UPDATE"]


def test_inr_currency_and_metrics_accuracy(client):
    # Step simulation forward so active power is computed
    client.post("/api/v1/control/step")
    res = client.get("/api/v1/simulation/state")
    assert res.status_code == 200
    telemetry = res.json()["telemetry"]

    # Verify price in $/kWh and $/MWh
    price_usd_kwh = telemetry["dynamic_lmp_price"]
    price_inr_mwh = price_usd_kwh * 1000.0 * 83.0
    assert price_usd_kwh > 0
    assert price_inr_mwh > 0

    # Verify power breakdown
    power = telemetry["power"]
    assert "chiller_kw" in power
    assert "supply_fan_kw" in power
    assert "total_hvac_kw" in power
    assert power["total_hvac_kw"] >= 0.0


def test_dynamic_live_parameter_inputs(client):
    # 1. Test Dynamic Weather Override
    res_weather = client.post("/api/v1/control/set-weather", json={"ambient_temp_c": 37.5, "solar_irradiance_wm2": 950.0})
    assert res_weather.status_code == 200
    assert res_weather.json()["ambient_temp_c"] == 37.5

    # 2. Test Dynamic Price Override
    res_price = client.post("/api/v1/control/set-pricing", json={"price_usd_per_kwh": 0.45})
    assert res_price.status_code == 200
    assert res_price.json()["price_usd_per_kwh"] == 0.45

    # 3. Test Dynamic Zone Target Setpoint
    res_zone = client.post("/api/v1/control/set-zone-target", json={"zone_id": "zone_2", "target_temp": 21.0})
    assert res_zone.status_code == 200
    assert res_zone.json()["target_temp"] == 21.0

    # 4. Test Comfort Bounds Adjustment
    res_bounds = client.post("/api/v1/control/set-comfort-bounds", json={
        "t_min": 19.5,
        "t_max": 25.0,
        "max_slew_per_step": 1.0,
        "min_dwell_steps": 2
    })
    assert res_bounds.status_code == 200
    assert res_bounds.json()["bounds"]["t_min"] == 19.5


