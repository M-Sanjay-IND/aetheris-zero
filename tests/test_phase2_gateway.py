"""
Unit and Integration Tests for Dev 2 Phase 2:
OpenADR 3.0 VEN, Telemetry Serializer, WebSocket Manager, and FastAPI Backend Gateway.
"""

import json
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from gateway.grid.openadr_ven import OpenADRVEN, EventStatus, SignalType
from gateway.streaming.telemetry_serializer import TelemetrySerializer, temp_to_hex_color
from gateway.streaming.ws_manager import ConnectionManager
from gateway.main import app, runtime


DATA_DIR = Path(__file__).parent.parent / "data"
RAW_BACNET_CSV = DATA_DIR / "raw_bacnet_dump.csv"


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_openadr_ven_lifecycle():
    ven = OpenADRVEN()
    assert ven.ven_id == "VEN_AETHERIS_001"

    # 1. Create DR Event (14:00 - 18:00)
    event = ven.create_event(
        start_hour=14.0,
        duration_hours=4.0,
        target_curtailment_kw=35.0,
        price_spike_usd=1.50,
        event_name="CAISO Stage 2 Curtailment",
    )
    assert event.id.startswith("evt_")
    assert event.end_hour == 18.0

    # 2. Check Event Status Transitions
    assert event.get_status_at(10.0) == EventStatus.FAR
    assert event.get_status_at(12.5) == EventStatus.NEAR      # In pre-cooling window
    assert event.get_status_at(15.0) == EventStatus.ACTIVE    # In active curtailment
    assert event.get_status_at(19.0) == EventStatus.COMPLETED # After event

    # 3. Check Active & Upcoming Querying
    assert ven.get_active_event(15.0) is not None
    assert ven.get_active_event(10.0) is None
    assert ven.get_upcoming_event(13.0) is not None

    # 4. Generate Compliance Report
    report = ven.generate_compliance_report(
        event_id=event.id,
        baseline_energy_kwh=320.0,
        actual_energy_kwh=195.0,
        peak_demand_shaved_kw=32.5,
        comfort_compliance_pct=99.2,
        settlement_rate_per_kwh=1.20,
    )
    assert report.event_id == event.id
    assert report.energy_curtailed_kwh == 125.0
    assert report.estimated_settlement_usd == 150.0  # 125 * 1.20 = 150.0
    assert report.curtailment_compliance_pct > 90.0

    reports = ven.list_reports()
    assert len(reports) == 1


def test_telemetry_serializer():
    serializer = TelemetrySerializer()

    # Test Color Hex Map
    c_cool = temp_to_hex_color(19.5)
    c_optimal = temp_to_hex_color(22.0)
    c_hot = temp_to_hex_color(26.5)
    assert c_cool.startswith("#")
    assert c_optimal.startswith("#")
    assert c_hot.startswith("#")

    # Sample Raw State
    raw_state = {
        "step": 42,
        "timestamp_hour": 14.5,
        "ambient_temp_c": 33.2,
        "solar_irradiance_wm2": 820.0,
        "dynamic_lmp_price": 1.50,
        "grid_dr_event_active": True,
        "zones": {
            "zone_1": {
                "name": "Core Zone",
                "temp_c": 22.2,
                "mass_temp_c": 21.9,
                "setpoint_c": 22.0,
                "pmv": 0.08,
                "ppd": 5.2,
                "comfort_compliant": True,
                "occupancy": 20,
                "cooling_load_kw": 18.5,
            }
        },
        "power": {
            "chiller_kw": 25.0,
            "fans_kw": 6.0,
            "total_hvac_kw": 36.0,
            "baseline_hvac_kw": 68.0,
            "demand_shaved_kw": 32.0,
        },
        "safety": {
            "intervention_active": False,
            "shield_status": "OPTIMAL",
            "dwell_time_remaining_sec": 0,
        },
        "metrics": {
            "cumulative_cost_actual": 85.20,
            "cumulative_cost_baseline": 142.50,
            "cumulative_savings_usd": 57.30,
            "cumulative_energy_actual_kwh": 210.0,
            "cumulative_energy_baseline_kwh": 340.0,
            "peak_demand_reduction_pct": 38.5,
        },
    }

    serialized = serializer.serialize_state(raw_state, dr_event_id="evt_test_123")
    assert serialized["step"] == 42
    assert serialized["time_display"] == "14:30"
    assert serialized["dynamic_lmp_price_mwh"] == 1500.0
    assert serialized["dr_event_id"] == "evt_test_123"
    assert "thermal_color" in serialized["zones"]["zone_1"]
    assert "heat_intensity" in serialized["zones"]["zone_1"]

    json_out = serializer.to_json(raw_state)
    assert isinstance(json_out, str)
    assert "Core Zone" in json_out


def test_fastapi_rest_health_and_ingestion(client):
    # Health
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json()["status"] == "healthy"

    # Parse Tags
    res = client.post("/api/v1/ingestion/parse-tags", json={"tags": ["AHU1_Z01_VAV_DAT_SP", "CHLR_CHW_STPT"]})
    assert res.status_code == 200
    data = res.json()
    assert data["count"] == 2
    assert data["points"][0]["brick_class"] == "Discharge_Air_Temperature_Setpoint"

    # Upload CSV
    csv_text = RAW_BACNET_CSV.read_text(encoding="utf-8")
    res = client.post("/api/v1/ingestion/upload-csv", content=csv_text, headers={"Content-Type": "text/plain"})
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "success"
    assert data["triples_generated"] > 50

    # Get Config & Turtle
    res_cfg = client.get("/api/v1/ingestion/config")
    assert res_cfg.status_code == 200
    assert "zones" in res_cfg.json()["config"]

    res_ttl = client.get("/api/v1/ingestion/graph-turtle")
    assert res_ttl.status_code == 200
    assert "@prefix brick:" in res_ttl.json()["turtle"]


def test_fastapi_rest_grid_and_openadr(client):
    # Current Tariff
    res_t = client.get("/api/v1/grid/tariff/current")
    assert res_t.status_code == 200
    assert "price_usd_per_kwh" in res_t.json()

    # Tariff Chart
    res_chart = client.get("/api/v1/grid/tariff/chart?resolution_minutes=15")
    assert res_chart.status_code == 200
    assert len(res_chart.json()["chart_data"]) == 96

    # Inject Spike
    res_sp = client.post("/api/v1/grid/tariff/inject-spike", json={"start_hour": 14.0, "duration_hours": 4.0, "spike_price": 2.20})
    assert res_sp.status_code == 200

    # OpenADR VEN & Events
    res_ven = client.get("/api/v1/grid/openadr/ven")
    assert res_ven.status_code == 200
    assert res_ven.json()["ven"]["ven_id"] == "VEN_AETHERIS_001"

    res_evts = client.get("/api/v1/grid/openadr/events")
    assert res_evts.status_code == 200
    assert len(res_evts.json()["events"]) >= 1

    res_new_evt = client.post("/api/v1/grid/openadr/events", json={
        "start_hour": 14.0,
        "duration_hours": 4.0,
        "target_curtailment_kw": 40.0,
        "price_spike_usd": 1.50,
        "event_name": "Test DR Dispatch",
    })
    assert res_new_evt.status_code == 200
    assert res_new_evt.json()["event"]["event_name"] == "Test DR Dispatch"

    res_rep = client.get("/api/v1/grid/openadr/reports")
    assert res_rep.status_code == 200


def test_fastapi_rest_simulation_and_controls(client):
    # Get State
    res_st = client.get("/api/v1/simulation/state")
    assert res_st.status_code == 200
    assert "zones" in res_st.json()["telemetry"]

    # Trigger DR Action
    res_dr = client.post("/api/v1/control/trigger-dr", json={"price_spike": 1.50, "start_hour": 14.0, "duration_hours": 4.0})
    assert res_dr.status_code == 200
    assert res_dr.json()["event"]["price_spike_usd"] == 1.50

    # Inject Fault Action & Verify CBF Safety Shield Active Intervention
    res_fault = client.post("/api/v1/control/inject-fault", json={"zone_id": "zone_1", "target_temp": 38.0})
    assert res_fault.status_code == 200
    telemetry = res_fault.json()["telemetry"]
    assert telemetry["safety"]["intervention_active"] is True
    assert telemetry["safety"]["shield_status"] in ["INTERVENED", "HARD_CLAMP"]

    # Toggle Shadow Mode
    res_shad = client.post("/api/v1/control/toggle-shadow", json={"enabled": True})
    assert res_shad.status_code == 200
    assert res_shad.json()["shadow_mode"] is True

    # Step Simulation
    res_step = client.post("/api/v1/control/step", json={"actions": None})
    assert res_step.status_code == 200
    assert res_step.json()["telemetry"]["step"] >= 1

    # PINN-FNO Forward Neural Surrogate Horizon Prediction
    res_pred = client.get("/api/v1/simulation/predict-horizon?horizon_steps=48")
    assert res_pred.status_code == 200
    assert res_pred.json()["status"] == "success"
    assert len(res_pred.json()["prediction_timeline"]) == 48

    # Reset Simulation
    res_reset = client.post("/api/v1/control/reset")
    assert res_reset.status_code == 200
    assert res_reset.json()["telemetry"]["step"] == 0


def test_websocket_stream_bidirectional(client):
    with client.websocket_connect("/ws/telemetry") as websocket:
        # Initial State
        init_data = websocket.receive_json()
        assert init_data["type"] == "INITIAL_STATE"
        assert "telemetry" in init_data
        assert len(init_data["telemetry"]["zones"]) == 5

        # Send Ping Command
        websocket.send_json({"action": "PING", "timestamp": 12345678})
        pong_data = websocket.receive_json()
        assert pong_data["status"] == "PONG"
        assert pong_data["timestamp"] == 12345678

        # Send Step Command
        websocket.send_json({"action": "STEP_SIMULATION"})
        step_data = websocket.receive_json()
        assert step_data["type"] == "TELEMETRY_UPDATE"
        assert step_data["telemetry"]["step"] >= 1

        # Send Trigger DR Command
        websocket.send_json({
            "action": "TRIGGER_OPENADR_EVENT",
            "params": {"start_hour": 14.0, "duration_hours": 4.0, "price_spike": 1.50}
        })
        dr_resp = websocket.receive_json()
        assert dr_resp["type"] == "TELEMETRY_UPDATE"

        # Send Malicious Setpoint Injection over WebSocket
        websocket.send_json({
            "action": "INJECT_MALICIOUS_SETPOINT",
            "params": {"zone_id": "zone_1", "target_temp": 40.0}
        })
        fault_resp = websocket.receive_json()
        assert fault_resp["type"] == "TELEMETRY_UPDATE"
        assert fault_resp["telemetry"]["safety"]["intervention_active"] is True

