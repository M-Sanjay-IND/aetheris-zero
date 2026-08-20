"""
Unit and Integration Tests for Dev 2 Phase 3:
WebSocket Bridge, Live Simulation Loop, PPO Safe-RL Arbitrage Integration, and 24h Benchmark Runner.
"""

import time
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from gateway.main import app, runtime


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_simulation_runtime_modes():
    runtime.initialize()
    assert runtime.arbitrage_engine is not None
    assert runtime.ppo_agent is not None

    # 1. RL Safe Arbitrage Mode
    runtime.controller_mode = "RL_SAFE_ARBITRAGE"
    st1 = runtime.step()
    assert st1["step"] == 1
    assert "power" in st1
    assert "safety" in st1

    # 2. Baseline Heuristic Mode
    runtime.controller_mode = "BASELINE_HEURISTIC"
    st2 = runtime.step()
    assert st2["step"] == 2

    # 3. Custom Override with CBF Protection
    st3 = runtime.step(actions={
        "zone_setpoints": {"zone_1": 39.0},
        "chiller_chw_setpoint": 6.5,
        "vav_damper_positions": {f"zone_{i}": 0.7 for i in range(1, 6)}
    })
    assert st3["safety"]["intervention_active"] is True
    assert st3["zones"]["zone_1"]["temp_c"] < 30.0


def test_fastapi_phase3_endpoints(client):
    # 1. Simulation Status
    res_st = client.get("/api/v1/simulation/status")
    assert res_st.status_code == 200
    data_st = res_st.json()
    assert "running" in data_st
    assert "controller_mode" in data_st

    # 2. Set Controller Mode
    res_m = client.post("/api/v1/control/set-mode", json={"mode": "RL_SAFE_ARBITRAGE"})
    assert res_m.status_code == 200
    assert res_m.json()["controller_mode"] == "RL_SAFE_ARBITRAGE"

    # 3. Run Fast-Forward Episode (48 steps = 4 hours)
    res_ep = client.post("/api/v1/simulation/run-episode", json={"total_steps": 48})
    assert res_ep.status_code == 200
    ep_data = res_ep.json()
    assert ep_data["status"] == "success"
    summary = ep_data["results"]["summary"]
    assert summary["total_steps"] == 48
    assert "cost_savings_usd" in summary

    # 4. Start & Stop Auto Loop
    res_start = client.post("/api/v1/simulation/start")
    assert res_start.status_code == 200
    assert res_start.json()["running"] is True

    time.sleep(0.3)

    res_stop = client.post("/api/v1/simulation/stop")
    assert res_stop.status_code == 200
    assert res_stop.json()["running"] is False


def test_websocket_phase3_commands(client):
    with client.websocket_connect("/ws/telemetry") as websocket:
        init_data = websocket.receive_json()
        assert init_data["type"] == "INITIAL_STATE"

        # Set Mode over WebSocket
        websocket.send_json({
            "action": "SET_CONTROLLER_MODE",
            "params": {"mode": "RL_SAFE_ARBITRAGE"}
        })

        # Run Episode over WebSocket
        websocket.send_json({
            "action": "RUN_EPISODE",
            "params": {"total_steps": 24}
        })
        ep_msg = websocket.receive_json()
        assert ep_msg["type"] in ["EPISODE_SUMMARY", "TELEMETRY_UPDATE"]

        # Step Simulation
        websocket.send_json({"action": "STEP_SIMULATION"})
        step_msg = websocket.receive_json()
        assert step_msg["type"] == "TELEMETRY_UPDATE"
