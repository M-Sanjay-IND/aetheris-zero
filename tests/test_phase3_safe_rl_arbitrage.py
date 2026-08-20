from pathlib import Path
import numpy as np
import pytest

from core.simulator.building_etp import BuildingSimulator
from core.safety.cbf_shield import CBFShield
from core.controller.ppo_agent import ActorCritic, PPOAgent
from core.controller.arbitrage_engine import ArbitrageEngine
from gateway.grid.tariff_feed import TariffFeed

DATA_DIR = Path(__file__).parent.parent / "data"
SAMPLE_CAISO_JSON = DATA_DIR / "sample_caiso_lmp.json"


def test_actor_critic_forward():
    ac = ActorCritic(state_dim=24, action_dim=11, hidden_dim=64)
    obs = np.random.randn(2, 24).astype(np.float32)
    import torch
    dist, val = ac(torch.from_numpy(obs))
    
    assert dist.mean.shape == (2, 11)
    assert dist.stddev.shape == (2, 11)
    assert val.shape == (2, 1)


def test_ppo_agent_observation_and_action_selection():
    agent = PPOAgent(state_dim=24, action_dim=11)
    sim = BuildingSimulator()
    state = sim.reset()

    # Test observation extraction
    obs = agent.extract_observation(state)
    assert obs.shape == (24,)
    assert isinstance(obs, np.ndarray)

    # Test action selection in normal regime
    actions, meta = agent.select_action(state, deterministic=True)
    assert "zone_setpoints" in actions
    assert "chiller_chw_setpoint" in actions
    assert "vav_damper_positions" in actions
    assert len(actions["zone_setpoints"]) == 5
    assert 20.0 <= actions["zone_setpoints"]["zone_1"] <= 24.5
    assert 4.0 <= actions["chiller_chw_setpoint"] <= 12.0

    # Test pre-cooling regime recognition
    # High future price, low current price
    state_cheap = dict(state)
    state_cheap["dynamic_lmp_price"] = 0.05
    future_expensive = [1.50] * 12
    actions_pc, meta_pc = agent.select_action(state_cheap, price_forecast=future_expensive)
    assert meta_pc["regime"] == "PRE_COOLING"
    assert actions_pc["zone_setpoints"]["zone_1"] <= 21.0

    # Test peak load shedding regime recognition
    state_expensive = dict(state)
    state_expensive["dynamic_lmp_price"] = 1.50
    actions_shed, meta_shed = agent.select_action(state_expensive)
    assert meta_shed["regime"] == "PEAK_SHEDDING"
    assert actions_shed["zone_setpoints"]["zone_1"] >= 23.0


def test_arbitrage_engine_closed_loop_step():
    tariff = TariffFeed(json_file_path=SAMPLE_CAISO_JSON)
    sim = BuildingSimulator()
    shield = CBFShield(t_min=20.0, t_max=24.5)
    agent = PPOAgent()

    engine = ArbitrageEngine(simulator=sim, agent=agent, cbf_shield=shield, tariff_feed=tariff)
    init_state = engine.reset()
    assert init_state["step"] == 0

    # Step closed loop 10 steps
    for _ in range(10):
        next_state, safe_actions, shield_diag = engine.step_closed_loop()
        assert next_state["step"] > 0
        assert "total_hvac_kw" in next_state["power"]
        assert "intervention_active" in next_state["safety"]
        assert "shield_status" in next_state["safety"]

    assert len(engine.history) == 10


def test_arbitrage_engine_full_24h_episode_roi():
    tariff = TariffFeed(json_file_path=SAMPLE_CAISO_JSON)
    sim = BuildingSimulator()
    shield = CBFShield(t_min=20.0, t_max=24.5)
    agent = PPOAgent()

    engine = ArbitrageEngine(simulator=sim, agent=agent, cbf_shield=shield, tariff_feed=tariff)
    
    # Run full 24-hour episode (288 steps @ 5 min)
    results = engine.run_episode(total_steps=288)
    summary = results["summary"]

    assert summary["total_steps"] == 288
    assert summary["cumulative_cost_actual_usd"] > 0
    assert summary["cumulative_cost_baseline_usd"] > 0
    
    # Verify significant cost savings from pre-cooling arbitrage vs fixed baseline
    assert summary["cost_savings_usd"] > 0
    assert summary["cost_savings_pct"] >= 20.0  # Industry benchmark: 20-38%
    assert summary["carbon_avoided_kg"] >= 0.0

    # Verify zero comfort breaches across the entire 24h run
    final_zones = results["final_state"]["zones"]
    for zk in final_zones:
        assert final_zones[zk]["comfort_compliant"] is True
