import os
from pathlib import Path
import numpy as np
import pytest
import torch

from core.device_utils import get_optimal_device
from data.datasets.dataset_generator import generate_slm_tag_corpus, generate_grid_weather_thermal_timeseries
from gateway.ingestion.neural_slm_model import AetherisBrickSLM, BMSTokenizer
from core.controller.vectorized_env import VectorizedBuildingEnv
from core.controller.ppo_agent import PPOAgent, ActorCritic


def test_device_utils_safety():
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    dev = get_optimal_device(force_cpu=True, verbose=False)
    assert dev == "cpu"


def test_dataset_generation_slm():
    tags = generate_slm_tag_corpus(num_samples=50, seed=123)
    assert len(tags) >= 50
    first = tags[0]
    assert "raw_tag" in first
    assert "brick_class" in first
    assert "equipment_type" in first
    assert "point_role" in first
    assert "subsystem" in first
    assert "zone_id" in first


def test_dataset_generation_grid():
    df = generate_grid_weather_thermal_timeseries(num_days=2, seed=123)
    assert len(df) == 2 * 288
    assert "dynamic_lmp_price_usd_per_kwh" in df.columns
    assert "ambient_temp_c" in df.columns
    assert "solar_irradiance_wm2" in df.columns
    assert "price_forecast_2h" in df.columns


def test_neural_slm_model_forward():
    model = AetherisBrickSLM(vocab_size=500, d_model=64, n_layers=2, n_heads=2)
    sample_tags = ["AHU1_SAT_SP", "CHLR1_CHW_SUP_T", "Z01_RAT_TEMP", "VAV_Z02_DMPR"]
    preds = model.predict_tags(sample_tags, device="cpu")
    assert len(preds) == 4
    for p in preds:
        assert "brick_class" in p
        assert "equipment_type" in p
        assert "confidence" in p
        assert 0.0 <= p["confidence"] <= 1.0


def test_vectorized_building_env():
    env = VectorizedBuildingEnv(num_envs=8, num_zones=5, device="cpu")
    obs = env.get_observations()
    assert obs.shape == (8, 32)

    # Step environment
    dummy_actions = torch.zeros(8, 11)
    next_obs, rewards, dones, info = env.step(dummy_actions)
    assert next_obs.shape == (8, 32)
    assert rewards.shape == (8,)
    assert dones.shape == (8,)
    assert "total_hvac_kw" in info


def test_ppo_agent_with_checkpoint():
    agent = PPOAgent(state_dim=32, action_dim=11)
    state = {
        "zones": {f"zone_{i}": {"temp_c": 22.5, "mass_temp_c": 22.0, "occupancy": 10} for i in range(1, 6)},
        "ambient_temp_c": 30.0,
        "solar_irradiance_wm2": 600.0,
        "dynamic_lmp_price": 0.25,
        "timestamp_hour": 14.0,
        "power": {"total_hvac_kw": 45.0}
    }
    actions, meta = agent.select_action(state, price_forecast=[0.25]*24)
    assert "zone_setpoints" in actions
    assert "chiller_chw_setpoint" in actions
    assert "vav_damper_positions" in actions
    assert len(actions["zone_setpoints"]) == 5
    assert 20.0 <= actions["zone_setpoints"]["zone_1"] <= 24.0
    assert 4.0 <= actions["chiller_chw_setpoint"] <= 12.0
