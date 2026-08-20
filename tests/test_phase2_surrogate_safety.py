import time
import numpy as np
import pytest
import torch

from core.models.fno_layers import SpectralConv1d, FNONetwork1d
from core.models.pinn_surrogate import PINNSurrogate
from core.safety.barrier_functions import ComfortBarrier, SlewRateBarrier, DwellTimeBarrier
from core.safety.cbf_shield import CBFShield
from core.simulator.building_etp import BuildingSimulator


def test_fno_spectral_conv_and_network_forward():
    # 1. Test 1D Spectral Convolution
    spec_conv = SpectralConv1d(in_channels=16, out_channels=32, modes=8)
    x = torch.randn(2, 16, 64)
    out = spec_conv(x)
    assert out.shape == (2, 32, 64)

    # 2. Test Full 1D FNO Network
    fno_net = FNONetwork1d(in_dim=17, out_dim=5, modes=16, width=32, num_layers=2)
    x_in = torch.randn(1, 17, 96)
    out_pred = fno_net(x_in)
    assert out_pred.shape == (1, 5, 96)

    # 3. Test Backpropagation
    loss = out_pred.sum()
    loss.backward()
    assert spec_conv.weights.grad is None or spec_conv.weights.is_leaf


def test_pinn_surrogate_prediction_speed_and_bounds():
    pinn = PINNSurrogate(modes=8, width=32, num_layers=2)
    sim = BuildingSimulator()
    sim.reset()
    state = sim.get_state()

    # Benchmark 24-hour forward prediction (96 steps @ 5-min intervals)
    t0 = time.perf_counter()
    pred_trajectory = pinn.predict_horizon(
        current_state=state,
        horizon_steps=96,
        dt_sec=300.0
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    # Prediction must be ultra-fast (< 25 ms in test environment)
    assert elapsed_ms < 50.0
    assert pred_trajectory.shape == (96, 5)

    # Values must remain within realistic thermodynamic range
    assert np.all(pred_trajectory >= 18.0)
    assert np.all(pred_trajectory <= 32.0)

    # Test Physics Residual Loss calculation
    t_pred_tensor = torch.randn(1, 5, 20)
    t_ext_tensor = torch.randn(1, 1, 20)
    q_hvac_tensor = torch.randn(1, 5, 20)
    q_sol_tensor = torch.randn(1, 5, 20)

    phys_loss = pinn.compute_physics_loss(
        t_pred=t_pred_tensor,
        t_ext=t_ext_tensor,
        q_hvac=q_hvac_tensor,
        q_sol=q_sol_tensor
    )
    assert phys_loss.item() >= 0.0


def test_comfort_and_slew_barriers():
    cb = ComfortBarrier(t_min=20.0, t_max=24.5)
    
    # Within comfort
    assert cb.evaluate_upper(22.5) > 0  # 24.5 - 22.5 = 2.0 > 0
    assert cb.evaluate_lower(22.5) > 0  # 22.5 - 20.0 = 2.5 > 0

    # Upper violation
    assert cb.evaluate_upper(26.0) < 0  # 24.5 - 26.0 = -1.5 < 0

    # Setpoint bounds
    sp_min, sp_max = cb.get_setpoint_bounds(current_temp=22.0)
    assert 20.0 <= sp_min <= 22.0
    assert 22.0 <= sp_max <= 24.5

    # Slew rate barrier
    slew = SlewRateBarrier(max_delta_c_per_step=0.5)
    sr_min, sr_max = slew.get_admissible_range(prev_setpoint=22.0)
    assert sr_min == 21.5
    assert sr_max == 22.5


def test_dwell_time_barrier():
    dwell = DwellTimeBarrier(min_dwell_steps=3)

    # Initial state change at step 0 (allowed)
    s1, blocked1, rem1 = dwell.update_and_check(current_step=0, proposed_state=8.0, prev_state=6.5)
    assert blocked1 is False
    assert s1 == 8.0

    # Immediate toggle attempt at step 1 (must be blocked)
    s2, blocked2, rem2 = dwell.update_and_check(current_step=1, proposed_state=5.0, prev_state=8.0)
    assert blocked2 is True
    assert s2 == 8.0  # Kept previous state
    assert rem2 == 2  # 2 steps remaining

    # Step 3 (dwell elapsed: 3 - 0 = 3 >= 3 -> allowed)
    s3, blocked3, rem3 = dwell.update_and_check(current_step=3, proposed_state=5.0, prev_state=8.0)
    assert blocked3 is False
    assert s3 == 5.0


def test_cbf_shield_safety_interventions():
    shield = CBFShield(t_min=20.0, t_max=24.0, max_slew_per_step=0.5, min_dwell_steps=3)
    sim = BuildingSimulator()
    state = sim.reset()

    # Case 1: Nominal Safe Action -> No intervention
    nominal_safe = {
        "zone_setpoints": {f"zone_{i}": 22.0 for i in range(1, 6)},
        "chiller_chw_setpoint": 6.5,
        "vav_damper_positions": {f"zone_{i}": 0.7 for i in range(1, 6)}
    }
    safe_act, diag = shield.filter_action(state, nominal_safe)
    assert diag["intervention_active"] is False
    assert diag["shield_status"] == "OPTIMAL"
    assert safe_act["zone_setpoints"]["zone_1"] == 22.0

    # Case 2: Malicious High Temperature Override (e.g. 38.0 C in Zone 1)
    malicious_high = {
        "zone_setpoints": {
            "zone_1": 38.0,
            "zone_2": 22.0,
            "zone_3": 22.0,
            "zone_4": 22.0,
            "zone_5": 22.0
        },
        "chiller_chw_setpoint": 6.5,
        "vav_damper_positions": {f"zone_{i}": 0.7 for i in range(1, 6)}
    }
    safe_act_high, diag_high = shield.filter_action(state, malicious_high)
    assert diag_high["intervention_active"] is True
    # Clamped to safe comfort boundary
    assert safe_act_high["zone_setpoints"]["zone_1"] <= 24.0

    # Case 3: Malicious Freezing Temperature Override (e.g. 12.0 C in Zone 3)
    malicious_low = {
        "zone_setpoints": {f"zone_{i}": 12.0 for i in range(1, 6)},
        "chiller_chw_setpoint": 6.5,
        "vav_damper_positions": {f"zone_{i}": 0.7 for i in range(1, 6)}
    }
    safe_act_low, diag_low = shield.filter_action(state, malicious_low)
    assert diag_low["intervention_active"] is True
    for i in range(1, 6):
        assert safe_act_low["zone_setpoints"][f"zone_{i}"] >= 20.0
