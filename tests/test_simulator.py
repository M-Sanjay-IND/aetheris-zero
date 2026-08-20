import pytest
from core.simulator.comfort import calculate_pmv, calculate_ppd, is_ashrae55_compliant
from core.simulator.baseline_scheduler import BaselineScheduler
from core.simulator.building_etp import BuildingSimulator

def test_pmv_ashrae55_benchmark():
    pmv_neutral = calculate_pmv(temp=23.5, humidity=50.0, air_velocity=0.1, met=1.1, clo=0.7)
    assert -0.5 <= pmv_neutral <= 0.5
    assert is_ashrae55_compliant(pmv_neutral) is True

    pmv_hot = calculate_pmv(temp=32.0, humidity=70.0, air_velocity=0.1, met=1.2, clo=0.8)
    assert pmv_hot > 1.0
    assert is_ashrae55_compliant(pmv_hot) is False

    pmv_cold = calculate_pmv(temp=15.0, humidity=40.0, air_velocity=0.2, met=1.0, clo=0.4)
    assert pmv_cold < -1.0
    assert is_ashrae55_compliant(pmv_cold) is False

    ppd_neutral = calculate_ppd(pmv_neutral)
    assert 5.0 <= ppd_neutral <= 15.0

def test_baseline_scheduler():
    scheduler = BaselineScheduler()
    
    assert scheduler.is_occupied_hour(12.0) is True
    assert scheduler.is_occupied_hour(2.0) is False
    
    assert scheduler.get_setpoint(14.0) == 22.0
    assert scheduler.get_setpoint(22.0) == 26.0

    actions = scheduler.get_actions(10.0, ["zone_1", "zone_2"])
    assert actions["zone_setpoints"]["zone_1"] == 22.0
    assert actions["chiller_chw_setpoint"] == 6.5
    assert actions["vav_damper_positions"]["zone_1"] == 0.8

def test_building_simulator_init_and_reset():
    sim = BuildingSimulator()
    state = sim.reset()
    
    assert state["step"] == 0
    assert state["timestamp_hour"] == 0.0
    assert len(state["zones"]) == 5
    assert "zone_1" in state["zones"]
    assert "power" in state
    assert "metrics" in state

def test_building_simulator_step_physics():
    sim = BuildingSimulator()
    sim.reset()
    
    actions = {
        "zone_setpoints": {
            "zone_1": 21.0,
            "zone_2": 21.0,
            "zone_3": 21.0,
            "zone_4": 21.0,
            "zone_5": 21.0
        },
        "chiller_chw_setpoint": 6.0,
        "vav_damper_positions": {
            "zone_1": 0.8,
            "zone_2": 0.8,
            "zone_3": 0.8,
            "zone_4": 0.8,
            "zone_5": 0.8
        }
    }
    
    state, reward, done, info = sim.step(actions)
    
    assert state["step"] == 1
    assert state["power"]["total_hvac_kw"] > 0
    assert state["power"]["chiller_kw"] > 0
    assert state["power"]["fans_kw"] > 0
    assert isinstance(reward, float)
    assert isinstance(done, bool)
    assert "step_cost_actual" in info
    assert state["metrics"]["cumulative_cost_actual"] > 0

def test_building_simulator_multi_step_trajectory():
    sim = BuildingSimulator()
    sim.reset()
    
    for _ in range(12):
        actions = {
            "zone_setpoints": {f"zone_{i}": 22.0 for i in range(1, 6)},
            "chiller_chw_setpoint": 6.5,
            "vav_damper_positions": {f"zone_{i}": 0.7 for i in range(1, 6)}
        }
        state, reward, done, info = sim.step(actions)
    
    assert state["step"] == 12
    assert state["timestamp_hour"] == 1.0
    assert state["metrics"]["cumulative_energy_actual_kwh"] > 0

def test_custom_config_ingestion():
    custom_cfg = {
        "simulation": {
            "time_step_sec": 60,
            "total_hours": 12,
            "start_hour": 6.0
        },
        "zones": [
            {
                "zone_id": "z_alpha",
                "name": "Alpha Zone",
                "C_z": 20000.0,
                "C_m": 80000.0,
                "R_ext": 3.0,
                "R_m": 0.4,
                "initial_temp": 24.0,
                "initial_mass_temp": 23.5,
                "occupancy_max": 30,
                "floor_area_sqm": 400.0,
                "solar_factor": 0.3,
                "max_cooling_kw": 50.0
            }
        ],
        "adjacencies": [],
        "equipment": {
            "chiller": {"capacity_kw": 200.0, "cop_base": 4.0, "chw_temp_min": 4.0, "chw_temp_max": 12.0},
            "ahu": {"max_airflow_m3s": 15.0, "fan_power_max_kw": 20.0},
            "vav_boxes": {"min_damper_position": 0.1, "max_damper_position": 1.0}
        }
    }
    
    sim = BuildingSimulator(custom_cfg)
    state = sim.reset()
    
    assert sim.dt == 60.0
    assert sim.start_hour == 6.0
    assert len(state["zones"]) == 1
    assert "z_alpha" in state["zones"]
    assert state["zones"]["z_alpha"]["temp_c"] == 24.0
    
    step_state, _, _, _ = sim.step({
        "zone_setpoints": {"z_alpha": 21.5},
        "chiller_chw_setpoint": 5.5,
        "vav_damper_positions": {"z_alpha": 0.9}
    })
    assert step_state["step"] == 1
    assert step_state["zones"]["z_alpha"]["temp_c"] < 24.0
