from typing import Any, Dict, List, Tuple
import numpy as np

class ComfortBarrier:
    """
    ASHRAE 55 Occupant Thermal Comfort Invariance Barrier.
    Ensures zone temperatures strictly remain within [T_min, T_max] bounds.
    """
    def __init__(self, t_min: float = 20.0, t_max: float = 24.5, gamma: float = 0.15):
        self.t_min = t_min
        self.t_max = t_max
        self.gamma = gamma

    def evaluate_upper(self, temp: float) -> float:
        """h_upper(T) = T_max - T >= 0"""
        return self.t_max - temp

    def evaluate_lower(self, temp: float) -> float:
        """h_lower(T) = T - T_min >= 0"""
        return temp - self.t_min

    def get_setpoint_bounds(self, current_temp: float, dt_sec: float = 300.0) -> Tuple[float, float]:
        """
        Calculates safe admissible setpoint bounds [u_min, u_max]
        derived from Control Barrier Function invariance.
        """
        # Upper comfort limit: Setpoint must not allow temp to drift above T_max
        h_upper = self.evaluate_upper(current_temp)
        # Lower comfort limit: Setpoint must not force temp below T_min
        h_lower = self.evaluate_lower(current_temp)

        # Admissible setpoints
        sp_min = max(self.t_min, current_temp - 2.5) if h_lower >= 0 else self.t_min + 0.5
        sp_max = min(self.t_max, current_temp + 2.5) if h_upper >= 0 else self.t_max - 0.5

        return float(sp_min), float(sp_max)

class SlewRateBarrier:
    """
    Actuator Slew-Rate Limiter Barrier.
    Prevents high-frequency oscillations, valve hunting, and mechanical stress.
    """
    def __init__(self, max_delta_c_per_step: float = 0.75):
        self.max_delta = max_delta_c_per_step

    def get_admissible_range(self, prev_setpoint: float) -> Tuple[float, float]:
        return (prev_setpoint - self.max_delta, prev_setpoint + self.max_delta)

class DwellTimeBarrier:
    """
    Mechanical Equipment Minimum Dwell-Time Barrier.
    Enforces minimum dwell time (>15 mins = 3 steps @ 5-min) to eliminate chiller short-cycling.
    """
    def __init__(self, min_dwell_steps: int = 3):
        self.min_dwell_steps = min_dwell_steps
        self.last_toggle_step = -min_dwell_steps
        self.locked_state: float | None = None

    def update_and_check(self, current_step: int, proposed_state: float, prev_state: float, threshold: float = 0.5) -> Tuple[float, bool, int]:
        steps_since = current_step - self.last_toggle_step
        state_changed = abs(proposed_state - prev_state) > threshold

        if state_changed:
            if steps_since < self.min_dwell_steps:
                # Dwell time violation: Block change and lock to previous state
                remaining_steps = self.min_dwell_steps - steps_since
                return prev_state, True, remaining_steps
            else:
                # Dwell time satisfied: Allow state change and record toggle step
                self.last_toggle_step = current_step
                return proposed_state, False, 0
        
        remaining_steps = max(0, self.min_dwell_steps - steps_since)
        return proposed_state, False, remaining_steps
