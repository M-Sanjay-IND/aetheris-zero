from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from core.simulator.building_etp import BuildingSimulator
from core.controller.ppo_agent import PPOAgent
from core.safety.cbf_shield import CBFShield
from gateway.grid.tariff_feed import TariffFeed

class ArbitrageEngine:
    """
    Transactive Virtual Battery & Safe Reinforcement Learning Arbitrage Engine.
    Coordinates structural thermal pre-cooling and load-shedding routines
    under dynamic CAISO/ERCOT wholesale LMP pricing with formal CBF-QP safety guarantees.
    """
    def __init__(
        self,
        simulator: Optional[BuildingSimulator] = None,
        agent: Optional[PPOAgent] = None,
        cbf_shield: Optional[CBFShield] = None,
        tariff_feed: Optional[TariffFeed] = None
    ):
        self.simulator = simulator or BuildingSimulator()
        self.agent = agent or PPOAgent()
        self.cbf_shield = cbf_shield or CBFShield(t_min=20.0, t_max=24.2, max_slew_per_step=0.75, min_dwell_steps=3)
        self.tariff_feed = tariff_feed or TariffFeed()

        # Connect tariff feed to simulator
        self.simulator.tariff_feed = self.tariff_feed

        self.history: List[Dict[str, Any]] = []
        self.reset()

    def reset(self) -> dict:
        self.history.clear()
        self.cbf_shield.reset()
        initial_state = self.simulator.reset()
        return initial_state

    def step_closed_loop(self) -> Tuple[dict, dict, dict]:
        """
        Executes one full closed-loop step:
        1. Query State & Tariff Forecast
        2. Policy Action Selection (PPO)
        3. OSQP Safety Filter Projection (CBF-QP)
        4. Physics State Evolution & Baselining
        """
        current_state = self.simulator.get_state()
        curr_step = int(current_state.get("step", 0))
        
        # 1. 24h (96 steps @ 5-min) lookahead price vector
        price_forecast = self.tariff_feed.get_forecast_horizon(
            current_step=curr_step,
            horizon_steps=96,
            step_sec=int(self.simulator.dt)
        )

        # 2. PPO Policy Action Selection
        nominal_actions, meta = self.agent.select_action(
            state=current_state,
            price_forecast=price_forecast,
            deterministic=True
        )

        # Merge any user-specified zone setpoint overrides
        if hasattr(self.simulator, "zone_target_overrides") and self.simulator.zone_target_overrides:
            for zid, custom_sp in self.simulator.zone_target_overrides.items():
                nominal_actions["zone_setpoints"][zid] = float(custom_sp)

        # 3. CBF Safety Shield Projection (Guarantees Comfort & Anti-Short-Cycling)
        safe_actions, shield_diag = self.cbf_shield.filter_action(
            state=current_state,
            nominal_actions=nominal_actions,
            dt_sec=self.simulator.dt
        )


        # 4. Physics Simulator Step
        next_state, reward, done, info = self.simulator.step(safe_actions)

        # Merge safety diagnostics into state output for WebSocket streaming
        next_state["safety"]["intervention_active"] = shield_diag["intervention_active"]
        next_state["safety"]["shield_status"] = shield_diag["shield_status"]
        next_state["safety"]["dwell_time_remaining_sec"] = shield_diag["dwell_time_remaining_sec"]

        step_record = {
            "step": curr_step,
            "hour": current_state["timestamp_hour"],
            "price": current_state["dynamic_lmp_price"],
            "power_actual_kw": next_state["power"]["total_hvac_kw"],
            "power_baseline_kw": next_state["power"]["baseline_hvac_kw"],
            "demand_shaved_kw": next_state["power"]["demand_shaved_kw"],
            "shield_status": shield_diag["shield_status"],
            "regime": meta["regime"]
        }
        self.history.append(step_record)

        return next_state, safe_actions, shield_diag

    def run_episode(self, total_steps: int = 288) -> Dict[str, Any]:
        """
        Runs a full 24-hour simulation episode (288 steps @ 5-min intervals)
        and computes complete benchmark statistics comparing AETHERIS-Zero vs Baseline.
        """
        self.reset()
        episode_states = []

        for _ in range(total_steps):
            state, safe_actions, shield_diag = self.step_closed_loop()
            episode_states.append(state)

        final_state = episode_states[-1]
        metrics = final_state.get("metrics", {})

        actual_cost = float(metrics.get("cumulative_cost_actual", 0.0))
        baseline_cost = float(metrics.get("cumulative_cost_baseline", 0.0))
        cost_savings_usd = max(0.0, baseline_cost - actual_cost)
        cost_savings_pct = (cost_savings_usd / baseline_cost * 100.0) if baseline_cost > 0 else 0.0

        peak_reduction_pct = float(metrics.get("peak_demand_reduction_pct", 0.0))
        actual_energy = float(metrics.get("cumulative_energy_actual_kwh", 0.0))
        baseline_energy = float(metrics.get("cumulative_energy_baseline_kwh", 0.0))
        energy_savings_kwh = max(0.0, baseline_energy - actual_energy)

        # Carbon avoidance: 0.385 kg CO2 per kWh avoided in CAISO grid
        carbon_avoided_kg = energy_savings_kwh * 0.385

        summary = {
            "total_steps": total_steps,
            "cumulative_cost_actual_usd": round(actual_cost, 2),
            "cumulative_cost_baseline_usd": round(baseline_cost, 2),
            "cost_savings_usd": round(cost_savings_usd, 2),
            "cost_savings_pct": round(cost_savings_pct, 1),
            "peak_demand_reduction_pct": round(peak_reduction_pct, 1),
            "energy_saved_kwh": round(energy_savings_kwh, 2),
            "carbon_avoided_kg": round(carbon_avoided_kg, 2),
            "total_safety_interventions": self.cbf_shield.total_interventions
        }

        return {
            "summary": summary,
            "final_state": final_state,
            "history": self.history
        }
