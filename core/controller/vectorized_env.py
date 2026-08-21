import os
from typing import Dict, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
from core.device_utils import get_optimal_device


class VectorizedBuildingEnv:
    """
    High-Throughput Vectorized Multi-Zone 2R2C Building Environment in pure PyTorch tensors.
    Simulates N parallel buildings simultaneously on CPU/GPU for ultra-fast Safe-RL training.
    """
    def __init__(
        self,
        num_envs: int = 64,
        num_zones: int = 5,
        dt_sec: float = 300.0,
        device: Optional[str] = None,
    ):
        self.num_envs = num_envs
        self.num_zones = num_zones
        self.dt_sec = dt_sec
        self.device = device or get_optimal_device(verbose=False)

        # Thermal parameters per zone (C_z: kJ/K, C_m: kJ/K, R_ext: K/kW, R_m: K/kW)
        self.cz = torch.tensor([15000.0, 12000.0, 12000.0, 11000.0, 11000.0], device=self.device).repeat(num_envs, 1)
        self.cm = torch.tensor([80000.0, 60000.0, 60000.0, 55000.0, 55000.0], device=self.device).repeat(num_envs, 1)
        self.rext = torch.tensor([2.5, 3.0, 3.0, 3.2, 3.2], device=self.device).repeat(num_envs, 1)
        self.rm = torch.tensor([0.8, 1.0, 1.0, 1.1, 1.1], device=self.device).repeat(num_envs, 1)
        self.solar_factor = torch.tensor([0.1, 0.4, 0.4, 0.45, 0.45], device=self.device).repeat(num_envs, 1)
        self.floor_area = torch.tensor([200.0, 120.0, 120.0, 100.0, 100.0], device=self.device).repeat(num_envs, 1)

        # State tensors: [num_envs, num_zones]
        self.t_zone = torch.zeros(num_envs, num_zones, device=self.device)
        self.t_mass = torch.zeros(num_envs, num_zones, device=self.device)
        self.step_counts = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self.hour = torch.zeros(num_envs, device=self.device)

        # Weather and Price buffers
        self.ambient_temp = torch.zeros(num_envs, device=self.device)
        self.solar_ghi = torch.zeros(num_envs, device=self.device)
        self.dynamic_price = torch.zeros(num_envs, device=self.device)
        self.p_2h = torch.zeros(num_envs, device=self.device)
        self.p_4h = torch.zeros(num_envs, device=self.device)
        self.p_6h = torch.zeros(num_envs, device=self.device)
        self.occupancy = torch.zeros(num_envs, num_zones, device=self.device)

        self.reset()

    def reset(self, env_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        if env_ids is None:
            # Reset all
            self.t_zone.uniform_(21.5, 23.0)
            self.t_mass.uniform_(21.5, 23.0)
            self.step_counts.zero_()
            self.hour.zero_()
            self._update_weather_and_grid(step_offset=0)
            return self.get_observations()
        else:
            # Partial reset
            self.t_zone[env_ids] = torch.empty((len(env_ids), self.num_zones), device=self.device).uniform_(21.5, 23.0)
            self.t_mass[env_ids] = torch.empty((len(env_ids), self.num_zones), device=self.device).uniform_(21.5, 23.0)
            self.step_counts[env_ids] = 0
            self.hour[env_ids] = 0.0
            return self.get_observations()

    def _update_weather_and_grid(self, step_offset: int = 0):
        # Diurnal weather curve simulation
        hr = (self.hour + step_offset * (self.dt_sec / 3600.0)) % 24.0

        # Ambient temperature (Peak at 15:00)
        self.ambient_temp = 22.0 + 8.0 * torch.sin(2.0 * np.pi * (hr - 9.0) / 24.0)

        # Solar GHI (Peak at 12:00)
        sol = torch.clamp(900.0 * torch.sin(np.pi * (hr - 6.0) / 12.0), min=0.0)
        sol = torch.where((hr >= 6.0) & (hr <= 18.0), sol, torch.zeros_like(sol))
        self.solar_ghi = sol

        # Dynamic CAISO LMP Tariff: base $0.15, duck curve drop at noon, evening peak $0.85
        base_price = 0.15
        noon_drop = torch.where((hr >= 10.0) & (hr <= 14.0), -0.06 * (self.solar_ghi / 900.0), torch.zeros_like(hr))
        eve_peak = torch.where((hr >= 17.0) & (hr <= 21.0), torch.full_like(hr, 0.70), torch.zeros_like(hr))
        self.dynamic_price = torch.clamp(base_price + noon_drop + eve_peak, min=0.02)

        self.p_2h = self.dynamic_price + 0.1 * torch.sin(2.0 * np.pi * (hr + 2.0) / 24.0)
        self.p_4h = self.dynamic_price + 0.15 * torch.sin(2.0 * np.pi * (hr + 4.0) / 24.0)
        self.p_6h = self.dynamic_price + 0.2 * torch.sin(2.0 * np.pi * (hr + 6.0) / 24.0)

        # Occupancy (0 to 20 people)
        occ_val = torch.clamp(20.0 * torch.sin(np.pi * (hr - 7.5) / 11.0), min=0.0)
        occ_val = torch.where((hr >= 7.5) & (hr <= 18.5), occ_val, torch.zeros_like(hr))
        self.occupancy = occ_val.unsqueeze(-1).repeat(1, self.num_zones)

    def get_observations(self) -> torch.Tensor:
        """
        Constructs normalized 32-dim state vector for all N environments.
        """
        tz_norm = (self.t_zone - 22.0) / 5.0
        tm_norm = (self.t_mass - 22.0) / 5.0
        text_norm = (self.ambient_temp - 28.0).unsqueeze(-1) / 15.0
        sol_norm = (self.solar_ghi / 1000.0).unsqueeze(-1)
        price_norm = self.dynamic_price.unsqueeze(-1)
        p2h_norm = self.p_2h.unsqueeze(-1)
        p4h_norm = self.p_4h.unsqueeze(-1)
        p6h_norm = self.p_6h.unsqueeze(-1)
        occ_norm = self.occupancy / 25.0

        sin_hr = torch.sin(2.0 * np.pi * self.hour / 24.0).unsqueeze(-1)
        cos_hr = torch.cos(2.0 * np.pi * self.hour / 24.0).unsqueeze(-1)
        dummy_power = torch.full((self.num_envs, 1), 0.4, device=self.device)
        dummy_diff = (self.p_2h - self.dynamic_price).unsqueeze(-1)
        day_enc = torch.zeros(self.num_envs, 7, device=self.device)
        day_enc[:, 2] = 1.0  # Wednesday default

        obs = torch.cat([
            tz_norm,        # 5
            tm_norm,        # 5
            text_norm,      # 1
            sol_norm,       # 1
            price_norm,     # 1
            p2h_norm,       # 1
            p4h_norm,       # 1
            p6h_norm,       # 1
            occ_norm,       # 5
            dummy_power,    # 1
            dummy_diff,     # 1
            sin_hr,         # 1
            cos_hr,         # 1
            day_enc         # 7
        ], dim=-1)         # Total = 32 dimensions

        return obs

    def step(self, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        """
        Actions shape: [num_envs, 11]
        [0:5]   -> Zone target setpoints (20.0 to 24.0 °C)
        [5]     -> Chiller CHW setpoint (4.0 to 12.0 °C)
        [6:11]  -> VAV damper positions (0.20 to 1.00)
        """
        # Map actions
        sp_zones = 20.0 + 4.0 * torch.sigmoid(actions[:, :5])
        chw_sp = 4.0 + 8.0 * torch.sigmoid(actions[:, 5:6])
        dampers = 0.2 + 0.8 * torch.sigmoid(actions[:, 6:11])

        # Thermal heat transfer physics
        # Q_ext = (T_ext - T_z) / R_ext
        q_ext = (self.ambient_temp.unsqueeze(-1) - self.t_zone) / self.rext
        # Q_mass = (T_mass - T_z) / R_m
        q_mass = (self.t_mass - self.t_zone) / self.rm
        # Q_sol = solar_factor * floor_area * solar_ghi
        q_sol = (self.solar_factor * self.floor_area * self.solar_ghi.unsqueeze(-1)) / 1000.0  # kW
        # Q_int = occupancy * 0.12 kW/person
        q_int = self.occupancy * 0.12

        # Cooling capacity delivered by VAV
        # Q_hvac = m_dot * c_p * (T_z - T_sup)
        supply_air_temp = torch.clamp(chw_sp + 6.0, min=12.0, max=18.0)
        air_flow = dampers * 1.5  # kg/s
        cooling_kw = torch.clamp(air_flow * 1.006 * (self.t_zone - supply_air_temp), min=0.0)

        # dT_z/dt = (Q_ext + Q_mass + Q_sol + Q_int - Q_hvac) / C_z
        dt_hours = self.dt_sec / 3600.0
        dt_z = ((q_ext + q_mass + q_sol + q_int - cooling_kw) / self.cz) * self.dt_sec
        # dT_mass/dt = (T_z - T_mass) / (R_m * C_m)
        dt_m = (((self.t_zone - self.t_mass) / self.rm) / self.cm) * self.dt_sec

        self.t_zone = self.t_zone + dt_z
        self.t_mass = self.t_mass + dt_m

        # Electrical power consumption
        cop_chiller = torch.clamp(6.0 - 0.15 * (self.ambient_temp - chw_sp.squeeze(-1)), min=2.5, max=6.5)
        total_cooling_kw = cooling_kw.sum(dim=-1)
        chiller_elec_kw = total_cooling_kw / cop_chiller
        fan_elec_kw = (dampers ** 2.5).sum(dim=-1) * 3.0
        total_hvac_kw = chiller_elec_kw + fan_elec_kw + 2.0  # + baseline pumps

        # Economic Cost ($ per 5-min step)
        energy_kwh = total_hvac_kw * dt_hours
        step_cost_usd = energy_kwh * self.dynamic_price

        # Comfort Penalty (ASHRAE 55 band: 20.0 to 24.0 °C)
        under_temp = torch.clamp(20.0 - self.t_zone, min=0.0)
        over_temp = torch.clamp(self.t_zone - 24.0, min=0.0)
        comfort_penalty = (under_temp ** 2 + over_temp ** 2).sum(dim=-1) * 5.0

        # Arbitrage pre-cooling incentive: reward chilling when electricity is cheap and upcoming is high
        precool_cond = (self.dynamic_price < 0.15) & (self.p_4h > 0.40) & (self.t_zone.mean(dim=-1) < 21.5)
        precool_incentive = torch.where(
            precool_cond,
            torch.full_like(step_cost_usd, 0.25),
            torch.zeros_like(step_cost_usd)
        )

        # Composite Reward
        rewards = -(step_cost_usd * 2.0) - comfort_penalty + precool_incentive

        # Step progression
        self.step_counts += 1
        self.hour = (self.hour + dt_hours) % 24.0
        self._update_weather_and_grid(step_offset=0)

        # Episode termination at 288 steps (24 hours)
        dones = self.step_counts >= 288
        if dones.any():
            env_reset_ids = torch.where(dones)[0]
            self.reset(env_reset_ids)

        obs = self.get_observations()
        info = {
            "total_hvac_kw": total_hvac_kw.detach().cpu().numpy(),
            "step_cost_usd": step_cost_usd.detach().cpu().numpy(),
            "comfort_penalty": comfort_penalty.detach().cpu().numpy(),
        }

        return obs, rewards, dones, info
