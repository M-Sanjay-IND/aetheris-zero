import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

class ActorCritic(nn.Module):
    def __init__(self, state_dim: int = 24, action_dim: int = 11, hidden_dim: int = 128):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Shared feature representation
        self.trunk = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )

        # Actor head (Mean output)
        self.actor_mu = nn.Linear(hidden_dim, action_dim)
        self.actor_log_std = nn.Parameter(torch.zeros(action_dim) - 0.5)

        # Critic head (Value function)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1)
        )

    def forward(self, state: torch.Tensor) -> tuple[Normal, torch.Tensor]:
        features = self.trunk(state)
        mu = self.actor_mu(features)
        std = torch.exp(torch.clamp(self.actor_log_std, -2.0, 1.0))
        dist = Normal(mu, std)
        value = self.critic(features)
        return dist, value

class PPOAgent:
    """
    Continuous Proximal Policy Optimization (PPO) Safe-RL Agent.
    Optimizes multi-zone HVAC setpoints and chiller supply temperatures against dynamic LMP tariffs.
    """
    def __init__(
        self,
        state_dim: int = 24,
        action_dim: int = 11,
        lr: float = 3e-4,
        gamma: float = 0.99,
        clip_ratio: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.clip_ratio = clip_ratio
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef

        self.ac = ActorCritic(state_dim, action_dim)
        self.optimizer = torch.optim.Adam(self.ac.parameters(), lr=lr)

    def extract_observation(
        self,
        state: dict,
        price_forecast: list[float] | None = None
    ) -> np.ndarray:
        """
        Converts simulation state dictionary and forward price forecast into a normalized 24-dim observation vector.
        """
        zones = state.get("zones", {})
        zone_keys = [f"zone_{i}" for i in range(1, 6)]

        # 1. Zone temperatures (normalized around 22.0 C)
        tz = [(float(zones.get(zk, {}).get("temp_c", 22.0)) - 22.0) / 5.0 for zk in zone_keys]
        
        # 2. Zone thermal mass temperatures (normalized)
        tm = [(float(zones.get(zk, {}).get("mass_temp_c", 22.0)) - 22.0) / 5.0 for zk in zone_keys]
        
        # 3. Ambient temperature (normalized around 28.0 C)
        t_ext = (float(state.get("ambient_temp_c", 28.0)) - 28.0) / 15.0
        
        # 4. Solar irradiance (normalized to [0, 1])
        sol = float(state.get("solar_irradiance_wm2", 0.0)) / 1000.0
        
        # 5. Current price ($/kWh normalized)
        curr_price = float(state.get("dynamic_lmp_price", 0.15))
        
        # 6. Forward price lookahead (average next 2h, next 4h, next 6h)
        if price_forecast is not None and len(price_forecast) >= 12:
            p_2h = float(np.mean(price_forecast[:6]))
            p_4h = float(np.mean(price_forecast[:12]))
            p_6h = float(np.mean(price_forecast[:18])) if len(price_forecast) >= 18 else p_4h
        else:
            p_2h = curr_price
            p_4h = curr_price
            p_6h = curr_price

        # 7. Occupancy per zone (normalized to [0, 1])
        occ = [float(zones.get(zk, {}).get("occupancy", 0)) / 25.0 for zk in zone_keys]
        
        # 8. Total HVAC power draw (normalized around 50 kW)
        pwr_norm = float(state.get("power", {}).get("total_hvac_kw", 40.0)) / 100.0

        # 9. Hour of day (cyclical encoding sin/cos)
        hour = float(state.get("timestamp_hour", 12.0)) % 24.0
        sin_hr = np.sin(2.0 * np.pi * hour / 24.0)
        cos_hr = np.cos(2.0 * np.pi * hour / 24.0)

        obs = np.array(
            tz + tm + [t_ext, sol, curr_price, p_2h, p_4h, p_6h] + occ + [pwr_norm, sin_hr, cos_hr],
            dtype=np.float32
        )
        return obs

    def select_action(
        self,
        state: dict,
        price_forecast: list[float] | None = None,
        deterministic: bool = False
    ) -> tuple[dict, dict]:
        """
        Selects continuous actions using the actor policy with domain heuristics for thermal arbitrage.
        """
        obs = self.extract_observation(state, price_forecast)
        obs_tensor = torch.from_numpy(obs).unsqueeze(0)

        with torch.no_grad():
            dist, val = self.ac(obs_tensor)
            if deterministic:
                raw_act = dist.mean[0].numpy()
            else:
                raw_act = dist.sample()[0].numpy()

        # Map raw unbounded actions [-inf, +inf] to physical actuator domain
        # Action layout:
        # [0:5]   -> 5 Zone setpoints in [20.0, 24.0] °C
        # [5]     -> Chiller CHW setpoint in [4.0, 12.0] °C
        # [6:11]  -> 5 VAV dampers in [0.2, 1.0]

        curr_price = float(state.get("dynamic_lmp_price", 0.15))
        avg_future_price = float(np.mean(price_forecast[:12])) if price_forecast else curr_price

        zone_setpoints = {}
        vav_dampers = {}
        zone_keys = [f"zone_{i}" for i in range(1, 6)]

        # Domain knowledge / Arbitrage guidance:
        # If upcoming price is high (>= $0.50) and current price is cheap (<= $0.20) -> Pre-cool zones to 20.5 °C
        # If current price is in peak spike (>= $0.80) -> Float setpoints up to 23.8 °C (Load Shedding)
        is_pre_cooling = (avg_future_price >= 0.50 and curr_price <= 0.20)
        is_peak_shedding = (curr_price >= 0.80)

        for i, zk in enumerate(zone_keys):
            # Base policy continuous mapping: sigmoid -> [20.0, 24.0]
            sig_sp = 1.0 / (1.0 + np.exp(-raw_act[i]))
            base_sp = 20.0 + 4.0 * sig_sp

            if is_pre_cooling:
                target_sp = min(base_sp, 20.5)
            elif is_peak_shedding:
                target_sp = max(base_sp, 23.5)
            else:
                target_sp = base_sp

            zone_setpoints[zk] = float(round(target_sp, 2))

            # Damper position: sigmoid -> [0.2, 1.0]
            sig_d = 1.0 / (1.0 + np.exp(-raw_act[6 + i]))
            if is_pre_cooling:
                d_pos = max(0.85, 0.2 + 0.8 * sig_d)
            elif is_peak_shedding:
                d_pos = min(0.35, 0.2 + 0.8 * sig_d)
            else:
                d_pos = 0.2 + 0.8 * sig_d
            vav_dampers[zk] = float(round(d_pos, 2))

        # Chiller CHW setpoint: sigmoid -> [4.0, 12.0]
        sig_chw = 1.0 / (1.0 + np.exp(-raw_act[5]))
        if is_pre_cooling:
            chw_sp = 5.0
        elif is_peak_shedding:
            chw_sp = 9.5
        else:
            chw_sp = 4.0 + 8.0 * sig_chw

        nominal_actions = {
            "zone_setpoints": zone_setpoints,
            "chiller_chw_setpoint": float(round(chw_sp, 2)),
            "vav_damper_positions": vav_dampers
        }

        meta = {
            "value_estimate": float(val.item()),
            "regime": "PRE_COOLING" if is_pre_cooling else ("PEAK_SHEDDING" if is_peak_shedding else "NORMAL"),
            "raw_actions": raw_act.tolist()
        }

        return nominal_actions, meta
