import time
import numpy as np
import torch
import torch.nn as nn
from core.models.fno_layers import FNONetwork1d

class PINNSurrogate(nn.Module):
    """
    Physics-Informed Neural Network (PINN) + Fourier Neural Operator (FNO)
    Digital Twin for ultra-fast (sub-millisecond) 24-hour multi-zone forward thermal state prediction.
    """
    def __init__(
        self,
        in_dim: int = 17,
        out_dim: int = 5,
        modes: int = 16,
        width: int = 64,
        num_layers: int = 3,
        lambda_phys: float = 0.2
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.lambda_phys = lambda_phys
        self.fno = FNONetwork1d(
            in_dim=in_dim,
            out_dim=out_dim,
            modes=modes,
            width=width,
            num_layers=num_layers
        )
        self.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fno(x)

    def compute_physics_loss(
        self,
        t_pred: torch.Tensor,
        t_ext: torch.Tensor,
        q_hvac: torch.Tensor,
        q_sol: torch.Tensor,
        dt_sec: float = 300.0,
        cz_base: float = 15000.0,
        rext_base: float = 2.5
    ) -> torch.Tensor:
        # Finite difference temporal derivative: d(T_z)/dt
        dt_pred = (t_pred[:, :, 1:] - t_pred[:, :, :-1]) / dt_sec
        
        # 2R2C lumped physics residual
        q_envelope = (t_ext[:, :, :-1] - t_pred[:, :, :-1]) / rext_base
        f_thermo = (q_envelope + 0.3 * q_sol[:, :, :-1] - q_hvac[:, :, :-1]) / cz_base
        
        res = dt_pred - f_thermo
        return torch.mean(res ** 2)

    def predict_horizon(
        self,
        current_state: dict,
        action_sequence: list[dict] | np.ndarray | None = None,
        weather_forecast: list[tuple[float, float]] | np.ndarray | None = None,
        horizon_steps: int = 96,
        dt_sec: float = 300.0
    ) -> np.ndarray:
        """
        Fast forward state predictor evaluating 24h (horizon_steps) multi-zone thermal trajectory.
        Target execution latency: < 5 ms.
        """
        start_time = time.perf_counter()
        
        # Extract starting conditions
        zones = current_state.get("zones", {})
        zone_keys = [f"zone_{i}" for i in range(1, 6)]
        init_temps = [zones.get(zk, {}).get("temp_c", 22.0) for zk in zone_keys]
        init_mass = [zones.get(zk, {}).get("mass_temp_c", 21.8) for zk in zone_keys]
        
        curr_hour = float(current_state.get("timestamp_hour", 0.0))
        
        # Build feature sequence: (batch=1, in_dim=17, length=horizon_steps)
        # Features per step:
        # - 5 zone current temps
        # - 5 zone target setpoints
        # - 5 VAV dampers
        # - Ambient temp (1)
        # - Solar irradiance (1)
        feat_matrix = np.zeros((1, self.in_dim, horizon_steps), dtype=np.float32)
        
        for step_i in range(horizon_steps):
            step_hour = (curr_hour + (step_i * dt_sec / 3600.0)) % 24.0
            
            # Weather
            if weather_forecast is not None and step_i < len(weather_forecast):
                t_ext_val, sol_val = weather_forecast[step_i]
            else:
                # Default diurnal profile
                t_ext_val = 28.0 + 8.0 * np.sin(2.0 * np.pi * (step_hour - 9.0) / 24.0)
                sol_val = max(0.0, 850.0 * np.sin(np.pi * (step_hour - 6.0) / 12.0)) if 6.0 <= step_hour <= 18.0 else 0.0
            
            # Actions
            if action_sequence is not None and step_i < len(action_sequence):
                act = action_sequence[step_i]
                if isinstance(act, dict):
                    sp_dict = act.get("zone_setpoints", {})
                    damp_dict = act.get("vav_damper_positions", {})
                    sp_vals = [float(sp_dict.get(zk, 22.0)) for zk in zone_keys]
                    damp_vals = [float(damp_dict.get(zk, 0.7)) for zk in zone_keys]
                else:
                    sp_vals = [22.0] * 5
                    damp_vals = [0.7] * 5
            else:
                sp_vals = [22.0] * 5
                damp_vals = [0.7] * 5
            
            feat_matrix[0, 0:5, step_i] = init_temps
            feat_matrix[0, 5:10, step_i] = sp_vals
            feat_matrix[0, 10:15, step_i] = damp_vals
            feat_matrix[0, 15, step_i] = t_ext_val
            feat_matrix[0, 16, step_i] = sol_val / 1000.0  # kW/m2 normalized

        with torch.no_grad():
            x_tensor = torch.from_numpy(feat_matrix)
            pred_delta = self.fno(x_tensor).numpy()[0]  # shape (5, horizon_steps)

        # Baseline analytical integration + learned spectral correction
        # Ensures smooth physical bounds matching the 2R2C baseline
        predicted_trajectory = np.zeros((horizon_steps, 5), dtype=np.float32)
        current_t = np.array(init_temps, dtype=np.float32)
        
        for s in range(horizon_steps):
            step_hour = (curr_hour + (s * dt_sec / 3600.0)) % 24.0
            t_ext_s = feat_matrix[0, 15, s]
            sp_s = feat_matrix[0, 5:10, s]
            
            # 2R2C analytical forward step
            cooling = np.maximum(0.0, (current_t - sp_s) * 0.8)
            decay = (t_ext_s - current_t) * (dt_sec / 18000.0)
            current_t = current_t + decay - cooling * (dt_sec / 15000.0) + pred_delta[:, s] * 0.05
            predicted_trajectory[s, :] = np.clip(current_t, 18.0, 32.0)

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        return predicted_trajectory
