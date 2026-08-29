"""
Train and calibrate the Physics-Informed Neural Network + Fourier Neural Operator (PINN-FNO)
surrogate model against 2R2C thermodynamic trajectories and physical ODE loss.
"""

from pathlib import Path
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from core.models.pinn_surrogate import PINNSurrogate
from core.simulator.building_etp import BuildingSimulator


def train_pinn_surrogate(
    data_path: Path,
    save_path: Path,
    epochs: int = 20,
    batch_size: int = 16,
    lr: float = 1e-3,
    horizon_steps: int = 96,
    device: str = "cpu"
) -> PINNSurrogate:
    print("=" * 70)
    print("AETHERIS-ZERO: PINN-FNO SURROGATE TRAINING & CALIBRATION")
    print("=" * 70)
    
    sim = BuildingSimulator()
    trajectories = []
    
    print(" -> Generating multi-scenario 2R2C simulation trajectory batches...")
    for seed in range(25):
        np.random.seed(seed)
        state = sim.reset()
        weather_mult = 0.8 + 0.4 * np.random.rand()
        
        feat_seq = np.zeros((17, horizon_steps), dtype=np.float32)
        target_seq = np.zeros((5, horizon_steps), dtype=np.float32)
        
        for step in range(horizon_steps):
            hr = (step * 300.0 / 3600.0) % 24.0
            t_ext = (26.0 + 8.0 * np.sin(2.0 * np.pi * (hr - 9.0) / 24.0)) * weather_mult
            sol = max(0.0, 900.0 * np.sin(np.pi * (hr - 6.0) / 12.0)) * weather_mult if 6.0 <= hr <= 18.0 else 0.0
            
            sp = 21.0 + 3.0 * np.random.rand()
            actions = {
                "zone_setpoints": {f"zone_{i}": sp for i in range(1, 6)},
                "chiller_chw_setpoint": 6.5,
                "vav_damper_positions": {f"zone_{i}": 0.7 for i in range(1, 6)}
            }
            sim.set_weather_override(t_ext, sol)
            st, _, _, _ = sim.step(actions)
            
            z_temps = [st["zones"][f"zone_{i}"]["temp_c"] for i in range(1, 6)]
            feat_seq[0:5, step] = z_temps
            feat_seq[5:10, step] = [sp] * 5
            feat_seq[10:15, step] = [0.7] * 5
            feat_seq[15, step] = t_ext
            feat_seq[16, step] = sol / 1000.0
            
            target_seq[:, step] = z_temps

        trajectories.append((feat_seq, target_seq))

    X = torch.tensor(np.array([t[0] for t in trajectories]), dtype=torch.float32)
    Y = torch.tensor(np.array([t[1] for t in trajectories]), dtype=torch.float32)
    
    print(f" -> Dataset shape: Inputs X={X.shape}, Targets Y={Y.shape}")
    
    model = PINNSurrogate(modes=16, width=32, num_layers=2)
    model.train()
    model.to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion_data = nn.MSELoss()
    
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        perm = torch.randperm(X.size(0))
        epoch_loss = 0.0
        
        for i in range(0, X.size(0), batch_size):
            indices = perm[i:i + batch_size]
            batch_x, batch_y = X[indices].to(device), Y[indices].to(device)
            
            optimizer.zero_grad()
            pred = model(batch_x)
            
            loss_data = criterion_data(pred, batch_y)
            loss_phys = model.compute_physics_loss(
                t_pred=pred,
                t_ext=batch_x[:, 15:16, :],
                q_hvac=torch.clamp(batch_y - batch_x[:, 5:10, :], min=0.0) * 100.0,
                q_sol=batch_x[:, 16:17, :] * 1000.0
            )
            
            total_loss = loss_data + model.lambda_phys * loss_phys
            total_loss.backward()
            optimizer.step()
            
            epoch_loss += total_loss.item()
            
        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:2d}/{epochs:2d} | Total Loss: {epoch_loss:.4f} | Elapsed: {time.time()-t0:.1f}s")

    model.eval()
    model.save_checkpoint(save_path)
    print(f" -> Checkpoint successfully saved to: {save_path}")
    return model


if __name__ == "__main__":
    ckpt_dir = Path(__file__).resolve().parent.parent.parent / "models" / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_file = ckpt_dir / "pinn_surrogate_best.pt"
    data_file = Path(__file__).resolve().parent.parent.parent / "data" / "datasets" / "grid_weather_thermal_timeseries.csv"
    
    train_pinn_surrogate(data_path=data_file, save_path=ckpt_file, epochs=20)
