#!/usr/bin/env python3
"""
AETHERIS-Zero: Master Machine Learning & Reinforcement Learning Training Pipeline.
Orchestrates Dataset Generation, Neural SLM Tag Classifier Training, and Vectorized Safe-RL PPO Agent Training.
Optimized for NVIDIA GeForce RTX 5070 Ti Laptop GPU & Multi-Core CPU Fallback.
"""

import argparse
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from core.device_utils import get_optimal_device
from data.datasets.dataset_generator import build_and_save_all_datasets
from gateway.ingestion.train_slm import train_neural_slm, export_generative_slm_instruction_dataset
from core.controller.train_ppo import train_safe_rl_ppo


BANNER = r"""
===================================================================================
     AETHERIS-ZERO :: NEURAL SLM & VECTORIZED SAFE-RL TRAINING PIPELINE
===================================================================================
"""


def main():
    parser = argparse.ArgumentParser(description="Aetheris-Zero Master ML/RL Trainer")
    parser.add_argument("--slm", action="store_true", default=True, help="Train Neural Brick-SLM Model")
    parser.add_argument("--rl", action="store_true", default=True, help="Train Vectorized Safe-RL PPO Controller")
    parser.add_argument("--slm-epochs", type=int, default=8, help="Epochs for SLM training")
    parser.add_argument("--rl-steps", type=int, default=30000, help="Timesteps for Safe-RL PPO training")
    parser.add_argument("--rl-envs", type=int, default=32, help="Number of parallel vectorized building envs")
    parser.add_argument("--force-cpu", action="store_true", help="Force CPU execution")
    args = parser.parse_args()

    print(BANNER)
    device = get_optimal_device(force_cpu=args.force_cpu)
    print(f"[*] Target Compute Device: {device.upper()}")
    print(f"[*] Workspace Root:        {PROJECT_ROOT}")
    print("-" * 83)

    # 1. Dataset Verification & Synthesis
    print("\n[PHASE 1/3] Verifying & Generating Curated Datasets...")
    slm_data_path = PROJECT_ROOT / "data" / "datasets" / "slm_bacnet_brick_corpus.jsonl"
    grid_data_path = PROJECT_ROOT / "data" / "datasets" / "grid_weather_thermal_timeseries.csv"

    if not slm_data_path.exists() or not grid_data_path.exists():
        build_and_save_all_datasets()
    else:
        print(f" -> Found SLM corpus: {slm_data_path}")
        print(f" -> Found 1-Year Grid & Weather Timeseries: {grid_data_path}")

    # 2. Neural SLM Training
    slm_ckpt_path = PROJECT_ROOT / "models" / "checkpoints" / "slm_brick_best.pt"
    if args.slm:
        print("\n[PHASE 2/3] Training Neural Brick-SLM Tag Classifier...")
        t0 = time.time()
        model, metrics = train_neural_slm(
            data_path=slm_data_path,
            save_checkpoint_path=slm_ckpt_path,
            epochs=args.slm_epochs,
            batch_size=64,
            lr=2e-3,
            force_cpu=args.force_cpu
        )
        export_generative_slm_instruction_dataset(
            jsonl_input=slm_data_path,
            output_path=PROJECT_ROOT / "data" / "datasets" / "slm_instruction_tuning.json"
        )
        print(f" -> SLM Training complete in {time.time() - t0:.1f}s | Checkpoint: {slm_ckpt_path}")

    # 3. Vectorized Safe-RL PPO Training
    rl_ckpt_path = PROJECT_ROOT / "models" / "checkpoints" / "ppo_safe_rl_best.pt"
    if args.rl:
        print("\n[PHASE 3/3] Training Vectorized Safe-RL PPO Agent (Parallel 2R2C Building Environments)...")
        t0 = time.time()
        actor_critic, rl_metrics = train_safe_rl_ppo(
            total_timesteps=args.rl_steps,
            num_envs=args.rl_envs,
            batch_size=128,
            save_checkpoint_path=str(rl_ckpt_path),
            force_cpu=args.force_cpu
        )
        print(f" -> Safe-RL PPO Training complete in {time.time() - t0:.1f}s | Checkpoint: {rl_ckpt_path}")

    print("\n" + "=" * 83)
    print("ALL MODELS TRAINED & READY-TO-USE OUT OF THE BOX!")
    print(f" - SLM Checkpoint:     {slm_ckpt_path}")
    print(f" - Safe-RL Checkpoint: {rl_ckpt_path}")
    print("=" * 83)


if __name__ == "__main__":
    main()
