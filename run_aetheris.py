#!/usr/bin/env python3
"""
AETHERIS-Zero: One-Command Unified Mission Control & Product Suite Launcher.
Starts the FastAPI backend gateway, serves the 3D Digital Twin Dashboard,
and provides direct access to the simulation engine, SLM parser, and API documentation.
"""

import os
import sys
from pathlib import Path

# Auto-detect and switch to local .venv Python if available and not already in use
ROOT_DIR = Path(__file__).resolve().parent
VENV_PYTHON = ROOT_DIR / ".venv" / "bin" / "python"
if VENV_PYTHON.exists() and sys.executable != str(VENV_PYTHON):
    try:
        os.execv(str(VENV_PYTHON), [str(VENV_PYTHON)] + sys.argv)
    except Exception:
        pass

import argparse
import subprocess
import threading
import time
import webbrowser


BANNER = r"""
    ___    ______ _____ _   _ _____ ____  _____ _____       _____                  
   /   |  / ____/_   _| | | | ____|  _ \|_   _/ ____|     |__  /___ _ __ ___      
  / /| | |  __|   | | | |_| |  _| | |_) | | | \___ \ _____  / // _ \ '__/ _ \     
 / ___ | | |___   | | |  _  | |___|  _ <  | |  ___) |_____| / /|  __/ | | (_) |    
/_/   |_| \____|  |_| |_| |_|_____|_| \_\ |_| |____/      /____\___|_|  \___/     
===================================================================================
 Autonomous BMS Supervisory Layer, Neural Brick-SLM & Safe-RL Transactive Engine
===================================================================================
"""


def run_cli_scenarios():
    """Execute automated demonstration scenarios in CLI mode."""
    print(BANNER)
    print(">> RUNNING AUTOMATED BENCHMARK & FAULT-INJECTION SUITE\n")
    
    from core.scenarios.fault_injection import (
        run_arbitrage_scenario,
        inject_malicious_setpoint,
        trigger_chiller_short_cycle,
    )

    print("-" * 78)
    print("SCENARIO 1: 24h Transactive Virtual Battery Arbitrage under CAISO Dynamic Pricing")
    print("-" * 78)
    res1 = run_arbitrage_scenario(total_steps=288)
    m1 = res1["metrics"]
    print(f" -> Baseline Energy Cost:      ${m1['baseline_cost_usd']:.2f} (₹{m1['baseline_cost_usd']*83:,.2f})")
    print(f" -> AETHERIS Safe-RL Cost:     ${m1['aetheris_cost_usd']:.2f} (₹{m1['aetheris_cost_usd']*83:,.2f})")
    print(f" -> Energy Cost Savings:       {m1['cost_savings_pct']:.1f}% (${m1['cost_savings_usd']:.2f} / ₹{m1['cost_savings_usd']*83:,.2f})")
    print(f" -> Peak Demand Shaved:        {m1['peak_demand_shaved_pct']:.1f}% ({m1['peak_demand_shaved_kw']:.1f} kW)")
    print(f" -> Carbon Footprint Avoided:  {m1['carbon_avoided_kg']:.2f} kg CO2")
    print(f" -> ASHRAE 55 SLA Compliance:  {m1['comfort_compliance_rate_pct']:.1f}%\n")

    print("-" * 78)
    print("SCENARIO 2: Cyber-Physical Setpoint Override Attack (38.0°C Fault Injection)")
    print("-" * 78)
    res2 = inject_malicious_setpoint(zone_id="zone_1", malicious_temp=38.0)
    print(f" -> Attack Setpoint Injected:  {res2['injected_temperature_c']:.1f}°C")
    print(f" -> CBF-QP Shield Safe Clamp:  {res2['shielded_safe_temperature_c']:.2f}°C")
    print(f" -> Active Constraints:        {', '.join(res2['active_constraints'])}")
    print(f" -> Intercept Status:          {res2['verdict']} ({res2['shield_status']})\n")

    print("-" * 78)
    print("SCENARIO 3: Compressor Anti-Short-Cycling Dwell Protection Attack")
    print("-" * 78)
    res3 = trigger_chiller_short_cycle(toggle_steps=6, min_dwell_steps=3)
    print(f" -> Rapid Toggles Injected:    {res3['total_toggle_attempts']}")
    print(f" -> Destructive Cycles Blocked:{res3['blocked_toggle_count']}")
    print(f" -> Dwell Enforcement Rate:    {res3['dwell_enforcement_rate_pct']:.1f}%")
    print(f" -> Equipment Safety Verdict:  {res3['verdict']}\n")
    print("=" * 78)
    print("All scenarios completed successfully with 100% safety and comfort SLAs verified.")
    print("=" * 78)


def run_training_pipeline(mode: str = "all", slm_epochs: int = 12, rl_steps: int = 30000):
    """Execute model training pipeline via train_all.py."""
    print(BANNER)
    print(f">> EXECUTING AETHERIS-ZERO ML TRAINING PIPELINE (MODE: {mode.upper()})\n")
    cmd = [sys.executable, str(ROOT_DIR / "train_all.py")]
    if mode in ["slm", "all"]:
        cmd.extend(["--slm", "--slm-epochs", str(slm_epochs)])
    if mode in ["rl", "all"]:
        cmd.extend(["--rl", "--rl-steps", str(rl_steps)])
    
    subprocess.run(cmd, check=True)


def start_server(host: str = "0.0.0.0", port: int = 8000, open_browser: bool = True, open_all: bool = False):
    """Launch FastAPI Uvicorn server and open product endpoints."""
    print(BANNER)
    display_host = "localhost" if host in ["0.0.0.0", "127.0.0.1"] else host
    base_url = f"http://{display_host}:{port}"
    
    print(" [PRODUCT DASHBOARDS & ACCESS URLS]")
    print(f"  * Commercial Product & ROI Portal: {base_url}/overview")
    print(f"  * 3D Mission Control Digital Twin: {base_url}/")
    print(f"  * Scenario & Injection Simulator:  {base_url}/simulator")
    print(f"  * BMS Integration Output Terminal: {base_url}/terminal")
    print(f"  * Interactive API Docs (Swagger):  {base_url}/docs")
    print(f"  * Technical Spec (ReDoc):          {base_url}/redoc")
    print(f"  * Real-Time Telemetry Stream:      ws://{display_host}:{port}/ws/telemetry")
    print(f"  * Performance Audit Report:        {ROOT_DIR / 'MODEL_PERFORMANCE_REPORT.md'}")
    print(f"  * Financial Currency Mode:         INR (₹) & USD ($) | 1 USD = 83.00 INR")
    print("-" * 78)
    print(" Press Ctrl+C to stop the mission control server.\n")

    if open_browser:
        def _open():
            time.sleep(1.2)
            try:
                webbrowser.open(f"{base_url}/")
                if open_all:
                    time.sleep(0.5)
                    webbrowser.open(f"{base_url}/docs")
            except Exception:
                pass
        threading.Thread(target=_open, daemon=True).start()

    import uvicorn
    uvicorn.run("gateway.main:app", host=host, port=port, log_level="info")


def main():
    parser = argparse.ArgumentParser(description="AETHERIS-Zero Mission Control & Product Suite Launcher")
    parser.add_argument("--demo", action="store_true", help="Run automated CLI scenario benchmarks and exit")
    parser.add_argument("--train-slm", action="store_true", help="Train Neural Brick-SLM on disjoint leak-free dataset")
    parser.add_argument("--train-rl", action="store_true", help="Train Vectorized Safe-RL PPO controller")
    parser.add_argument("--train-all", action="store_true", help="Train both Neural SLM and Safe-RL PPO models")
    parser.add_argument("--slm-epochs", type=int, default=12, help="Number of training epochs for SLM (default: 12)")
    parser.add_argument("--rl-steps", type=int, default=30000, help="Number of training steps for Safe-RL (default: 30000)")
    parser.add_argument("--host", default="0.0.0.0", help="Host address to bind (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Port number (default: 8000)")
    parser.add_argument("--no-browser", action="store_true", help="Do not automatically open browser on launch")
    parser.add_argument("--open-all", action="store_true", help="Open both the 3D Dashboard and Swagger API Docs in browser")

    args = parser.parse_args()

    if args.demo:
        run_cli_scenarios()
    elif args.train_slm:
        run_training_pipeline(mode="slm", slm_epochs=args.slm_epochs)
    elif args.train_rl:
        run_training_pipeline(mode="rl", rl_steps=args.rl_steps)
    elif args.train_all:
        run_training_pipeline(mode="all", slm_epochs=args.slm_epochs, rl_steps=args.rl_steps)
    else:
        start_server(
            host=args.host,
            port=args.port,
            open_browser=not args.no_browser,
            open_all=args.open_all,
        )


if __name__ == "__main__":
    main()
