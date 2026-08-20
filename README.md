# AETHERIS-Zero
### Autonomous Physics-Informed Safe-RL & Transactive Virtual Power Plant (VPP) Engine

[![Tests](https://img.shields.io/badge/pytest-39%20passed-brightgreen.svg)](tests/)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Standards](https://img.shields.io/badge/standards-OpenADR%203.0%20%7C%20Brick%20Schema%20v1.3%20%7C%20ASHRAE%2055-orange.svg)](https://brickschema.org/)

AETHERIS-Zero turns commercial buildings into **Transactive Virtual Power Plants (VPPs)** by exploiting structural thermal inertia as a "Virtual Battery" using Continuous PPO Reinforcement Learning, OSQP Control Barrier Functions (CBF-QP), and OpenADR 3.0.

---

## ⚡ Key Highlights & Capabilities

- 🏢 **Thermal Battery Arbitrage:** Autonomous pre-cooling before peak tariff windows and deep load-shedding during price spikes ($\ge \$0.50$/kWh).
- 🛡️ **Mathematical Safety Shield:** Real-time OSQP quadratic program solving Control Barrier Functions in $<1.5\text{ ms}$ to guarantee 100% ASHRAE 55 occupant comfort and prevent equipment short-cycling.
- ⚡ **OpenADR 3.0 VEN:** Native OpenADR 3.0 Virtual End Node with dynamic dispatch event registration, telemetry reporting, and automated DR compliance.
- 🧠 **Fourier Neural Operator Twin:** PINN-FNO neural surrogate model predicting multi-zone thermal horizons in $<5\text{ ms}$.
- 🌐 **3D Three.js Digital Twin:** High-frequency spatial heatmap, airflow particle dynamics, and analytics charts denominated in Indian Rupee (**₹ INR**).
- 🧱 **Semantic Ingestion:** Zero-shot SLM point tag parser generating Brick Schema v1.3 RDF Turtle graphs and extracting physical simulator priors via SPARQL.

---

## 🚀 Quick Start (60 Seconds)

### 1. Launch Mission Control & Digital Twin
```bash
# Start backend and launch interactive 3D dashboard
python run_aetheris.py
```
- **Mission Control Dashboard:** Open [http://localhost:8000/](http://localhost:8000/)
- **Interactive REST API Docs:** Open [http://localhost:8000/docs](http://localhost:8000/docs)

### 2. Run Automated Scenario Demonstrations in CLI
```bash
python run_aetheris.py --demo
```

### 3. Run Automated Tests
```bash
.venv/bin/pytest tests/ -v
```

---

## 📖 Complete Documentation & Guides

- 📘 [**Full User & Operator Guide (USER_GUIDE.md)**](USER_GUIDE.md) — Comprehensive operator walkthrough, demonstration scenarios, API reference, and keyboard shortcuts.
- 📐 [**Design System & UI Specification**](stitch_aetheris_zero_digital_twin_dashboard/DESIGN.md) — Violet Dusk design tokens, color palette, and typography.
- 🧪 [**Test Suite**](tests/) — 39 integration and unit tests covering all system layers.

---

## 🏗️ Repository Architecture

```
aetheris-zero/
├── core/                                # Dev 1: Physics, Safe-RL & Controls
│   ├── simulator/                       # 5-zone 2R2C thermal simulator & ASHRAE 55 engine
│   ├── models/                          # PINN-FNO neural surrogate twin
│   ├── safety/                          # OSQP CBF-QP safety shield & barrier functions
│   ├── controller/                      # Continuous PPO agent & arbitrage engine
│   └── scenarios/                       # Demonstration fault-injection scenarios
├── gateway/                             # Dev 2: Ingestion, Grid & Digital Twin API
│   ├── ingestion/                       # SLM tag parser, Brick Schema builder, SPARQL extractor
│   ├── grid/                            # OpenADR 3.0 VEN & CAISO dynamic tariff feed
│   ├── streaming/                       # Telemetry serializer & WebSocket connection manager
│   ├── templates/                       # Embedded Mission Control HTML dashboard
│   └── main.py                          # FastAPI unified backend & WebSocket stream
├── dashboard/                           # Next.js React Digital Twin Dashboard
├── tests/                               # 39 Comprehensive automated tests (Phases 1-4)
├── run_aetheris.py                      # One-command CLI & server launcher
├── USER_GUIDE.md                        # Operator guide and scenario manual
└── README.md                            # Main project overview
```

---

## 📜 License
MIT License. Developed for advanced autonomous building energy optimization.
