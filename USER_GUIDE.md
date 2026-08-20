# AETHERIS-Zero: Operator & User Guide
### Autonomous Physics-Informed Safe-RL & Transactive Virtual Power Plant (VPP) Engine

---

## 1. Executive Summary

**AETHERIS-Zero** is an autonomous cyber-physical intelligence platform designed for commercial smart buildings and Virtual Power Plants (VPPs). It treats the building’s physical envelope (concrete walls, structural core, internal air capacitance) as a **Transactive Virtual Battery**. 

By orchestrating HVAC chillers, supply fans, and Variable Air Volume (VAV) dampers through **Continuous PPO Reinforcement Learning** and mathematically rigorous **Control Barrier Functions (CBFs)**, AETHERIS-Zero:
1. **Pre-cools** structural thermal mass during cheap off-peak hours.
2. **Deep-sheds** electrical power during extreme wholesale LMP grid spikes and OpenADR 3.0 demand response events.
3. **Guarantees 100% Occupant Comfort & Equipment Protection** via real-time Quadratic Program (QP) safety shields solved in under $1.5\text{ ms}$.
4. **Delivers 30%–35% Energy Cost Reduction** with zero comfort violations.

All metrics are natively localized to the **Indian Rupee (₹ INR)** ($1\text{ USD} = 83\text{ INR}$).

---

## 2. System Architecture

```
                               ┌──────────────────────────────────────────────┐
                               │  External Grid / Wholesale Market (CAISO)    │
                               │  OpenADR 3.0 DRAS Signals & LMP Real-Time    │
                               └──────────────────────┬───────────────────────┘
                                                      │
                                                      ▼
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                       AETHERIS-ZERO PLATFORM                                           │
│                                                                                                        │
│  ┌─────────────────────────────┐    ┌─────────────────────────────┐    ┌────────────────────────────┐  │
│  │   Semantic Ingestion        │    │    Physics & Safety Core    │    │    Autonomous Controller   │  │
│  │ - Zero-Shot SLM Tag Parser  │───►│ - 5-Zone 2R2C Ground Truth  │───►│ - Continuous PPO Agent     │  │
│  │ - Brick Schema v1.3 Graph   │    │ - ASHRAE 55 PMV/PPD Comfort │    │ - Virtual Battery Arbitrage│  │
│  │ - SPARQL Config Extractor   │    │ - OSQP CBF-QP Safety Shield │    │ - PINN-FNO Neural Surrogate│  │
│  └─────────────────────────────┘    └─────────────────────────────┘    └────────────────────────────┘  │
│                                                     │                                                  │
│                                                     ▼                                                  │
│  ┌──────────────────────────────────────────────────────────────────────────────────────────────────┐  │
│  │                               FastAPI Gateway & WebSocket Telemetry Stream                       │  │
│  └──────────────────────────────────────────────────┬───────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────┼──────────────────────────────────────────────────┘
                                                      │ (10-60 Hz Live Stream)
                                                      ▼
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                           Mission Control & 3D Three.js Digital Twin Dashboard                         │
│                                                                                                        │
│  ┌───────────────────────────────────────┐   ┌──────────────────────────────────────────────────────┐  │
│  │  3D Digital Twin Stage (60%)          │   │  Analytics Rail (40%)                                │  │
│  │  - Interactive Three.js Envelope      │   │  - Baseline vs. AETHERIS Load Analysis (kW)          │  │
│  │  - 5 Thermal Slices (Heatmap Hex)     │   │  - CAISO LMP Wholesale Price Curve (₹/MWh)           │  │
│  │  - Dynamic Airflow Vector Particles   │   │  - Accumulated Savings (₹4,060,360.45 / 30.2% ROI)   │  │
│  │  - Live Safety Barrier HUD Status     │   │  - Semantic Ingestion & RDF Graph Inspector          │  │
│  └───────────────────────────────────────┘   └──────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────────────────────────────────────────────────┐  │
│  │  Mission Control Deck: [Play/Pause] [Step] [Reset] [24h ROI Run] [Shadow Mode] [Override] [DR]   │  │
│  └──────────────────────────────────────────────────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Quickstart Guide (Get Up & Running in 60 Seconds)

### Prerequisites
- **Python 3.10+** (Tested on Python 3.12)
- Modern Web Browser (Chrome, Edge, Firefox, Safari)

### Option A: One-Command Interactive Launcher (Recommended)
Launch the unified server and automatically open the digital twin dashboard:
```bash
python run_aetheris.py
```
- **Mission Control Dashboard:** Open [`http://localhost:8000/`](http://localhost:8000/)
- **Interactive API Documentation:** Open [`http://localhost:8000/docs`](http://localhost:8000/docs)

### Option B: Automated Terminal Scenario Demonstrator
Run the full automated benchmark and cyber-physical fault injection suite directly in your terminal:
```bash
python run_aetheris.py --demo
```

### Option C: Run Automated Test Suite
Execute the entire test suite covering all 4 phases:
```bash
.venv/bin/pytest tests/ -v
```
*(All 39 tests should pass in under 3.5 seconds.)*

---

## 4. Mission Control Dashboard Operator Guide

The Mission Control dashboard adopts the **"Violet Dusk Precision"** theme with high contrast, semantic styling, and real-time WebGL rendering:

### 1. Top Navigation Bar
- **Brand Identity:** `AETHERIS-Zero: Safe-RL Transactive VPP`.
- **Mode Selectors:**
  - `PPO Safe-RL`: Activates closed-loop autonomous transactive battery arbitrage.
  - `Baseline`: Switches to conventional static building automation schedule.
  - `Brick Schema`: Opens the semantic metadata and RDF graph inspector.
- **Operator Guide (`[H]` / `[?]`):** Opens in-app modal with instructions and live shortcut legend.
- **WebSocket Status Indicator:** Displays live connection health (`LIVE WS` with pulsing emerald dot).

### 2. 3D Digital Twin Stage (Left 60%)
- **Interactive Three.js Wireframe:** Click and drag to orbit around the 5-story commercial building.
- **Thermal Heatmap Slices:** Each floor represents an active thermal zone dynamically colored by temperature:
  - 🟦 **Cool:** $< 20.5^\circ\text{C}$ (`#06b6d4`)
  - 🟩 / 🍑 **Optimal Comfort:** $20.5^\circ\text{C} - 23.0^\circ\text{C}$ (`#10b981` / `#ffa586`)
  - 🟧 **Warm:** $23.0^\circ\text{C} - 24.5^\circ\text{C}$ (`#f59e0b`)
  - 🟥 **Overheated:** $> 24.5^\circ\text{C}$ (`#b51a2b`)
- **Airflow Particles:** Real-time particle system whose speed and density dynamically correlate with active HVAC fan and chiller loads.
- **Safety Barrier HUD:** Real-time shield diagnostics tag (`OPTIMAL` in green/peach or pulsing red `INTERVENED` when safety barriers clamp inputs).

### 3. Analytics Rail (Right 40%)
- **Load Analysis:** Dual comparative curves displaying **Baseline Load (Dashed Gray)** vs. **AETHERIS Load (Solid Peach)** with active shaved demand counters (kW).
- **CAISO LMP Wholesale Price Curve:** Gradient area chart showing wholesale LMP dynamics in **₹/MWh**. When wholesale price spikes occur ($\ge \$0.50$/kWh or ₹41,500/MWh), an animated **`SPIKE DETECTED`** banner activates.
- **Accumulated Savings Card:** Glassmorphic ROI counter displaying total financial savings in Indian Rupee (`₹4,060,360.45`), peak reduction percentage ($31.0\%$), and ASHRAE 55 SLA compliance ($100\%$).

### 4. Mission Control Deck (Bottom Panel)
- **Status Readout:** Shows current simulation hour, step number, controller mode, and safety status.
- **Playback Controls:**
  - `Start / Pause Loop`: Toggle continuous real-time background simulation stepping.
  - `Step`: Advance simulation by one discrete 5-minute timestep.
  - `Reset`: Reset simulation, thermal state, and historical accumulators.
- **Action Triggers:**
  - `24h ROI Run`: Run a complete 288-step fast-forward day and display quantitative ROI summary.
  - `Shadow Mode`: Toggle AI passive shadow monitoring mode.
  - `Override (38°C)`: Simulate a cyber-physical fault or operator error to witness real-time CBF safety clamping.
  - `Trigger OpenADR DR`: Dispatch an automated OpenADR 3.0 demand response event.

---

## 5. Demonstration Scenarios

### Scenario 1: Transactive Virtual Battery Arbitrage
1. Navigate to the dashboard at `http://localhost:8000/`.
2. Click **Start Loop** (or press `Space`).
3. Observe how the PPO Agent detects cheap electricity hours (08:00–13:00) and lowers zone setpoints to $21.0^\circ\text{C}$, charging the thermal mass.
4. When the 14:00–18:00 CAISO peak price spike arrives, the chiller throttles down, discharging stored coolness while indoor temperatures remain comfortably below $24.5^\circ\text{C}$.

### Scenario 2: Cyber-Physical Setpoint Override & CBF Protection
1. While the simulation is running, click **Override (38°C)** (or press `O`).
2. Notice the **Safety Barrier Tag** immediately flashes red (`CBF SHIELD INTERVENED`).
3. The underlying OSQP solver clamps the $38.0^\circ\text{C}$ override to the safe upper comfort bound ($\le 24.5^\circ\text{C}$), completely preventing thermal runaway and occupant discomfort.

### Scenario 3: Compressor Anti-Short-Cycling Dwell Protection
1. Execute rapid alternating setpoint commands between $4.0^\circ\text{C}$ and $12.0^\circ\text{C}$.
2. The **Dwell-Time Barrier** activates, enforcing a minimum 3-step ($15\text{ min}$) rest period between compressor reversals.
3. Rapid cycling is blocked, protecting physical equipment from premature failure.

### Scenario 4: OpenADR 3.0 Automated Demand Response Dispatch
1. Click **Trigger OpenADR DR** (or press `D`).
2. An OpenADR 3.0 event payload with a 4-hour $\$1.50$/kWh price spike is dispatched.
3. The Virtual End Node (VEN) immediately receives the dispatch, notifies the virtual battery, and deep-curtails building HVAC demand by $>30\%$.

### Scenario 5: Semantic Ingestion & Brick Schema v1.3
1. Click **Brick Schema** in the top navigation bar.
2. In the modal, input unstructured BACnet tags (e.g. `AHU1_SAT, VAV101_ZN_T, CHW_SUP_T`).
3. Click **Parse Tags & Extract Brick Priors**.
4. View the extracted Brick Schema RDF Turtle ontology graph and 5-zone thermodynamic physical priors.

---

## 6. Keyboard Shortcuts & Accessibility

| Key | Action |
| :--- | :--- |
| <kbd>Space</kbd> | Toggle Continuous Simulation Loop (Play / Pause) |
| <kbd>S</kbd> | Advance Simulation by 1 Step (5 minutes) |
| <kbd>R</kbd> | Reset Simulation & Safety Barrier States |
| <kbd>O</kbd> | Inject Malicious Setpoint Override (38.0°C Fault) |
| <kbd>D</kbd> | Trigger OpenADR 3.0 Demand Response Event |
| <kbd>H</kbd> or <kbd>?</kbd> | Open Operator Guide & System Overview Modal |
| <kbd>Esc</kbd> | Close Any Open Modal Window |

- **Accessibility Support:** High contrast WCAG 2.1 AA colors, semantic ARIA landmarks (`role="main"`, `role="navigation"`, `role="region"`), keyboard focus rings, and screen-reader status announcements.

---

## 7. REST & WebSocket API Reference

### Key REST Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/` | Serves the interactive 3D Digital Twin Mission Control Dashboard |
| `GET` | `/health` | Health check endpoint and connected client count |
| `GET` | `/api/v1/simulation/state` | Retrieves the latest serialized telemetry frame |
| `POST` | `/api/v1/control/step` | Advances simulation by one step |
| `POST` | `/api/v1/control/reset` | Resets the simulation environment |
| `POST` | `/api/v1/control/inject-fault` | Injects malicious setpoint to test CBF safety shield |
| `POST` | `/api/v1/control/trigger-dr` | Dispatches an OpenADR 3.0 Demand Response event |
| `POST` | `/api/v1/simulation/run-episode` | Executes full 24h fast-forward benchmark episode |
| `POST` | `/api/v1/ingestion/parse-tags` | Zero-shot parses BACnet / Modbus tags to Brick Schema classes |
| `GET` | `/api/v1/ingestion/graph-turtle` | Exports Brick Schema v1.3 RDF graph in Turtle format |

### Real-Time WebSocket (`/ws/telemetry`)

- **Connection URL:** `ws://localhost:8000/ws/telemetry`
- **Supported Client Commands:**
  - `{"action": "START_SIMULATION"}`
  - `{"action": "STOP_SIMULATION"}`
  - `{"action": "STEP_SIMULATION"}`
  - `{"action": "RESET_SIMULATION"}`
  - `{"action": "INJECT_MALICIOUS_SETPOINT", "params": {"zone_id": "zone_1", "target_temp": 38.0}}`
  - `{"action": "TRIGGER_OPENADR_EVENT", "params": {"price_spike": 1.50, "start_hour": 14.0, "duration_hours": 4.0}}`
  - `{"action": "RUN_EPISODE", "params": {"total_steps": 288}}`
  - `{"action": "SET_CONTROLLER_MODE", "params": {"mode": "RL_SAFE_ARBITRAGE"}}`

---

## 8. Troubleshooting & FAQ

**Q: The dashboard says `OFFLINE` in the top right.**
> **A:** Ensure the FastAPI server is running (`python run_aetheris.py`). The dashboard automatically retries WebSocket connection every 2 seconds.

**Q: Can I run this in headless or server environments without a browser?**
> **A:** Yes! Run `python run_aetheris.py --no-browser --host 0.0.0.0 --port 8000` to run in headless container or remote server mode.

**Q: How do I change the currency exchange rate?**
> **A:** The system defaults to standard $1\text{ USD} = 83\text{ INR}$. Conversion logic is centrally managed in `gateway/streaming/telemetry_serializer.py` and `gateway/templates/dashboard.html`.

---

© 2026 AETHERIS-Zero Team. All rights reserved.
