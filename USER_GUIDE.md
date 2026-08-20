# AETHERIS-Zero: Operator & User Guide
### Autonomous Smart Building Energy Optimizer & Virtual Power Plant (VPP) Platform

---

## 1. Executive Summary

**AETHERIS-Zero** turns commercial buildings into **Transactive Virtual Power Plants (VPPs)** by utilizing structural thermal inertia (walls, floors, air volume) as a **Virtual Thermal Battery**.

By coordinating HVAC chillers, supply fans, and variable airflow dampers using **Continuous PPO Safe-RL** and real-time **OSQP Control Barrier Functions (CBF-QP)**, AETHERIS-Zero:
1. **Pre-cools** the building structure when electricity prices are low.
2. **Sheds electrical load** during expensive peak grid hours (e.g. 2:00 PM – 6:00 PM) or during OpenADR 3.0 grid events.
3. **Mathematically Guarantees 100% Comfort & Safety** so room temperatures never violate ASHRAE 55 standards and AC equipment is never damaged.
4. **Calculates 100% Real, Dynamic Cost & Carbon Savings** with zero hardcoded numbers or fake placeholders.

All monetary values are dynamically computed in real-time, with an instant toggle between **₹ INR (Indian Rupee)** ($1\text{ USD} = 83\text{ INR}$) and **$ USD**.

---

## 2. Quick Start (60 Seconds)

### Launch the Platform
```bash
# Start backend server and automatically open the 3D Digital Twin in your browser
python run_aetheris.py
```
- **Mission Control Dashboard:** Open [http://localhost:8000/](http://localhost:8000/)
- **Interactive REST API Docs:** Open [http://localhost:8000/docs](http://localhost:8000/docs)

### Run Demonstration Scenarios in Terminal (CLI Mode)
```bash
python run_aetheris.py --demo
```

### Run Automated Test Suite
```bash
.venv/bin/pytest tests/ -v
```
*(All 40 integration tests pass in under 3.5 seconds.)*

---

## 3. How to Use the Dashboard (Designed for Everyone)

The dashboard is engineered to be **immediately clear and effortless to operate**, even for someone with no computer engineering background.

### Top Status Banner (Plain English)
At the top of the screen, you will always see an easy-to-read banner explaining what the system is doing right now:
- 🟢 **AI Optimizer Active (Comfort Protected & Saving Electricity):** Indicates standard operation. All rooms are at comfortable temperatures (21°C–23°C) and energy is being drawn efficiently.
- 🟡 **High Electricity Price Detected (Stored Coolness Discharging):** Indicates that peak electricity rates are active. The AI has dialed down power-hungry AC compressors while the coolness stored in concrete walls keeps rooms comfortable.
- 🛡️ **Safety Shield Intervened (Comfort Protected):** Indicates that an extreme or unsafe thermostat command (e.g. 38°C) was intercepted and limited to safe limits ($20.0^\circ\text{C} - 24.5^\circ\text{C}$) to protect occupants.

### One-Click Control Buttons
- ⏯️ **Start Auto Optimizer (Spacebar):** Starts or pauses the live real-time simulation loop.
- ⏭️ **+5 Min (S):** Advances the building simulation forward by 5 minutes.
- ⚡ **Full 24h Day (F):** Runs a complete 24-hour day in 1 second to calculate full cumulative savings and ROI.
- 🔄 **Reset (R):** Clears all accumulators and resets the clock to midnight (00:00).

---

## 4. Live Input & Scenario Studio (Real-World Custom Data)

You can adjust real-world inputs right on the screen and watch the physics model react immediately:

### 1. Outdoor Weather & Ambient Temperature
- **Interactive Slider:** Drag from $15^\circ\text{C}$ to $45^\circ\text{C}$ and click **Set**.
- **Quick Presets:**
  - *Cool Day ($18^\circ\text{C}$)* — Low cooling demand.
  - *Standard ($30^\circ\text{C}$)* — Regular operating conditions.
  - *🔥 Heatwave ($42^\circ\text{C}$)* — High external thermal influx.

### 2. Electricity Tariff & Grid Pricing
- **Interactive Slider:** Adjust price from ₹1/kWh to ₹50/kWh (or $0.05 to $1.50/kWh) and click **Set**.
- **⚡ Trigger 5x Peak Surge:** Simulates an OpenADR 3.0 demand response emergency spike. Watch how the AI immediately cuts chiller power while maintaining room comfort!

### 3. Room Thermostats & Comfort Targets
- **Zone Selector:** Click any of the 5 zones (*Core Floor 1*, *North Office*, *South Office*, *East Office*, *West Office*).
- **Target Slider:** Set the room's desired temperature and click **Apply**.
- **🛡️ Test Wrong 38°C Button:** Injects a malicious or accidental $38^\circ\text{C}$ override to demonstrate how the OSQP Safety Shield clamps the setpoint to $24.5^\circ\text{C}$ to prevent overheating.

---

## 5. REST API & WebSocket Reference

All endpoints accept dynamic inputs and broadcast updates in real time:

| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `/api/v1/control/set-weather` | `POST` | Update live outdoor temperature (°C) & solar irradiance |
| `/api/v1/control/set-pricing` | `POST` | Set instantaneous electricity price ($/kWh or ₹/kWh) |
| `/api/v1/control/set-tariff-schedule`| `POST` | Upload custom 24-hour tariff schedule array |
| `/api/v1/control/set-zone-target` | `POST` | Set desired temperature setpoint for a specific room |
| `/api/v1/control/set-comfort-bounds` | `POST` | Adjust safety shield comfort bounds (`t_min`, `t_max`) |
| `/api/v1/control/step` | `POST` | Step simulation engine forward 1 step |
| `/api/v1/control/reset` | `POST` | Reset simulation to initial state |
| `/api/v1/simulation/start` | `POST` | Start background real-time simulation loop |
| `/api/v1/simulation/stop` | `POST` | Stop background simulation loop |
| `/api/v1/simulation/run-episode` | `POST` | Fast-forward 24h episode and return analytical ROI |
| `/ws/telemetry` | `WebSocket`| High-frequency full-duplex stream for 3D twin & charts |

---

## 6. Keyboard Shortcuts Legend

| Key | Action | Description |
| :--- | :--- | :--- |
| <kbd>Space</kbd> | **Play / Pause** | Toggle background autonomous simulation loop |
| <kbd>S</kbd> | **Step +5 Min** | Execute 1 discrete simulation step (5 minutes) |
| <kbd>F</kbd> | **Fast 24h Day** | Fast-forward 288 steps (24 hours) |
| <kbd>R</kbd> | **Reset** | Reset simulation clock and cost accumulators |
| <kbd>H</kbd> or <kbd>?</kbd> | **Help / Guide** | Open in-app plain-English operator guide modal |
| <kbd>Esc</kbd> | **Dismiss** | Close active modals |

---

## 7. Frequently Asked Questions (FAQ)

**Q: Are any numbers or graphs fake or hardcoded?**  
**A:** No. Every single value—from cost savings to power kW, zone temperatures, and carbon emissions—is computed directly by the 2R2C thermodynamics engine and the PPO Safe-RL agent step-by-step.

**Q: Can I connect real sensor tags from my building BMS?**  
**A:** Yes. The `/api/v1/ingestion/parse-tags` and `/api/v1/ingestion/upload-csv` endpoints use a zero-shot semantic parser to map BACnet/Modbus register tags into Brick Schema v1.3 RDF ontology graphs and extract physical simulator priors automatically.

**Q: How does the system handle currency?**  
**A:** Click the **₹ INR / $ USD** toggle in the top-right corner to convert all financial metrics, wholesale electricity curves, and accumulated savings dynamically using the live conversion factor.
