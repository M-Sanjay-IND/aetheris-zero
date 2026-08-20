# AETHERIS-Zero: Operational & Deployment Guidelines
### Best Practices for Autonomous Smart Building Energy Management & VPP Operations

---

## 1. Safety & Thermal Comfort Guidelines

AETHERIS-Zero utilizes a mathematical **Control Barrier Function (CBF-QP)** filter that intercepts all setpoint adjustments before dispatching commands to physical BACnet/Modbus actuators.

### A. Comfort Bands (ASHRAE 55 Standards)
- **Standard Occupied Hours (08:00 – 18:00):** Maintain zone air temperature between **$20.0^\circ\text{C}$ and $24.5^\circ\text{C}$**.
- **Unoccupied Hours (18:00 – 08:00):** Setback allowed between **$18.0^\circ\text{C}$ and $27.0^\circ\text{C}$** for energy conservation.
- **Predicted Mean Vote (PMV):** All occupied zones must maintain $-0.5 \le \text{PMV} \le +0.5$ (Predicted Percentage Dissatisfied $\text{PPD} \le 10\%$).

### B. Equipment Longevity & Actuator Health
- **Chiller Minimum Dwell Time:** Chillers and DX compressors must remain in a given operating state for at least **15 minutes (3 simulation steps / 900 seconds)** to prevent short-cycling damage and thermal shock.
- **Maximum Setpoint Slew Rate:** Never change zone temperature setpoints faster than **$0.75^\circ\text{C}$ per 5-minute timestep** ($9.0^\circ\text{C}/\text{hour}$) to prevent duct pressure surging and VAV damper hunting.
- **Supply Chilled Water Temperature Range:** Maintain Chilled Water Supply (CHWST) between **$4.0^\circ\text{C}$ and $12.0^\circ\text{C}$**.

---

## 2. Dynamic Tariff & Real-World Pricing Configuration

### A. Wholesale LMP Ingestion
- **Standard Feed:** AETHERIS-Zero continuously polls CAISO / PJM / Indian Energy Exchange (IEX) day-ahead and 5-minute real-time LMP prices.
- **Currency Configuration:**
  - Base calculations: Universal USD ($) normalized.
  - Regional Localization: Live converted to Indian Rupee (₹ INR) at $1\text{ USD} = 83\text{ INR}$.
- **Pre-Cooling Timing:**
  - When off-peak prices are $\le \$0.10/\text{kWh}$ (₹8.30/kWh), the Virtual Battery engine pre-cools structural mass to $20.5^\circ\text{C} - 21.0^\circ\text{C}$ between **10:00 AM and 1:30 PM**.
  - During peak hours (**14:00 – 18:00** where prices exceed $\$0.50/\text{kWh}$ / ₹41.50/kWh), chiller electrical draw is curtailed by $40\% - 70\%$.

---

## 3. OpenADR 3.0 Demand Response (DR) Operation

### A. Virtual End Node (VEN) Registration
- **VEN Name:** `aetheris-ven-01`
- **Supported Programs:** `CAPACITY_BIDDING`, `DYNAMIC_LMP_PRICING`, `FAST_FREQUENCY_RESERVE`.
- **Target Curtailment:** Configured per facility (Default: $35.0\text{ kW}$ baseline reduction).

### B. Automated Dispatch Execution
1. Upon receipt of an OpenADR 3.0 `active` event signal from the Grid DRAS:
   - System registers price spike override immediately.
   - PPO Agent activates deep virtual battery discharge.
   - Safety Shield maintains strict floor of $24.5^\circ\text{C}$ max zone temperature.
2. At event completion:
   - Gradual recovery ramp ($0.5^\circ\text{C}$/step) is executed to avoid rebound coincident demand peaks.

---

## 4. Ingestion & Point Tagging Guidelines

### A. Tag Extraction from Legacy BMS
When connecting a new commercial building:
1. Export point register CSV dump containing point names, descriptions, and units.
2. Post CSV to `/api/v1/ingestion/upload-csv` or upload via the Ingestion API.
3. The Zero-Shot SLM Tag Parser standardizes points into:
   - `Zone_Air_Temperature_Sensor`
   - `Zone_Air_Temperature_Setpoint`
   - `Chilled_Water_Supply_Temperature_Setpoint`
   - `Damper_Position_Command`
   - `Electrical_Power_Sensor`
4. Review generated Brick Schema v1.3 RDF Turtle graph (`GET /api/v1/ingestion/graph-turtle`) before enabling autonomous actuation.

---

## 5. Daily Operator Procedures

1. **Morning System Health Check (08:00):**
   - Verify WebSocket connection status is **Live Connected** (Green badge).
   - Check that all 5 thermal zone tiles show **Comfortable / Compliant**.
2. **Pre-Peak Verification (13:30):**
   - Confirm pre-cooling phase has stored sufficient thermal energy (core zone at ~21°C).
   - Verify upcoming peak tariff schedule on the 24-hour grid price curve.
3. **End-of-Day Savings Audit (18:30):**
   - Check total accumulated financial savings (₹ INR) and CO₂ emissions avoided.
   - Run `/api/v1/simulation/predict-horizon` to inspect overnight thermal decay forecast.

---

## 6. Incident Response & Troubleshooting

| Symptom / Alert | Likely Cause | Recommended Operator Action |
| :--- | :--- | :--- |
| **Safety Shield Active (Yellow Banner)** | Thermostat target was manually set too high (>24.5°C) or too low (<20.0°C). | No action required; the CBF Shield automatically clamped the command to safe limits. |
| **Zone Warm Warning (>24.0°C during peak)** | Extreme outdoor heatwave exceeding chiller design capacity. | Use the Weather slider to verify load or temporarily increase chiller capacity setting. |
| **WebSocket Reconnecting** | Gateway service restart or network interruption. | Browser automatically reconnects every 2 seconds. Verify `run_aetheris.py` process is running. |
