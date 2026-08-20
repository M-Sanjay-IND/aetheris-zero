**Dev 1: ML, Physics Simulation & Control Engineer** is solely responsible for building the underlying physics simulation, the neural surrogate predictive models, the safety filter, the reinforcement learning policy, and the live demonstration stress-test routines.

  

### Core Role & Ownership

- **Thermodynamic Environment:** Designing the multi-zone virtual building simulation that acts as the real-time ground truth for temperatures, power draw, and comfort metrics.
    
      
    
- **Neural Surrogate Modeling:** Building a fast surrogate model to predict future thermal states exponentially faster than classical building physics engines.
    
      
    
- **Hard Safety Invariance:** Implementing the real-time mathematical safety shield that intercepts and overrides unsafe control commands before they reach actuators.
    
      
    
- **Optimal Energy Arbitrage:** Developing the reinforcement learning agent that leverages the building's thermal mass as a virtual battery to pre-cool zones before peak utility pricing spikes.
    
      
    
- **Demonstration Stress Tests:** Crafting scripted fault-injection scenarios (malicious temperature overrides, extreme pricing spikes, equipment short-cycling attacks) for the live pitch.
    
      
    

### Phase-by-Phase Task Breakdown

**Phase 1: Environment & Physics Simulation Core (Hours 00–08)**

  

- **Multi-Zone Building Simulator:** Build a 5-zone state-space building thermal simulation engine tracking internal zone temperatures, ambient outdoor weather, solar radiation, occupancy schedules, and active HVAC heating/cooling loads.
    
      
    
- **Standardized Environment Interface:** Create standard initialization, step, and reset functions that take HVAC actuator actions as inputs and return updated thermal states, total power draw, and comfort indices.
    
      
    
- **Thermal Comfort & Energy Baselining:** Implement calculations for standard comfort compliance (Predicted Mean Vote / ASHRAE 55) and fixed-schedule baseline energy consumption to measure future efficiency gains.
    
      
    
- **Metadata Ingestion Pipeline:** Ensure the simulation configuration accepts structured building parameters (envelope thermal resistance, zone capacitance, spatial adjacencies) directly from Dev 2's ontology ingestion output.
    
      
    

**Phase 2: Surrogate Model & Safety Shield Architecture (Hours 08–20)**

  

- **Fast Predictive Surrogate:** Train a physics-informed surrogate model capable of predicting multi-zone thermal dynamics 24 hours into the future in under a few milliseconds.
    
      
    
- **Control Barrier Function (CBF) Shield:** Build a convex quadratic optimization safety filter that continuously evaluates proposed control actions against safe operating boundaries.
    
      
    
- **Hard Constraint Enforcement:**
    
      
    - Enforce strict occupant comfort bounds to keep zone temperatures within acceptable thermal limits.
        
          
        
    - Enforce mechanical equipment protection limits, including minimum equipment dwell times (preventing rapid chiller short-cycling) and actuator rate-of-change ceilings to avoid mechanical wear.
        
          
        
- **Action Intervention Logic:** Create a pass-through layer where any nominal control action is projected to the closest safe alternative if a boundary violation is detected.
    
      
    

**Phase 3: Safe Reinforcement Learning & Closed-Loop Controller (Hours 20–28)**

  

- **Energy Arbitrage Policy:** Develop and train a continuous reinforcement learning policy (such as PPO) optimized to minimize electricity costs under dynamic tariffs while maintaining comfort.
    
      
    
- **Thermal Mass Pre-Cooling Logic:** Tune the agent to recognize upcoming dynamic pricing events and autonomously pre-cool zones during cheap or high-renewable hours, storing thermal energy in the structural concrete and chilled water loops.
    
      
    
- **Safety-Shielded Training & Execution:** Wrap the policy output directly with the CBF safety shield so all exploratory and deployed actions are provably safe.
    
      
    
- **Dynamic Grid Signal Ingestion:** Connect the controller to real-time wholesale price trajectories and demand response signals supplied by Dev 2's grid API.
    
      
    

**Phase 4: Fault Injections, Demo Scenarios & Pitch Validation (Hours 28–36)**

  

- **Live Fault-Injection Scripting:** Build three interactive live-demo scenarios:
    
      
    1. _Standard Arbitrage:_ Showing the baseline schedule vs. autonomous pre-cooling load-shedding during peak pricing hours.
        
          
        
    2. _Malicious Setpoint Injection:_ Forcing an extreme setpoint override (e.g., trying to shut off cooling during peak occupancy) and showing the safety shield overriding it live.
        
          
        
    3. _Equipment Protection:_ Forcing high-frequency toggling to demonstrate the dwell-time limiter stopping chiller short-cycling.
        
          
        
- **Quantitative Benchmark Extraction:** Generate verifiable before-and-after comparison metrics (percentage peak demand shaved, total operational cost saved, and zero comfort violations achieved) to populate the frontend dashboard.
    
      
    
- **Pipeline Stress Testing:** Run end-to-end simulation loops with Dev 2’s WebSocket server to eliminate execution lag during the live pitch.
    
      
    

### Dev 1 Deliverables Handed to Dev 2

| **Deliverable**                 | **Description**                                                                            | **Used By Dev 2 For**                                                           |
| ------------------------------- | ------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------- |
| **Simulation Core Engine**      | Callable module returning time-series states, temperatures, and power draw.                | Hooking into the FastAPI backend and driving live WebSocket data streams.       |
| **Control & Safety Controller** | Packaged module that accepts pricing/weather and outputs safe actuator setpoints.          | Running the real-time optimization loop against OpenADR demand response events. |
| **Interactive Event Triggers**  | Executable functions for demand-response events, price spikes, and malicious faults.       | Wiring up UI buttons ("Trigger DR Event," "Inject Fault") on the dashboard.     |
| **Comparative Metrics**         | Structured output logs comparing baseline vs. optimized cost, kW load, and comfort bounds. | Powering the analytical charts and ROI displays on the frontend.                |

Repo Architecture
```
aetheris-zero/
├── README.md
├── docker-compose.yml
├── requirements.txt
│
├── core/                                   # [DEV 1 PRIMARY DOMAIN] ML, Physics & Safe-RL Engine
│   ├── __init__.py
│   ├── simulator/
│   │   ├── __init__.py
│   │   ├── building_etp.py                # 5-zone 2R2C state-space thermal differential equation solver[cite: 1]
│   │   ├── comfort.py                     # ASHRAE 55 Predicted Mean Vote (PMV) & PPD index calculation[cite: 1]
│   │   └── baseline_scheduler.py          # Fixed-schedule rule-based baseline comparator[cite: 1]
│   ├── models/
│   │   ├── __init__.py
│   │   ├── pinn_surrogate.py              # Physics-Informed Neural Network 24h forward state predictor[cite: 1]
│   │   └── fno_layers.py                  # Fourier Neural Operator temporal dynamics layer[cite: 1]
│   ├── safety/
│   │   ├── __init__.py
│   │   ├── cbf_shield.py                  # OSQP Quadratic Program Differentiable Safety Filter[cite: 1]
│   │   └── barrier_functions.py           # Comfort invariants & compressor dwell-time constraint math[cite: 1]
│   ├── controller/
│   │   ├── __init__.py
│   │   ├── ppo_agent.py                   # Continuous Proximal Policy Optimization RL policy[cite: 1]
│   │   └── arbitrage_engine.py            # Virtual battery structural pre-cooling coordinator[cite: 1]
│   └── scenarios/
│       ├── __init__.py
│       └── fault_injection.py             # Malicious setpoints, price spike triggers & trip tests[cite: 1]
│
├── gateway/                                # [DEV 2 PRIMARY DOMAIN] Ingestion, Grid Protocols & Backend
│   ├── __init__.py
│   ├── main.py                            # FastAPI entrypoint linking core engine to WebSocket streams
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── slm_tag_parser.py              # Zero-shot SLM BACnet register string normalizer[cite: 1]
│   │   ├── schema_builder.py              # Brick Schema RDF Turtle/JSON-LD graph generator[cite: 1]
│   │   └── sparql_extractor.py            # RDF graph query engine extracting building thermal priors[cite: 1]
│   ├── grid/
│   │   ├── __init__.py
│   │   ├── openadr_ven.py                 # OpenADR 3.0 Virtual End Node REST/JSON endpoint[cite: 1]
│   │   └── tariff_feed.py                 # Real-time CAISO/ERCOT wholesale LMP price generator[cite: 1]
│   └── streaming/
│       ├── __init__.py
│       ├── ws_manager.py                  # Real-time full-duplex WebSocket connection broker
│       └── telemetry_serializer.py        # Normalizer for step-by-step kW, temps & safety status[cite: 1]
│
├── dashboard/                              # [DEV 2 PRIMARY DOMAIN] Next.js + Three.js Digital Twin
│   ├── package.json
│   ├── tsconfig.json
│   ├── tailwind.config.js
│   ├── src/
│   │   ├── app/
│   │   │   ├── layout.tsx
│   │   │   └── page.tsx                   # Main split-screen operational control room layout[cite: 1]
│   │   ├── components/
│   │   │   ├── 3d/
│   │   │   │   ├── BuildingScene.tsx      # Three.js / React Three Fiber canvas viewport[cite: 1]
│   │   │   │   ├── ZoneThermalMesh.tsx    # Color-interpolated dynamic heatmaps per zone[cite: 1]
│   │   │   │   └── AirflowVectors.tsx     # Animated damper and chiller status indicators[cite: 1]
│   │   │   ├── analytics/
│   │   │   │   ├── DemandLoadChart.tsx    # Baseline vs. AETHERIS-Zero real-time load (kW)[cite: 1]
│   │   │   │   ├── PriceCurveChart.tsx    # Dynamic wholesale electricity LMP price curve[cite: 1]
│   │   │   │   └── SavingsCard.tsx        # Accumulated financial savings & carbon shaving[cite: 1]
│   │   │   └── controls/
│   │   │       ├── ActionPanel.tsx        # Buttons: Trigger OpenADR DR, Inject Fault, Toggle Shadow[cite: 1]
│   │   │       └── IngestionUpload.tsx    # Raw BACnet CSV upload & instant RDF graph renderer[cite: 1]
│   │   └── hooks/
│   │       └── useSimulationStream.ts     # WebSocket client hook updating live frontend states
│
└── data/                                   # Shared Sample Datasets & Mappings
    ├── raw_bacnet_dump.csv                # Messy unstandardized sensor point tags[cite: 1]
    ├── sample_caiso_lmp.json              # Historical wholesale electricity tariff curves[cite: 1]
    └── building_templates/
        └── 5zone_office.ttl               # Standard Brick Schema validation model[cite: 1]
```

### Branch Architecture & Git Strategy

To maintain continuous development without merge conflicts, both developers work out of dedicated feature branches, synchronize through an intermediate `staging` branch, and merge clean, tested milestones directly into `main` after every phase.

  

```
main (Pitch-Ready Golden Branch)
  └── staging (Phase Integration & End-to-End Testing)
        ├── dev1/p1-simulator-core      ├── dev2/p1-ingest-pricing
        ├── dev1/p2-pinn-cbf-shield     ├── dev2/p2-openadr-dashboard
        ├── dev1/p3-rl-controller       ├── dev2/p3-websocket-bridge
        └── dev1/p4-fault-scenarios     └── dev2/p4-3d-twin-polish
```


Techstack Utilization:
**Edge Runtime & Infrastructure**

* **Containerization & Deployment:** Docker, K3s.


* **Execution Runtimes:** Containerized Python, Rust Daemon.


* **Field Automation Protocols:** BACnet/IP (via `BAC0`), Modbus TCP (via `pymodbus`), MQTT Sparkplug B.



**Semantic Knowledge Graph & Ingestion**

* **Semantic Ontologies:** Brick Schema v1.3, Project Haystack.


* **Graph Databases & Processing:** `rdflib`, Neo4j, SPARQL.


* **Ingestion Architecture:** Fine-tuned Small Language Models (SLM) with Graph Retrieval-Augmented Generation (Graph RAG).



**Physics Modeling, Machine Learning & Safe Control**

* **Deep Learning Framework:** PyTorch.


* **Neural Surrogates:** Physics-Informed Neural Networks (PINN), Fourier Neural Operators (FNO).


* **Physics Dynamics:** $2R2C$ Equivalent Thermal Parameter (ETP) state-space differential equations.


* **Reinforcement Learning:** Proximal Policy Optimization (PPO), Ray RLlib.


* **Mathematical Optimization & Safety Filters:** Differentiable Control Barrier Functions (CBF-QP), `scipy.optimize` (OSQP).



**Grid Interoperability & Market Protocols**

* **Demand Response Standards:** OpenADR 3.0 (REST/JSON API for Virtual End Nodes).


* **Smart Grid Communication:** IEEE 2030.5.


* **Wholesale Market Integration:** Locational Marginal Pricing (LMP) feeds for CAISO and ERCOT.



**Frontend & 3D Digital Twin**

* **Web Framework & Styling:** Next.js, React, TailwindCSS.


* **3D Visualization:** Three.js, WebGL.