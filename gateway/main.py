"""
FastAPI Main Application Entrypoint for AETHERIS-Zero.
Exposes REST APIs for semantic ingestion, OpenADR 3.0 VEN, wholesale tariff feeds,
and high-frequency WebSocket streams for the Next.js / Three.js 3D digital twin dashboard.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Query, Body
from fastapi.responses import HTMLResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from gateway.ingestion.slm_tag_parser import SLMTagParser, ParsedPoint
from gateway.ingestion.schema_builder import SchemaBuilder
from gateway.ingestion.sparql_extractor import SPARQLExtractor
from gateway.grid.tariff_feed import TariffFeed
from gateway.grid.openadr_ven import OpenADRVEN, EventStatus
from gateway.streaming.telemetry_serializer import TelemetrySerializer
from gateway.streaming.ws_manager import ws_manager
from core.simulator.building_etp import BuildingSimulator
from core.simulator.baseline_scheduler import BaselineScheduler
from core.safety.cbf_shield import CBFShield
from core.models.pinn_surrogate import PINNSurrogate
from core.controller.ppo_agent import PPOAgent
from core.controller.arbitrage_engine import ArbitrageEngine


# File paths
DATA_DIR = Path(__file__).parent.parent / "data"
RAW_BACNET_CSV = DATA_DIR / "raw_bacnet_dump.csv"
SAMPLE_CAISO_JSON = DATA_DIR / "sample_caiso_lmp.json"
TEMPLATE_TTL = DATA_DIR / "building_templates" / "5zone_office.ttl"
DASHBOARD_HTML_PATH = Path(__file__).parent / "templates" / "dashboard.html"



class SimulationRuntime:
    """Global singleton maintaining the state of the active simulation, safety shield, RL arbitrage, and services."""
    def __init__(self):
        self.parser = SLMTagParser()
        self.builder = SchemaBuilder()
        self.extractor = SPARQLExtractor()
        self.serializer = TelemetrySerializer()
        self.tariff_feed = TariffFeed(json_file_path=SAMPLE_CAISO_JSON)
        self.openadr_ven = OpenADRVEN()
        self.baseline_scheduler = BaselineScheduler()
        self.cbf_shield = CBFShield(t_min=20.0, t_max=24.5, max_slew_per_step=0.75, min_dwell_steps=3)
        self.pinn_surrogate = PINNSurrogate(modes=8, width=32, num_layers=2)
        self.ppo_agent = PPOAgent()
        
        self.sim_config: Dict[str, Any] = {}
        self.simulator: Optional[BuildingSimulator] = None
        self.arbitrage_engine: Optional[ArbitrageEngine] = None
        
        self.controller_mode: str = "RL_SAFE_ARBITRAGE"  # "RL_SAFE_ARBITRAGE" | "BASELINE_HEURISTIC" | "SHADOW_MODE"
        self.last_shield_diagnostics: Dict[str, Any] = {
            "intervention_active": False,
            "shield_status": "OPTIMAL",
            "dwell_time_remaining_sec": 0,
            "solve_time_ms": 0.0,
            "active_constraints": []
        }
        self.is_running_auto_loop: bool = False
        self.loop_task: Optional[asyncio.Task] = None
        self.step_delay_sec: float = 0.25  # Stream step speed for demo

    def initialize(self) -> None:
        """Initialize RDF ontology, extract config, and spin up BuildingSimulator and ArbitrageEngine."""
        if TEMPLATE_TTL.exists():
            graph = self.builder.load_ttl(TEMPLATE_TTL)
        elif RAW_BACNET_CSV.exists():
            graph = self.builder.build_from_csv(RAW_BACNET_CSV)
        else:
            graph = self.builder.build_from_parsed_points([])

        self.sim_config = self.extractor.extract_building_config(graph)
        self.sim_config["tariff_feed"] = self.tariff_feed
        self.simulator = BuildingSimulator(config=self.sim_config)
        self.simulator.reset()
        self.cbf_shield.reset()
        
        self.arbitrage_engine = ArbitrageEngine(
            simulator=self.simulator,
            agent=self.ppo_agent,
            cbf_shield=self.cbf_shield,
            tariff_feed=self.tariff_feed
        )

    def get_serialized_state(self) -> Dict[str, Any]:
        """Return serialized state frame enriched with safety shield diagnostics."""
        if not self.simulator:
            return {}
        raw_state = self.simulator.get_state()
        
        # Inject live CBF shield diagnostics
        raw_state["safety"]["intervention_active"] = self.last_shield_diagnostics.get("intervention_active", False)
        raw_state["safety"]["shield_status"] = self.last_shield_diagnostics.get("shield_status", "OPTIMAL")
        raw_state["safety"]["dwell_time_remaining_sec"] = self.last_shield_diagnostics.get("dwell_time_remaining_sec", 0)
        raw_state["safety"]["solve_time_ms"] = self.last_shield_diagnostics.get("solve_time_ms", 1.15)
        raw_state["safety"]["active_constraints"] = self.last_shield_diagnostics.get("active_constraints", [])
        raw_state["safety"]["t_min_bound"] = self.cbf_shield.t_min
        raw_state["safety"]["t_max_bound"] = self.cbf_shield.t_max
        raw_state["safety"]["max_slew_per_step"] = self.cbf_shield.max_slew_per_step

        active_evt = self.openadr_ven.get_active_event(self.simulator.current_hour)
        evt_id = active_evt.id if active_evt else None
        return self.serializer.serialize_state(raw_state, dr_event_id=evt_id)


    def step(self, actions: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute one step through the active control mode (RL Arbitrage / Baseline / Custom override)."""
        if not self.simulator:
            raise RuntimeError("Simulator not initialized")

        current_raw_state = self.simulator.get_state()

        if actions is not None:
            # Custom Action (e.g. from Fault Injection) -> Pass through CBF Safety Shield
            safe_actions, diagnostics = self.cbf_shield.filter_action(
                state=current_raw_state,
                nominal_actions=actions,
                dt_sec=self.simulator.dt
            )
            self.last_shield_diagnostics = diagnostics
            self.simulator.step(safe_actions)
        elif self.controller_mode == "RL_SAFE_ARBITRAGE" and self.arbitrage_engine:
            # RL Transactive Arbitrage Closed Loop
            next_state, safe_actions, diagnostics = self.arbitrage_engine.step_closed_loop()
            self.last_shield_diagnostics = diagnostics
        else:
            # Baseline Heuristic Schedule
            nominal_actions = self.baseline_scheduler.get_actions(
                self.simulator.current_hour, self.simulator.zone_ids
            )
            safe_actions, diagnostics = self.cbf_shield.filter_action(
                state=current_raw_state,
                nominal_actions=nominal_actions,
                dt_sec=self.simulator.dt
            )
            self.last_shield_diagnostics = diagnostics
            self.simulator.step(safe_actions)
        
        # Check active OpenADR event sync
        active_evt = self.openadr_ven.get_active_event(self.simulator.current_hour)
        if active_evt:
            self.simulator.dr_event_active = True
            self.simulator.dr_price_override = active_evt.price_spike_usd
        else:
            self.simulator.dr_event_active = False
            self.simulator.dr_price_override = None

        return self.get_serialized_state()

    def reset(self) -> Dict[str, Any]:
        """Reset simulation and safety barrier states."""
        self.is_running_auto_loop = False
        if self.loop_task and not self.loop_task.done():
            self.loop_task.cancel()
        self.loop_task = None

        if self.arbitrage_engine:
            self.arbitrage_engine.reset()
        elif self.simulator:
            self.simulator.reset()
        self.cbf_shield.reset()
        self.tariff_feed.reset()
        self.last_shield_diagnostics = {
            "intervention_active": False,
            "shield_status": "OPTIMAL",
            "dwell_time_remaining_sec": 0,
            "solve_time_ms": 0.0,
            "active_constraints": []
        }
        return self.get_serialized_state()

    def run_episode(self, total_steps: int = 288) -> Dict[str, Any]:
        """Run full 24h simulation episode and return analytical ROI summary."""
        self.is_running_auto_loop = False
        if self.loop_task and not self.loop_task.done():
            self.loop_task.cancel()
        self.loop_task = None

        if not self.arbitrage_engine:
            self.initialize()
        results = self.arbitrage_engine.run_episode(total_steps=total_steps)
        return results

    def set_weather_override(self, ambient_temp: Optional[float] = None, solar_irradiance: Optional[float] = None) -> None:
        """Dynamically override weather conditions (ambient temp & solar irradiance)."""
        if self.simulator:
            self.simulator.set_weather_override(ambient_temp, solar_irradiance)

    def set_price_override(self, price_usd: Optional[float] = None) -> None:
        """Dynamically override electricity price."""
        if self.simulator:
            self.simulator.set_price_override(price_usd)

    def set_tariff_schedule(self, hourly_prices: List[float]) -> None:
        """Set custom 24-hour tariff schedule array."""
        if self.simulator:
            self.simulator.set_custom_tariff_schedule(hourly_prices)

    def set_zone_target(self, zone_id: str, target_temp: float) -> None:
        """Set individual zone target temperature."""
        if self.simulator:
            self.simulator.set_zone_target(zone_id, target_temp)

    def set_comfort_bounds(self, t_min: float = 20.0, t_max: float = 24.5, max_slew: float = 0.75, min_dwell_steps: int = 3) -> None:
        """Adjust CBF safety shield comfort and equipment constraints."""
        self.cbf_shield.t_min = float(t_min)
        self.cbf_shield.t_max = float(t_max)
        self.cbf_shield.max_slew_per_step = float(max_slew)
        self.cbf_shield.min_dwell_steps = int(min_dwell_steps)

    async def start_simulation_loop(self) -> None:
        """Start asynchronous background loop stepping simulation and broadcasting frames."""
        if self.is_running_auto_loop:
            return
        self.is_running_auto_loop = True
        await ws_manager.broadcast({"type": "LOOP_STARTED", "running": True})

        async def _loop_worker():
            try:
                while self.is_running_auto_loop:
                    state = self.step()
                    await ws_manager.broadcast({"type": "TELEMETRY_UPDATE", "telemetry": state})
                    await asyncio.sleep(self.step_delay_sec)
            except asyncio.CancelledError:
                pass
            finally:
                self.is_running_auto_loop = False
                await ws_manager.broadcast({"type": "LOOP_STOPPED", "running": False})

        self.loop_task = asyncio.create_task(_loop_worker())

    async def stop_simulation_loop(self) -> None:
        """Stop background simulation loop."""
        self.is_running_auto_loop = False
        if self.loop_task and not self.loop_task.done():
            self.loop_task.cancel()
            try:
                await self.loop_task
            except asyncio.CancelledError:
                pass
        self.loop_task = None
        await ws_manager.broadcast({"type": "LOOP_STOPPED", "running": False})


runtime = SimulationRuntime()



@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    runtime.initialize()
    yield
    # Shutdown
    if runtime.loop_task and not runtime.loop_task.done():
        runtime.loop_task.cancel()


app = FastAPI(
    title="AETHERIS-Zero Gateway API",
    description="Autonomous Physics-Informed Safe-RL & Transactive VPP Engine API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Health, Root & Digital Twin Dashboard Routes
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
@app.get("/dashboard", response_class=HTMLResponse)
def get_dashboard_html():
    """Serve the interactive AETHERIS-Zero Digital Twin Mission Control Dashboard."""
    if DASHBOARD_HTML_PATH.exists():
        with open(DASHBOARD_HTML_PATH, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(
        content="<h1>AETHERIS-Zero Digital Twin Dashboard</h1><p>Dashboard template not found.</p>",
        status_code=200
    )


@app.get("/favicon.ico", include_in_schema=False)
def get_favicon():
    return Response(status_code=204)


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "aetheris-gateway",
        "version": "1.0.0",
        "connected_ws_clients": ws_manager.client_count,
        "simulator_ready": runtime.simulator is not None,
    }



# ---------------------------------------------------------------------------
# Semantic Ingestion Routes
# ---------------------------------------------------------------------------

class ParseTagsRequest(BaseModel):
    tags: List[str]


@app.post("/api/v1/ingestion/parse-tags")
def parse_tags(payload: ParseTagsRequest):
    """Parse unstructured point tag strings into canonical Brick Schema entities."""
    parsed = [runtime.parser.parse_point_name(tag=t).to_dict() for t in payload.tags]
    return {"status": "success", "count": len(parsed), "points": parsed}


@app.post("/api/v1/ingestion/upload-csv")
def upload_csv_content(csv_content: str = Body(..., media_type="text/plain")):
    """Ingest raw CSV register dump, build Brick Schema graph, and extract simulation priors."""
    points = runtime.parser.parse_csv(csv_content)
    builder = SchemaBuilder()
    graph = builder.build_from_parsed_points(points)
    config = runtime.extractor.extract_building_config(graph)
    ttl_str = builder.serialize(format="turtle")
    return {
        "status": "success",
        "points_parsed": len(points),
        "triples_generated": len(graph),
        "building_config": config,
        "turtle": ttl_str,
    }


@app.get("/api/v1/ingestion/config")
def get_extracted_config():
    """Retrieve the validated 5-zone thermodynamic building config extracted via SPARQL."""
    return {"status": "success", "config": runtime.sim_config}


@app.get("/api/v1/ingestion/graph-turtle")
def get_graph_turtle():
    """Retrieve the current Brick Schema v1.3 RDF graph in Turtle format."""
    return {"status": "success", "turtle": runtime.builder.serialize(format="turtle")}


# ---------------------------------------------------------------------------
# Grid & Dynamic Tariff Routes
# ---------------------------------------------------------------------------

@app.get("/api/v1/grid/tariff/current")
def get_current_tariff():
    """Retrieve instantaneous wholesale electricity price ($/kWh)."""
    current_hour = runtime.simulator.current_hour if runtime.simulator else 12.0
    price = runtime.tariff_feed.get_price_by_hour(current_hour)
    return {
        "hour": round(current_hour, 2),
        "price_usd_per_kwh": price,
        "price_usd_per_mwh": round(price * 1000.0, 2),
    }


@app.get("/api/v1/grid/tariff/chart")
def get_tariff_chart(resolution_minutes: int = Query(default=15)):
    """Retrieve 24-hour tariff profile for frontend charting."""
    data = runtime.tariff_feed.to_chart_data(resolution_minutes=resolution_minutes)
    return {"status": "success", "resolution_minutes": resolution_minutes, "chart_data": data}


class PriceSpikeRequest(BaseModel):
    start_hour: float = Field(default=14.0, description="Start hour of spike [0-23]")
    duration_hours: float = Field(default=4.0, description="Duration in hours")
    spike_price: float = Field(default=1.50, description="Spike price in $/kWh")


@app.post("/api/v1/grid/tariff/inject-spike")
def inject_price_spike(payload: PriceSpikeRequest):
    """Dynamically inject an extreme price spike event into the tariff feed."""
    runtime.tariff_feed.inject_spike(
        start_hour=payload.start_hour,
        duration_hours=payload.duration_hours,
        spike_price=payload.spike_price,
    )
    return {"status": "success", "message": f"Injected spike ${payload.spike_price}/kWh from hour {payload.start_hour} for {payload.duration_hours}h"}


# ---------------------------------------------------------------------------
# OpenADR 3.0 Virtual End Node (VEN) Routes
# ---------------------------------------------------------------------------

@app.get("/api/v1/grid/openadr/ven")
def get_openadr_ven_info():
    """Retrieve OpenADR 3.0 VEN registration info."""
    return {"status": "success", "ven": runtime.openadr_ven.to_dict()}


@app.get("/api/v1/grid/openadr/events")
def list_openadr_events():
    """List all registered OpenADR 3.0 demand response events."""
    current_hour = runtime.simulator.current_hour if runtime.simulator else 0.0
    events = runtime.openadr_ven.list_events(current_hour=current_hour)
    return {"status": "success", "events": events}


class CreateEventRequest(BaseModel):
    start_hour: float = 14.0
    duration_hours: float = 4.0
    target_curtailment_kw: float = 35.0
    price_spike_usd: float = 1.50
    event_name: str = "Live Pitch DR Event"


@app.post("/api/v1/grid/openadr/events")
def create_openadr_event(payload: CreateEventRequest):
    """Register a new OpenADR 3.0 Demand Response Event."""
    event = runtime.openadr_ven.create_event(
        start_hour=payload.start_hour,
        duration_hours=payload.duration_hours,
        target_curtailment_kw=payload.target_curtailment_kw,
        price_spike_usd=payload.price_spike_usd,
        event_name=payload.event_name,
    )
    # Also mirror into tariff feed
    runtime.tariff_feed.inject_spike(payload.start_hour, payload.duration_hours, payload.price_spike_usd)
    return {"status": "success", "event": event.to_dict()}


@app.get("/api/v1/grid/openadr/reports")
def list_openadr_reports():
    """List all settlement and compliance verification reports."""
    return {"status": "success", "reports": runtime.openadr_ven.list_reports()}


# ---------------------------------------------------------------------------
# Interactive Simulation & Fault Injection Control Routes
# ---------------------------------------------------------------------------

class TriggerDRRequest(BaseModel):
    price_spike: float = 1.50
    start_hour: float = 14.0
    duration_hours: float = 4.0


@app.post("/api/v1/control/trigger-dr")
async def trigger_dr_event(payload: TriggerDRRequest):
    """Dashboard Action: Trigger OpenADR 3.0 Demand Response Event."""
    evt = runtime.openadr_ven.create_event(
        start_hour=payload.start_hour,
        duration_hours=payload.duration_hours,
        price_spike_usd=payload.price_spike,
        event_name="Interactive Demo DR Dispatch",
    )
    runtime.tariff_feed.inject_spike(payload.start_hour, payload.duration_hours, payload.price_spike)
    state = runtime.get_serialized_state()
    await ws_manager.broadcast({"type": "EVENT_TRIGGERED", "event": evt.to_dict(), "telemetry": state})
    return {"status": "success", "event": evt.to_dict(), "telemetry": state}


class InjectFaultRequest(BaseModel):
    zone_id: str = "zone_1"
    target_temp: float = 38.0


@app.post("/api/v1/control/inject-fault")
async def inject_malicious_setpoint(payload: InjectFaultRequest):
    """Dashboard Action: Inject malicious temperature setpoint override to test CBF safety shield."""
    if not runtime.simulator:
        raise HTTPException(status_code=500, detail="Simulator not ready")
    
    # Formulate action override
    actions = runtime.baseline_scheduler.get_actions(
        runtime.simulator.current_hour, runtime.simulator.zone_ids
    )
    actions["zone_setpoints"][payload.zone_id] = payload.target_temp
    
    state = runtime.step(actions)
    await ws_manager.broadcast({"type": "FAULT_INJECTED", "zone_id": payload.zone_id, "target_temp": payload.target_temp, "telemetry": state})
    return {"status": "success", "zone_id": payload.zone_id, "target_temp": payload.target_temp, "telemetry": state}


class SetModeRequest(BaseModel):
    mode: str = Field(..., description="'RL_SAFE_ARBITRAGE' | 'BASELINE_HEURISTIC' | 'SHADOW_MODE'")


@app.post("/api/v1/control/set-mode")
async def set_controller_mode(payload: SetModeRequest):
    """Set active control mode: RL_SAFE_ARBITRAGE, BASELINE_HEURISTIC, or SHADOW_MODE."""
    mode = payload.mode.upper()
    if mode not in ["RL_SAFE_ARBITRAGE", "BASELINE_HEURISTIC", "SHADOW_MODE"]:
        raise HTTPException(status_code=400, detail=f"Invalid controller mode: {payload.mode}")
    runtime.controller_mode = mode
    await ws_manager.broadcast({"type": "MODE_CHANGED", "controller_mode": runtime.controller_mode})
    return {"status": "success", "controller_mode": runtime.controller_mode}


class ToggleShadowRequest(BaseModel):
    enabled: bool


@app.post("/api/v1/control/toggle-shadow")
async def toggle_shadow_mode(payload: ToggleShadowRequest):
    """Dashboard Action: Toggle AI Shadow Mode."""
    runtime.controller_mode = "SHADOW_MODE" if payload.enabled else "RL_SAFE_ARBITRAGE"
    await ws_manager.broadcast({"type": "SHADOW_MODE_TOGGLED", "shadow_mode": payload.enabled, "controller_mode": runtime.controller_mode})
    return {"status": "success", "shadow_mode": payload.enabled, "controller_mode": runtime.controller_mode}



class WeatherOverrideRequest(BaseModel):
    ambient_temp_c: Optional[float] = Field(default=None, description="Outdoor temperature in °C")
    solar_irradiance_wm2: Optional[float] = Field(default=None, description="Solar irradiance in W/m²")


@app.post("/api/v1/control/set-weather")
async def set_weather(payload: WeatherOverrideRequest):
    """Dynamically update real-world outdoor weather / ambient temperature."""
    runtime.set_weather_override(payload.ambient_temp_c, payload.solar_irradiance_wm2)
    state = runtime.step()
    await ws_manager.broadcast({"type": "TELEMETRY_UPDATE", "telemetry": state})
    return {"status": "success", "ambient_temp_c": payload.ambient_temp_c, "solar_irradiance_wm2": payload.solar_irradiance_wm2, "telemetry": state}


class PriceOverrideRequest(BaseModel):
    price_usd_per_kwh: Optional[float] = Field(default=None, description="Electricity price in $/kWh")


@app.post("/api/v1/control/set-pricing")
async def set_pricing(payload: PriceOverrideRequest):
    """Dynamically update current electricity price in real-time."""
    runtime.set_price_override(payload.price_usd_per_kwh)
    state = runtime.step()
    await ws_manager.broadcast({"type": "TELEMETRY_UPDATE", "telemetry": state})
    return {"status": "success", "price_usd_per_kwh": payload.price_usd_per_kwh, "telemetry": state}


class CustomTariffScheduleRequest(BaseModel):
    hourly_prices: List[float] = Field(..., description="24-hour tariff schedule in $/kWh")


@app.post("/api/v1/control/set-tariff-schedule")
async def set_tariff_schedule(payload: CustomTariffScheduleRequest):
    """Update full 24-hour wholesale electricity pricing array."""
    runtime.set_tariff_schedule(payload.hourly_prices)
    state = runtime.step()
    await ws_manager.broadcast({"type": "TELEMETRY_UPDATE", "telemetry": state})
    return {"status": "success", "hourly_prices": payload.hourly_prices, "telemetry": state}


class ZoneTargetRequest(BaseModel):
    zone_id: str
    target_temp: float


@app.post("/api/v1/control/set-zone-target")
async def set_zone_target(payload: ZoneTargetRequest):
    """Set custom comfort temperature setpoint for a specific zone."""
    runtime.set_zone_target(payload.zone_id, payload.target_temp)
    state = runtime.step({"zone_setpoints": {payload.zone_id: payload.target_temp}})
    await ws_manager.broadcast({"type": "TELEMETRY_UPDATE", "telemetry": state})
    return {"status": "success", "zone_id": payload.zone_id, "target_temp": payload.target_temp, "telemetry": state}



class ComfortBoundsRequest(BaseModel):
    t_min: float = 20.0
    t_max: float = 24.5
    max_slew_per_step: float = 0.75
    min_dwell_steps: int = 3


@app.post("/api/v1/control/set-comfort-bounds")
async def set_comfort_bounds(payload: ComfortBoundsRequest):
    """Configure safety shield comfort thresholds and equipment slew rate limits."""
    runtime.set_comfort_bounds(payload.t_min, payload.t_max, payload.max_slew_per_step, payload.min_dwell_steps)
    return {"status": "success", "bounds": payload.dict()}


class StepRequest(BaseModel):
    actions: Optional[Dict[str, Any]] = None


@app.post("/api/v1/control/step")
async def step_simulation(payload: Optional[StepRequest] = None):
    """Step the simulation engine one step forward."""
    actions = payload.actions if payload else None
    state = runtime.step(actions)
    await ws_manager.broadcast({"type": "TELEMETRY_UPDATE", "telemetry": state})
    return {"status": "success", "telemetry": state}


@app.post("/api/v1/control/reset")
async def reset_simulation():
    """Reset the simulation environment."""
    state = runtime.reset()
    await ws_manager.broadcast({"type": "SIMULATION_RESET", "telemetry": state})
    return {"status": "success", "telemetry": state}


@app.post("/api/v1/simulation/start")
async def start_simulation_auto_loop():
    """Start continuous real-time simulation loop broadcasting to WebSocket clients."""
    await runtime.start_simulation_loop()
    await ws_manager.broadcast({"type": "LOOP_STARTED", "running": runtime.is_running_auto_loop})
    return {"status": "success", "running": runtime.is_running_auto_loop}


@app.post("/api/v1/simulation/stop")
async def stop_simulation_auto_loop():
    """Stop continuous real-time simulation loop."""
    await runtime.stop_simulation_loop()
    await ws_manager.broadcast({"type": "LOOP_STOPPED", "running": runtime.is_running_auto_loop})
    return {"status": "success", "running": runtime.is_running_auto_loop}


class RunEpisodeRequest(BaseModel):
    total_steps: int = Field(default=288, ge=12, le=576)


@app.post("/api/v1/simulation/run-episode")
async def run_simulation_episode(payload: Optional[RunEpisodeRequest] = None):
    """Execute a fast-forward full 24h simulation episode and return ROI summary."""
    steps = payload.total_steps if payload else 288
    results = runtime.run_episode(total_steps=steps)
    await ws_manager.broadcast({"type": "EPISODE_SUMMARY", "summary": results["summary"]})
    return {"status": "success", "results": results}


@app.get("/api/v1/simulation/status")
def get_simulation_status():
    """Get active runtime loop status and controller configuration."""
    return {
        "status": "success",
        "running": runtime.is_running_auto_loop,
        "controller_mode": runtime.controller_mode,
        "step": runtime.simulator.current_step if runtime.simulator else 0,
        "hour": round(runtime.simulator.current_hour, 2) if runtime.simulator else 0.0,
        "connected_ws_clients": ws_manager.client_count,
    }


@app.get("/api/v1/simulation/state")
def get_simulation_state():
    """Get latest serialized telemetry state."""
    return {"status": "success", "telemetry": runtime.get_serialized_state()}


@app.get("/api/v1/simulation/predict-horizon")
def predict_horizon(horizon_steps: int = Query(default=96, ge=1, le=288)):
    """
    Use Dev 1's PINN-FNO Neural Surrogate to predict 24h future multi-zone
    thermal state trajectories in < 5ms.
    """
    if not runtime.simulator:
        raise HTTPException(status_code=500, detail="Simulator not initialized")
    
    current_state = runtime.simulator.get_state()
    pred_trajectory = runtime.pinn_surrogate.predict_horizon(
        current_state=current_state,
        horizon_steps=horizon_steps,
        dt_sec=runtime.simulator.dt
    )
    zone_keys = list(current_state.get("zones", {}).keys()) or [f"zone_{i}" for i in range(1, 6)]
    
    timeline = []
    for step_idx in range(horizon_steps):
        step_hour = current_state["timestamp_hour"] + (step_idx * runtime.simulator.dt / 3600.0)
        step_entry = {
            "step": step_idx,
            "hour": round(step_hour % 24.0, 2),
            "zones": {zk: round(float(pred_trajectory[step_idx, z_i]), 2) for z_i, zk in enumerate(zone_keys)}
        }
        timeline.append(step_entry)
        
    return {
        "status": "success",
        "horizon_steps": horizon_steps,
        "prediction_timeline": timeline
    }


# ---------------------------------------------------------------------------
# WebSocket Telemetry Stream
# ---------------------------------------------------------------------------

@app.websocket("/ws/telemetry")
async def websocket_telemetry_endpoint(websocket: WebSocket):
    """
    High-frequency full-duplex WebSocket stream for 3D Three.js digital twin
    and real-time analytics dashboards.
    """
    await ws_manager.connect(websocket)

    # Handlers for client-side actions
    command_handlers = {
        "TRIGGER_OPENADR_EVENT": lambda p: (
            runtime.openadr_ven.create_event(
                start_hour=p.get("start_hour", 14.0),
                duration_hours=p.get("duration_hours", 4.0),
                price_spike_usd=p.get("price_spike", 1.50),
            ),
            runtime.tariff_feed.inject_spike(p.get("start_hour", 14.0), p.get("duration_hours", 4.0), p.get("price_spike", 1.50)),
            runtime.step()
        )[-1],
        "INJECT_MALICIOUS_SETPOINT": lambda p: (
            runtime.set_zone_target(p.get("zone_id", "zone_1"), p.get("target_temp", 38.0)),
            runtime.step({
                "zone_setpoints": {p.get("zone_id", "zone_1"): p.get("target_temp", 38.0)},
                "chiller_chw_setpoint": 6.5,
                "vav_damper_positions": {f"zone_{i}": 0.7 for i in range(1, 6)},
            })
        )[-1],
        "INJECT_DWELL_ATTACK": lambda p: runtime.step({
            "zone_setpoints": {f"zone_{i}": 22.0 for i in range(1, 6)},
            "chiller_chw_setpoint": 4.0 if runtime.simulator.current_step % 2 == 0 else 12.0,
            "vav_damper_positions": {f"zone_{i}": 0.7 for i in range(1, 6)},
        }),
        "SET_WEATHER_OVERRIDE": lambda p: (
            runtime.set_weather_override(p.get("ambient_temp_c"), p.get("solar_irradiance_wm2")),
            runtime.step()
        )[-1],
        "SET_PRICE_OVERRIDE": lambda p: (
            runtime.set_price_override(p.get("price_usd_per_kwh")),
            runtime.step()
        )[-1],
        "SET_TARIFF_SCHEDULE": lambda p: (
            runtime.set_tariff_schedule(p.get("hourly_prices", [])),
            runtime.step()
        )[-1],
        "SET_ZONE_TARGET": lambda p: (
            runtime.set_zone_target(p.get("zone_id", "zone_1"), p.get("target_temp", 22.0)),
            runtime.step({"zone_setpoints": {p.get("zone_id", "zone_1"): p.get("target_temp", 22.0)}})
        )[-1],
        "SET_COMFORT_BOUNDS": lambda p: runtime.set_comfort_bounds(p.get("t_min", 20.0), p.get("t_max", 24.5), p.get("max_slew", 0.75), p.get("min_dwell", 3)),
        "SET_CONTROLLER_MODE": lambda p: setattr(runtime, "controller_mode", str(p.get("mode", "RL_SAFE_ARBITRAGE")).upper()),
        "TOGGLE_SHADOW_MODE": lambda p: setattr(runtime, "controller_mode", "SHADOW_MODE" if p.get("enabled", True) else "RL_SAFE_ARBITRAGE"),
        "RESET_SIMULATION": lambda p: runtime.reset(),
        "STEP_SIMULATION": lambda p: runtime.step(p.get("actions")),
        "START_SIMULATION": lambda p: runtime.start_simulation_loop(),
        "STOP_SIMULATION": lambda p: runtime.stop_simulation_loop(),
        "RUN_EPISODE": lambda p: runtime.run_episode(total_steps=int(p.get("total_steps", 288))),
    }



    try:
        # Send initial state immediately
        initial_frame = runtime.get_serialized_state()
        await websocket.send_json({"type": "INITIAL_STATE", "telemetry": initial_frame})

        while True:
            data = await websocket.receive_json()
            response = await ws_manager.handle_client_command(data, command_handlers)
            
            # Broadcast state update if a step/action occurred
            if response.get("status") == "SUCCESS":
                current_state = runtime.get_serialized_state()
                await ws_manager.broadcast({"type": "TELEMETRY_UPDATE", "telemetry": current_state})
            else:
                await ws_manager.send_personal_message(response, websocket)

    except WebSocketDisconnect:
        await ws_manager.disconnect(websocket)
    except Exception as e:
        await ws_manager.disconnect(websocket)

