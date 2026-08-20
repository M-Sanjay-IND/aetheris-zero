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


# File paths
DATA_DIR = Path(__file__).parent.parent / "data"
RAW_BACNET_CSV = DATA_DIR / "raw_bacnet_dump.csv"
SAMPLE_CAISO_JSON = DATA_DIR / "sample_caiso_lmp.json"
TEMPLATE_TTL = DATA_DIR / "building_templates" / "5zone_office.ttl"


class SimulationRuntime:
    """Global singleton maintaining the state of the active simulation and services."""
    def __init__(self):
        self.parser = SLMTagParser()
        self.builder = SchemaBuilder()
        self.extractor = SPARQLExtractor()
        self.serializer = TelemetrySerializer()
        self.tariff_feed = TariffFeed(json_file_path=SAMPLE_CAISO_JSON)
        self.openadr_ven = OpenADRVEN()
        self.baseline_scheduler = BaselineScheduler()
        
        self.sim_config: Dict[str, Any] = {}
        self.simulator: Optional[BuildingSimulator] = None
        self.is_running_auto_loop: bool = False
        self.shadow_mode: bool = False
        self.loop_task: Optional[asyncio.Task] = None
        self.step_delay_sec: float = 0.5  # Stream step speed for demo

    def initialize(self) -> None:
        """Initialize RDF ontology, extract config, and spin up BuildingSimulator."""
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

    def get_serialized_state(self) -> Dict[str, Any]:
        """Return serialized state frame."""
        if not self.simulator:
            return {}
        raw_state = self.simulator.get_state()
        active_evt = self.openadr_ven.get_active_event(self.simulator.current_hour)
        evt_id = active_evt.id if active_evt else None
        return self.serializer.serialize_state(raw_state, dr_event_id=evt_id)

    def step(self, actions: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Perform one simulation step and return serialized state."""
        if not self.simulator:
            raise RuntimeError("Simulator not initialized")

        if actions is None:
            # Use baseline scheduler actions or default setpoints
            actions = self.baseline_scheduler.get_actions(
                self.simulator.current_hour, self.simulator.zone_ids
            )

        raw_state, reward, done, info = self.simulator.step(actions)
        
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
        """Reset simulation state."""
        if self.simulator:
            self.simulator.reset()
        self.tariff_feed.reset()
        return self.get_serialized_state()


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
# Health & General Routes
# ---------------------------------------------------------------------------

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


class ToggleShadowRequest(BaseModel):
    enabled: bool


@app.post("/api/v1/control/toggle-shadow")
async def toggle_shadow_mode(payload: ToggleShadowRequest):
    """Dashboard Action: Toggle AI Shadow Mode."""
    runtime.shadow_mode = payload.enabled
    await ws_manager.broadcast({"type": "SHADOW_MODE_TOGGLED", "shadow_mode": runtime.shadow_mode})
    return {"status": "success", "shadow_mode": runtime.shadow_mode}


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


@app.get("/api/v1/simulation/state")
def get_simulation_state():
    """Get latest serialized telemetry state."""
    return {"status": "success", "telemetry": runtime.get_serialized_state()}


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
        "TRIGGER_OPENADR_EVENT": lambda p: runtime.openadr_ven.create_event(
            start_hour=p.get("start_hour", 14.0),
            duration_hours=p.get("duration_hours", 4.0),
            price_spike_usd=p.get("price_spike", 1.50),
        ).to_dict(),
        "INJECT_MALICIOUS_SETPOINT": lambda p: runtime.step({
            "zone_setpoints": {p.get("zone_id", "zone_1"): p.get("target_temp", 38.0)},
            "chiller_chw_setpoint": 6.5,
            "vav_damper_positions": {f"zone_{i}": 0.7 for i in range(1, 6)},
        }),
        "TOGGLE_SHADOW_MODE": lambda p: setattr(runtime, "shadow_mode", bool(p.get("enabled", True))),
        "RESET_SIMULATION": lambda p: runtime.reset(),
        "STEP_SIMULATION": lambda p: runtime.step(p.get("actions")),
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
