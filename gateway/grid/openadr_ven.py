"""
OpenADR 3.0 Virtual End Node (VEN) Gateway.
Implements OpenADR 3.0 REST/JSON data models for Demand Response Automation Servers (DRAS),
including automated event handling, load-curtailment dispatch, and verification reporting.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union
import uuid


class EventStatus(str, Enum):
    FAR = "FAR"              # Scheduled far in advance (> 24 hours)
    NEAR = "NEAR"            # Imminent event (< 2 hours, pre-cooling window)
    ACTIVE = "ACTIVE"        # Currently executing curtailment
    COMPLETED = "COMPLETED"  # Event has finished
    CANCELLED = "CANCELLED"  # Event cancelled by DRAS


class SignalType(str, Enum):
    PRICE_MULTIPLIER = "PRICE_MULTIPLIER"
    PRICE_ABSOLUTE = "PRICE_ABSOLUTE"
    SETPOINT = "SETPOINT"
    LOAD_DISPATCH = "LOAD_DISPATCH"


@dataclass
class OpenADREvent:
    """OpenADR 3.0 Event representation."""
    id: str = field(default_factory=lambda: f"evt_{uuid.uuid4().hex[:8]}")
    program_id: str = "CAISO_CRITICAL_PEAK_PRICING"
    event_name: str = "Emergency Demand Curtailment"
    start_hour: float = 14.0       # Hour of day (0.0 - 24.0)
    duration_hours: float = 4.0    # Duration in hours
    target_curtailment_kw: float = 35.0  # Desired peak kW reduction
    price_spike_usd: float = 1.50  # Dynamic tariff during event ($/kWh)
    signal_type: SignalType = SignalType.PRICE_ABSOLUTE
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @property
    def end_hour(self) -> float:
        return self.start_hour + self.duration_hours

    def get_status_at(self, current_hour: float) -> EventStatus:
        norm_hour = current_hour % 24.0
        if self.start_hour <= norm_hour < self.end_hour:
            return EventStatus.ACTIVE
        elif (self.start_hour - 2.0) <= norm_hour < self.start_hour:
            return EventStatus.NEAR
        elif norm_hour >= self.end_hour:
            return EventStatus.COMPLETED
        return EventStatus.FAR

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["end_hour"] = self.end_hour
        d["signal_type"] = self.signal_type.value
        return d


@dataclass
class OpenADRReport:
    """OpenADR 3.0 Telemetry and Settlement Compliance Report."""
    id: str = field(default_factory=lambda: f"rep_{uuid.uuid4().hex[:8]}")
    event_id: str = ""
    ven_id: str = "VEN_AETHERIS_001"
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    baseline_energy_kwh: float = 0.0
    actual_energy_kwh: float = 0.0
    energy_curtailed_kwh: float = 0.0
    peak_demand_shaved_kw: float = 0.0
    curtailment_compliance_pct: float = 100.0
    comfort_sla_compliance_pct: float = 98.5
    estimated_settlement_usd: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class OpenADRVEN:
    """
    OpenADR 3.0 Virtual End Node (VEN).
    Communicates with Utility DRAS, listens for demand response dispatch signals,
    coordinates with the BuildingSimulator / RL Controller, and generates settlement reports.
    """

    def __init__(
        self,
        ven_id: str = "VEN_AETHERIS_001",
        ven_name: str = "AETHERIS Zero Virtual Power Plant Node",
        dras_url: str = "https://dras.caiso-mock.net/openadr/v3",
    ):
        self.ven_id = ven_id
        self.ven_name = ven_name
        self.dras_url = dras_url
        self.events: Dict[str, OpenADREvent] = {}
        self.reports: Dict[str, OpenADRReport] = {}
        self.active_listeners: List[Callable[[OpenADREvent, EventStatus], None]] = []

        # Populate a default reference event matching the pitch scenario (14:00 - 18:00)
        default_evt = OpenADREvent(
            id="evt_caiso_peak_default",
            program_id="CAISO_CPP_2026",
            event_name="Summer Peak Demand Response Event",
            start_hour=14.0,
            duration_hours=4.0,
            target_curtailment_kw=35.0,
            price_spike_usd=1.50,
        )
        self.events[default_evt.id] = default_evt

    def register_listener(self, callback: Callable[[OpenADREvent, EventStatus], None]) -> None:
        """Register a callback to be triggered when event status transitions occur."""
        self.active_listeners.append(callback)

    def create_event(
        self,
        start_hour: float = 14.0,
        duration_hours: float = 4.0,
        target_curtailment_kw: float = 35.0,
        price_spike_usd: float = 1.50,
        event_name: str = "Dynamic DR Dispatch",
        program_id: str = "CAISO_DYNAMIC_VPP",
    ) -> OpenADREvent:
        """Create and register a new OpenADR 3.0 event."""
        event = OpenADREvent(
            program_id=program_id,
            event_name=event_name,
            start_hour=start_hour,
            duration_hours=duration_hours,
            target_curtailment_kw=target_curtailment_kw,
            price_spike_usd=price_spike_usd,
        )
        self.events[event.id] = event
        return event

    def get_event(self, event_id: str) -> Optional[OpenADREvent]:
        return self.events.get(event_id)

    def list_events(self, current_hour: Optional[float] = None) -> List[Dict[str, Any]]:
        """List all events with their live status at current_hour."""
        res = []
        for evt in self.events.values():
            d = evt.to_dict()
            if current_hour is not None:
                d["current_status"] = evt.get_status_at(current_hour).value
            res.append(d)
        return res

    def get_active_event(self, current_hour: float) -> Optional[OpenADREvent]:
        """Return the currently active OpenADR event if any."""
        for evt in self.events.values():
            if evt.get_status_at(current_hour) == EventStatus.ACTIVE:
                return evt
        return None

    def get_upcoming_event(self, current_hour: float) -> Optional[OpenADREvent]:
        """Return upcoming event in the NEAR pre-cooling window."""
        for evt in self.events.values():
            if evt.get_status_at(current_hour) == EventStatus.NEAR:
                return evt
        return None

    def cancel_event(self, event_id: str) -> bool:
        if event_id in self.events:
            del self.events[event_id]
            return True
        return False

    def generate_compliance_report(
        self,
        event_id: str,
        baseline_energy_kwh: float,
        actual_energy_kwh: float,
        peak_demand_shaved_kw: float,
        comfort_compliance_pct: float = 98.0,
        settlement_rate_per_kwh: float = 1.20,
    ) -> OpenADRReport:
        """Generate OpenADR 3.0 compliance and settlement report for DRAS submission."""
        curtailed = max(0.0, baseline_energy_kwh - actual_energy_kwh)
        event = self.get_event(event_id)
        target_kw = event.target_curtailment_kw if event else 30.0
        
        compliance_pct = 100.0
        if target_kw > 0 and peak_demand_shaved_kw > 0:
            compliance_pct = min(150.0, (peak_demand_shaved_kw / target_kw) * 100.0)

        settlement_usd = round(curtailed * settlement_rate_per_kwh, 2)

        report = OpenADRReport(
            event_id=event_id,
            ven_id=self.ven_id,
            baseline_energy_kwh=round(baseline_energy_kwh, 2),
            actual_energy_kwh=round(actual_energy_kwh, 2),
            energy_curtailed_kwh=round(curtailed, 2),
            peak_demand_shaved_kw=round(peak_demand_shaved_kw, 2),
            curtailment_compliance_pct=round(compliance_pct, 1),
            comfort_sla_compliance_pct=round(comfort_compliance_pct, 1),
            estimated_settlement_usd=settlement_usd,
        )
        self.reports[report.id] = report
        return report

    def list_reports(self) -> List[Dict[str, Any]]:
        return [rep.to_dict() for rep in self.reports.values()]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ven_id": self.ven_id,
            "ven_name": self.ven_name,
            "dras_url": self.dras_url,
            "total_events_registered": len(self.events),
            "total_reports_generated": len(self.reports),
        }
