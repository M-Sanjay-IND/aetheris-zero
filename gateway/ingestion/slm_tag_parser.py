"""
SLMTagParser: Zero-Shot Tag Parser & Semantic Normalizer for BACnet / Modbus Registers.
Normalizes raw, unstandardized point tags into canonical Brick Schema entity mappings.
"""

from __future__ import annotations

import csv
import io
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


@dataclass
class ParsedPoint:
    """Canonical representation of a parsed telemetry / control / parameter point."""
    raw_tag: str
    canonical_id: str
    brick_class: str
    point_role: str  # 'sensor', 'setpoint', 'command', 'parameter', 'meter', 'status'
    subsystem: str   # 'hvac', 'thermal_model', 'electrical', 'environment'
    equipment_type: Optional[str] = None  # 'Chiller', 'Air_Handling_Unit', 'Variable_Air_Volume_Box', etc.
    equipment_id: Optional[str] = None
    zone_id: Optional[str] = None
    value: Optional[float] = None
    unit: Optional[str] = None
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "raw_tag": self.raw_tag,
            "canonical_id": self.canonical_id,
            "brick_class": self.brick_class,
            "point_role": self.point_role,
            "subsystem": self.subsystem,
            "equipment_type": self.equipment_type,
            "equipment_id": self.equipment_id,
            "zone_id": self.zone_id,
            "value": self.value,
            "unit": self.unit,
            "confidence": self.confidence,
            "metadata": self.metadata,
        }


class SLMTagParser:
    """
    Zero-Shot semantic tag parser for messy BACnet, Modbus, and building automation tags.
    Supports heuristic tokenization, regular expression decomposition, semantic classification,
    and extensible hooks for local SLM/LLM inference.
    """

    # Vocabulary dictionaries for entity extraction
    ZONE_PATTERNS = {
        r"(?:z(?:one)?[\._\-]?0?1|core)": ("zone_1", "Core"),
        r"(?:z(?:one)?[\._\-]?0?2|north)": ("zone_2", "Perimeter North"),
        r"(?:z(?:one)?[\._\-]?0?3|south)": ("zone_3", "Perimeter South"),
        r"(?:z(?:one)?[\._\-]?0?4|east)": ("zone_4", "Perimeter East"),
        r"(?:z(?:one)?[\._\-]?0?5|west)": ("zone_5", "Perimeter West"),
    }

    EQUIPMENT_KEYWORDS = {
        "CHLR": "Chiller",
        "CHILLER": "Chiller",
        "AHU": "Air_Handling_Unit",
        "RTU": "Rooftop_Unit",
        "VAV": "Variable_Air_Volume_Box",
        "BLDG": "Building",
        "BUILDING": "Building",
        "BESS": "Battery_Energy_Storage_System",
        "EVSE": "Electric_Vehicle_Supply_Equipment",
    }

    POINT_PATTERNS = [
        # Building Level / Environmental
        (r"BLDG.*PWR.*KW|BUILDING.*POWER", "Electric_Power_Sensor", "meter", "electrical", "kW"),
        (r"OAT|OUTDOOR.*TEMP|WEATHER.*TEMP", "Outside_Air_Temperature_Sensor", "sensor", "environment", "deg_C"),
        (r"SOL.*IRRAD|SOLAR|GHI", "Solar_Radiance_Sensor", "sensor", "environment", "W/m2"),

        # Chiller Points
        (r"CHLR.*CHW.*STPT|CHILLER.*SETPOINT|CHW_STPT", "Chilled_Water_Supply_Temperature_Setpoint", "setpoint", "hvac", "deg_C"),
        (r"CHLR.*PWR|CHILLER.*KW", "Electric_Power_Sensor", "meter", "electrical", "kW"),
        (r"CHLR.*SUP.*TEMP|CHW.*SUP", "Chilled_Water_Supply_Temperature_Sensor", "sensor", "hvac", "deg_C"),
        (r"CHLR.*RET.*TEMP|CHW.*RET", "Chilled_Water_Return_Temperature_Sensor", "sensor", "hvac", "deg_C"),

        # AHU Points
        (r"AHU.*FAN.*PWR|FAN.*KW", "Electric_Power_Sensor", "meter", "electrical", "kW"),
        (r"AHU.*SUP.*AIR.*TEMP.*SP|AHU.*SAT.*SP", "Supply_Air_Temperature_Setpoint", "setpoint", "hvac", "deg_C"),
        (r"AHU.*SUP.*AIR.*TEMP|AHU.*SAT", "Supply_Air_Temperature_Sensor", "sensor", "hvac", "deg_C"),
        (r"AHU.*AIR.*FLOW|AIR_FLOW", "Air_Flow_Sensor", "sensor", "hvac", "m3/s"),

        # VAV & Zone Points
        (r"VAV.*DAT.*SP|DISCHARGE.*TEMP.*SP", "Discharge_Air_Temperature_Setpoint", "setpoint", "hvac", "deg_C"),
        (r"VAV.*DMPR.*POS|DAMPER.*POS", "Damper_Position_Command", "command", "hvac", "ratio"),
        (r"AIR.*TEMP.*SP|TEMP.*SP|ROOM.*TEMP.*SP", "Zone_Air_Temperature_Setpoint", "setpoint", "hvac", "deg_C"),
        (r"AIR.*TEMP|ROOM.*TEMP|ZONE.*TEMP", "Zone_Air_Temperature_Sensor", "sensor", "hvac", "deg_C"),
        (r"OCC.*COUNT|OCC.*SENS|OCCUPANCY", "Occupancy_Sensor", "sensor", "environment", "count"),

        # 2R2C Model Thermal Parameters
        (r"CZ.*PARAM|CAPACITANCE.*AIR", "Thermal_Capacitance_Air_Parameter", "parameter", "thermal_model", "kJ/K"),
        (r"CM.*PARAM|CAPACITANCE.*MASS", "Thermal_Capacitance_Mass_Parameter", "parameter", "thermal_model", "kJ/K"),
        (r"REXT.*PARAM|RESISTANCE.*EXT", "Envelope_Thermal_Resistance_Parameter", "parameter", "thermal_model", "K/kW"),
        (r"RM.*PARAM|RESISTANCE.*MASS", "Mass_Thermal_Resistance_Parameter", "parameter", "thermal_model", "K/kW"),
        (r"RADJ.*PARAM|RESISTANCE.*ADJ", "Interzone_Thermal_Resistance_Parameter", "parameter", "thermal_model", "K/kW"),
        (r"AREA.*SQM|FLOOR.*AREA", "Floor_Area_Parameter", "parameter", "thermal_model", "m2"),
        (r"SOLAR.*FRAC|SOLAR.*FACTOR", "Solar_Factor_Parameter", "parameter", "thermal_model", "ratio"),
    ]

    def __init__(self, use_slm_llm: bool = False, model_name: Optional[str] = None):
        self.use_slm_llm = use_slm_llm
        self.model_name = model_name

    def tokenize_tag(self, tag: str) -> List[str]:
        """Split a raw point tag by common delimiters and camelCase boundaries."""
        # Convert camelCase to snake_case
        s1 = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", tag)
        s2 = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s1)
        # Split on non-alphanumeric
        tokens = [t.upper() for t in re.split(r"[\._\-\s\/]+", s2) if t]
        return tokens

    def infer_zone(self, tag: str) -> Optional[tuple[str, str]]:
        """Identify if a tag belongs to a specific zone."""
        tag_upper = tag.upper()
        for pattern, (zone_id, zone_name) in self.ZONE_PATTERNS.items():
            if re.search(pattern, tag_upper, re.IGNORECASE):
                return zone_id, zone_name
        return None

    def infer_equipment(self, tag: str, zone_id: Optional[str] = None) -> tuple[Optional[str], Optional[str]]:
        """Identify equipment type and unique ID from tag."""
        tag_upper = tag.upper()
        if "CHLR" in tag_upper or "CHILLER" in tag_upper:
            return "Chiller", "Chiller1"
        if "AHU" in tag_upper:
            ahu_match = re.search(r"AHU(\d+)", tag_upper)
            ahu_id = f"AHU{ahu_match.group(1)}" if ahu_match else "AHU1"
            if "VAV" in tag_upper and zone_id:
                return "Variable_Air_Volume_Box", f"VAV_{zone_id.upper()}"
            return "Air_Handling_Unit", ahu_id
        if "VAV" in tag_upper:
            vav_id = f"VAV_{zone_id.upper()}" if zone_id else "VAV_Generic"
            return "Variable_Air_Volume_Box", vav_id
        if "BLDG" in tag_upper or "BUILDING" in tag_upper:
            return "Building", "CommercialOffice_5Zone"
        return None, None

    def parse_point_name(
        self,
        tag: str,
        value: Optional[Union[float, int, str]] = None,
        unit: Optional[str] = None,
        description: Optional[str] = None,
    ) -> ParsedPoint:
        """Parse a single raw point tag into a canonical ParsedPoint."""
        tag_clean = tag.strip()
        num_val: Optional[float] = None
        if value is not None:
            try:
                num_val = float(value)
            except (ValueError, TypeError):
                pass

        zone_match = self.infer_zone(tag_clean)
        zone_id = zone_match[0] if zone_match else None

        eq_type, eq_id = self.infer_equipment(tag_clean, zone_id)

        brick_class = "Point"
        point_role = "sensor"
        subsystem = "hvac"
        inferred_unit = unit or "unknown"
        confidence = 0.85
        metadata: Dict[str, Any] = {}

        if zone_match:
            metadata["zone_name"] = zone_match[1]

        # Match against Point Patterns
        for pat, b_class, role, sub, def_unit in self.POINT_PATTERNS:
            if re.search(pat, tag_clean, re.IGNORECASE):
                brick_class = b_class
                point_role = role
                subsystem = sub
                if not unit or unit == "unknown":
                    inferred_unit = def_unit
                confidence = 0.98
                break

        # Check for Inter-zone adjacency parameter: e.g., Z01_Z02_RADJ_PARAM
        adj_match = re.search(r"Z(?:ONE)?0?(\d+)_Z(?:ONE)?0?(\d+)_RADJ", tag_clean, re.IGNORECASE)
        if adj_match:
            from_z = f"zone_{adj_match.group(1)}"
            to_z = f"zone_{adj_match.group(2)}"
            metadata["from_zone"] = from_z
            metadata["to_zone"] = to_z
            metadata["param_key"] = "R_adj"
            brick_class = "Interzone_Thermal_Resistance_Parameter"
            point_role = "parameter"
            subsystem = "thermal_model"
            inferred_unit = "K/kW"
            confidence = 1.0

        # Parameter classification details
        if "CZ_PARAM" in tag_clean.upper():
            metadata["param_key"] = "C_z"
        elif "CM_PARAM" in tag_clean.upper():
            metadata["param_key"] = "C_m"
        elif "REXT_PARAM" in tag_clean.upper():
            metadata["param_key"] = "R_ext"
        elif "RM_PARAM" in tag_clean.upper():
            metadata["param_key"] = "R_m"
        elif "AREA_SQM" in tag_clean.upper():
            metadata["param_key"] = "floor_area_sqm"
        elif "SOLAR_FRAC" in tag_clean.upper():
            metadata["param_key"] = "solar_factor"

        if description:
            metadata["description"] = description

        return ParsedPoint(
            raw_tag=tag_clean,
            canonical_id=tag_clean,
            brick_class=brick_class,
            point_role=point_role,
            subsystem=subsystem,
            equipment_type=eq_type,
            equipment_id=eq_id,
            zone_id=zone_id,
            value=num_val,
            unit=inferred_unit,
            confidence=confidence,
            metadata=metadata,
        )

    def parse_csv(self, file_path_or_content: Union[str, Path, io.StringIO]) -> List[ParsedPoint]:
        """Parse a CSV dump containing raw BACnet point tags."""
        rows: List[Dict[str, str]] = []
        if isinstance(file_path_or_content, (str, Path)) and "\n" not in str(file_path_or_content) and Path(file_path_or_content).exists():
            with open(file_path_or_content, mode="r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
        elif isinstance(file_path_or_content, str):
            reader = csv.DictReader(io.StringIO(file_path_or_content))
            rows = list(reader)
        elif isinstance(file_path_or_content, io.StringIO):
            reader = csv.DictReader(file_path_or_content)
            rows = list(reader)

        results: List[ParsedPoint] = []
        for row in rows:
            point_name = row.get("point_name") or row.get("tag") or row.get("name") or ""
            if not point_name:
                continue
            val = row.get("value")
            unit = row.get("unit")
            desc = row.get("description")
            parsed = self.parse_point_name(tag=point_name, value=val, unit=unit, description=desc)
            results.append(parsed)

        return results

    def generate_slm_prompt(self, raw_tags: List[str]) -> str:
        """Construct a zero-shot prompt for an external SLM / LLM classifier."""
        tags_formatted = "\n".join([f"- {t}" for t in raw_tags])
        prompt = (
            "You are an expert building ontology specialist. Map each raw BACnet point tag below "
            "into Brick Schema v1.3 classes, equipment types, zone assignments, and point roles.\n\n"
            f"Input Tags:\n{tags_formatted}\n\n"
            "Output JSON format:\n"
            '{"mappings": [{"raw_tag": str, "brick_class": str, "equipment_type": str, "zone_id": str, "point_role": str}]}'
        )
        return prompt
