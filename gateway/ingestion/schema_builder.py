"""
SchemaBuilder: RDF Knowledge Graph Generator conforming to Brick Schema v1.3.
Constructs multi-zone building topologies, HVAC feeding hierarchies, and 2R2C thermal parameters.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Union

import rdflib
from rdflib import Graph, Literal, Namespace, RDF, RDFS, URIRef
from rdflib.namespace import XSD

from gateway.ingestion.slm_tag_parser import ParsedPoint, SLMTagParser


# Namespaces
BRICK = Namespace("https://brickschema.org/schema/Brick#")
EX = Namespace("http://example.com/building#")


class SchemaBuilder:
    """
    Constructs and serializes Brick Schema v1.3 RDF knowledge graphs.
    Maps parsed sensor/actuator points and structural thermal metadata into a queryable semantic graph.
    """

    def __init__(self, building_id: str = "CommercialOffice_5Zone", building_label: str = "5-Zone Commercial Office"):
        self.building_id = building_id
        self.building_label = building_label
        self.graph = Graph()
        self._bind_namespaces()

    def _bind_namespaces(self) -> None:
        """Bind standard ontology namespaces to graph."""
        self.graph.bind("brick", BRICK)
        self.graph.bind("ex", EX)
        self.graph.bind("rdf", RDF)
        self.graph.bind("rdfs", RDFS)
        self.graph.bind("xsd", XSD)

    def load_ttl(self, file_path: Union[str, Path]) -> Graph:
        """Load an existing Turtle (.ttl) model into graph."""
        self.graph.parse(str(file_path), format="turtle")
        return self.graph

    def build_from_parsed_points(
        self,
        parsed_points: List[ParsedPoint],
        default_zones: Optional[Dict[str, Dict]] = None,
    ) -> Graph:
        """
        Build complete Brick RDF graph from parsed telemetry and parameter points.
        """
        bldg_uri = EX[self.building_id]
        self.graph.add((bldg_uri, RDF.type, BRICK.Building))
        self.graph.add((bldg_uri, RDFS.label, Literal(self.building_label, datatype=XSD.string)))

        # Default 5-Zone Specifications (if not fully overridden by parameter points)
        zone_defaults = default_zones or {
            "zone_1": {
                "name": "Core Zone", "C_z": 15000.0, "C_m": 85000.0, "R_ext": 2.5, "R_m": 0.5,
                "floor_area_sqm": 250.0, "solar_factor": 0.05, "occupancy_max": 20,
                "initial_temp": 22.5, "initial_mass_temp": 22.0,
                "adjacencies": ["zone_2", "zone_3", "zone_4", "zone_5"]
            },
            "zone_2": {
                "name": "Perimeter North Zone", "C_z": 12000.0, "C_m": 70000.0, "R_ext": 1.8, "R_m": 0.6,
                "floor_area_sqm": 180.0, "solar_factor": 0.15, "occupancy_max": 12,
                "initial_temp": 22.0, "initial_mass_temp": 21.8,
                "adjacencies": ["zone_1"]
            },
            "zone_3": {
                "name": "Perimeter South Zone", "C_z": 12000.0, "C_m": 70000.0, "R_ext": 1.6, "R_m": 0.6,
                "floor_area_sqm": 180.0, "solar_factor": 0.35, "occupancy_max": 15,
                "initial_temp": 22.5, "initial_mass_temp": 22.2,
                "adjacencies": ["zone_1"]
            },
            "zone_4": {
                "name": "Perimeter East Zone", "C_z": 11000.0, "C_m": 65000.0, "R_ext": 1.7, "R_m": 0.6,
                "floor_area_sqm": 160.0, "solar_factor": 0.25, "occupancy_max": 10,
                "initial_temp": 22.0, "initial_mass_temp": 21.9,
                "adjacencies": ["zone_1"]
            },
            "zone_5": {
                "name": "Perimeter West Zone", "C_z": 11000.0, "C_m": 65000.0, "R_ext": 1.7, "R_m": 0.6,
                "floor_area_sqm": 160.0, "solar_factor": 0.25, "occupancy_max": 14,
                "initial_temp": 22.2, "initial_mass_temp": 22.0,
                "adjacencies": ["zone_1"]
            },
        }

        # Override zone parameters with parsed parameter points
        for p in parsed_points:
            if p.point_role == "parameter" and p.zone_id and p.zone_id in zone_defaults:
                param_key = p.metadata.get("param_key")
                if param_key and p.value is not None:
                    zone_defaults[p.zone_id][param_key] = p.value

        # Instantiate Zones
        for z_id, z_data in zone_defaults.items():
            z_uri = EX[f"Zone_{z_id.split('_')[-1]}"]
            self.graph.add((z_uri, RDF.type, BRICK.HVAC_Zone))
            self.graph.add((z_uri, RDFS.label, Literal(z_data.get("name", z_id), datatype=XSD.string)))
            self.graph.add((z_uri, EX.zoneId, Literal(z_id, datatype=XSD.string)))

            # Thermal Parameters (2R2C Model)
            self.graph.add((z_uri, EX.thermalCapacitanceAir, Literal(float(z_data["C_z"]), datatype=XSD.float)))
            self.graph.add((z_uri, EX.thermalCapacitanceMass, Literal(float(z_data["C_m"]), datatype=XSD.float)))
            self.graph.add((z_uri, EX.envelopeResistance, Literal(float(z_data["R_ext"]), datatype=XSD.float)))
            self.graph.add((z_uri, EX.internalMassResistance, Literal(float(z_data["R_m"]), datatype=XSD.float)))
            self.graph.add((z_uri, EX.floorAreaSqm, Literal(float(z_data["floor_area_sqm"]), datatype=XSD.float)))
            self.graph.add((z_uri, EX.solarFactor, Literal(float(z_data["solar_factor"]), datatype=XSD.float)))
            self.graph.add((z_uri, EX.occupancyMax, Literal(int(z_data["occupancy_max"]), datatype=XSD.integer)))
            self.graph.add((z_uri, EX.initialTemp, Literal(float(z_data["initial_temp"]), datatype=XSD.float)))
            self.graph.add((z_uri, EX.initialMassTemp, Literal(float(z_data["initial_mass_temp"]), datatype=XSD.float)))

            # Adjacencies
            for adj_id in z_data.get("adjacencies", []):
                adj_uri = EX[f"Zone_{adj_id.split('_')[-1]}"]
                self.graph.add((z_uri, BRICK.hasAdjacentZone, adj_uri))

        # Inter-zone adjacency resistances
        radj_overrides: Dict[tuple[str, str], float] = {}
        for p in parsed_points:
            if p.brick_class == "Interzone_Thermal_Resistance_Parameter" and p.value is not None:
                from_z = p.metadata.get("from_zone")
                to_z = p.metadata.get("to_zone")
                if from_z and to_z:
                    radj_overrides[(from_z, to_z)] = p.value

        # Default adjacencies if not in parsed points
        if not radj_overrides:
            radj_overrides = {
                ("zone_1", "zone_2"): 1.2,
                ("zone_1", "zone_3"): 1.2,
                ("zone_1", "zone_4"): 1.2,
                ("zone_1", "zone_5"): 1.2,
            }

        for (from_z, to_z), r_val in radj_overrides.items():
            from_idx = from_z.split('_')[-1]
            to_idx = to_z.split('_')[-1]
            adj_node = EX[f"Adj_Z0{from_idx}_Z0{to_idx}"]
            self.graph.add((adj_node, RDF.type, EX.ZoneAdjacency))
            self.graph.add((adj_node, EX.fromZone, EX[f"Zone_{from_idx}"]))
            self.graph.add((adj_node, EX.toZone, EX[f"Zone_{to_idx}"]))
            self.graph.add((adj_node, EX.resistanceAdj, Literal(float(r_val), datatype=XSD.float)))

        # Central Plant: Chiller
        chlr_uri = EX.Chiller1
        self.graph.add((chlr_uri, RDF.type, BRICK.Chiller))
        self.graph.add((chlr_uri, RDFS.label, Literal("Central Water-Cooled Chiller Plant", datatype=XSD.string)))
        self.graph.add((chlr_uri, EX.coolingCapacityKw, Literal(120.0, datatype=XSD.float)))
        self.graph.add((chlr_uri, EX.copBase, Literal(3.8, datatype=XSD.float)))
        self.graph.add((chlr_uri, EX.chwTempMin, Literal(4.0, datatype=XSD.float)))
        self.graph.add((chlr_uri, EX.chwTempMax, Literal(12.0, datatype=XSD.float)))

        # AHU & VAV Hierarchy
        ahu_uri = EX.AHU1
        self.graph.add((ahu_uri, RDF.type, BRICK.Air_Handling_Unit))
        self.graph.add((ahu_uri, RDFS.label, Literal("Main Variable Air Volume AHU", datatype=XSD.string)))
        self.graph.add((ahu_uri, EX.maxAirflowM3s, Literal(8.5, datatype=XSD.float)))
        self.graph.add((ahu_uri, EX.fanPowerMaxKw, Literal(15.0, datatype=XSD.float)))
        self.graph.add((ahu_uri, EX.supplyAirTempMin, Literal(12.0, datatype=XSD.float)))
        self.graph.add((ahu_uri, EX.supplyAirTempMax, Literal(24.0, datatype=XSD.float)))
        self.graph.add((chlr_uri, BRICK.feeds, ahu_uri))

        for i in range(1, 6):
            vav_uri = EX[f"VAV_Z0{i}"]
            z_uri = EX[f"Zone_{i}"]
            self.graph.add((vav_uri, RDF.type, BRICK.Variable_Air_Volume_Box))
            self.graph.add((ahu_uri, BRICK.feeds, vav_uri))
            self.graph.add((vav_uri, BRICK.feeds, z_uri))

        # Add all parsed telemetry and control points
        for p in parsed_points:
            # Skip pure parameter tags from point attachment if handled above
            if p.point_role == "parameter":
                continue

            pt_uri = EX[p.canonical_id]
            brick_type = getattr(BRICK, p.brick_class, BRICK.Point)
            self.graph.add((pt_uri, RDF.type, brick_type))
            if p.metadata.get("description"):
                self.graph.add((pt_uri, RDFS.label, Literal(p.metadata["description"], datatype=XSD.string)))

            # Attach point to corresponding equipment or zone or building
            if p.equipment_id and p.equipment_id.startswith("CHLR") or p.equipment_type == "Chiller":
                self.graph.add((chlr_uri, BRICK.hasPoint, pt_uri))
            elif p.equipment_id and p.equipment_id.startswith("AHU") and "VAV" not in p.raw_tag:
                self.graph.add((ahu_uri, BRICK.hasPoint, pt_uri))
            elif p.zone_id:
                z_num = p.zone_id.split("_")[-1]
                if "VAV" in p.raw_tag:
                    self.graph.add((EX[f"VAV_Z0{z_num}"], BRICK.hasPoint, pt_uri))
                else:
                    self.graph.add((EX[f"Zone_{z_num}"], BRICK.hasPoint, pt_uri))
            else:
                self.graph.add((bldg_uri, BRICK.hasPoint, pt_uri))

        return self.graph

    def build_from_csv(self, csv_path: Union[str, Path]) -> Graph:
        """Convenience method: parses CSV dump and builds graph."""
        parser = SLMTagParser()
        points = parser.parse_csv(csv_path)
        return self.build_from_parsed_points(points)

    def serialize(self, format: str = "turtle") -> str:
        """Serialize graph to string (turtle, json-ld, xml, nt)."""
        return self.graph.serialize(format=format)

    def save_to_file(self, file_path: Union[str, Path], format: str = "turtle") -> None:
        """Save graph to file."""
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        self.graph.serialize(destination=str(file_path), format=format)
