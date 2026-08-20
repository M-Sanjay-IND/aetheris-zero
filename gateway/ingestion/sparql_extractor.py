"""
SPARQLExtractor: Queries Brick Schema RDF graphs using SPARQL to extract structured
thermodynamic priors, zone adjacencies, HVAC equipment specs, and telemetry mappings
for Dev 1's BuildingSimulator.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import rdflib
from pydantic import BaseModel, Field


class SimulationSettings(BaseModel):
    time_step_sec: int = Field(default=300, description="Simulation step interval in seconds")
    total_hours: int = Field(default=24, description="Episode horizon in hours")
    start_hour: int = Field(default=0, description="Starting hour of day [0-23]")


class ZoneConfig(BaseModel):
    zone_id: str
    name: str
    C_z: float = Field(..., description="Zone air capacitance in kJ/K")
    C_m: float = Field(..., description="Structural thermal mass capacitance in kJ/K")
    R_ext: float = Field(..., description="Envelope thermal resistance in K/kW")
    R_m: float = Field(..., description="Internal air-to-mass resistance in K/kW")
    initial_temp: float = Field(default=22.0, description="Initial indoor air temperature in °C")
    initial_mass_temp: float = Field(default=21.8, description="Initial structural mass temperature in °C")
    occupancy_max: int = Field(default=15, description="Peak occupant count")
    floor_area_sqm: float = Field(default=180.0, description="Zone floor area in m²")
    solar_factor: float = Field(default=0.15, description="Solar gain penetration factor")


class AdjacencyConfig(BaseModel):
    from_zone: str
    to_zone: str
    R_adj: float = Field(..., description="Inter-zone thermal resistance in K/kW")


class ChillerConfig(BaseModel):
    capacity_kw: float = Field(default=120.0, description="Maximum cooling capacity in kW")
    cop_base: float = Field(default=3.8, description="Baseline COP")
    chw_temp_min: float = Field(default=4.0, description="Minimum CHW supply temperature in °C")
    chw_temp_max: float = Field(default=12.0, description="Maximum CHW supply temperature in °C")


class AHUConfig(BaseModel):
    max_airflow_m3s: float = Field(default=8.5, description="Maximum supply air flow in m³/s")
    fan_power_max_kw: float = Field(default=15.0, description="Maximum fan power in kW")
    supply_air_temp_min: float = Field(default=12.0, description="Minimum supply air temp in °C")
    supply_air_temp_max: float = Field(default=24.0, description="Maximum supply air temp in °C")


class VAVBoxesConfig(BaseModel):
    min_damper_position: float = Field(default=0.2, description="Minimum ventilation position [0-1]")
    max_damper_position: float = Field(default=1.0, description="Maximum damper position [0-1]")


class EquipmentConfig(BaseModel):
    chiller: ChillerConfig = Field(default_factory=ChillerConfig)
    ahu: AHUConfig = Field(default_factory=AHUConfig)
    vav_boxes: VAVBoxesConfig = Field(default_factory=VAVBoxesConfig)


class BuildingSimulatorConfig(BaseModel):
    simulation: SimulationSettings = Field(default_factory=SimulationSettings)
    zones: List[ZoneConfig]
    adjacencies: List[AdjacencyConfig]
    equipment: EquipmentConfig = Field(default_factory=EquipmentConfig)
    points: Dict[str, Dict[str, Any]] = Field(default_factory=dict)


class SPARQLExtractor:
    """
    Executes SPARQL queries against Brick Schema RDF knowledge graphs
    and returns validated configuration structures for Dev 1's BuildingSimulator.
    """

    PREFIXES = """
        PREFIX brick: <https://brickschema.org/schema/Brick#>
        PREFIX ex: <http://example.com/building#>
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
    """

    def __init__(self):
        pass

    def extract_zones(self, graph: rdflib.Graph) -> List[Dict[str, Any]]:
        """Extract all HVAC zones and their 2R2C thermal parameters."""
        query = self.PREFIXES + """
            SELECT ?zone ?zoneId ?label ?cz ?cm ?rext ?rm ?area ?solar ?occ ?tInit ?tMassInit
            WHERE {
                ?zone a brick:HVAC_Zone .
                OPTIONAL { ?zone ex:zoneId ?zoneId . }
                OPTIONAL { ?zone rdfs:label ?label . }
                OPTIONAL { ?zone ex:thermalCapacitanceAir ?cz . }
                OPTIONAL { ?zone ex:thermalCapacitanceMass ?cm . }
                OPTIONAL { ?zone ex:envelopeResistance ?rext . }
                OPTIONAL { ?zone ex:internalMassResistance ?rm . }
                OPTIONAL { ?zone ex:floorAreaSqm ?area . }
                OPTIONAL { ?zone ex:solarFactor ?solar . }
                OPTIONAL { ?zone ex:occupancyMax ?occ . }
                OPTIONAL { ?zone ex:initialTemp ?tInit . }
                OPTIONAL { ?zone ex:initialMassTemp ?tMassInit . }
            }
            ORDER BY ?zoneId
        """
        results = graph.query(query)
        zones = []
        for row in results:
            z_uri = str(row.zone)
            local_id = z_uri.split("#")[-1].replace("Zone_", "zone_").lower()
            zone_id = str(row.zoneId) if row.zoneId else local_id
            name = str(row.label) if row.label else f"Zone {zone_id}"
            
            # Default fallbacks if omitted in graph
            cz = float(row.cz) if row.cz else 15000.0
            cm = float(row.cm) if row.cm else 85000.0
            rext = float(row.rext) if row.rext else 2.0
            rm = float(row.rm) if row.rm else 0.5
            area = float(row.area) if row.area else 180.0
            solar = float(row.solar) if row.solar else 0.15
            occ = int(row.occ) if row.occ else 15
            t_init = float(row.tInit) if row.tInit else 22.0
            t_mass_init = float(row.tMassInit) if row.tMassInit else 21.8

            zones.append({
                "zone_id": zone_id,
                "name": name,
                "C_z": cz,
                "C_m": cm,
                "R_ext": rext,
                "R_m": rm,
                "initial_temp": t_init,
                "initial_mass_temp": t_mass_init,
                "occupancy_max": occ,
                "floor_area_sqm": area,
                "solar_factor": solar,
            })
        return zones

    def extract_adjacencies(self, graph: rdflib.Graph) -> List[Dict[str, Any]]:
        """Extract inter-zone adjacency relationships and resistances."""
        query = self.PREFIXES + """
            SELECT ?fromZone ?toZone ?radj
            WHERE {
                ?adj a ex:ZoneAdjacency ;
                     ex:fromZone ?from ;
                     ex:toZone ?to ;
                     ex:resistanceAdj ?radj .
                OPTIONAL { ?from ex:zoneId ?fromZone . }
                OPTIONAL { ?to ex:zoneId ?toZone . }
            }
        """
        results = graph.query(query)
        adjacencies = []
        for row in results:
            from_z = str(row.fromZone) if row.fromZone else "zone_1"
            to_z = str(row.toZone) if row.toZone else "zone_2"
            radj = float(row.radj) if row.radj else 1.2
            adjacencies.append({
                "from_zone": from_z,
                "to_zone": to_z,
                "R_adj": radj,
            })

        # Fallback if no explicit ZoneAdjacency nodes were created
        if not adjacencies:
            adjacencies = [
                {"from_zone": "zone_1", "to_zone": "zone_2", "R_adj": 1.2},
                {"from_zone": "zone_1", "to_zone": "zone_3", "R_adj": 1.2},
                {"from_zone": "zone_1", "to_zone": "zone_4", "R_adj": 1.2},
                {"from_zone": "zone_1", "to_zone": "zone_5", "R_adj": 1.2},
            ]
        return adjacencies

    def extract_equipment(self, graph: rdflib.Graph) -> Dict[str, Any]:
        """Extract chiller, AHU, and VAV box engineering parameters."""
        # Chiller query
        chiller_query = self.PREFIXES + """
            SELECT ?cap ?cop ?chwMin ?chwMax
            WHERE {
                ?chiller a brick:Chiller .
                OPTIONAL { ?chiller ex:coolingCapacityKw ?cap . }
                OPTIONAL { ?chiller ex:copBase ?cop . }
                OPTIONAL { ?chiller ex:chwTempMin ?chwMin . }
                OPTIONAL { ?chiller ex:chwTempMax ?chwMax . }
            }
            LIMIT 1
        """
        ch_res = list(graph.query(chiller_query))
        if ch_res:
            row = ch_res[0]
            chiller_data = {
                "capacity_kw": float(row.cap) if row.cap else 120.0,
                "cop_base": float(row.cop) if row.cop else 3.8,
                "chw_temp_min": float(row.chwMin) if row.chwMin else 4.0,
                "chw_temp_max": float(row.chwMax) if row.chwMax else 12.0,
            }
        else:
            chiller_data = {"capacity_kw": 120.0, "cop_base": 3.8, "chw_temp_min": 4.0, "chw_temp_max": 12.0}

        # AHU query
        ahu_query = self.PREFIXES + """
            SELECT ?flow ?pwr ?satMin ?satMax
            WHERE {
                ?ahu a brick:Air_Handling_Unit .
                OPTIONAL { ?ahu ex:maxAirflowM3s ?flow . }
                OPTIONAL { ?ahu ex:fanPowerMaxKw ?pwr . }
                OPTIONAL { ?ahu ex:supplyAirTempMin ?satMin . }
                OPTIONAL { ?ahu ex:supplyAirTempMax ?satMax . }
            }
            LIMIT 1
        """
        ahu_res = list(graph.query(ahu_query))
        if ahu_res:
            row = ahu_res[0]
            ahu_data = {
                "max_airflow_m3s": float(row.flow) if row.flow else 8.5,
                "fan_power_max_kw": float(row.pwr) if row.pwr else 15.0,
                "supply_air_temp_min": float(row.satMin) if row.satMin else 12.0,
                "supply_air_temp_max": float(row.satMax) if row.satMax else 24.0,
            }
        else:
            ahu_data = {"max_airflow_m3s": 8.5, "fan_power_max_kw": 15.0, "supply_air_temp_min": 12.0, "supply_air_temp_max": 24.0}

        return {
            "chiller": chiller_data,
            "ahu": ahu_data,
            "vav_boxes": {
                "min_damper_position": 0.2,
                "max_damper_position": 1.0,
            },
        }

    def extract_point_mappings(self, graph: rdflib.Graph) -> Dict[str, Dict[str, Any]]:
        """Extract mapping between raw BACnet point tags and simulation telemetry keys."""
        query = self.PREFIXES + """
            SELECT ?entity ?point ?pType
            WHERE {
                ?entity brick:hasPoint ?point .
                ?point a ?pType .
            }
        """
        results = graph.query(query)
        points_map = {}
        for row in results:
            pt_name = str(row.point).split("#")[-1]
            pt_type = str(row.pType).split("#")[-1]
            
            # Classify point mapping for Dev 1 BuildingSimulator
            if "DAT_SP" in pt_name or "VAV.*DAT" in pt_name:
                zone_key = None
                for i in range(1, 6):
                    if f"Z0{i}" in pt_name or f"Z{i}" in pt_name:
                        zone_key = f"zone_{i}"
                        break
                points_map[pt_name] = {"type": "zone_setpoint", "zone_id": zone_key or "zone_1"}
            elif "CHW_STPT" in pt_name:
                points_map[pt_name] = {"type": "chiller_chw_setpoint"}
            elif "PWR_TOT" in pt_name or "BLDG_PWR" in pt_name:
                points_map[pt_name] = {"type": "total_power_sensor"}
            elif "AIR_TEMP" in pt_name:
                zone_key = None
                for i in range(1, 6):
                    if f"Z0{i}" in pt_name or f"Z{i}" in pt_name:
                        zone_key = f"zone_{i}"
                        break
                points_map[pt_name] = {"type": "zone_temp_sensor", "zone_id": zone_key or "zone_1"}
            elif "DMPR_POS" in pt_name:
                zone_key = None
                for i in range(1, 6):
                    if f"Z0{i}" in pt_name or f"Z{i}" in pt_name:
                        zone_key = f"zone_{i}"
                        break
                points_map[pt_name] = {"type": "damper_command", "zone_id": zone_key or "zone_1"}
            else:
                points_map[pt_name] = {"type": pt_type}

        return points_map

    def extract_building_config(self, graph: rdflib.Graph) -> Dict[str, Any]:
        """
        Main extraction function returning a validated dictionary matching Dev 1's
        BuildingSimulator(config) contract.
        """
        zones_data = self.extract_zones(graph)
        adj_data = self.extract_adjacencies(graph)
        eq_data = self.extract_equipment(graph)
        pts_data = self.extract_point_mappings(graph)

        config_model = BuildingSimulatorConfig(
            simulation=SimulationSettings(),
            zones=[ZoneConfig(**z) for z in zones_data],
            adjacencies=[AdjacencyConfig(**a) for a in adj_data],
            equipment=EquipmentConfig(
                chiller=ChillerConfig(**eq_data["chiller"]),
                ahu=AHUConfig(**eq_data["ahu"]),
                vav_boxes=VAVBoxesConfig(**eq_data["vav_boxes"]),
            ),
            points=pts_data,
        )

        return config_model.model_dump()
