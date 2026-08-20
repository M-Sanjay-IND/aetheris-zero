"""
Unit and Integration Tests for Dev 2 Phase 1:
Semantic Ingestion, Brick Schema v1.3 Knowledge Graph, SPARQL Extractor, and Tariff Feed.
"""

from pathlib import Path
import pytest
import rdflib

from gateway.ingestion.slm_tag_parser import SLMTagParser, ParsedPoint
from gateway.ingestion.schema_builder import SchemaBuilder
from gateway.ingestion.sparql_extractor import SPARQLExtractor, BuildingSimulatorConfig
from gateway.grid.tariff_feed import TariffFeed
from core.simulator.building_etp import BuildingSimulator


DATA_DIR = Path(__file__).parent.parent / "data"
RAW_BACNET_CSV = DATA_DIR / "raw_bacnet_dump.csv"
SAMPLE_CAISO_JSON = DATA_DIR / "sample_caiso_lmp.json"
TEMPLATE_TTL = DATA_DIR / "building_templates" / "5zone_office.ttl"


def test_slm_tag_parser_individual_tags():
    parser = SLMTagParser()

    # Zone Setpoint
    p1 = parser.parse_point_name("AHU1_Z01_VAV_DAT_SP", value=22.0, unit="deg_C")
    assert p1.brick_class == "Discharge_Air_Temperature_Setpoint"
    assert p1.point_role == "setpoint"
    assert p1.zone_id == "zone_1"
    assert p1.equipment_type == "Variable_Air_Volume_Box"

    # Chiller Setpoint
    p2 = parser.parse_point_name("CHLR_CHW_STPT", value=6.5, unit="deg_C")
    assert p2.brick_class == "Chilled_Water_Supply_Temperature_Setpoint"
    assert p2.point_role == "setpoint"
    assert p2.equipment_type == "Chiller"

    # Building Power Meter
    p3 = parser.parse_point_name("BLDG_PWR_TOT_KW", value=150.0, unit="kW")
    assert p3.brick_class == "Electric_Power_Sensor"
    assert p3.point_role == "meter"
    assert p3.subsystem == "electrical"

    # 2R2C Parameter: Zone Capacitance Cz
    p4 = parser.parse_point_name("Z01_CORE_CZ_PARAM", value=15000.0, unit="kJ/K")
    assert p4.brick_class == "Thermal_Capacitance_Air_Parameter"
    assert p4.point_role == "parameter"
    assert p4.metadata.get("param_key") == "C_z"
    assert p4.zone_id == "zone_1"

    # Inter-zone Adjacency: Radj
    p5 = parser.parse_point_name("Z01_Z02_RADJ_PARAM", value=1.2, unit="K/kW")
    assert p5.brick_class == "Interzone_Thermal_Resistance_Parameter"
    assert p5.point_role == "parameter"
    assert p5.metadata.get("from_zone") == "zone_1"
    assert p5.metadata.get("to_zone") == "zone_2"
    assert p5.metadata.get("param_key") == "R_adj"


def test_slm_tag_parser_csv():
    parser = SLMTagParser()
    points = parser.parse_csv(RAW_BACNET_CSV)
    assert len(points) >= 30

    # Ensure all 5 zones are captured
    zones_found = {p.zone_id for p in points if p.zone_id}
    for i in range(1, 6):
        assert f"zone_{i}" in zones_found


def test_schema_builder_from_csv():
    builder = SchemaBuilder()
    graph = builder.build_from_csv(RAW_BACNET_CSV)
    assert len(graph) > 100

    # Check for Brick classes in graph
    BRICK = rdflib.Namespace("https://brickschema.org/schema/Brick#")
    zones = list(graph.subjects(rdflib.RDF.type, BRICK.HVAC_Zone))
    assert len(zones) == 5

    chillers = list(graph.subjects(rdflib.RDF.type, BRICK.Chiller))
    assert len(chillers) == 1

    ahus = list(graph.subjects(rdflib.RDF.type, BRICK.Air_Handling_Unit))
    assert len(ahus) == 1

    vavs = list(graph.subjects(rdflib.RDF.type, BRICK.Variable_Air_Volume_Box))
    assert len(vavs) == 5

    # Check Turtle serialization
    ttl_str = builder.serialize(format="turtle")
    assert "@prefix brick:" in ttl_str
    assert "Zone_1" in ttl_str


def test_schema_builder_load_reference_ttl():
    builder = SchemaBuilder()
    graph = builder.load_ttl(TEMPLATE_TTL)
    assert len(graph) >= 150

    BRICK = rdflib.Namespace("https://brickschema.org/schema/Brick#")
    zones = list(graph.subjects(rdflib.RDF.type, BRICK.HVAC_Zone))
    assert len(zones) == 5


def test_sparql_extractor_full_contract():
    builder = SchemaBuilder()
    graph = builder.load_ttl(TEMPLATE_TTL)

    extractor = SPARQLExtractor()
    config_dict = extractor.extract_building_config(graph)

    # Validate against Pydantic schema
    validated_model = BuildingSimulatorConfig(**config_dict)
    assert validated_model is not None

    # Check exact contract expected by Dev 1 BuildingSimulator
    assert "simulation" in config_dict
    assert config_dict["simulation"]["time_step_sec"] == 300
    assert config_dict["simulation"]["total_hours"] == 24

    assert "zones" in config_dict
    assert len(config_dict["zones"]) == 5
    for z in config_dict["zones"]:
        assert "zone_id" in z
        assert "name" in z
        assert "C_z" in z and z["C_z"] > 0
        assert "C_m" in z and z["C_m"] > 0
        assert "R_ext" in z and z["R_ext"] > 0
        assert "R_m" in z and z["R_m"] > 0
        assert "initial_temp" in z
        assert "initial_mass_temp" in z
        assert "occupancy_max" in z
        assert "floor_area_sqm" in z
        assert "solar_factor" in z

    assert "adjacencies" in config_dict
    assert len(config_dict["adjacencies"]) >= 4
    for adj in config_dict["adjacencies"]:
        assert "from_zone" in adj
        assert "to_zone" in adj
        assert "R_adj" in adj and adj["R_adj"] > 0

    assert "equipment" in config_dict
    assert "chiller" in config_dict["equipment"]
    assert "ahu" in config_dict["equipment"]
    assert "vav_boxes" in config_dict["equipment"]
    assert config_dict["equipment"]["chiller"]["capacity_kw"] == 120.0

    assert "points" in config_dict
    assert len(config_dict["points"]) > 0


def test_tariff_feed_caiso_profile():
    feed = TariffFeed(json_file_path=SAMPLE_CAISO_JSON)

    # Midday solar dip (hour 12) should be low price
    price_midday = feed.get_price_by_hour(12.0)
    assert price_midday <= 0.06

    # Critical DR spike (hour 15) should be $1.50/kWh
    price_spike = feed.get_price_by_hour(15.0)
    assert price_spike == 1.50

    # Timestamp lookup (hour 12 = 12 * 3600 = 43200s)
    assert feed.get_price(43200) == price_midday

    # Forecast horizon (96 steps of 5 min = 24h)
    horizon = feed.get_forecast_horizon(current_step=0, horizon_steps=96, step_sec=300)
    assert len(horizon) == 96
    assert all(isinstance(p, float) for p in horizon)

    # Dynamic Spike Injection
    feed.inject_spike(start_hour=8.0, duration_hours=2.0, spike_price=2.00)
    assert feed.get_price_by_hour(8.5) == 2.00

    # Reset
    feed.reset()
    assert feed.get_price_by_hour(8.5) < 2.00

    # Chart data structure
    chart_data = feed.to_chart_data(resolution_minutes=15)
    assert len(chart_data) == 96
    assert "price_usd_per_kwh" in chart_data[0]
    assert "time_str" in chart_data[0]


def test_end_to_end_phase1_pipeline():
    """
    End-to-end Phase 1 verification:
    Raw BACnet CSV -> SLM Parser -> Schema Builder -> SPARQL Extractor -> BuildingSimulator(config)
    """
    # 1. Parse CSV
    parser = SLMTagParser()
    parsed_points = parser.parse_csv(RAW_BACNET_CSV)
    assert len(parsed_points) > 0

    # 2. Build Brick Schema RDF Graph
    builder = SchemaBuilder(building_id="CommercialOffice_5Zone", building_label="5-Zone Commercial Office")
    graph = builder.build_from_parsed_points(parsed_points)
    assert len(graph) > 100

    # 3. Extract & Validate Config via SPARQL
    extractor = SPARQLExtractor()
    sim_config = extractor.extract_building_config(graph)

    # 4. Dev 1 Simulator Initialization Check
    sim = BuildingSimulator(config=sim_config)
    assert sim.config == sim_config

    # 5. Tariff Feed Horizon Check for RL
    tariff = TariffFeed(json_file_path=SAMPLE_CAISO_JSON)
    forecast_24h = tariff.get_forecast_horizon(0, 96, 300)
    assert len(forecast_24h) == 96
