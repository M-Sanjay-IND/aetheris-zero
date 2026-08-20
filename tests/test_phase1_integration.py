from pathlib import Path
import pytest
from gateway.ingestion.slm_tag_parser import SLMTagParser
from gateway.ingestion.schema_builder import SchemaBuilder
from gateway.ingestion.sparql_extractor import SPARQLExtractor
from gateway.grid.tariff_feed import TariffFeed
from core.simulator.building_etp import BuildingSimulator
from core.simulator.comfort import calculate_pmv, is_ashrae55_compliant

DATA_DIR = Path(__file__).parent.parent / "data"
RAW_BACNET_CSV = DATA_DIR / "raw_bacnet_dump.csv"
SAMPLE_CAISO_JSON = DATA_DIR / "sample_caiso_lmp.json"


def test_full_cross_dev_phase1_integration():
    # 1. Dev 2 SLM Ingestion from raw BACnet tags
    parser = SLMTagParser()
    parsed_points = parser.parse_csv(RAW_BACNET_CSV)
    assert len(parsed_points) >= 30

    # 2. Dev 2 Brick Schema RDF Graph Construction
    builder = SchemaBuilder(building_id="Office_5Z", building_label="5-Zone Commercial Office")
    graph = builder.build_from_parsed_points(parsed_points)
    assert len(graph) > 100

    # 3. Dev 2 SPARQL Extraction of Building Priors
    extractor = SPARQLExtractor()
    sim_config = extractor.extract_building_config(graph)
    assert len(sim_config["zones"]) == 5
    assert len(sim_config["adjacencies"]) >= 4

    # 4. Dev 2 Tariff Feed
    tariff_feed = TariffFeed(json_file_path=SAMPLE_CAISO_JSON)
    sim_config["tariff_feed"] = tariff_feed

    # 5. Dev 1 Physics Engine Initialization
    sim = BuildingSimulator(config=sim_config)
    initial_state = sim.reset()
    assert initial_state["step"] == 0
    assert len(initial_state["zones"]) == 5

    # 6. Step through morning, solar surplus, and afternoon peak price window
    for step in range(200):
        actions = {
            "zone_setpoints": {f"zone_{i}": 22.0 for i in range(1, 6)},
            "chiller_chw_setpoint": 6.5,
            "vav_damper_positions": {f"zone_{i}": 0.7 for i in range(1, 6)},
        }
        state, reward, done, info = sim.step(actions)

        # Check telemetry integrity
        assert state["step"] == step + 1
        assert "total_hvac_kw" in state["power"]
        assert state["metrics"]["cumulative_energy_actual_kwh"] > 0
        assert "dynamic_lmp_price" in state

        # Check comfort in core zone
        pmv = state["zones"]["zone_1"]["pmv"]
        assert -1.0 <= pmv <= 1.0

    # 7. Check price spike injection impact on cost
    tariff_feed.inject_spike(start_hour=sim.current_hour, duration_hours=2.0, spike_price=2.50)
    current_price = sim.get_dynamic_lmp_price(sim.current_hour)
    assert current_price == 2.50
