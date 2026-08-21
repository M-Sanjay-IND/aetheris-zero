import csv
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

DATASET_DIR = Path(__file__).parent.resolve()
SLM_JSONL_PATH = DATASET_DIR / "slm_bacnet_brick_corpus.jsonl"
SLM_CSV_PATH = DATASET_DIR / "slm_bacnet_brick_corpus.csv"
SLM_TRAIN_PATH = DATASET_DIR / "slm_train.jsonl"
SLM_VAL_PATH = DATASET_DIR / "slm_val.jsonl"
SLM_TEST_OOD_PATH = DATASET_DIR / "slm_test_ood.jsonl"
GRID_CSV_PATH = DATASET_DIR / "grid_weather_thermal_timeseries.csv"
GRID_PARQUET_PATH = DATASET_DIR / "grid_weather_thermal_timeseries.parquet"

BRICK_CLASSES = [
    "Supply_Air_Temperature_Sensor",
    "Supply_Air_Temperature_Setpoint",
    "Discharge_Air_Temperature_Setpoint",
    "Outside_Air_Temperature_Sensor",
    "Zone_Air_Temperature_Sensor",
    "Zone_Air_Temperature_Setpoint",
    "Electric_Power_Sensor",
    "Air_Flow_Sensor",
    "Damper_Position_Command",
    "Chilled_Water_Supply_Temperature_Sensor",
    "Chilled_Water_Supply_Temperature_Setpoint",
    "Chilled_Water_Return_Temperature_Sensor",
    "Solar_Radiance_Sensor",
    "Occupancy_Sensor",
    "Thermal_Capacitance_Air_Parameter",
    "Thermal_Capacitance_Mass_Parameter",
    "Envelope_Thermal_Resistance_Parameter",
    "Mass_Thermal_Resistance_Parameter",
    "Interzone_Thermal_Resistance_Parameter",
    "Floor_Area_Parameter",
    "Solar_Factor_Parameter",
    "CO2_Level_Sensor",
    "Relative_Humidity_Sensor",
    "Fan_Status",
    "Filter_Differential_Pressure_Sensor",
]

EQUIPMENT_TYPES = [
    "Air_Handling_Unit",
    "Chiller",
    "Variable_Air_Volume_Box",
    "Building",
    "Fan_Coil_Unit",
    "Boiler",
    "Cooling_Tower",
    "Pump",
    "Battery_Energy_Storage_System",
    "Electric_Vehicle_Supply_Equipment",
]

POINT_ROLES = ["sensor", "setpoint", "command", "parameter", "meter", "status"]
SUBSYSTEMS = ["hvac", "thermal_model", "electrical", "environment"]
UNITS = ["deg_C", "kW", "m3/s", "ratio", "count", "K/kW", "kJ/K", "m2", "W/m2", "ppm", "%", "kPa", "unknown"]


def generate_slm_tag_corpus_leak_free(seed: int = 42) -> Tuple[List[dict], List[dict], List[dict]]:
    """
    Generates a realistic, multi-vendor building tag corpus with strict Disjoint Facility/Domain splitting
    to guarantee zero data leakage between Train, Validation, and Out-of-Distribution (OOD) Test sets.
    """
    rng = np.random.default_rng(seed)

    # Disjoint Facility Domains (Zero overlap across splits)
    train_facilities = ["CAMPUS_ENG", "BLDG1", "FACILITY_A", "TOWER_NORTH", "PLANT_EAST"]
    val_facilities = ["HQ_CAMPUS", "HOSPITAL_MAIN"]
    test_ood_facilities = ["DATACENTER_WEST", "AUTOMATED_LOGIC_SITE7", "METASYS_UNSEEN_FACILITY", "SIEMENS_DESIGO_LAB"]

    # Point Archetype definitions with realistic abbreviation variants
    archetypes = [
        # (Templates, brick_class, equipment_type, point_role, subsystem, unit, param_key)
        (
            ["{fac}_{zone}_RAT", "{fac}_{zone}_ZN_T", "{fac}_{zone}_ROOM_TEMP", "{fac}_{zone}_TEMP_SENS", "{fac}_{zone}_AIR_TEMP"],
            "Zone_Air_Temperature_Sensor", "Variable_Air_Volume_Box", "sensor", "hvac", "deg_C", None
        ),
        (
            ["{fac}_{zone}_TEMP_SP", "{fac}_{zone}_STPT", "{fac}_{zone}_ZN_STPT", "{fac}_{zone}_AIR_TEMP_SP", "{fac}_{zone}_SETPT"],
            "Zone_Air_Temperature_Setpoint", "Variable_Air_Volume_Box", "setpoint", "hvac", "deg_C", None
        ),
        (
            ["{fac}_{zone}_OCC", "{fac}_{zone}_OCCUPANCY", "{fac}_{zone}_OCC_COUNT", "{fac}_{zone}_OCC_SENS"],
            "Occupancy_Sensor", "Variable_Air_Volume_Box", "sensor", "environment", "count", None
        ),
        (
            ["{fac}_{zone}_CO2", "{fac}_{zone}_CO2_PPM", "{fac}_{zone}_CARBON_DIOXIDE"],
            "CO2_Level_Sensor", "Variable_Air_Volume_Box", "sensor", "environment", "ppm", None
        ),
        (
            ["{fac}_{zone}_RH", "{fac}_{zone}_RH_SENS", "{fac}_{zone}_HUMIDITY_PCT"],
            "Relative_Humidity_Sensor", "Variable_Air_Volume_Box", "sensor", "environment", "%", None
        ),
        (
            ["{fac}_VAV_{zone}_DMPR_POS", "{fac}_VAV_{zone}_DMP_CMD", "{fac}_VAV_{zone}_DAMPER_POS", "{fac}_VAV_{zone}_DMPR"],
            "Damper_Position_Command", "Variable_Air_Volume_Box", "command", "hvac", "ratio", None
        ),
        (
            ["{fac}_VAV_{zone}_FLOW", "{fac}_VAV_{zone}_AIR_FLOW", "{fac}_VAV_{zone}_CFM", "{fac}_VAV_{zone}_M3S"],
            "Air_Flow_Sensor", "Variable_Air_Volume_Box", "sensor", "hvac", "m3/s", None
        ),
        (
            ["{fac}_VAV_{zone}_DAT_SP", "{fac}_VAV_{zone}_DISCH_TEMP_SP", "{fac}_VAV_{zone}_DAT_SETPT"],
            "Discharge_Air_Temperature_Setpoint", "Variable_Air_Volume_Box", "setpoint", "hvac", "deg_C", None
        ),
        (
            ["{fac}_AHU{ahu}_SAT", "{fac}_AHU{ahu}_SUPPLY_TEMP", "{fac}_AHU{ahu}_SA_TEMP", "{fac}_AHU{ahu}_SUP_AIR_T"],
            "Supply_Air_Temperature_Sensor", "Air_Handling_Unit", "sensor", "hvac", "deg_C", None
        ),
        (
            ["{fac}_AHU{ahu}_SAT_SP", "{fac}_AHU{ahu}_SUP_TEMP_SP", "{fac}_AHU{ahu}_SA_STPT", "{fac}_AHU{ahu}_SUPPLY_AIR_STPT"],
            "Supply_Air_Temperature_Setpoint", "Air_Handling_Unit", "setpoint", "hvac", "deg_C", None
        ),
        (
            ["{fac}_AHU{ahu}_FAN_KW", "{fac}_AHU{ahu}_FAN_PWR", "{fac}_AHU{ahu}_SUP_FAN_KW", "{fac}_AHU{ahu}_ELEC_KW"],
            "Electric_Power_Sensor", "Air_Handling_Unit", "meter", "electrical", "kW", None
        ),
        (
            ["{fac}_AHU{ahu}_FAN_STAT", "{fac}_AHU{ahu}_FAN_STATUS", "{fac}_AHU{ahu}_RUN_STAT"],
            "Fan_Status", "Air_Handling_Unit", "status", "hvac", "unknown", None
        ),
        (
            ["{fac}_AHU{ahu}_FLTR_DP", "{fac}_AHU{ahu}_FILTER_DIFF_P", "{fac}_AHU{ahu}_DP_SENS"],
            "Filter_Differential_Pressure_Sensor", "Air_Handling_Unit", "sensor", "hvac", "kPa", None
        ),
        (
            ["{fac}_CHLR{chlr}_CHW_SUP_T", "{fac}_CHLR{chlr}_CHW_SUP", "{fac}_CHLR{chlr}_SUPPLY_TEMP", "{fac}_CHLR{chlr}_CHW_ST"],
            "Chilled_Water_Supply_Temperature_Sensor", "Chiller", "sensor", "hvac", "deg_C", None
        ),
        (
            ["{fac}_CHLR{chlr}_CHW_RET_T", "{fac}_CHLR{chlr}_CHW_RET", "{fac}_CHLR{chlr}_RETURN_TEMP", "{fac}_CHLR{chlr}_CHW_RT"],
            "Chilled_Water_Return_Temperature_Sensor", "Chiller", "sensor", "hvac", "deg_C", None
        ),
        (
            ["{fac}_CHLR{chlr}_CHW_STPT", "{fac}_CHLR{chlr}_CHW_SP", "{fac}_CHLR{chlr}_SETPOINT", "{fac}_CHLR{chlr}_CHW_SETPT"],
            "Chilled_Water_Supply_Temperature_Setpoint", "Chiller", "setpoint", "hvac", "deg_C", None
        ),
        (
            ["{fac}_CHLR{chlr}_KW", "{fac}_CHLR{chlr}_POWER_KW", "{fac}_CHLR{chlr}_ELEC_KW", "{fac}_CHLR{chlr}_MTR_KW"],
            "Electric_Power_Sensor", "Chiller", "meter", "electrical", "kW", None
        ),
        (
            ["{fac}_BLDG_TOTAL_KW", "{fac}_MAIN_MTR_KW", "{fac}_BUILDING_POWER", "{fac}_GRID_KW"],
            "Electric_Power_Sensor", "Building", "meter", "electrical", "kW", None
        ),
        (
            ["{fac}_OAT", "{fac}_OUTDOOR_TEMP", "{fac}_AMB_TEMP", "{fac}_OA_T", "{fac}_WEATHER_TEMP"],
            "Outside_Air_Temperature_Sensor", "Building", "sensor", "environment", "deg_C", None
        ),
        (
            ["{fac}_SOLAR_GHI", "{fac}_SOLAR_IRRAD", "{fac}_GHI_SENS", "{fac}_SOL_IRRAD"],
            "Solar_Radiance_Sensor", "Building", "sensor", "environment", "W/m2", None
        ),
        (
            ["{fac}_{zone}_CZ_PARAM", "{fac}_{zone}_CAPACITANCE_AIR"],
            "Thermal_Capacitance_Air_Parameter", "Variable_Air_Volume_Box", "parameter", "thermal_model", "kJ/K", "C_z"
        ),
        (
            ["{fac}_{zone}_CM_PARAM", "{fac}_{zone}_CAPACITANCE_MASS"],
            "Thermal_Capacitance_Mass_Parameter", "Variable_Air_Volume_Box", "parameter", "thermal_model", "kJ/K", "C_m"
        ),
        (
            ["{fac}_{zone}_REXT_PARAM", "{fac}_{zone}_RESISTANCE_EXT"],
            "Envelope_Thermal_Resistance_Parameter", "Variable_Air_Volume_Box", "parameter", "thermal_model", "K/kW", "R_ext"
        ),
        (
            ["{fac}_{zone}_RM_PARAM", "{fac}_{zone}_RESISTANCE_MASS"],
            "Mass_Thermal_Resistance_Parameter", "Variable_Air_Volume_Box", "parameter", "thermal_model", "K/kW", "R_m"
        ),
        (
            ["{fac}_{zone}_AREA_SQM", "{fac}_{zone}_FLOOR_AREA"],
            "Floor_Area_Parameter", "Variable_Air_Volume_Box", "parameter", "thermal_model", "m2", "floor_area_sqm"
        ),
        (
            ["{fac}_{zone}_SOLAR_FRAC", "{fac}_{zone}_SOLAR_FACTOR"],
            "Solar_Factor_Parameter", "Variable_Air_Volume_Box", "parameter", "thermal_model", "ratio", "solar_factor"
        ),
    ]

    zones = ["zone_1", "zone_2", "zone_3", "zone_4", "zone_5"]
    zone_aliases = {
        "zone_1": ["Z01", "ZN1", "ZONE_1", "CORE", "Z1", "FL01_Z1", "ZN_CORE"],
        "zone_2": ["Z02", "ZN2", "ZONE_2", "NORTH", "PERIM_N", "Z2", "FL01_Z2", "ZN_NORTH"],
        "zone_3": ["Z03", "ZN3", "ZONE_3", "SOUTH", "PERIM_S", "Z3", "FL01_Z3", "ZN_SOUTH"],
        "zone_4": ["Z04", "ZN4", "ZONE_4", "EAST", "PERIM_E", "Z4", "FL01_Z4", "ZN_EAST"],
        "zone_5": ["Z05", "ZN5", "ZONE_5", "WEST", "PERIM_W", "Z5", "FL01_Z5", "ZN_WEST"],
    }

    delimiters = ["_", "-", ".", "/", " "]
    case_styles = ["upper", "lower", "mixed", "camel"]

    def _generate_for_facilities(facility_list: List[str], num_points: int, split_name: str) -> List[dict]:
        dataset = []
        for _ in range(num_points):
            arch = archetypes[rng.integers(0, len(archetypes))]
            templates, b_class, eq_type, role, sub, unit, p_key = arch

            template = rng.choice(templates)
            fac = rng.choice(facility_list)
            z_idx = rng.choice(zones)
            z_alias = rng.choice(zone_aliases[z_idx])
            ahu_num = rng.integers(1, 5)
            chlr_num = rng.integers(1, 4)

            tag = template.format(
                fac=fac,
                zone=z_alias,
                ahu=ahu_num,
                chlr=chlr_num
            )

            # Delimiter variation
            delim = rng.choice(delimiters)
            if delim != "_":
                tag = tag.replace("_", delim)

            # Casing variation
            style = rng.choice(case_styles)
            if style == "lower":
                tag = tag.lower()
            elif style == "camel":
                parts = tag.split(delim if delim != "_" else "_")
                tag = "".join(p.capitalize() for p in parts)
            elif style == "mixed" and rng.random() > 0.5:
                tag = tag.title()

            # Optional Noise / Typo Injection (5% chance in Train, 15% in Test OOD to test robustness)
            noise_rate = 0.15 if split_name == "test_ood" else 0.05
            if rng.random() < noise_rate and len(tag) > 6:
                idx = rng.integers(1, len(tag) - 1)
                # Random char swap or duplication
                if rng.random() > 0.5:
                    tag = tag[:idx] + tag[idx+1] + tag[idx] + tag[idx+2:]
                else:
                    tag = tag[:idx] + tag[idx] + tag[idx:]

            eq_id = f"{eq_type}_{ahu_num}" if "AHU" in eq_type else (
                f"Chiller_{chlr_num}" if "Chiller" in eq_type else (
                    f"VAV_{z_idx.upper()}" if "VAV" in eq_type else "Building_Main"
                )
            )

            assigned_zone = z_idx if ("RAT" in template or "VAV" in template or "PARAM" in template or "zone" in template or "OCC" in template or "ZN" in template) else "unassigned"

            dataset.append({
                "raw_tag": tag,
                "canonical_id": tag,
                "brick_class": b_class,
                "equipment_type": eq_type,
                "equipment_id": eq_id,
                "point_role": role,
                "subsystem": sub,
                "zone_id": assigned_zone,
                "unit": unit,
                "param_key": p_key or "none",
                "split": split_name,
                "facility": fac,
                "description": f"Standard {b_class} point for {eq_id} in {assigned_zone}"
            })

        # Inter-zone adjacency parameters
        for z1 in range(1, 6):
            for z2 in range(1, 6):
                if z1 != z2:
                    fac = rng.choice(facility_list)
                    tag = f"{fac}_Z0{z1}_Z0{z2}_RADJ_PARAM"
                    dataset.append({
                        "raw_tag": tag,
                        "canonical_id": tag,
                        "brick_class": "Interzone_Thermal_Resistance_Parameter",
                        "equipment_type": "Variable_Air_Volume_Box",
                        "equipment_id": f"VAV_ZONE_{z1}",
                        "point_role": "parameter",
                        "subsystem": "thermal_model",
                        "zone_id": f"zone_{z1}",
                        "unit": "K/kW",
                        "param_key": "R_adj",
                        "split": split_name,
                        "facility": fac,
                        "description": f"Inter-zone thermal resistance between zone_{z1} and zone_{z2}"
                    })

        return dataset

    train_data = _generate_for_facilities(train_facilities, num_points=4500, split_name="train")
    val_data = _generate_for_facilities(val_facilities, num_points=800, split_name="val")
    test_ood_data = _generate_for_facilities(test_ood_facilities, num_points=1000, split_name="test_ood")

    return train_data, val_data, test_ood_data


def generate_slm_tag_corpus(num_samples: int = 1000, seed: int = 42) -> List[dict]:
    train_d, val_d, test_d = generate_slm_tag_corpus_leak_free(seed=seed)
    combined = train_d + val_d + test_d
    return combined[:num_samples]


def generate_grid_weather_thermal_timeseries(num_days: int = 365, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    steps_per_day = 288  # 5-minute intervals
    total_steps = num_days * steps_per_day
    dt_hours = 5.0 / 60.0

    step_indices = np.arange(total_steps)
    hour_of_day = (step_indices * dt_hours) % 24.0
    day_of_year = (step_indices // steps_per_day) % 365
    day_of_week = (day_of_year + 3) % 7
    is_weekend = (day_of_week >= 5).astype(int)

    # Seasonal temperature envelope
    seasonal_temp_offset = 12.0 * np.sin(2.0 * np.pi * (day_of_year - 100) / 365.0)  # Peak in summer
    diurnal_temp_swing = 7.0 * np.sin(2.0 * np.pi * (hour_of_day - 9.0) / 24.0)
    weather_noise = rng.normal(0, 1.2, total_steps)
    ambient_temp = 20.0 + seasonal_temp_offset + diurnal_temp_swing + weather_noise

    # Solar Irradiance (GHI and DNI)
    solar_elevation = np.maximum(0.0, np.sin(np.pi * (hour_of_day - 6.0) / 12.0))
    solar_elevation = np.where((hour_of_day >= 6.0) & (hour_of_day <= 18.0), solar_elevation, 0.0)
    cloud_cover = np.clip(rng.normal(0.15, 0.1, total_steps), 0.0, 0.9)
    solar_ghi = np.maximum(0.0, 950.0 * solar_elevation * (1.0 - cloud_cover) + rng.normal(0, 5.0, total_steps))
    solar_dni = np.maximum(0.0, 850.0 * (solar_elevation ** 1.2) * (1.0 - cloud_cover * 1.3))

    # Relative Humidity (%)
    rel_humidity = np.clip(60.0 - 25.0 * np.sin(2.0 * np.pi * (hour_of_day - 8.0) / 24.0) + rng.normal(0, 4.0, total_steps), 15.0, 95.0)

    # Dynamic LMP Electricity Price ($/kWh)
    base_lmp = 0.12 + 0.04 * (seasonal_temp_offset > 5.0)
    solar_curtailment_drop = np.where((hour_of_day >= 10.0) & (hour_of_day <= 15.0), -0.06 * (solar_ghi / 900.0), 0.0)
    evening_peak = np.where(
        (hour_of_day >= 17.0) & (hour_of_day <= 21.0),
        0.35 + 0.25 * (seasonal_temp_offset > 6.0),
        0.0
    )

    spike_prob = np.where((hour_of_day >= 16.0) & (hour_of_day <= 20.0) & (ambient_temp > 33.0), 0.08, 0.005)
    random_spikes = (rng.random(total_steps) < spike_prob) * rng.uniform(0.60, 1.80, total_steps)

    spring_negative = np.where(
        (day_of_year >= 60) & (day_of_year <= 130) & (hour_of_day >= 12.0) & (hour_of_day <= 14.0) & (rng.random(total_steps) < 0.15),
        -0.08,
        0.0
    )

    dynamic_lmp = np.maximum(-0.05, base_lmp + solar_curtailment_drop + evening_peak + random_spikes + spring_negative)

    p_2h = np.zeros(total_steps)
    p_4h = np.zeros(total_steps)
    p_6h = np.zeros(total_steps)
    for t in range(total_steps):
        p_2h[t] = np.mean(dynamic_lmp[t : min(t + 24, total_steps)])
        p_4h[t] = np.mean(dynamic_lmp[t : min(t + 48, total_steps)])
        p_6h[t] = np.mean(dynamic_lmp[t : min(t + 72, total_steps)])

    dr_active = np.where((dynamic_lmp > 0.65) | (random_spikes > 0.5), 1, 0)

    occ_curve = np.where(
        (is_weekend == 0) & (hour_of_day >= 7.5) & (hour_of_day <= 18.5),
        np.sin(np.pi * (hour_of_day - 7.5) / 11.0),
        0.05
    )
    occ_z1 = np.clip(np.round(22.0 * occ_curve + rng.integers(-2, 3, total_steps)), 0, 25)
    occ_z2 = np.clip(np.round(18.0 * occ_curve + rng.integers(-2, 3, total_steps)), 0, 20)
    occ_z3 = np.clip(np.round(19.0 * occ_curve + rng.integers(-2, 3, total_steps)), 0, 20)
    occ_z4 = np.clip(np.round(16.0 * occ_curve + rng.integers(-2, 3, total_steps)), 0, 18)
    occ_z5 = np.clip(np.round(17.0 * occ_curve + rng.integers(-2, 3, total_steps)), 0, 18)

    internal_load = (occ_z1 + occ_z2 + occ_z3 + occ_z4 + occ_z5) * 0.12 + np.where(occ_curve > 0.2, 8.5, 1.2)

    df = pd.DataFrame({
        "step": step_indices,
        "timestamp_hour": np.round(hour_of_day, 4),
        "day_of_year": day_of_year,
        "day_of_week": day_of_week,
        "is_weekend": is_weekend,
        "ambient_temp_c": np.round(ambient_temp, 2),
        "solar_irradiance_wm2": np.round(solar_ghi, 1),
        "direct_normal_irradiance_wm2": np.round(solar_dni, 1),
        "relative_humidity_pct": np.round(rel_humidity, 1),
        "dynamic_lmp_price_usd_per_kwh": np.round(dynamic_lmp, 4),
        "price_forecast_2h": np.round(p_2h, 4),
        "price_forecast_4h": np.round(p_4h, 4),
        "price_forecast_6h": np.round(p_6h, 4),
        "dr_event_active": dr_active,
        "occupancy_zone_1": occ_z1.astype(int),
        "occupancy_zone_2": occ_z2.astype(int),
        "occupancy_zone_3": occ_z3.astype(int),
        "occupancy_zone_4": occ_z4.astype(int),
        "occupancy_zone_5": occ_z5.astype(int),
        "internal_sensible_load_kw": np.round(internal_load, 2),
    })

    return df


def build_and_save_all_datasets():
    DATASET_DIR.mkdir(parents=True, exist_ok=True)

    print("Generating Leak-Free Disjoint BMS Tag Datasets...")
    train_data, val_data, test_ood_data = generate_slm_tag_corpus_leak_free(seed=42)
    all_data = train_data + val_data + test_ood_data

    # Save disjoint split JSONLs
    for p, d in [(SLM_TRAIN_PATH, train_data), (SLM_VAL_PATH, val_data), (SLM_TEST_OOD_PATH, test_ood_data), (SLM_JSONL_PATH, all_data)]:
        with open(p, "w", encoding="utf-8") as f:
            for item in d:
                f.write(json.dumps(item) + "\n")
        print(f" -> Saved {len(d)} items to {p.name}")

    # Save full CSV
    keys = list(all_data[0].keys())
    with open(SLM_CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(all_data)
    print(f" -> Saved full CSV dataset to {SLM_CSV_PATH.name}")

    print("Generating 1-Year 5-minute Grid & Weather Thermal Timeseries dataset...")
    df_grid = generate_grid_weather_thermal_timeseries(num_days=365, seed=42)
    df_grid.to_csv(GRID_CSV_PATH, index=False)
    print(f" -> Saved {len(df_grid)} steps to {GRID_CSV_PATH.name}")

    try:
        df_grid.to_parquet(GRID_PARQUET_PATH, index=False)
        print(f" -> Saved Parquet dataset to {GRID_PARQUET_PATH.name}")
    except Exception:
        pass


if __name__ == "__main__":
    build_and_save_all_datasets()
