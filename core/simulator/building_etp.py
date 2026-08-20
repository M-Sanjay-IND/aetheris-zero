import math
import numpy as np
from core.simulator.comfort import calculate_pmv, calculate_ppd, is_ashrae55_compliant
from core.simulator.baseline_scheduler import BaselineScheduler

class BuildingSimulator:
    def __init__(self, config: dict | None = None):
        self.config = config or {}
        
        sim_cfg = self.config.get("simulation", {})
        self.dt = float(sim_cfg.get("time_step_sec", 300))
        self.total_hours = float(sim_cfg.get("total_hours", 24))
        self.start_hour = float(sim_cfg.get("start_hour", 0.0))
        
        self.zones_cfg = self.config.get("zones", self._default_zones_config())
        self.adjacencies_cfg = self.config.get("adjacencies", self._default_adjacencies_config())
        self.equipment_cfg = self.config.get("equipment", self._default_equipment_config())
        self.points_mapping = self.config.get("points", {})
        
        self.baseline_scheduler = BaselineScheduler()
        self.zone_ids = [z["zone_id"] for z in self.zones_cfg]
        
        self.weather_cfg = self.config.get("weather", {
            "t_mean": 28.0,
            "t_amp": 8.0,
            "sol_peak": 850.0
        })
        
        self.reset()

    def _default_zones_config(self) -> list[dict]:
        return [
            {
                "zone_id": "zone_1",
                "name": "Core Zone",
                "C_z": 18000.0,
                "C_m": 90000.0,
                "R_ext": 4.5,
                "R_m": 0.6,
                "initial_temp": 22.5,
                "initial_mass_temp": 22.0,
                "occupancy_max": 25,
                "floor_area_sqm": 300.0,
                "solar_factor": 0.05,
                "max_cooling_kw": 35.0
            },
            {
                "zone_id": "zone_2",
                "name": "Perimeter North",
                "C_z": 12000.0,
                "C_m": 60000.0,
                "R_ext": 2.2,
                "R_m": 0.5,
                "initial_temp": 22.8,
                "initial_mass_temp": 22.2,
                "occupancy_max": 15,
                "floor_area_sqm": 180.0,
                "solar_factor": 0.2,
                "max_cooling_kw": 25.0
            },
            {
                "zone_id": "zone_3",
                "name": "Perimeter South",
                "C_z": 12000.0,
                "C_m": 60000.0,
                "R_ext": 2.0,
                "R_m": 0.5,
                "initial_temp": 23.0,
                "initial_mass_temp": 22.5,
                "occupancy_max": 15,
                "floor_area_sqm": 180.0,
                "solar_factor": 0.45,
                "max_cooling_kw": 30.0
            },
            {
                "zone_id": "zone_4",
                "name": "Perimeter East",
                "C_z": 12000.0,
                "C_m": 60000.0,
                "R_ext": 2.1,
                "R_m": 0.5,
                "initial_temp": 22.7,
                "initial_mass_temp": 22.1,
                "occupancy_max": 15,
                "floor_area_sqm": 180.0,
                "solar_factor": 0.35,
                "max_cooling_kw": 28.0
            },
            {
                "zone_id": "zone_5",
                "name": "Perimeter West",
                "C_z": 12000.0,
                "C_m": 60000.0,
                "R_ext": 2.1,
                "R_m": 0.5,
                "initial_temp": 22.9,
                "initial_mass_temp": 22.3,
                "occupancy_max": 15,
                "floor_area_sqm": 180.0,
                "solar_factor": 0.4,
                "max_cooling_kw": 28.0
            }
        ]

    def _default_adjacencies_config(self) -> list[dict]:
        return [
            {"from_zone": "zone_1", "to_zone": "zone_2", "R_adj": 1.2},
            {"from_zone": "zone_1", "to_zone": "zone_3", "R_adj": 1.2},
            {"from_zone": "zone_1", "to_zone": "zone_4", "R_adj": 1.2},
            {"from_zone": "zone_1", "to_zone": "zone_5", "R_adj": 1.2},
            {"from_zone": "zone_2", "to_zone": "zone_4", "R_adj": 2.5},
            {"from_zone": "zone_2", "to_zone": "zone_5", "R_adj": 2.5},
            {"from_zone": "zone_3", "to_zone": "zone_4", "R_adj": 2.5},
            {"from_zone": "zone_3", "to_zone": "zone_5", "R_adj": 2.5}
        ]

    def _default_equipment_config(self) -> dict:
        return {
            "chiller": {
                "capacity_kw": 150.0,
                "cop_base": 3.8,
                "chw_temp_min": 4.0,
                "chw_temp_max": 12.0
            },
            "ahu": {
                "max_airflow_m3s": 10.0,
                "fan_power_max_kw": 12.0,
                "supply_air_temp_min": 12.0,
                "supply_air_temp_max": 24.0
            },
            "vav_boxes": {
                "min_damper_position": 0.2,
                "max_damper_position": 1.0
            }
        }

    def reset(self, seed: int | None = None) -> dict:
        if seed is not None:
            np.random.seed(seed)

        self.current_step = 0
        self.sim_time_sec = 0.0
        self.current_hour = self.start_hour

        self.t_zones = {}
        self.t_mass = {}
        for z in self.zones_cfg:
            zid = z["zone_id"]
            self.t_zones[zid] = float(z.get("initial_temp", 22.5))
            self.t_mass[zid] = float(z.get("initial_mass_temp", 22.0))

        self.baseline_t_zones = dict(self.t_zones)
        self.baseline_t_mass = dict(self.t_mass)

        self.cum_cost_actual = 0.0
        self.cum_cost_baseline = 0.0
        self.cum_energy_actual_kwh = 0.0
        self.cum_energy_baseline_kwh = 0.0
        self.peak_kw_actual = 0.0
        self.peak_kw_baseline = 0.0

        self.last_actions = {}
        self.dr_event_active = False
        self.dr_price_override = None

        return self.get_state()

    def get_weather(self, hour: float) -> tuple[float, float]:
        norm_hour = hour % 24.0
        t_mean = self.weather_cfg.get("t_mean", 28.0)
        t_amp = self.weather_cfg.get("t_amp", 8.0)
        sol_peak = self.weather_cfg.get("sol_peak", 850.0)

        t_ext = t_mean + t_amp * math.sin(2.0 * math.pi * (norm_hour - 9.0) / 24.0)
        
        if 6.0 <= norm_hour <= 18.0:
            solar_irradiance = sol_peak * math.sin(math.pi * (norm_hour - 6.0) / 12.0)
        else:
            solar_irradiance = 0.0

        return t_ext, max(0.0, solar_irradiance)

    def get_occupancy(self, hour: float, zone_cfg: dict) -> int:
        norm_hour = hour % 24.0
        max_occ = zone_cfg.get("occupancy_max", 15)
        if 8.0 <= norm_hour <= 17.5:
            profile = math.sin(math.pi * (norm_hour - 8.0) / 9.5)
            return max(1, int(round(max_occ * max(0.0, profile))))
        elif 7.0 <= norm_hour < 8.0 or 17.5 < norm_hour <= 19.0:
            return max(1, int(round(max_occ * 0.25)))
        return 0

    def get_dynamic_lmp_price(self, hour: float) -> float:
        if self.dr_price_override is not None:
            return float(self.dr_price_override)
        
        norm_hour = hour % 24.0
        if 14.0 <= norm_hour < 18.0:
            return 0.85
        elif 11.0 <= norm_hour < 14.0 or 18.0 <= norm_hour < 21.0:
            return 0.35
        elif 0.0 <= norm_hour < 6.0:
            return 0.08
        return 0.18

    def _compute_interzone_flux(self, zone_id: str, t_dict: dict) -> float:
        flux = 0.0
        tz = t_dict[zone_id]
        for adj in self.adjacencies_cfg:
            from_z = adj["from_zone"]
            to_z = adj["to_zone"]
            r_adj = float(adj.get("R_adj", 1.5))
            if from_z == zone_id and to_z in t_dict:
                flux += (t_dict[to_z] - tz) / r_adj
            elif to_z == zone_id and from_z in t_dict:
                flux += (t_dict[from_z] - tz) / r_adj
        return flux

    def _compute_hvac_cooling(
        self,
        zone_cfg: dict,
        current_temp: float,
        target_setpoint: float,
        dt_sec: float
    ) -> float:
        cz = float(zone_cfg.get("C_z", 15000.0))
        max_cool_kw = float(zone_cfg.get("max_cooling_kw", 30.0))
        
        if current_temp <= target_setpoint:
            return 0.0
        
        needed_cooling_kw = (cz * (current_temp - target_setpoint)) / dt_sec
        return float(min(max_cool_kw, max(0.0, needed_cooling_kw)))

    def _compute_power_draw(
        self,
        total_cooling_kw: float,
        chw_setpoint: float,
        t_ext: float,
        damper_positions: list[float]
    ) -> tuple[float, float, float]:
        chiller_cfg = self.equipment_cfg.get("chiller", {})
        cop_base = float(chiller_cfg.get("cop_base", 3.8))
        cop = cop_base * (1.0 + 0.025 * (chw_setpoint - 6.5) - 0.015 * (t_ext - 30.0))
        cop = max(1.5, min(6.5, cop))

        chiller_elec_kw = total_cooling_kw / cop if total_cooling_kw > 0 else 1.2
        
        ahu_cfg = self.equipment_cfg.get("ahu", {})
        fan_power_max = float(ahu_cfg.get("fan_power_max_kw", 12.0))
        avg_damper = float(np.mean(damper_positions)) if damper_positions else 0.5
        fan_elec_kw = fan_power_max * (avg_damper ** 3)

        base_elec_kw = 5.0
        total_elec_kw = chiller_elec_kw + fan_elec_kw + base_elec_kw
        return float(chiller_elec_kw), float(fan_elec_kw), float(total_elec_kw)

    def step(self, actions: dict) -> tuple[dict, float, bool, dict]:
        t_ext, solar_irradiance = self.get_weather(self.current_hour)
        price = self.get_dynamic_lmp_price(self.current_hour)
        
        zone_setpoints = actions.get("zone_setpoints", {})
        chw_setpoint = float(actions.get("chiller_chw_setpoint", 6.5))
        damper_positions_dict = actions.get("vav_damper_positions", {})

        actual_cooling_loads = {}
        actual_dampers = []
        
        for z in self.zones_cfg:
            zid = z["zone_id"]
            sp = float(zone_setpoints.get(zid, 22.0))
            cooling_kw = self._compute_hvac_cooling(z, self.t_zones[zid], sp, self.dt)
            actual_cooling_loads[zid] = cooling_kw
            damper = float(damper_positions_dict.get(zid, 0.6))
            actual_dampers.append(damper)

        total_cooling_actual_kw = sum(actual_cooling_loads.values())
        chiller_kw, fan_kw, total_kw = self._compute_power_draw(
            total_cooling_actual_kw, chw_setpoint, t_ext, actual_dampers
        )

        for z in self.zones_cfg:
            zid = z["zone_id"]
            cz = float(z.get("C_z", 15000.0))
            cm = float(z.get("C_m", 75000.0))
            r_ext = float(z.get("R_ext", 2.5))
            r_m = float(z.get("R_m", 0.5))
            sol_factor = float(z.get("solar_factor", 0.2))

            occ = self.get_occupancy(self.current_hour, z)
            q_occ = occ * 0.10
            q_sol = solar_irradiance * 0.001 * (float(z.get("floor_area_sqm", 200.0)) * 0.15) * sol_factor
            q_hvac = -actual_cooling_loads[zid]

            q_adj = self._compute_interzone_flux(zid, self.t_zones)
            q_envelope = (t_ext - self.t_zones[zid]) / r_ext
            q_mass_air = (self.t_mass[zid] - self.t_zones[zid]) / r_m

            dt_z_dt = (q_envelope + q_mass_air + q_adj + 0.3 * q_sol + q_occ + q_hvac) / cz
            dt_m_dt = ((self.t_zones[zid] - self.t_mass[zid]) / r_m + 0.7 * q_sol) / cm

            self.t_zones[zid] += float(dt_z_dt * self.dt)
            self.t_mass[zid] += float(dt_m_dt * self.dt)

        base_actions = self.baseline_scheduler.get_actions(self.current_hour, self.zone_ids)
        base_cooling_loads = {}
        base_dampers = []
        for z in self.zones_cfg:
            zid = z["zone_id"]
            b_sp = float(base_actions["zone_setpoints"].get(zid, 22.0))
            b_cool = self._compute_hvac_cooling(z, self.baseline_t_zones[zid], b_sp, self.dt)
            base_cooling_loads[zid] = b_cool
            base_dampers.append(float(base_actions["vav_damper_positions"].get(zid, 0.5)))

        total_cooling_base_kw = sum(base_cooling_loads.values())
        _, _, total_base_kw = self._compute_power_draw(
            total_cooling_base_kw, base_actions["chiller_chw_setpoint"], t_ext, base_dampers
        )

        for z in self.zones_cfg:
            zid = z["zone_id"]
            cz = float(z.get("C_z", 15000.0))
            cm = float(z.get("C_m", 75000.0))
            r_ext = float(z.get("R_ext", 2.5))
            r_m = float(z.get("R_m", 0.5))
            sol_factor = float(z.get("solar_factor", 0.2))

            occ = self.get_occupancy(self.current_hour, z)
            q_occ = occ * 0.10
            q_sol = solar_irradiance * 0.001 * (float(z.get("floor_area_sqm", 200.0)) * 0.15) * sol_factor
            q_hvac = -base_cooling_loads[zid]

            q_adj = self._compute_interzone_flux(zid, self.baseline_t_zones)
            q_envelope = (t_ext - self.baseline_t_zones[zid]) / r_ext
            q_mass_air = (self.baseline_t_mass[zid] - self.baseline_t_zones[zid]) / r_m

            dt_z_dt = (q_envelope + q_mass_air + q_adj + 0.3 * q_sol + q_occ + q_hvac) / cz
            dt_m_dt = ((self.baseline_t_zones[zid] - self.baseline_t_mass[zid]) / r_m + 0.7 * q_sol) / cm

            self.baseline_t_zones[zid] += float(dt_z_dt * self.dt)
            self.baseline_t_mass[zid] += float(dt_m_dt * self.dt)

        hours_fraction = self.dt / 3600.0
        step_cost_actual = total_kw * hours_fraction * price
        step_cost_baseline = total_base_kw * hours_fraction * price

        self.cum_cost_actual += step_cost_actual
        self.cum_cost_baseline += step_cost_baseline
        self.cum_energy_actual_kwh += total_kw * hours_fraction
        self.cum_energy_baseline_kwh += total_base_kw * hours_fraction
        self.peak_kw_actual = max(self.peak_kw_actual, total_kw)
        self.peak_kw_baseline = max(self.peak_kw_baseline, total_base_kw)

        self.current_step += 1
        self.sim_time_sec += self.dt
        self.current_hour = self.start_hour + (self.sim_time_sec / 3600.0)

        discomfort_penalty = 0.0
        for z in self.zones_cfg:
            zid = z["zone_id"]
            tz = self.t_zones[zid]
            pmv = calculate_pmv(tz, humidity=50.0)
            if not is_ashrae55_compliant(pmv):
                discomfort_penalty += (abs(pmv) - 0.5) ** 2

        reward = - (step_cost_actual + 2.0 * discomfort_penalty)
        done = self.current_hour >= (self.start_hour + self.total_hours)

        state = self.get_state()
        state["power"]["chiller_kw"] = round(chiller_kw, 2)
        state["power"]["fans_kw"] = round(fan_kw, 2)
        state["power"]["total_hvac_kw"] = round(total_kw, 2)
        state["power"]["baseline_hvac_kw"] = round(total_base_kw, 2)
        state["power"]["demand_shaved_kw"] = round(max(0.0, total_base_kw - total_kw), 2)

        info = {
            "step_cost_actual": step_cost_actual,
            "step_cost_baseline": step_cost_baseline,
            "discomfort_penalty": discomfort_penalty
        }

        return state, float(reward), bool(done), info

    def get_state(self) -> dict:
        t_ext, solar_irradiance = self.get_weather(self.current_hour)
        price = self.get_dynamic_lmp_price(self.current_hour)

        zones_state = {}
        for z in self.zones_cfg:
            zid = z["zone_id"]
            tz = self.t_zones[zid]
            tm = self.t_mass[zid]
            pmv = calculate_pmv(tz, humidity=50.0)
            ppd = calculate_ppd(pmv)
            occ = self.get_occupancy(self.current_hour, z)

            zones_state[zid] = {
                "name": z.get("name", zid),
                "temp_c": round(tz, 2),
                "mass_temp_c": round(tm, 2),
                "setpoint_c": 22.0,
                "pmv": pmv,
                "ppd": ppd,
                "comfort_compliant": is_ashrae55_compliant(pmv),
                "occupancy": occ,
                "cooling_load_kw": 0.0
            }

        savings_usd = max(0.0, self.cum_cost_baseline - self.cum_cost_actual)
        peak_shave_pct = 0.0
        if self.peak_kw_baseline > 0:
            peak_shave_pct = max(0.0, ((self.peak_kw_baseline - self.peak_kw_actual) / self.peak_kw_baseline) * 100.0)

        return {
            "step": self.current_step,
            "timestamp_hour": round(self.current_hour, 3),
            "ambient_temp_c": round(t_ext, 2),
            "solar_irradiance_wm2": round(solar_irradiance, 1),
            "dynamic_lmp_price": round(price, 3),
            "grid_dr_event_active": self.dr_event_active,
            "zones": zones_state,
            "power": {
                "chiller_kw": 0.0,
                "fans_kw": 0.0,
                "total_hvac_kw": 0.0,
                "baseline_hvac_kw": 0.0,
                "demand_shaved_kw": 0.0
            },
            "safety": {
                "intervention_active": False,
                "shield_status": "OPTIMAL",
                "dwell_time_remaining_sec": 0
            },
            "metrics": {
                "cumulative_cost_actual": round(self.cum_cost_actual, 2),
                "cumulative_cost_baseline": round(self.cum_cost_baseline, 2),
                "cumulative_savings_usd": round(savings_usd, 2),
                "cumulative_energy_actual_kwh": round(self.cum_energy_actual_kwh, 2),
                "cumulative_energy_baseline_kwh": round(self.cum_energy_baseline_kwh, 2),
                "peak_demand_reduction_pct": round(peak_shave_pct, 1)
            }
        }
