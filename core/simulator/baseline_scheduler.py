class BaselineScheduler:
    def __init__(
        self,
        occupied_start_hour: float = 7.0,
        occupied_end_hour: float = 19.0,
        cooling_setpoint_occ: float = 22.0,
        cooling_setpoint_unocc: float = 26.0,
        heating_setpoint_occ: float = 20.0,
        heating_setpoint_unocc: float = 16.0,
        chw_setpoint: float = 6.5
    ):
        self.occupied_start = occupied_start_hour
        self.occupied_end = occupied_end_hour
        self.t_cool_occ = cooling_setpoint_occ
        self.t_cool_unocc = cooling_setpoint_unocc
        self.t_heat_occ = heating_setpoint_occ
        self.t_heat_unocc = heating_setpoint_unocc
        self.chw_setpoint = chw_setpoint

    def is_occupied_hour(self, current_hour: float) -> bool:
        norm_hour = current_hour % 24.0
        return self.occupied_start <= norm_hour < self.occupied_end

    def get_setpoint(self, current_hour: float) -> float:
        if self.is_occupied_hour(current_hour):
            return self.t_cool_occ
        return self.t_cool_unocc

    def get_actions(self, current_hour: float, zone_ids: list[str]) -> dict:
        target_sp = self.get_setpoint(current_hour)
        zone_setpoints = {zid: target_sp for zid in zone_ids}
        vav_dampers = {zid: 0.8 if self.is_occupied_hour(current_hour) else 0.2 for zid in zone_ids}

        return {
            "zone_setpoints": zone_setpoints,
            "chiller_chw_setpoint": self.chw_setpoint,
            "vav_damper_positions": vav_dampers
        }
