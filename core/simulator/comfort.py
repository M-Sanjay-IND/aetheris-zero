def calculate_pmv(temp: float, humidity: float, rad_temp: float, air_velocity: float) -> float:
    """Calculates Predicted Mean Vote."""
    return 0.0

def is_ashrae55_compliant(pmv: float) -> bool:
    """Checks ASHRAE 55 compliance."""
    return -0.5 <= pmv <= 0.5
