import math

def calculate_pmv(
    temp: float,
    humidity: float = 50.0,
    rad_temp: float | None = None,
    air_velocity: float = 0.1,
    met: float = 1.1,
    clo: float = 0.7,
    wme: float = 0.0
) -> float:
    if rad_temp is None:
        rad_temp = temp

    ta = float(temp)
    tr = float(rad_temp)
    vel = max(float(air_velocity), 0.01)
    rh = max(min(float(humidity), 100.0), 0.0)

    m = met * 58.15
    w = wme * 58.15
    mw = m - w
    icl = 0.155 * clo

    if icl <= 0.078:
        fcl = 1.0 + 1.29 * icl
    else:
        fcl = 1.05 + 0.645 * icl

    pa = rh * 10.0 * math.exp(16.6536 - 4030.183 / (ta + 235.0))

    tra = tr + 273.15
    taa = ta + 273.15
    tcla = taa + (35.5 - ta) / (3.5 * (6.45 * icl) + 0.1)

    p1 = icl * fcl
    p2 = p1 * 3.96 * 1e-8
    p4 = 308.7 - 0.028 * mw + p2 * (tra ** 4)

    tcl = tcla
    for _ in range(150):
        tcl_old = tcl
        hc = 12.1 * math.sqrt(vel)
        hcf = 2.38 * (abs(tcl - taa) ** 0.25)
        if hcf > hc:
            hc = hcf
        tcl = (p4 + p1 * hc * taa - p2 * (tcl ** 4)) / (1.0 + p1 * hc)
        if abs(tcl - tcl_old) < 0.0001:
            break

    hl1 = 3.05 * 0.001 * (5733.0 - 6.99 * mw - pa)
    hl2 = 0.42 * (mw - 58.15) if mw > 58.15 else 0.0
    hl3 = 1.7 * 1e-5 * m * (5867.0 - pa)
    hl4 = 0.0014 * m * (34.0 - ta)
    hl5 = 3.96 * 1e-8 * fcl * ((tcl ** 4) - (tra ** 4))
    hl6 = fcl * hc * (tcl - taa)

    thermal_load = mw - hl1 - hl2 - hl3 - hl4 - hl5 - hl6
    pmv = (0.303 * math.exp(-0.036 * m) + 0.028) * thermal_load
    return round(float(pmv), 3)

def calculate_ppd(pmv: float) -> float:
    pmv_val = float(pmv)
    ppd = 100.0 - 95.0 * math.exp(-0.03353 * (pmv_val ** 4) - 0.2179 * (pmv_val ** 2))
    return round(float(ppd), 2)

def is_ashrae55_compliant(pmv: float) -> bool:
    return -0.5 <= pmv <= 0.5
