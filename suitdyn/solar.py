"""Solar rotation used to separate spacecraft pointing from the motion of solar features."""
import numpy as np

# Snodgrass & Ulrich (1990) sidereal rotation of magnetic features, deg/day: A + B sin²φ + C sin⁴φ
SU90 = (14.713, -2.396, -1.787)
# Earth's (and L1's) mean orbital rate, deg/day, subtracted to get the synodic rate seen from L1
ORBIT = 0.9856


def synodic_deg_per_day(lat_deg):
    s2 = np.sin(np.deg2rad(lat_deg)) ** 2
    return SU90[0] + SU90[1] * s2 + SU90[2] * s2 ** 2 - ORBIT


def disk_centre_motion_px(R_px, b0_deg, dt_s, crota2_deg):
    """Image displacement (dx, dy) of features at disk centre over dt_s, in detector pixels."""
    w = np.deg2rad(synodic_deg_per_day(b0_deg)) / 86400.0
    v = R_px * np.cos(np.deg2rad(b0_deg)) * w * dt_s
    a = np.deg2rad(-crota2_deg)
    return v * np.cos(a), v * np.sin(a)
