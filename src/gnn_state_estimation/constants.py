"""Physical constants for Earth-orbit simulation."""

MU_EARTH = 3.986004418e14  # m^3 / s^2
R_EARTH = 6378.1363e3  # m
J2 = 1.08262668e-3  # unitless
# Higher-order zonal geopotential coefficients (unnormalised, EGM-class
# nominal values) used by the higher-fidelity precise-reference propagator.
J3 = -2.53265649e-6  # unitless
J4 = -1.61962159e-6  # unitless
J5 = -2.27295e-7  # unitless (EGM-class nominal value)
J6 = 5.40681e-7  # unitless (EGM-class nominal value)
EARTH_ROTATION_RATE = 7.2921159e-5  # rad / s
R_SUN = 695_700_000.0  # m

# Simplified exponential atmosphere for LEO drag modeling.
ATMOSPHERE_SURFACE_DENSITY = 1.225  # kg / m^3
ATMOSPHERE_SCALE_HEIGHT = 8500.0  # m

# Third-body and SRP constants for higher-fidelity perturbations.
MU_SUN = 1.32712440018e20  # m^3 / s^2
MU_MOON = 4.9048695e12  # m^3 / s^2
AU = 149_597_870_700.0  # m
MOON_ORBIT_RADIUS = 384_400_000.0  # m
MOON_MEAN_MOTION_RADPS = 2.6617e-6  # rad / s
EARTH_ORBIT_MEAN_MOTION_RADPS = 1.9910213e-7  # rad / s
SOLAR_RADIATION_PRESSURE_1AU = 4.56e-6  # N / m^2
