"""Fundamental and atomic physical constants used by the BEC models."""

from math import pi
from typing import Final


class PhysicalConstants:
    """Namespace for SI physical constants and rubidium-87 atomic data.

    The class is intentionally not meant to be instantiated. Its attributes
    preserve the historical numerical values used in the original notebooks and
    scripts while separating them from experimental parameters and derived
    quantities.
    """

    BOLTZMANN_CONSTANT: Final[float] = 1.380649e-23  # J/K
    REDUCED_PLANCK_CONSTANT: Final[float] = 1.05457180013e-34  # J.s
    SPEED_OF_LIGHT: Final[float] = 2.99792458e8  # m/s
    STANDARD_GRAVITY: Final[float] = 9.80665  # m/s^2

    RUBIDIUM_87_MASS: Final[float] = 1.4431606483768263e-25  # kg
    RUBIDIUM_87_SCATTERING_LENGTH: Final[float] = 100 * 0.529e-10  # m

    RUBIDIUM_D1_WAVELENGTH: Final[float] = 7.94978851156e-7  # m
    RUBIDIUM_D1_ANGULAR_FREQUENCY: Final[float] = (2 * pi * 3.77107463380e14)  # rad/s
    RUBIDIUM_D1_LINEWIDTH: Final[float] = 2 * pi * 5.7500e6  # rad/s

    RUBIDIUM_D2_WAVELENGTH: Final[float] = 7.80241209686e-7  # m
    RUBIDIUM_D2_ANGULAR_FREQUENCY: Final[float] = (2 * pi * 3.842304844685e14)  # rad/s
    RUBIDIUM_D2_LINEWIDTH: Final[float] = 2 * pi * 6.0666e6  # rad/s
