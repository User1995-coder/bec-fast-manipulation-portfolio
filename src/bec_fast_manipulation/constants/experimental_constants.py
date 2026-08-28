"""Fixed parameters of the historical experimental setup."""

from math import pi
from typing import Final


class ExperimentalConstants:
    """Namespace for fixed historical experimental setup parameters.

    All quantities are expressed in SI units.
    """

    LASER_WAVELENGTH: Final[float] = 1.064e-6  # m
    HORIZONTAL_BEAM_WAIST: Final[float] = 50e-6  # m
    VERTICAL_BEAM_WAIST: Final[float] = 50e-6  # m
    TRAP_LASER_WAVELENGTH_M: Final[float] = LASER_WAVELENGTH  # m
    TRAP_BEAM_HORIZONTAL_WAIST_M: Final[float] = HORIZONTAL_BEAM_WAIST  # m
    TRAP_BEAM_VERTICAL_WAIST_M: Final[float] = VERTICAL_BEAM_WAIST  # m
    CONDENSATE_ATOM_NUMBER: Final[int] = 100_000

    INITIAL_TRAP_ANGULAR_FREQUENCY_X: Final[float] = 2 * pi * 500  # rad/s
    INITIAL_TRAP_ANGULAR_FREQUENCY_Y: Final[float] = 2 * pi * 600  # rad/s
    INITIAL_TRAP_ANGULAR_FREQUENCY_Z: Final[float] = 2 * pi * 700  # rad/s

    FINAL_TRAP_ANGULAR_FREQUENCY_X: Final[float] = 2 * pi * 5  # rad/s
    FINAL_TRAP_ANGULAR_FREQUENCY_Y: Final[float] = 2 * pi * 5  # rad/s
    FINAL_TRAP_ANGULAR_FREQUENCY_Z: Final[float] = 2 * pi * 60  # rad/s
