"""Control inversion for crossed painted dipole potentials."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import numpy as np
from scipy.optimize import least_squares

from bec_fast_manipulation.painted_potential.model import CrossedPaintedDipolePotential


@dataclass(frozen=True)
class PaintedPotentialControls:
    """Result of the inverse painted-potential control problem."""

    power_w: float
    amplitude_m: float
    horizontal_frequency_hz: float
    vertical_frequency_hz: float
    residuals: np.ndarray
    success: bool
    message: str


class PaintedPotentialControl:
    """Invert local trap frequencies to experimental controls P and h."""

    def __init__(self, potential: CrossedPaintedDipolePotential) -> None:
        if not isinstance(potential, CrossedPaintedDipolePotential):
            raise TypeError("potential must be a CrossedPaintedDipolePotential.")
        self.potential = potential

    def frequencies_from_controls(self, power_w: float, amplitude_m: float, *, include_gravity: bool = True) -> dict:
        """Return horizontal and vertical frequencies for controls P and h."""
        return self.potential.trap_frequencies(power_w, amplitude_m, include_gravity=include_gravity)

    def controls_from_frequencies(
        self,
        horizontal_frequency_hz: float,
        vertical_frequency_hz: float,
        *,
        initial_guess: tuple[float, float],
        power_bounds: tuple[float, float],
        amplitude_bounds: tuple[float, float],
        include_gravity: bool = True,
    ) -> PaintedPotentialControls:
        """Solve the bounded inverse problem from target frequencies to P,h."""
        horizontal_target = self._validate_positive_finite(horizontal_frequency_hz, "horizontal_frequency_hz")
        vertical_target = self._validate_positive_finite(vertical_frequency_hz, "vertical_frequency_hz")
        power_bounds = self._validate_bounds(power_bounds, "power_bounds", positive=True)
        amplitude_bounds = self._validate_bounds(amplitude_bounds, "amplitude_bounds", positive=False)
        initial = np.array(
            [
                self._validate_positive_finite(initial_guess[0], "initial_guess[0]"),
                self._validate_non_negative_finite(initial_guess[1], "initial_guess[1]"),
            ],
            dtype=float,
        )

        def residuals(variables):
            frequencies = self.frequencies_from_controls(
                variables[0],
                variables[1],
                include_gravity=include_gravity,
            )
            return np.array(
                [
                    (frequencies["horizontal_frequency_hz"] - horizontal_target) / horizontal_target,
                    (frequencies["vertical_frequency_hz"] - vertical_target) / vertical_target,
                ],
                dtype=float,
            )

        result = least_squares(
            residuals,
            initial,
            bounds=([power_bounds[0], amplitude_bounds[0]], [power_bounds[1], amplitude_bounds[1]]),
            xtol=1e-10,
            ftol=1e-10,
            gtol=1e-10,
        )
        frequencies = self.frequencies_from_controls(result.x[0], result.x[1], include_gravity=include_gravity)
        final_residuals = np.array(
            [
                (frequencies["horizontal_frequency_hz"] - horizontal_target) / horizontal_target,
                (frequencies["vertical_frequency_hz"] - vertical_target) / vertical_target,
            ],
            dtype=float,
        )
        return PaintedPotentialControls(
            power_w=float(result.x[0]),
            amplitude_m=float(result.x[1]),
            horizontal_frequency_hz=float(frequencies["horizontal_frequency_hz"]),
            vertical_frequency_hz=float(frequencies["vertical_frequency_hz"]),
            residuals=final_residuals,
            success=bool(result.success),
            message=str(result.message),
        )

    @staticmethod
    def _validate_bounds(bounds: tuple[float, float], name: str, *, positive: bool) -> tuple[float, float]:
        if len(bounds) != 2:
            raise ValueError(f"{name} must contain exactly two values.")
        lower = float(bounds[0])
        upper = float(bounds[1])
        if not isfinite(lower) or not isfinite(upper):
            raise ValueError(f"{name} must contain only finite values.")
        if positive and lower <= 0.0:
            raise ValueError(f"{name}[0] must be strictly positive.")
        if not positive and lower < 0.0:
            raise ValueError(f"{name}[0] must be non-negative.")
        if upper <= lower:
            raise ValueError(f"{name}[1] must be greater than {name}[0].")
        return lower, upper

    @staticmethod
    def _validate_positive_finite(value: float, name: str) -> float:
        value = float(value)
        if not isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and strictly positive.")
        return value

    @staticmethod
    def _validate_non_negative_finite(value: float, name: str) -> float:
        value = float(value)
        if not isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and non-negative.")
        return value
