"""Analytical retro-sinusoidal inverse engineering for Castin-Dum scaling."""

from __future__ import annotations

from math import isfinite, pi

import numpy as np

from bec_fast_manipulation.castin_dum import CastinDumModel
from bec_fast_manipulation.constants import ExperimentalConstants


class RetroSinusoidalProtocol:
    """Imposed scaling factors and inverse Castin-Dum angular frequencies.

    The direct Castin-Dum model maps trap angular frequencies to scaling
    factors. This class represents the inverse analytical construction:
    imposed ``lambda_i(t)`` profiles are used to reconstruct
    ``omega_i_squared(t)``. It never integrates differential equations.
    """

    _MONOTONIC_TOLERANCE = 1e-12
    _OMEGA_SQUARED_RELATIVE_TOLERANCE = 1e-12

    def __init__(
        self,
        final_time: float,
        a: float,
        b: float,
        lambda_x_initial: float = 1.0,
        lambda_y_initial: float = 1.0,
        lambda_z_initial: float = 1.0,
        lambda_x_final: float | None = None,
        lambda_y_final: float | None = None,
        lambda_z_final: float | None = None,
        omega_x_initial: float = ExperimentalConstants.INITIAL_TRAP_ANGULAR_FREQUENCY_X,
        omega_y_initial: float = ExperimentalConstants.INITIAL_TRAP_ANGULAR_FREQUENCY_Y,
        omega_z_initial: float = ExperimentalConstants.INITIAL_TRAP_ANGULAR_FREQUENCY_Z,
        require_monotonic_phase: bool = False,
    ) -> None:
        self.final_time = self._validate_positive_finite(final_time, "final_time")
        self.a = self._validate_finite(a, "a")
        self.b = self._validate_finite(b, "b")
        self.phase_denominator = 1.0 + self.a + self.b
        if not isfinite(self.phase_denominator) or np.isclose(self.phase_denominator, 0.0):
            raise ValueError("1 + a + b must be non-zero.")

        self.lambda_x_initial = self._validate_positive_finite(lambda_x_initial, "lambda_x_initial")
        self.lambda_y_initial = self._validate_positive_finite(lambda_y_initial, "lambda_y_initial")
        self.lambda_z_initial = self._validate_positive_finite(lambda_z_initial, "lambda_z_initial")

        final_values = (lambda_x_final, lambda_y_final, lambda_z_final)
        if all(value is None for value in final_values):
            (
                self.lambda_x_final,
                self.lambda_y_final,
                self.lambda_z_final,
            ) = CastinDumModel().final_scaling_factors()
        elif any(value is None for value in final_values):
            raise ValueError("lambda_x_final, lambda_y_final, and lambda_z_final must be provided together.")
        else:
            self.lambda_x_final = self._validate_positive_finite(lambda_x_final, "lambda_x_final")
            self.lambda_y_final = self._validate_positive_finite(lambda_y_final, "lambda_y_final")
            self.lambda_z_final = self._validate_positive_finite(lambda_z_final, "lambda_z_final")

        self.omega_x_initial = self._validate_positive_finite(omega_x_initial, "omega_x_initial")
        self.omega_y_initial = self._validate_positive_finite(omega_y_initial, "omega_y_initial")
        self.omega_z_initial = self._validate_positive_finite(omega_z_initial, "omega_z_initial")

        if require_monotonic_phase and not self.is_phase_monotonic():
            raise ValueError("phase must be monotone increasing when require_monotonic_phase is True.")

    def phase(self, t):
        """Return the retro-sinusoidal phase phi(t), in radians."""
        time, scalar_input = self._validate_time(t)
        u = time / self.final_time
        result = 2.0 * pi * u * (1.0 + self.a * u + self.b * u**2) / self.phase_denominator
        return self._restore_type(result, scalar_input)

    def phase_velocity(self, t):
        """Return phi_dot(t), in 1/s."""
        time, scalar_input = self._validate_time(t)
        u = time / self.final_time
        result = (
            (2.0 * pi / self.final_time)
            * (1.0 + 2.0 * self.a * u + 3.0 * self.b * u**2)
            / self.phase_denominator
        )
        return self._restore_type(result, scalar_input)

    def phase_acceleration(self, t):
        """Return phi_ddot(t), in 1/s^2."""
        time, scalar_input = self._validate_time(t)
        u = time / self.final_time
        result = (
            (2.0 * pi / self.final_time**2)
            * (2.0 * self.a + 6.0 * self.b * u)
            / self.phase_denominator
        )
        return self._restore_type(result, scalar_input)

    def scaling_factors(self, t):
        """Return lambda_x(t), lambda_y(t), and lambda_z(t)."""
        phase = self.phase(t)
        return self._scaling_profile(phase, scalar_input=np.isscalar(phase))

    def scaling_velocities(self, t):
        """Return lambda_x_dot(t), lambda_y_dot(t), and lambda_z_dot(t), in 1/s."""
        phase = self.phase(t)
        phase_velocity = self.phase_velocity(t)
        scalar_input = np.isscalar(phase)
        profile = (6.0 - 8.0 * np.cos(phase) + 2.0 * np.cos(2.0 * phase)) * phase_velocity
        return self._scale_axis_profile(profile, scalar_input)

    def scaling_accelerations(self, t):
        """Return lambda_x_ddot(t), lambda_y_ddot(t), and lambda_z_ddot(t), in 1/s^2."""
        phase = self.phase(t)
        phase_velocity = self.phase_velocity(t)
        phase_acceleration = self.phase_acceleration(t)
        scalar_input = np.isscalar(phase)
        profile = (
            6.0 * phase_acceleration
            - 8.0 * (phase_acceleration * np.cos(phase) - phase_velocity**2 * np.sin(phase))
            + 2.0
            * (
                phase_acceleration * np.cos(2.0 * phase)
                - 2.0 * phase_velocity**2 * np.sin(2.0 * phase)
            )
        )
        return self._scale_axis_profile(profile, scalar_input)

    def angular_frequency_squared(self, t):
        """Return inverse Castin-Dum omega_i_squared(t), preserving negative values."""
        lambda_x, lambda_y, lambda_z = self.scaling_factors(t)
        lambda_x_ddot, lambda_y_ddot, lambda_z_ddot = self.scaling_accelerations(t)
        omega_x_squared = self.omega_x_initial**2 / (lambda_x**3 * lambda_y * lambda_z) - lambda_x_ddot / lambda_x
        omega_y_squared = self.omega_y_initial**2 / (lambda_x * lambda_y**3 * lambda_z) - lambda_y_ddot / lambda_y
        omega_z_squared = self.omega_z_initial**2 / (lambda_x * lambda_y * lambda_z**3) - lambda_z_ddot / lambda_z
        return omega_x_squared, omega_y_squared, omega_z_squared

    def angular_frequencies(self, t):
        """Return physical angular frequencies, in rad/s.

        The retro-sinusoidal protocol is admissible in this project only when
        every reconstructed ``omega_squared`` is non-negative. Tiny negative
        values compatible with floating-point noise are set to zero before the
        square root; genuinely negative values raise ``ValueError``.
        """
        omega_squared = self.angular_frequency_squared(t)
        tolerance = self._omega_squared_tolerance()
        invalid = [
            (axis, float(np.min(values)))
            for axis, values in zip(("x", "y", "z"), omega_squared)
            if np.min(values) < -tolerance
        ]
        if invalid:
            axis, value = min(invalid, key=lambda item: item[1])
            raise ValueError(
                "Retro-sinusoidal protocol requires anti-confining curvature: "
                "omega_squared must remain non-negative for a physical trapping "
                f"protocol. Minimum on axis {axis}: {value}."
            )
        return tuple(self._sqrt_non_negative(values) for values in omega_squared)

    def is_trapping_protocol(self, t) -> bool:
        """Return whether reconstructed omega_squared is non-negative on t."""
        tolerance = self._omega_squared_tolerance()
        return all(np.min(values) >= -tolerance for values in self.angular_frequency_squared(t))

    def is_phase_monotonic(self) -> bool:
        """Return whether phi_dot(t) is non-negative over the full interval."""
        candidates = [0.0, 1.0]
        if self.b != 0.0:
            vertex = -self.a / (3.0 * self.b)
            if 0.0 <= vertex <= 1.0:
                candidates.append(vertex)
        values = [
            (1.0 + 2.0 * self.a * u + 3.0 * self.b * u**2) / self.phase_denominator
            for u in candidates
        ]
        return min(values) >= -self._MONOTONIC_TOLERANCE

    def _scaling_profile(self, phase, *, scalar_input: bool):
        profile = 6.0 * phase - 8.0 * np.sin(phase) + np.sin(2.0 * phase)
        factor = 1.0 / (12.0 * pi)
        values = (
            self.lambda_x_initial + (self.lambda_x_final - self.lambda_x_initial) * factor * profile,
            self.lambda_y_initial + (self.lambda_y_final - self.lambda_y_initial) * factor * profile,
            self.lambda_z_initial + (self.lambda_z_final - self.lambda_z_initial) * factor * profile,
        )
        if scalar_input:
            return tuple(float(value) for value in values)
        return values

    def _scale_axis_profile(self, profile, scalar_input: bool):
        factor = 1.0 / (12.0 * pi)
        values = (
            (self.lambda_x_final - self.lambda_x_initial) * factor * profile,
            (self.lambda_y_final - self.lambda_y_initial) * factor * profile,
            (self.lambda_z_final - self.lambda_z_initial) * factor * profile,
        )
        if scalar_input:
            return tuple(float(value) for value in values)
        return values

    def _validate_time(self, t) -> tuple[np.ndarray, bool]:
        array = np.asarray(t, dtype=float)
        scalar_input = array.ndim == 0
        if scalar_input:
            array = array.reshape(1)
        elif array.ndim != 1:
            raise ValueError("t must be a scalar or a 1D array.")
        if array.size == 0:
            raise ValueError("t must not be empty.")
        if not np.all(np.isfinite(array)):
            raise ValueError("t must contain only finite values.")
        if np.any(array < 0.0) or np.any(array > self.final_time):
            raise ValueError("t must satisfy 0 <= t <= final_time.")
        return array, scalar_input

    @staticmethod
    def _restore_type(values: np.ndarray, scalar_input: bool):
        if scalar_input:
            return float(values[0])
        return values

    def _omega_squared_tolerance(self) -> float:
        scale = max(
            self.omega_x_initial**2,
            self.omega_y_initial**2,
            self.omega_z_initial**2,
        )
        return self._OMEGA_SQUARED_RELATIVE_TOLERANCE * scale

    @staticmethod
    def _sqrt_non_negative(values):
        adjusted = np.where(values < 0.0, 0.0, values)
        result = np.sqrt(adjusted)
        if np.isscalar(values):
            return float(result)
        return result

    @staticmethod
    def _validate_finite(value: float, name: str) -> float:
        value = float(value)
        if not isfinite(value):
            raise ValueError(f"{name} must be finite.")
        return value

    @classmethod
    def _validate_positive_finite(cls, value: float, name: str) -> float:
        value = cls._validate_finite(value, name)
        if value <= 0.0:
            raise ValueError(f"{name} must be strictly positive.")
        return value
