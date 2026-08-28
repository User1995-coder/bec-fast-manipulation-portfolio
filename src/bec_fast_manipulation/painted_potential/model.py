"""Exact crossed painted dipole potential model."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, pi, sqrt

import numpy as np
from scipy.optimize import minimize, minimize_scalar

from bec_fast_manipulation.constants import ExperimentalConstants, PhysicalConstants
from bec_fast_manipulation.painted_potential.modulation import harmonic_painting_modulation


@dataclass(frozen=True)
class PotentialMinimum:
    """Local metastable minimum connected to the optical trap center."""

    position_m: np.ndarray
    potential_J: float
    success: bool
    message: str


@dataclass(frozen=True)
class TrapModes:
    """Local normal modes of the potential at a stable minimum."""

    angular_frequencies_rad_s: np.ndarray
    frequencies_hz: np.ndarray
    eigenvectors: np.ndarray
    minimum_position_m: np.ndarray
    horizontal_frequency_hz: float
    vertical_frequency_hz: float


class CrossedPaintedDipolePotential:
    """Two orthogonal Gaussian beams painted with a normalized modulation."""

    def __init__(
        self,
        *,
        laser_wavelength_m: float = ExperimentalConstants.TRAP_LASER_WAVELENGTH_M,
        horizontal_waist_m: float = ExperimentalConstants.TRAP_BEAM_HORIZONTAL_WAIST_M,
        vertical_waist_m: float = ExperimentalConstants.TRAP_BEAM_VERTICAL_WAIST_M,
        quadrature_order: int = 128,
    ) -> None:
        self.laser_wavelength_m = self._validate_positive_finite(laser_wavelength_m, "laser_wavelength_m")
        self.horizontal_waist_m = self._validate_positive_finite(horizontal_waist_m, "horizontal_waist_m")
        self.vertical_waist_m = self._validate_positive_finite(vertical_waist_m, "vertical_waist_m")
        self.quadrature_order = self._validate_int_at_least(quadrature_order, "quadrature_order", 8)

        nodes, weights = np.polynomial.legendre.leggauss(self.quadrature_order)
        self._phase_nodes = 0.5 * (nodes + 1.0)
        self._phase_weights = 0.5 * weights
        self._modulation_values = harmonic_painting_modulation(self._phase_nodes)
        self._dipole_coefficient = self.dipole_potential_coefficient()

    def laser_angular_frequency(self) -> float:
        """Return the laser angular frequency in rad/s."""
        return 2.0 * pi * PhysicalConstants.SPEED_OF_LIGHT / self.laser_wavelength_m

    def dipole_potential_coefficient(self) -> float:
        """Return the Rb87 D1/D2 scalar dipole coefficient in J m^2/W."""
        omega_laser = self.laser_angular_frequency()
        return -((pi * PhysicalConstants.SPEED_OF_LIGHT**2) / 2.0) * (
            (2.0 * PhysicalConstants.RUBIDIUM_D2_LINEWIDTH / PhysicalConstants.RUBIDIUM_D2_ANGULAR_FREQUENCY**3)
            * (
                1.0 / (PhysicalConstants.RUBIDIUM_D2_ANGULAR_FREQUENCY - omega_laser)
                + 1.0 / (PhysicalConstants.RUBIDIUM_D2_ANGULAR_FREQUENCY + omega_laser)
            )
            + (PhysicalConstants.RUBIDIUM_D1_LINEWIDTH / PhysicalConstants.RUBIDIUM_D1_ANGULAR_FREQUENCY**3)
            * (
                1.0 / (PhysicalConstants.RUBIDIUM_D1_ANGULAR_FREQUENCY - omega_laser)
                + 1.0 / (PhysicalConstants.RUBIDIUM_D1_ANGULAR_FREQUENCY + omega_laser)
            )
        )

    def instantaneous_intensity(self, position, phase, power_w: float, amplitude_m: float):
        """Return the instantaneous crossed-beam intensity in W/m^2."""
        power_w = self._validate_positive_finite(power_w, "power_w")
        amplitude_m = self._validate_non_negative_finite(amplitude_m, "amplitude_m")
        x, y, z = self._position_components(position)
        modulation = harmonic_painting_modulation(phase)
        intensity = self._beam1_intensity(x, y, z, power_w, amplitude_m, modulation) + self._beam2_intensity(
            x,
            y,
            z,
            power_w,
            amplitude_m,
            modulation,
        )
        if np.asarray(intensity).size == 1:
            return float(np.asarray(intensity).reshape(()))
        return intensity

    def averaged_intensity(self, position, power_w: float, amplitude_m: float):
        """Return the cycle-averaged painted intensity in W/m^2."""
        beam1, beam2 = self.averaged_beam_intensities(position, power_w, amplitude_m)
        return beam1 + beam2

    def averaged_beam_intensities(self, position, power_w: float, amplitude_m: float):
        """Return the cycle-averaged beam 1 and beam 2 intensities in W/m^2."""
        power_w = self._validate_positive_finite(power_w, "power_w")
        amplitude_m = self._validate_non_negative_finite(amplitude_m, "amplitude_m")
        x, y, z = self._position_components(position)
        modulation = self._modulation_values
        beam1 = self._cycle_average(self._beam1_intensity(x, y, z, power_w, amplitude_m, modulation))
        beam2 = self._cycle_average(
            self._beam2_intensity(
                x,
                y,
                z,
                power_w,
                amplitude_m,
                modulation,
            )
        )
        return beam1, beam2

    def averaged_beam1_intensity(self, position, power_w: float, amplitude_m: float):
        """Return the cycle-averaged beam 1 intensity in W/m^2."""
        beam1, _ = self.averaged_beam_intensities(position, power_w, amplitude_m)
        return beam1

    def averaged_beam2_intensity(self, position, power_w: float, amplitude_m: float):
        """Return the cycle-averaged beam 2 intensity in W/m^2."""
        _, beam2 = self.averaged_beam_intensities(position, power_w, amplitude_m)
        return beam2

    def _cycle_average(self, intensity):
        weights = self._phase_weights.reshape((-1,) + (1,) * (np.ndim(intensity) - 1))
        return np.sum(weights * intensity, axis=0)

    def optical_potential(self, position, power_w: float, amplitude_m: float):
        """Return the optical dipole potential in J."""
        return self._dipole_coefficient * self.averaged_intensity(position, power_w, amplitude_m)

    def total_potential(self, position, power_w: float, amplitude_m: float, *, include_gravity: bool = True):
        """Return optical plus optional gravitational potential in J.

        The vertical convention is z positive upward, so ``U_g = m g z``.
        """
        potential = self.optical_potential(position, power_w, amplitude_m)
        if include_gravity:
            z = self._position_components(position)[2]
            potential = potential + PhysicalConstants.RUBIDIUM_87_MASS * PhysicalConstants.STANDARD_GRAVITY * z
        return potential

    def find_minimum(
        self,
        power_w: float,
        amplitude_m: float,
        *,
        include_gravity: bool = True,
        initial_guess=None,
    ) -> PotentialMinimum:
        """Return the local trap minimum near the crossed-beam center.

        With gravity, this is not a global minimum because ``m*g*z`` is
        unbounded below for ``z -> -infinity``.
        """
        return self.find_local_minimum(
            power_w,
            amplitude_m,
            include_gravity=include_gravity,
            initial_guess=initial_guess,
        )

    def find_local_minimum(
        self,
        power_w: float,
        amplitude_m: float,
        *,
        include_gravity: bool = True,
        initial_guess=None,
    ) -> PotentialMinimum:
        """Find the local metastable minimum connected to the trap center."""
        power_w = self._validate_positive_finite(power_w, "power_w")
        amplitude_m = self._validate_non_negative_finite(amplitude_m, "amplitude_m")
        if initial_guess is None:
            if not include_gravity:
                position = np.zeros(3, dtype=float)
                return PotentialMinimum(
                    position_m=position,
                    potential_J=float(self.total_potential(position, power_w, amplitude_m, include_gravity=False)),
                    success=True,
                    message="Symmetric no-gravity minimum.",
                )
            return self._find_symmetric_local_minimum(power_w, amplitude_m)
        guess = self._as_position_vector(initial_guess)
        return self._find_local_minimum_3d(power_w, amplitude_m, include_gravity=include_gravity, initial_guess=guess)

    def _find_symmetric_local_minimum(self, power_w: float, amplitude_m: float) -> PotentialMinimum:
        span_z = 0.75 * self.vertical_waist_m
        step = self.vertical_waist_m / 400.0

        def potential_z(z_value: float) -> float:
            return float(self.total_potential([0.0, 0.0, z_value], power_w, amplitude_m, include_gravity=True))

        def derivative_z(z_value: float) -> float:
            return (potential_z(z_value + step) - potential_z(z_value - step)) / (2.0 * step)

        scale = max(abs(float(self.optical_potential([0.0, 0.0, 0.0], power_w, amplitude_m))), 1e-40)
        result = minimize_scalar(
            lambda z_value: potential_z(z_value) / scale,
            bounds=(-span_z, 0.0),
            method="bounded",
            options={"xatol": 1e-13, "maxiter": 100},
        )
        z_minimum = float(result.x)
        distance_to_boundary = min(abs(z_minimum + span_z), abs(z_minimum))
        curvature = (potential_z(z_minimum + step) - 2.0 * potential_z(z_minimum) + potential_z(z_minimum - step)) / step**2
        derivative = derivative_z(z_minimum)
        force_scale = max(abs(derivative_z(0.0)), 1e-30)
        if (
            result.success
            and distance_to_boundary > 5.0 * step
            and curvature > 0.0
            and abs(derivative) <= max(1e-24, 1e-5 * force_scale)
        ):
            position = np.array([0.0, 0.0, z_minimum], dtype=float)
            return PotentialMinimum(
                position_m=position,
                potential_J=potential_z(z_minimum),
                success=True,
                message="Symmetric local metastable minimum.",
            )
        return PotentialMinimum(
            position_m=np.array([0.0, 0.0, float("nan")], dtype=float),
            potential_J=float("nan"),
            success=False,
            message="No stable local minimum found near the optical trap center.",
        )

    def _find_local_minimum_3d(
        self,
        power_w: float,
        amplitude_m: float,
        *,
        include_gravity: bool,
        initial_guess: np.ndarray,
    ) -> PotentialMinimum:
        span_xy = max(4.0 * self.horizontal_waist_m, 2.0 * amplitude_m + 2.0 * self.horizontal_waist_m)
        span_z = 4.0 * self.vertical_waist_m
        bounds = [(-span_xy, span_xy), (-span_xy, span_xy), (-span_z, span_z)]
        scale = max(abs(float(self.optical_potential([0.0, 0.0, 0.0], power_w, amplitude_m))), 1e-40)

        result = minimize(
            lambda point: float(self.total_potential(point, power_w, amplitude_m, include_gravity=include_gravity)) / scale,
            initial_guess,
            method="L-BFGS-B",
            bounds=bounds,
            options={"ftol": 1e-14, "gtol": 1e-10, "maxiter": 1000},
        )
        return PotentialMinimum(
            position_m=np.asarray(result.x, dtype=float),
            potential_J=float(self.total_potential(result.x, power_w, amplitude_m, include_gravity=include_gravity)),
            success=bool(result.success),
            message=str(result.message),
        )

    def hessian_at_minimum(
        self,
        power_w: float,
        amplitude_m: float,
        *,
        include_gravity: bool = True,
        minimum: PotentialMinimum | None = None,
        step_m: float | None = None,
    ) -> np.ndarray:
        """Return a central-difference Hessian at the local trap minimum."""
        if minimum is None:
            minimum = self.find_local_minimum(power_w, amplitude_m, include_gravity=include_gravity)
        if not minimum.success:
            raise RuntimeError(f"Could not find local potential minimum: {minimum.message}")
        point = self._as_position_vector(minimum.position_m)
        step = self._validate_positive_finite(
            step_m if step_m is not None else min(self.horizontal_waist_m, self.vertical_waist_m) / 120.0,
            "step_m",
        )
        hessian = np.zeros((3, 3), dtype=float)
        basis = np.eye(3)
        center = float(self.total_potential(point, power_w, amplitude_m, include_gravity=include_gravity))
        for i in range(3):
            ei = basis[i]
            forward = float(self.total_potential(point + step * ei, power_w, amplitude_m, include_gravity=include_gravity))
            backward = float(self.total_potential(point - step * ei, power_w, amplitude_m, include_gravity=include_gravity))
            hessian[i, i] = (forward - 2.0 * center + backward) / step**2
            for j in range(i + 1, 3):
                ej = basis[j]
                fpp = float(self.total_potential(point + step * ei + step * ej, power_w, amplitude_m, include_gravity=include_gravity))
                fpm = float(self.total_potential(point + step * ei - step * ej, power_w, amplitude_m, include_gravity=include_gravity))
                fmp = float(self.total_potential(point - step * ei + step * ej, power_w, amplitude_m, include_gravity=include_gravity))
                fmm = float(self.total_potential(point - step * ei - step * ej, power_w, amplitude_m, include_gravity=include_gravity))
                value = (fpp - fpm - fmp + fmm) / (4.0 * step**2)
                hessian[i, j] = value
                hessian[j, i] = value
        return 0.5 * (hessian + hessian.T)

    def trap_modes(self, power_w: float, amplitude_m: float, *, include_gravity: bool = True) -> TrapModes:
        """Return local normal modes and axis-oriented summary frequencies."""
        minimum = self.find_local_minimum(power_w, amplitude_m, include_gravity=include_gravity)
        if not minimum.success:
            raise ValueError(f"Potential minimum is not stable: {minimum.message}")
        hessian = self.hessian_at_minimum(
            power_w,
            amplitude_m,
            include_gravity=include_gravity,
            minimum=minimum,
        )
        eigenvalues, eigenvectors = np.linalg.eigh(hessian / PhysicalConstants.RUBIDIUM_87_MASS)
        if np.any(eigenvalues <= 0.0):
            raise ValueError("Potential minimum is not stable: Hessian has non-positive mode.")
        angular_frequencies = np.sqrt(eigenvalues)
        frequencies_hz = angular_frequencies / (2.0 * pi)
        vertical_index = int(np.argmax(np.abs(eigenvectors[2, :])))
        horizontal_indices = [index for index in range(3) if index != vertical_index]
        return TrapModes(
            angular_frequencies_rad_s=angular_frequencies,
            frequencies_hz=frequencies_hz,
            eigenvectors=eigenvectors,
            minimum_position_m=minimum.position_m,
            horizontal_frequency_hz=float(np.mean(frequencies_hz[horizontal_indices])),
            vertical_frequency_hz=float(frequencies_hz[vertical_index]),
        )

    def trap_frequencies(self, power_w: float, amplitude_m: float, *, include_gravity: bool = True) -> dict[str, object]:
        """Return local trap frequencies for the symmetric crossed trap."""
        modes = self.trap_modes(power_w, amplitude_m, include_gravity=include_gravity)
        return {
            "frequencies_hz": modes.frequencies_hz,
            "angular_frequencies_rad_s": modes.angular_frequencies_rad_s,
            "horizontal_frequency_hz": modes.horizontal_frequency_hz,
            "vertical_frequency_hz": modes.vertical_frequency_hz,
            "eigenvectors": modes.eigenvectors,
            "minimum_position_m": modes.minimum_position_m,
        }

    def _beam1_intensity(self, x, y, z, power_w: float, amplitude_m: float, modulation):
        w_horizontal = self._waist(self.horizontal_waist_m, x)
        w_vertical = self._waist(self.vertical_waist_m, x)
        shifted_y = np.expand_dims(y, axis=0) - amplitude_m * np.asarray(modulation).reshape((-1,) + (1,) * np.ndim(y))
        z_values = np.expand_dims(z, axis=0)
        prefactor = 2.0 * power_w / (pi * np.expand_dims(w_horizontal, axis=0) * np.expand_dims(w_vertical, axis=0))
        return prefactor * np.exp(-2.0 * shifted_y**2 / np.expand_dims(w_horizontal, axis=0) ** 2) * np.exp(
            -2.0 * z_values**2 / np.expand_dims(w_vertical, axis=0) ** 2
        )

    def _beam2_intensity(self, x, y, z, power_w: float, amplitude_m: float, modulation):
        w_horizontal = self._waist(self.horizontal_waist_m, y)
        w_vertical = self._waist(self.vertical_waist_m, y)
        shifted_x = np.expand_dims(x, axis=0) - amplitude_m * np.asarray(modulation).reshape((-1,) + (1,) * np.ndim(x))
        z_values = np.expand_dims(z, axis=0)
        prefactor = 2.0 * power_w / (pi * np.expand_dims(w_horizontal, axis=0) * np.expand_dims(w_vertical, axis=0))
        return prefactor * np.exp(-2.0 * shifted_x**2 / np.expand_dims(w_horizontal, axis=0) ** 2) * np.exp(
            -2.0 * z_values**2 / np.expand_dims(w_vertical, axis=0) ** 2
        )

    def _waist(self, waist_0_m: float, propagation_coordinate_m):
        z_rayleigh = pi * waist_0_m**2 / self.laser_wavelength_m
        return waist_0_m * np.sqrt(1.0 + (np.asarray(propagation_coordinate_m) / z_rayleigh) ** 2)

    @staticmethod
    def _position_components(position):
        array = np.asarray(position, dtype=float)
        if not np.all(np.isfinite(array)):
            raise ValueError("position must contain only finite values.")
        if array.shape == (3,):
            return array[0], array[1], array[2]
        if array.ndim >= 1 and array.shape[0] == 3:
            return array[0], array[1], array[2]
        raise ValueError("position must have shape (3,) or (3, ...).")

    @staticmethod
    def _as_position_vector(position) -> np.ndarray:
        array = np.asarray(position, dtype=float)
        if array.shape != (3,):
            raise ValueError("position must contain exactly 3 coordinates.")
        if not np.all(np.isfinite(array)):
            raise ValueError("position must contain only finite values.")
        return array

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

    @staticmethod
    def _validate_int_at_least(value: int, name: str, minimum: int) -> int:
        if isinstance(value, bool):
            raise ValueError(f"{name} must be an integer.")
        value = int(value)
        if value < minimum:
            raise ValueError(f"{name} must be at least {minimum}.")
        return value
