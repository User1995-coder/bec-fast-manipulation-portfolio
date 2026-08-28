"""Thomas-Fermi initial-state model for the condensate."""

from __future__ import annotations

import math

import numpy as np

from bec_fast_manipulation.constants import ExperimentalConstants, PhysicalConstants


class ThomasFermiModel:
    """Compute Thomas-Fermi quantities and convert Castin-Dum scaling factors."""

    def __init__(
        self,
        atom_number: int | float = ExperimentalConstants.CONDENSATE_ATOM_NUMBER,
        omega_x: float = ExperimentalConstants.INITIAL_TRAP_ANGULAR_FREQUENCY_X,
        omega_y: float = ExperimentalConstants.INITIAL_TRAP_ANGULAR_FREQUENCY_Y,
        omega_z: float = ExperimentalConstants.INITIAL_TRAP_ANGULAR_FREQUENCY_Z,
    ) -> None:
        self.atom_number = self._validate_positive_scalar(atom_number, "atom_number")
        self.omega_x = self._validate_positive_scalar(omega_x, "omega_x")
        self.omega_y = self._validate_positive_scalar(omega_y, "omega_y")
        self.omega_z = self._validate_positive_scalar(omega_z, "omega_z")

    def geometric_mean_frequency(self) -> float:
        """Return the geometric mean angular frequency in rad/s."""
        return float((self.omega_x * self.omega_y * self.omega_z) ** (1 / 3))

    def chemical_potential(self) -> float:
        """Return the historical Thomas-Fermi chemical potential in J."""
        hbar = PhysicalConstants.REDUCED_PLANCK_CONSTANT
        mass = PhysicalConstants.RUBIDIUM_87_MASS
        scattering_length = PhysicalConstants.RUBIDIUM_87_SCATTERING_LENGTH
        omega_bar = self.geometric_mean_frequency()
        interaction_term = (
            15
            * self.atom_number
            * scattering_length
            * math.sqrt(mass * omega_bar / hbar)
        )
        return float((hbar * omega_bar / 2) * interaction_term ** (2 / 5))

    def initial_radii(self) -> tuple[float, float, float]:
        """Return initial Thomas-Fermi radii along x, y, z in m."""
        mass = PhysicalConstants.RUBIDIUM_87_MASS
        mu = self.chemical_potential()
        radius_x = math.sqrt(2 * mu / (mass * self.omega_x**2))
        radius_y = math.sqrt(2 * mu / (mass * self.omega_y**2))
        radius_z = math.sqrt(2 * mu / (mass * self.omega_z**2))
        return float(radius_x), float(radius_y), float(radius_z)

    def radii_from_scaling_factors(self, lambda_x, lambda_y, lambda_z):
        """Return physical radii R_i(t) = R_i0 * lambda_i(t) in m."""
        lambda_arrays, scalar_input = self._validate_triplet(
            (lambda_x, lambda_y, lambda_z),
            ("lambda_x", "lambda_y", "lambda_z"),
            strictly_positive=True,
        )
        radii = self.initial_radii()
        results = tuple(radius * lambda_array for radius, lambda_array in zip(radii, lambda_arrays))
        return self._restore_triplet(results, scalar_input)

    def radius_velocities_from_scaling_velocities(
        self,
        lambda_x_dot,
        lambda_y_dot,
        lambda_z_dot,
    ):
        """Return radius velocities R_dot_i(t) = R_i0 * lambda_dot_i(t) in m/s."""
        lambda_dot_arrays, scalar_input = self._validate_triplet(
            (lambda_x_dot, lambda_y_dot, lambda_z_dot),
            ("lambda_x_dot", "lambda_y_dot", "lambda_z_dot"),
            strictly_positive=False,
        )
        radii = self.initial_radii()
        results = tuple(radius * lambda_dot_array for radius, lambda_dot_array in zip(radii, lambda_dot_arrays))
        return self._restore_triplet(results, scalar_input)

    @staticmethod
    def _validate_positive_scalar(value, name: str) -> float:
        value_float = float(value)
        if not math.isfinite(value_float):
            raise ValueError(f"{name} must be finite.")
        if value_float <= 0:
            raise ValueError(f"{name} must be strictly positive.")
        return value_float

    @classmethod
    def _validate_triplet(
        cls,
        values,
        names: tuple[str, str, str],
        *,
        strictly_positive: bool,
    ) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray], bool]:
        arrays = tuple(cls._as_scalar_or_1d_array(value, name) for value, name in zip(values, names))
        scalar_input = all(array.shape == () for array in arrays)
        if not scalar_input:
            for array, name in zip(arrays, names):
                if array.ndim != 1:
                    raise ValueError(f"{name} must be a scalar or a 1D array.")
            shapes = {array.shape for array in arrays}
            if len(shapes) != 1:
                raise ValueError("Input arrays must have identical shapes.")
        for array, name in zip(arrays, names):
            if not np.all(np.isfinite(array)):
                raise ValueError(f"{name} must contain only finite values.")
            if strictly_positive and not np.all(array > 0):
                raise ValueError(f"{name} must contain strictly positive values.")
        return arrays, scalar_input

    @staticmethod
    def _as_scalar_or_1d_array(value, name: str) -> np.ndarray:
        array = np.asarray(value, dtype=float)
        if array.ndim > 1:
            raise ValueError(f"{name} must be a scalar or a 1D array.")
        if array.ndim == 1 and array.size == 0:
            raise ValueError(f"{name} must not be empty.")
        return array

    @staticmethod
    def _restore_triplet(values: tuple[np.ndarray, np.ndarray, np.ndarray], scalar_input: bool):
        if scalar_input:
            return tuple(float(value) for value in values)
        return values
