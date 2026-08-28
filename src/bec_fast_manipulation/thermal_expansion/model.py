"""Effective expansion temperature model."""

from __future__ import annotations

import numpy as np

from bec_fast_manipulation.constants import PhysicalConstants


class ThermalExpansionModel:
    """Convert radius expansion velocities to effective expansion temperatures."""

    def directional_temperatures(
        self,
        velocity_x_m_s,
        velocity_y_m_s,
        velocity_z_m_s,
    ):
        """Return effective directional expansion temperatures in K."""
        velocity_arrays, scalar_input = self._validate_triplet(
            (velocity_x_m_s, velocity_y_m_s, velocity_z_m_s),
            ("velocity_x_m_s", "velocity_y_m_s", "velocity_z_m_s"),
        )
        factor = PhysicalConstants.RUBIDIUM_87_MASS / PhysicalConstants.BOLTZMANN_CONSTANT
        temperatures = tuple(factor * velocity_array**2 for velocity_array in velocity_arrays)
        return self._restore_triplet(temperatures, scalar_input)

    def temperature_3d(self, temperature_x_K, temperature_y_K, temperature_z_K):
        """Return the effective 3D expansion temperature in K."""
        temperature_arrays, scalar_input = self._validate_triplet(
            (temperature_x_K, temperature_y_K, temperature_z_K),
            ("temperature_x_K", "temperature_y_K", "temperature_z_K"),
        )
        temperature_3d = sum(temperature_arrays) / 3
        if scalar_input:
            return float(temperature_3d)
        return temperature_3d

    def temperatures_from_radius_velocities(
        self,
        velocity_x_m_s,
        velocity_y_m_s,
        velocity_z_m_s,
    ) -> dict[str, float | np.ndarray]:
        """Return directional and 3D effective expansion temperatures."""
        temperature_x, temperature_y, temperature_z = self.directional_temperatures(
            velocity_x_m_s,
            velocity_y_m_s,
            velocity_z_m_s,
        )
        return {
            "x": temperature_x,
            "y": temperature_y,
            "z": temperature_z,
            "3d": self.temperature_3d(temperature_x, temperature_y, temperature_z),
        }

    @classmethod
    def _validate_triplet(
        cls,
        values,
        names: tuple[str, str, str],
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
