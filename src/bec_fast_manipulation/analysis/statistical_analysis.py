"""Generic numerical analysis helpers."""

from __future__ import annotations

import numpy as np


class StatisticalAnalysis:
    """Numerical comparisons without plotting, printing, or physics formulas."""

    @staticmethod
    def safe_ratio(numerator, denominator, *, atol: float = 1e-12):
        numerator_array = np.asarray(numerator, dtype=float)
        denominator_array = np.asarray(denominator, dtype=float)
        ratio = np.divide(
            numerator_array,
            denominator_array,
            out=np.full(np.broadcast_shapes(numerator_array.shape, denominator_array.shape), np.nan),
            where=~np.isclose(denominator_array, 0.0, atol=atol),
        )
        return float(ratio) if ratio.shape == () else ratio

    @classmethod
    def relative_change(cls, value, reference):
        ratio = cls.safe_ratio(value - np.asarray(reference, dtype=float), reference)
        return ratio

    @classmethod
    def reduction_fraction(cls, value, reference):
        ratio = cls.safe_ratio(value, reference)
        return 1.0 - ratio

    @classmethod
    def reduction_percent(cls, value, reference):
        return 100.0 * cls.reduction_fraction(value, reference)

    @classmethod
    def compare_axis_values(
        cls,
        values,
        references,
        axis_names: tuple[str, ...] = ("x", "y", "z"),
    ) -> dict[str, dict[str, float]]:
        values_array = cls._as_1d_array(values, "values")
        references_array = cls._as_1d_array(references, "references")
        if len(axis_names) != values_array.size:
            raise ValueError("axis_names must have the same length as values.")
        if references_array.shape != values_array.shape:
            raise ValueError("references must have the same shape as values.")
        return {
            axis: cls.compare_scalar_values(value, reference)
            for axis, value, reference in zip(axis_names, values_array, references_array)
        }

    @classmethod
    def compare_scalar_values(cls, value, reference) -> dict[str, float]:
        value_float = float(value)
        reference_float = float(reference)
        return {
            "value": value_float,
            "reference": reference_float,
            "ratio": cls.safe_ratio(value_float, reference_float),
            "reduction_fraction": cls.reduction_fraction(value_float, reference_float),
            "reduction_percent": cls.reduction_percent(value_float, reference_float),
        }

    @classmethod
    def final_axis_values(cls, x_values, y_values, z_values) -> dict[str, float]:
        return {
            "x": float(cls._as_1d_array(x_values, "x_values")[-1]),
            "y": float(cls._as_1d_array(y_values, "y_values")[-1]),
            "z": float(cls._as_1d_array(z_values, "z_values")[-1]),
        }

    @classmethod
    def rms(cls, values) -> float:
        array = cls._as_1d_array(values, "values")
        return float(np.sqrt(np.mean(array**2)))

    @staticmethod
    def _as_1d_array(values, name: str) -> np.ndarray:
        array = np.asarray(values, dtype=float)
        if array.size == 0:
            raise ValueError(f"{name} must not be empty.")
        if array.ndim != 1:
            raise ValueError(f"{name} must be a 1D array.")
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must contain only finite values.")
        return array
