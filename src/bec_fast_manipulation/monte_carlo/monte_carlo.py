"""Generic Monte Carlo perturbations and ensemble statistics."""

from __future__ import annotations

from math import isfinite

import numpy as np


class MonteCarlo:
    """Generate noisy realizations around a nominal signal.

    Parameters
    ----------
    n_simulations:
        Number of independent realizations to generate. Must be at least one.
    relative_noise:
        Multiplicative noise amplitude expressed as a fraction. For example,
        ``0.05`` means 5 percent relative noise.
    distribution:
        Noise law. ``"uniform"`` draws
        ``epsilon ~ U(-relative_noise, +relative_noise)``. ``"gaussian"``
        draws ``epsilon ~ N(0, relative_noise)``. Values are not clipped or
        forced to remain positive.
    seed:
        Seed passed to ``np.random.default_rng``. Two instances created with
        the same seed and parameters produce the same first ensemble for the
        same signal. Repeated calls on the same instance advance the RNG state.

    Notes
    -----
    The generated signal is always ``nominal_signal * (1 + epsilon)``. This
    class is purely mathematical: it does not know whether a negative perturbed
    value is physically admissible.
    """

    _DISTRIBUTIONS = {"uniform", "gaussian"}

    def __init__(
        self,
        n_simulations: int,
        relative_noise: float,
        *,
        distribution: str = "uniform",
        seed: int | None = None,
    ) -> None:
        self.n_simulations = self._validate_n_simulations(n_simulations)
        self.relative_noise = self._validate_relative_noise(relative_noise)
        self.distribution = self._validate_distribution(distribution)
        self.seed = seed
        self._rng = np.random.default_rng(seed)

    def generate(self, signal, *, independent_points: bool = True) -> np.ndarray:
        """Return noisy realizations of a scalar or one-dimensional signal.

        Parameters
        ----------
        signal:
            Nominal scalar or one-dimensional NumPy-compatible signal. A vector
            with shape ``(N,)`` returns an ensemble with shape
            ``(n_simulations, N)``. A scalar returns shape ``(n_simulations,)``.
            Empty arrays and arrays containing NaN or infinity are rejected.
        independent_points:
            If ``True``, each simulation and each time point receives its own
            independent multiplicative noise coefficient. This represents
            point-to-point technical noise. If ``False``, one coefficient is
            drawn per simulation and applied to the whole trajectory. This
            represents a global calibration or gain error.

        Returns
        -------
        numpy.ndarray
            Noisy ensemble generated as ``signal * (1 + epsilon)``.
        """
        signal_array, scalar_input = self._validate_signal(signal)
        noise_shape = (self.n_simulations,) if scalar_input or not independent_points else (self.n_simulations, signal_array.size)
        epsilon = self._draw_noise(noise_shape)

        if scalar_input:
            return signal_array.reshape(())[()] * (1.0 + epsilon)
        if independent_points:
            return signal_array.reshape(1, -1) * (1.0 + epsilon)
        return signal_array.reshape(1, -1) * (1.0 + epsilon.reshape(-1, 1))

    @staticmethod
    def mean(ensemble, axis=0):
        """Return the ensemble mean along ``axis``."""
        return np.mean(MonteCarlo._validate_ensemble(ensemble), axis=axis)

    @staticmethod
    def std(ensemble, axis=0):
        """Return the ensemble standard deviation along ``axis``."""
        return np.std(MonteCarlo._validate_ensemble(ensemble), axis=axis)

    @staticmethod
    def statistics(ensemble, axis=0) -> dict[str, np.ndarray]:
        """Return empirical ensemble statistics along ``axis``.

        The returned dictionary contains ``mean``, ``std``, ``min``, ``max``,
        ``median``, ``q05``, and ``q95``. The ``q05`` and ``q95`` entries are
        empirical 5 percent and 95 percent quantiles, not confidence intervals.
        """
        array = MonteCarlo._validate_ensemble(ensemble)
        return {
            "mean": np.mean(array, axis=axis),
            "std": np.std(array, axis=axis),
            "min": np.min(array, axis=axis),
            "max": np.max(array, axis=axis),
            "median": np.median(array, axis=axis),
            "q05": np.quantile(array, 0.05, axis=axis),
            "q95": np.quantile(array, 0.95, axis=axis),
        }

    def _draw_noise(self, shape):
        if self.relative_noise == 0.0:
            return np.zeros(shape, dtype=float)
        if self.distribution == "uniform":
            return self._rng.uniform(-self.relative_noise, self.relative_noise, size=shape)
        return self._rng.normal(0.0, self.relative_noise, size=shape)

    @staticmethod
    def _validate_n_simulations(value: int) -> int:
        if isinstance(value, bool):
            raise ValueError("n_simulations must be an integer greater than or equal to 1.")
        try:
            integer_value = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("n_simulations must be an integer greater than or equal to 1.") from exc
        if integer_value != value or integer_value < 1:
            raise ValueError("n_simulations must be an integer greater than or equal to 1.")
        return integer_value

    @staticmethod
    def _validate_relative_noise(value: float) -> float:
        try:
            float_value = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("relative_noise must be finite and non-negative.") from exc
        if not isfinite(float_value) or float_value < 0.0:
            raise ValueError("relative_noise must be finite and non-negative.")
        return float_value

    @classmethod
    def _validate_distribution(cls, value: str) -> str:
        if value not in cls._DISTRIBUTIONS:
            allowed = ", ".join(sorted(cls._DISTRIBUTIONS))
            raise ValueError(f"distribution must be one of: {allowed}.")
        return value

    @staticmethod
    def _validate_signal(signal) -> tuple[np.ndarray, bool]:
        array = np.asarray(signal, dtype=float)
        scalar_input = array.ndim == 0
        if not scalar_input and array.ndim != 1:
            raise ValueError("signal must be a scalar or a one-dimensional array.")
        if array.size == 0:
            raise ValueError("signal must not be empty.")
        if not np.all(np.isfinite(array)):
            raise ValueError("signal must contain only finite values.")
        return array, scalar_input

    @staticmethod
    def _validate_ensemble(ensemble) -> np.ndarray:
        array = np.asarray(ensemble, dtype=float)
        if array.size == 0:
            raise ValueError("ensemble must not be empty.")
        if not np.all(np.isfinite(array)):
            raise ValueError("ensemble must contain only finite values.")
        return array
