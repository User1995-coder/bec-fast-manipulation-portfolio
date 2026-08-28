"""Continuous optimizer for retro-sinusoidal STA trap protocols."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, pi, sqrt
from typing import Callable

import numpy as np
from scipy.optimize import Bounds, NonlinearConstraint, differential_evolution, minimize_scalar

from bec_fast_manipulation.retro_sinusoidal import RetroSinusoidalProtocol

AXES = ("x", "y", "z")


@dataclass(frozen=True)
class STACandidateEvaluation:
    """Diagnostic evaluation of one STA candidate."""

    final_time: float
    a: float
    b: float
    phase_monotonic: bool
    minimum_omega_squared_x: float
    minimum_omega_squared_y: float
    minimum_omega_squared_z: float
    minimum_frequency_x_hz: float
    minimum_frequency_y_hz: float
    minimum_frequency_z_hz: float
    trapping_constraints_satisfied: bool
    z_constraint_satisfied: bool
    frequency_decompression_satisfied: bool
    maximum_omega_squared_increase_horizontal: float
    maximum_omega_squared_increase_vertical: float
    feasible: bool
    message: str = ""


@dataclass(frozen=True)
class STAOptimizationResult:
    """Result of a continuous STA optimization."""

    final_time: float
    a: float
    b: float
    minimum_frequency_x_hz: float
    minimum_frequency_y_hz: float
    minimum_frequency_z_hz: float
    minimum_omega_squared_x: float
    minimum_omega_squared_y: float
    minimum_omega_squared_z: float
    phase_monotonic: bool
    success: bool
    message: str
    objective_value: float
    number_of_function_evaluations: int | None = None
    frequency_decompression_satisfied: bool = True


class STAOptimizer:
    """Find the fastest feasible retro-sinusoidal STA protocol."""

    def __init__(
        self,
        *,
        final_time_bounds_s: tuple[float, float],
        a_bounds: tuple[float, float],
        b_bounds: tuple[float, float],
        minimum_z_frequency_hz: float = 50.0,
        require_monotonic_phase: bool = True,
        require_frequency_decompression: bool = False,
        minimum_search_samples: int = 151,
        minimum_search_xatol: float = 1e-10,
        maxiter: int = 80,
        popsize: int = 12,
        tolerance: float = 1e-7,
        polish: bool = True,
        seed: int | None = 12345,
        protocol_kwargs: dict | None = None,
    ) -> None:
        self.final_time_bounds_s = self._validate_bounds(final_time_bounds_s, "final_time_bounds_s", positive=True)
        self.a_bounds = self._validate_bounds(a_bounds, "a_bounds")
        self.b_bounds = self._validate_bounds(b_bounds, "b_bounds")
        self.minimum_z_frequency_hz = self._validate_positive_finite(
            minimum_z_frequency_hz,
            "minimum_z_frequency_hz",
        )
        self.require_monotonic_phase = bool(require_monotonic_phase)
        self.require_frequency_decompression = bool(require_frequency_decompression)
        self.minimum_search_samples = self._validate_int_at_least(
            minimum_search_samples,
            "minimum_search_samples",
            5,
        )
        self.minimum_search_xatol = self._validate_positive_finite(
            minimum_search_xatol,
            "minimum_search_xatol",
        )
        self.maxiter = self._validate_int_at_least(maxiter, "maxiter", 1)
        self.popsize = self._validate_int_at_least(popsize, "popsize", 1)
        self.tolerance = self._validate_positive_finite(tolerance, "tolerance")
        self.polish = bool(polish)
        self.seed = self._validate_seed(seed)
        self.protocol_kwargs = {} if protocol_kwargs is None else dict(protocol_kwargs)

    def evaluate_candidate(self, final_time: float, a: float, b: float) -> STACandidateEvaluation:
        """Evaluate one candidate without running the optimizer."""
        final_time = self._validate_positive_finite(final_time, "final_time")
        a = self._validate_finite(a, "a")
        b = self._validate_finite(b, "b")
        protocol = self._build_protocol(final_time, a, b)
        if protocol is None:
            return self._invalid_candidate(final_time, a, b, "1 + a + b must be non-zero.")

        phase_monotonic = protocol.is_phase_monotonic()
        minimum_omega_squared = self._minimum_omega_squared_by_axis(protocol)
        frequencies_hz = tuple(self._frequency_from_omega_squared(value) for value in minimum_omega_squared)
        threshold = self._z_omega_squared_threshold()
        tolerance = self._omega_squared_tolerance(protocol)

        x_closed = minimum_omega_squared[0] >= -tolerance
        y_closed = minimum_omega_squared[1] >= -tolerance
        x_strict = minimum_omega_squared[0] > 0.0
        y_strict = minimum_omega_squared[1] > 0.0
        z_constraint_satisfied = minimum_omega_squared[2] >= threshold - tolerance
        trapping_constraints_satisfied = x_closed and y_closed
        monotonicity_satisfied = phase_monotonic or not self.require_monotonic_phase
        maximum_increases = self._maximum_decompression_increases(protocol)
        decompression_tolerance = self._decompression_increase_tolerance(protocol)
        frequency_decompression_satisfied = (
            maximum_increases[0] <= decompression_tolerance
            and maximum_increases[1] <= decompression_tolerance
        ) or not self.require_frequency_decompression
        feasible = (
            trapping_constraints_satisfied
            and x_strict
            and y_strict
            and z_constraint_satisfied
            and monotonicity_satisfied
            and frequency_decompression_satisfied
        )

        return STACandidateEvaluation(
            final_time=final_time,
            a=a,
            b=b,
            phase_monotonic=phase_monotonic,
            minimum_omega_squared_x=minimum_omega_squared[0],
            minimum_omega_squared_y=minimum_omega_squared[1],
            minimum_omega_squared_z=minimum_omega_squared[2],
            minimum_frequency_x_hz=frequencies_hz[0],
            minimum_frequency_y_hz=frequencies_hz[1],
            minimum_frequency_z_hz=frequencies_hz[2],
            trapping_constraints_satisfied=trapping_constraints_satisfied,
            z_constraint_satisfied=z_constraint_satisfied,
            frequency_decompression_satisfied=frequency_decompression_satisfied,
            maximum_omega_squared_increase_horizontal=maximum_increases[0],
            maximum_omega_squared_increase_vertical=maximum_increases[1],
            feasible=feasible,
            message="feasible" if feasible else "constraints not satisfied",
        )

    def optimize(self, *, seed: int | None = None) -> STAOptimizationResult:
        """Minimize final_time with nonlinear feasibility constraints."""
        optimizer_seed = self.seed if seed is None else self._validate_seed(seed)
        bounds = Bounds(
            [self.final_time_bounds_s[0], self.a_bounds[0], self.b_bounds[0]],
            [self.final_time_bounds_s[1], self.a_bounds[1], self.b_bounds[1]],
        )
        constraint_size = 6 if self.require_frequency_decompression else 4
        constraints = (
            NonlinearConstraint(
                self._constraint_values,
                lb=np.zeros(constraint_size),
                ub=np.full(constraint_size, np.inf),
            ),
        )
        result = differential_evolution(
            self._objective,
            bounds=bounds,
            constraints=constraints,
            maxiter=self.maxiter,
            popsize=self.popsize,
            tol=self.tolerance,
            polish=self.polish,
            seed=optimizer_seed,
        )
        evaluation = self.evaluate_candidate(*result.x)
        success = bool(result.success and evaluation.feasible)
        message = str(result.message)
        if not evaluation.feasible:
            message = f"{message}; final candidate is not feasible"
        return STAOptimizationResult(
            final_time=evaluation.final_time,
            a=evaluation.a,
            b=evaluation.b,
            minimum_frequency_x_hz=evaluation.minimum_frequency_x_hz,
            minimum_frequency_y_hz=evaluation.minimum_frequency_y_hz,
            minimum_frequency_z_hz=evaluation.minimum_frequency_z_hz,
            minimum_omega_squared_x=evaluation.minimum_omega_squared_x,
            minimum_omega_squared_y=evaluation.minimum_omega_squared_y,
            minimum_omega_squared_z=evaluation.minimum_omega_squared_z,
            phase_monotonic=evaluation.phase_monotonic,
            success=success,
            message=message,
            objective_value=self._objective(result.x),
            number_of_function_evaluations=getattr(result, "nfev", None),
            frequency_decompression_satisfied=evaluation.frequency_decompression_satisfied,
        )

    def optimize_fixed_final_time(self, final_time_s: float, *, seed: int | None = None) -> STAOptimizationResult:
        """Optimize only a,b at fixed final time with the same constraints."""
        final_time_s = self._validate_positive_finite(final_time_s, "final_time_s")
        optimizer_seed = self.seed if seed is None else self._validate_seed(seed)
        bounds = Bounds([self.a_bounds[0], self.b_bounds[0]], [self.a_bounds[1], self.b_bounds[1]])
        constraint_size = 6 if self.require_frequency_decompression else 4
        constraints = (
            NonlinearConstraint(
                lambda variables: self._fixed_time_constraint_values(final_time_s, variables),
                lb=np.zeros(constraint_size),
                ub=np.full(constraint_size, np.inf),
            ),
        )
        result = differential_evolution(
            lambda variables: self._fixed_time_objective(final_time_s, variables),
            bounds=bounds,
            constraints=constraints,
            maxiter=self.maxiter,
            popsize=self.popsize,
            tol=self.tolerance,
            polish=self.polish,
            seed=optimizer_seed,
        )
        evaluation = self.evaluate_candidate(final_time_s, float(result.x[0]), float(result.x[1]))
        success = bool(result.success and evaluation.feasible)
        message = str(result.message)
        if not evaluation.feasible:
            message = f"{message}; final candidate is not feasible"
        return STAOptimizationResult(
            final_time=evaluation.final_time,
            a=evaluation.a,
            b=evaluation.b,
            minimum_frequency_x_hz=evaluation.minimum_frequency_x_hz,
            minimum_frequency_y_hz=evaluation.minimum_frequency_y_hz,
            minimum_frequency_z_hz=evaluation.minimum_frequency_z_hz,
            minimum_omega_squared_x=evaluation.minimum_omega_squared_x,
            minimum_omega_squared_y=evaluation.minimum_omega_squared_y,
            minimum_omega_squared_z=evaluation.minimum_omega_squared_z,
            phase_monotonic=evaluation.phase_monotonic,
            success=success,
            message=message,
            objective_value=self._fixed_time_objective(final_time_s, result.x),
            number_of_function_evaluations=getattr(result, "nfev", None),
            frequency_decompression_satisfied=evaluation.frequency_decompression_satisfied,
        )

    @staticmethod
    def _objective(variables) -> float:
        return float(variables[0])

    def _constraint_values(self, variables) -> np.ndarray:
        final_time, a, b = (float(value) for value in variables)
        try:
            evaluation = self.evaluate_candidate(final_time, a, b)
        except ValueError:
            return np.full(6 if self.require_frequency_decompression else 4, -1e300)
        monotonicity = 1.0 if (evaluation.phase_monotonic or not self.require_monotonic_phase) else -1.0
        values = [
            evaluation.minimum_omega_squared_x,
            evaluation.minimum_omega_squared_y,
            evaluation.minimum_omega_squared_z - self._z_omega_squared_threshold(),
            monotonicity,
        ]
        if self.require_frequency_decompression:
            tolerance = self._decompression_increase_tolerance_for_evaluation(evaluation)
            values.extend(
                [
                    tolerance - evaluation.maximum_omega_squared_increase_horizontal,
                    tolerance - evaluation.maximum_omega_squared_increase_vertical,
                ]
            )
        return np.array(values, dtype=float)

    def _fixed_time_constraint_values(self, final_time_s: float, variables) -> np.ndarray:
        a, b = (float(value) for value in variables)
        return self._constraint_values([final_time_s, a, b])

    def _fixed_time_objective(self, final_time_s: float, variables) -> float:
        a, b = (float(value) for value in variables)
        evaluation = self.evaluate_candidate(final_time_s, a, b)
        penalty = 0.0 if evaluation.feasible else 1e6
        return penalty + a**2 + b**2

    def _minimum_omega_squared_by_axis(self, protocol: RetroSinusoidalProtocol) -> tuple[float, float, float]:
        return tuple(
            self._continuous_minimum(lambda u, axis_index=axis_index: self._axis_omega_squared(protocol, axis_index, u))
            for axis_index in range(3)
        )

    def _continuous_minimum(self, function: Callable[[float], float]) -> float:
        u_grid = np.linspace(0.0, 1.0, self.minimum_search_samples)
        values = np.array([function(float(u)) for u in u_grid], dtype=float)
        if not np.all(np.isfinite(values)):
            raise ValueError("omega_squared minimum search produced non-finite values.")

        candidate_indices = {0, values.size - 1}
        local_minima = np.where((values[1:-1] <= values[:-2]) & (values[1:-1] <= values[2:]))[0] + 1
        candidate_indices.update(int(index) for index in local_minima)

        minima = [float(values[0]), float(values[-1])]
        for index in sorted(candidate_indices):
            if index == 0 or index == values.size - 1:
                continue
            lower = float(u_grid[index - 1])
            upper = float(u_grid[index + 1])
            refined = minimize_scalar(
                function,
                bounds=(lower, upper),
                method="bounded",
                options={"xatol": self.minimum_search_xatol},
            )
            if not refined.success:
                minima.append(float(values[index]))
            else:
                minima.append(float(refined.fun))
        return min(minima)

    def _maximum_decompression_increases(self, protocol: RetroSinusoidalProtocol) -> tuple[float, float]:
        horizontal = lambda u: 0.5 * (
            self._axis_omega_squared(protocol, 0, u) + self._axis_omega_squared(protocol, 1, u)
        )
        vertical = lambda u: self._axis_omega_squared(protocol, 2, u)
        return self._maximum_increase(horizontal), self._maximum_increase(vertical)

    def _maximum_increase(self, function: Callable[[float], float]) -> float:
        u_grid = np.linspace(0.0, 1.0, self.minimum_search_samples)
        values = np.array([function(float(u)) for u in u_grid], dtype=float)
        if not np.all(np.isfinite(values)):
            raise ValueError("omega_squared decompression search produced non-finite values.")
        increments = values[1:] - values[:-1]
        maximum = float(np.max(increments))
        candidate_indices = set(np.where(increments > 0.5 * maximum)[0].tolist()) if maximum > 0.0 else set()
        for index in sorted(candidate_indices):
            lower = float(u_grid[index])
            upper = float(u_grid[index + 1])
            refined = minimize_scalar(
                lambda u: -self._local_increase(function, u),
                bounds=(lower, upper),
                method="bounded",
                options={"xatol": self.minimum_search_xatol},
            )
            if refined.success:
                maximum = max(maximum, float(-refined.fun))
        return maximum

    @staticmethod
    def _local_increase(function: Callable[[float], float], u: float) -> float:
        delta = 1e-4
        lower = max(0.0, float(u) - delta)
        upper = min(1.0, float(u) + delta)
        if upper == lower:
            return 0.0
        return function(upper) - function(lower)

    @staticmethod
    def _axis_omega_squared(protocol: RetroSinusoidalProtocol, axis_index: int, u: float) -> float:
        u = min(1.0, max(0.0, float(u)))
        omega_squared = protocol.angular_frequency_squared(u * protocol.final_time)
        return float(omega_squared[axis_index])

    def _build_protocol(self, final_time: float, a: float, b: float) -> RetroSinusoidalProtocol | None:
        try:
            return RetroSinusoidalProtocol(final_time=final_time, a=a, b=b, **self.protocol_kwargs)
        except ValueError as error:
            if "1 + a + b must be non-zero" in str(error):
                return None
            raise

    def _invalid_candidate(self, final_time: float, a: float, b: float, message: str) -> STACandidateEvaluation:
        return STACandidateEvaluation(
            final_time=final_time,
            a=a,
            b=b,
            phase_monotonic=False,
            minimum_omega_squared_x=float("-inf"),
            minimum_omega_squared_y=float("-inf"),
            minimum_omega_squared_z=float("-inf"),
            minimum_frequency_x_hz=float("-inf"),
            minimum_frequency_y_hz=float("-inf"),
            minimum_frequency_z_hz=float("-inf"),
            trapping_constraints_satisfied=False,
            z_constraint_satisfied=False,
            frequency_decompression_satisfied=False,
            maximum_omega_squared_increase_horizontal=float("inf"),
            maximum_omega_squared_increase_vertical=float("inf"),
            feasible=False,
            message=message,
        )

    def _z_omega_squared_threshold(self) -> float:
        return (2 * pi * self.minimum_z_frequency_hz) ** 2

    @staticmethod
    def _decompression_increase_tolerance(protocol: RetroSinusoidalProtocol) -> float:
        scale = max(
            protocol.omega_x_initial**2,
            protocol.omega_y_initial**2,
            protocol.omega_z_initial**2,
            1.0,
        )
        return 1e-9 * scale

    @staticmethod
    def _decompression_increase_tolerance_for_evaluation(evaluation: STACandidateEvaluation) -> float:
        scale = max(
            abs(evaluation.minimum_omega_squared_x),
            abs(evaluation.minimum_omega_squared_y),
            abs(evaluation.minimum_omega_squared_z),
            1.0,
        )
        return 1e-7 * scale

    @staticmethod
    def _frequency_from_omega_squared(value: float) -> float:
        if value < 0.0:
            return -sqrt(abs(value)) / (2 * pi)
        return sqrt(value) / (2 * pi)

    @staticmethod
    def _omega_squared_tolerance(protocol: RetroSinusoidalProtocol) -> float:
        return float(protocol._omega_squared_tolerance())

    @staticmethod
    def _validate_bounds(bounds: tuple[float, float], name: str, *, positive: bool = False) -> tuple[float, float]:
        if len(bounds) != 2:
            raise ValueError(f"{name} must contain exactly two values.")
        lower = STAOptimizer._validate_finite(bounds[0], f"{name}[0]")
        upper = STAOptimizer._validate_finite(bounds[1], f"{name}[1]")
        if positive and lower <= 0.0:
            raise ValueError(f"{name}[0] must be strictly positive.")
        if upper <= lower:
            raise ValueError(f"{name}[1] must be greater than {name}[0].")
        return lower, upper

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

    @staticmethod
    def _validate_int_at_least(value: int, name: str, minimum: int) -> int:
        if isinstance(value, bool):
            raise ValueError(f"{name} must be an integer.")
        value = int(value)
        if value < minimum:
            raise ValueError(f"{name} must be at least {minimum}.")
        return value

    @staticmethod
    def _validate_seed(seed: int | None) -> int | None:
        if seed is None:
            return None
        if isinstance(seed, bool):
            raise ValueError("seed must be an integer or None.")
        seed = int(seed)
        if seed < 0:
            raise ValueError("seed must be non-negative.")
        return seed
