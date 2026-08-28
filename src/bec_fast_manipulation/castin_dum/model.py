"""Canonical Castin-Dum scaling model."""

from collections.abc import Callable, Sequence
from math import isfinite, sqrt

import numpy as np
from scipy.integrate import solve_ivp

from bec_fast_manipulation.constants import ExperimentalConstants

STATE_SIZE = 6


class CastinDumModel:
    """Castin-Dum scaling dynamics for a trapped condensate.

    The canonical state ordering is
    ``[lambda_x, lambda_x_dot, lambda_y, lambda_y_dot, lambda_z, lambda_z_dot]``.
    All angular frequencies are expressed in rad/s.
    """

    def __init__(
        self,
        omega_x_initial: float = (
            ExperimentalConstants.INITIAL_TRAP_ANGULAR_FREQUENCY_X
        ),
        omega_y_initial: float = (
            ExperimentalConstants.INITIAL_TRAP_ANGULAR_FREQUENCY_Y
        ),
        omega_z_initial: float = (
            ExperimentalConstants.INITIAL_TRAP_ANGULAR_FREQUENCY_Z
        ),
        omega_x_final: float = ExperimentalConstants.FINAL_TRAP_ANGULAR_FREQUENCY_X,
        omega_y_final: float = ExperimentalConstants.FINAL_TRAP_ANGULAR_FREQUENCY_Y,
        omega_z_final: float = ExperimentalConstants.FINAL_TRAP_ANGULAR_FREQUENCY_Z,
    ) -> None:
        self.omega_x_initial = self._validate_positive_finite(
            omega_x_initial,
            "omega_x_initial",
        )
        self.omega_y_initial = self._validate_positive_finite(
            omega_y_initial,
            "omega_y_initial",
        )
        self.omega_z_initial = self._validate_positive_finite(
            omega_z_initial,
            "omega_z_initial",
        )
        self.omega_x_final = self._validate_positive_finite(
            omega_x_final,
            "omega_x_final",
        )
        self.omega_y_final = self._validate_positive_finite(
            omega_y_final,
            "omega_y_final",
        )
        self.omega_z_final = self._validate_positive_finite(
            omega_z_final,
            "omega_z_final",
        )

    def rhs(
        self,
        t: float,
        state: Sequence[float],
        omega_x: float,
        omega_y: float,
        omega_z: float,
    ) -> list[float]:
        """Return the Castin-Dum time derivative for instantaneous trap rates."""
        del t
        state_array = self._validate_state(state)
        omega_x_current = self._validate_finite(omega_x, "omega_x")
        omega_y_current = self._validate_finite(omega_y, "omega_y")
        omega_z_current = self._validate_finite(omega_z, "omega_z")

        lambda_x, lambda_x_dot, lambda_y, lambda_y_dot, lambda_z, lambda_z_dot = (
            state_array
        )

        lambda_x_ddot = (
            self.omega_x_initial**2 / (lambda_x**2 * lambda_y * lambda_z)
            - omega_x_current**2 * lambda_x
        )
        lambda_y_ddot = (
            self.omega_y_initial**2 / (lambda_x * lambda_y**2 * lambda_z)
            - omega_y_current**2 * lambda_y
        )
        lambda_z_ddot = (
            self.omega_z_initial**2 / (lambda_x * lambda_y * lambda_z**2)
            - omega_z_current**2 * lambda_z
        )

        return [
            lambda_x_dot,
            lambda_x_ddot,
            lambda_y_dot,
            lambda_y_ddot,
            lambda_z_dot,
            lambda_z_ddot,
        ]

    def rhs_free(self, t: float, state: Sequence[float]) -> list[float]:
        """Return the Castin-Dum derivative after the trap is switched off."""
        return self.rhs(t, state, omega_x=0.0, omega_y=0.0, omega_z=0.0)

    @staticmethod
    def equilibrium_initial_state() -> np.ndarray:
        """Return the canonical equilibrium initial state.

        The returned state is ordered as
        ``[lambda_x, lambda_x_dot, lambda_y, lambda_y_dot, lambda_z, lambda_z_dot]``.
        It corresponds to unit scaling factors and no initial expansion or
        compression velocity.
        """
        return np.array(
            [1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
            dtype=float,
        )

    def integrate(
        self,
        t_eval: Sequence[float],
        omega_x: Callable[[float], float],
        omega_y: Callable[[float], float],
        omega_z: Callable[[float], float],
        initial_state: Sequence[float] | None = None,
        *,
        method: str = "RK45",
        rtol: float = 1e-9,
        atol: float = 1e-12,
    ):
        """Integrate the Castin-Dum equations with time-dependent trap rates."""
        t_eval_array = self._validate_t_eval(t_eval)
        initial_state_array = self._validated_initial_state(initial_state)

        solution = solve_ivp(
            fun=lambda t, y: self.rhs(t, y, omega_x(t), omega_y(t), omega_z(t)),
            t_span=(t_eval_array[0], t_eval_array[-1]),
            y0=initial_state_array,
            t_eval=t_eval_array,
            method=method,
            rtol=rtol,
            atol=atol,
        )
        if not solution.success:
            raise RuntimeError(f"Castin-Dum integration failed: {solution.message}")
        return solution

    def integrate_free(
        self,
        t_eval: Sequence[float],
        initial_state: Sequence[float] | None = None,
        *,
        method: str = "RK45",
        rtol: float = 1e-9,
        atol: float = 1e-12,
    ):
        """Integrate the Castin-Dum equations after trap switch-off."""
        t_eval_array = self._validate_t_eval(t_eval)
        initial_state_array = self._validated_initial_state(initial_state)

        solution = solve_ivp(
            fun=self.rhs_free,
            t_span=(t_eval_array[0], t_eval_array[-1]),
            y0=initial_state_array,
            t_eval=t_eval_array,
            method=method,
            rtol=rtol,
            atol=atol,
        )
        if not solution.success:
            raise RuntimeError(f"Free Castin-Dum integration failed: {solution.message}")
        return solution

    def characteristic_time(self) -> float:
        """Return the historical characteristic time used by the project."""
        tc_x = abs(1 / self.omega_x_final - 1 / self.omega_x_initial) / (
            4 * sqrt(2)
        )
        tc_y = abs(1 / self.omega_y_final - 1 / self.omega_y_initial) / (
            4 * sqrt(2)
        )
        tc_z = abs(1 / self.omega_z_final - 1 / self.omega_z_initial) / (
            4 * sqrt(2)
        )
        return max(tc_x, tc_y, tc_z)

    def final_scaling_factors(self) -> tuple[float, float, float]:
        """Return the analytical final Castin-Dum scaling factors."""
        ratio_x = self.omega_x_initial / self.omega_x_final
        ratio_y = self.omega_y_initial / self.omega_y_final
        ratio_z = self.omega_z_initial / self.omega_z_final

        lambda_x_final = ratio_x ** (4 / 5) / (ratio_y * ratio_z) ** (1 / 5)
        lambda_y_final = ratio_y ** (4 / 5) / (ratio_x * ratio_z) ** (1 / 5)
        lambda_z_final = ratio_z ** (4 / 5) / (ratio_x * ratio_y) ** (1 / 5)

        return lambda_x_final, lambda_y_final, lambda_z_final

    @staticmethod
    def _validate_positive_finite(value: float, name: str) -> float:
        value = float(value)
        if not isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and strictly positive.")
        return value

    @staticmethod
    def _validate_finite(value: float, name: str) -> float:
        value = float(value)
        if not isfinite(value):
            raise ValueError(f"{name} must be finite.")
        return value

    @staticmethod
    def _validate_state(state: Sequence[float]) -> np.ndarray:
        state_array = np.asarray(state, dtype=float)
        if state_array.shape != (STATE_SIZE,):
            raise ValueError(
                "state must contain exactly 6 values ordered as "
                "[lambda_x, lambda_x_dot, lambda_y, lambda_y_dot, "
                "lambda_z, lambda_z_dot]."
            )
        if not np.all(np.isfinite(state_array)):
            raise ValueError("state values must be finite.")
        if np.any(state_array[[0, 2, 4]] <= 0):
            raise ValueError("lambda_x, lambda_y, and lambda_z must be positive.")
        return state_array

    @classmethod
    def _validated_initial_state(cls, state: Sequence[float] | None) -> np.ndarray:
        if state is None:
            return cls.equilibrium_initial_state()
        return cls._validate_state(state)

    @staticmethod
    def _validate_t_eval(t_eval: Sequence[float]) -> np.ndarray:
        t_eval_array = np.asarray(t_eval, dtype=float)
        if t_eval_array.ndim != 1:
            raise ValueError("t_eval must be a one-dimensional array.")
        if t_eval_array.size < 2:
            raise ValueError("t_eval must contain at least two points.")
        if not np.all(np.isfinite(t_eval_array)):
            raise ValueError("t_eval values must be finite.")
        if np.any(np.diff(t_eval_array) <= 0):
            raise ValueError("t_eval values must be strictly increasing.")
        return t_eval_array
