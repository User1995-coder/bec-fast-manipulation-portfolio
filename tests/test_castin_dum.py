from math import inf, isclose, nan, pi, sqrt

import numpy as np
import pytest

from bec_fast_manipulation.castin_dum import CastinDumModel
from bec_fast_manipulation.constants import ExperimentalConstants


def make_model():
    return CastinDumModel(
        omega_x_initial=2 * pi * 500,
        omega_y_initial=2 * pi * 600,
        omega_z_initial=2 * pi * 700,
    )


def test_default_initial_frequencies_use_experimental_constants():
    model = CastinDumModel()

    assert (
        model.omega_x_initial
        == ExperimentalConstants.INITIAL_TRAP_ANGULAR_FREQUENCY_X
    )
    assert (
        model.omega_y_initial
        == ExperimentalConstants.INITIAL_TRAP_ANGULAR_FREQUENCY_Y
    )
    assert (
        model.omega_z_initial
        == ExperimentalConstants.INITIAL_TRAP_ANGULAR_FREQUENCY_Z
    )
    assert model.omega_x_final == ExperimentalConstants.FINAL_TRAP_ANGULAR_FREQUENCY_X
    assert model.omega_y_final == ExperimentalConstants.FINAL_TRAP_ANGULAR_FREQUENCY_Y
    assert model.omega_z_final == ExperimentalConstants.FINAL_TRAP_ANGULAR_FREQUENCY_Z


def test_explicit_initial_frequencies_override_nominal_defaults():
    model = CastinDumModel(
        omega_x_initial=2 * pi * 10,
        omega_y_initial=2 * pi * 20,
        omega_z_initial=2 * pi * 30,
        omega_x_final=2 * pi * 1,
        omega_y_final=2 * pi * 2,
        omega_z_final=2 * pi * 3,
    )

    assert model.omega_x_initial == 2 * pi * 10
    assert model.omega_y_initial == 2 * pi * 20
    assert model.omega_z_initial == 2 * pi * 30
    assert model.omega_x_final == 2 * pi * 1
    assert model.omega_y_final == 2 * pi * 2
    assert model.omega_z_final == 2 * pi * 3


def test_equilibrium_initial_state_values_shape_and_dtype():
    state = CastinDumModel.equilibrium_initial_state()

    np.testing.assert_array_equal(state, [1.0, 0.0, 1.0, 0.0, 1.0, 0.0])
    assert state.shape == (6,)
    assert np.issubdtype(state.dtype, np.floating)
    assert np.all(np.isfinite(state))


def test_equilibrium_initial_state_returns_independent_arrays():
    model = make_model()
    state_1 = model.equilibrium_initial_state()
    state_2 = model.equilibrium_initial_state()

    state_1[0] = 42.0

    np.testing.assert_array_equal(state_2, [1.0, 0.0, 1.0, 0.0, 1.0, 0.0])


def test_positive_initial_frequencies_create_model():
    model = make_model()

    assert model.omega_x_initial == 2 * pi * 500
    assert model.omega_y_initial == 2 * pi * 600
    assert model.omega_z_initial == 2 * pi * 700


@pytest.mark.parametrize("bad_value", [0.0, -1.0, nan, inf])
def test_invalid_initial_frequencies_are_rejected(bad_value):
    with pytest.raises(ValueError, match="omega_x_initial"):
        CastinDumModel(bad_value, 2 * pi * 600, 2 * pi * 700)

    with pytest.raises(ValueError, match="omega_y_initial"):
        CastinDumModel(2 * pi * 500, bad_value, 2 * pi * 700)

    with pytest.raises(ValueError, match="omega_z_initial"):
        CastinDumModel(2 * pi * 500, 2 * pi * 600, bad_value)


@pytest.mark.parametrize("bad_value", [0.0, -1.0, nan, inf])
def test_invalid_final_frequencies_are_rejected(bad_value):
    with pytest.raises(ValueError, match="omega_x_final"):
        CastinDumModel(
            2 * pi * 500,
            2 * pi * 600,
            2 * pi * 700,
            bad_value,
            2 * pi * 5,
            2 * pi * 60,
        )

    with pytest.raises(ValueError, match="omega_y_final"):
        CastinDumModel(
            2 * pi * 500,
            2 * pi * 600,
            2 * pi * 700,
            2 * pi * 5,
            bad_value,
            2 * pi * 60,
        )

    with pytest.raises(ValueError, match="omega_z_final"):
        CastinDumModel(
            2 * pi * 500,
            2 * pi * 600,
            2 * pi * 700,
            2 * pi * 5,
            2 * pi * 5,
            bad_value,
        )


def test_equilibrium_state_has_zero_acceleration_in_constant_initial_trap():
    model = make_model()
    rhs = model.rhs(
        0.0,
        [1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
        model.omega_x_initial,
        model.omega_y_initial,
        model.omega_z_initial,
    )

    np.testing.assert_allclose(rhs, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], atol=1e-12)


def test_rhs_matches_manual_anisotropic_accelerations():
    model = make_model()
    state = [1.2, 0.3, 0.9, -0.1, 1.5, 0.2]
    omega_x = 2 * pi * 450
    omega_y = 2 * pi * 650
    omega_z = 2 * pi * 550

    rhs = model.rhs(0.0, state, omega_x, omega_y, omega_z)
    lambda_x, lambda_x_dot, lambda_y, lambda_y_dot, lambda_z, lambda_z_dot = state
    expected_x_ddot = (
        model.omega_x_initial**2 / (lambda_x**2 * lambda_y * lambda_z)
        - omega_x**2 * lambda_x
    )
    expected_y_ddot = (
        model.omega_y_initial**2 / (lambda_x * lambda_y**2 * lambda_z)
        - omega_y**2 * lambda_y
    )
    expected_z_ddot = (
        model.omega_z_initial**2 / (lambda_x * lambda_y * lambda_z**2)
        - omega_z**2 * lambda_z
    )

    np.testing.assert_allclose(
        rhs,
        [
            lambda_x_dot,
            expected_x_ddot,
            lambda_y_dot,
            expected_y_ddot,
            lambda_z_dot,
            expected_z_ddot,
        ],
        rtol=1e-14,
    )


def test_free_rhs_initial_accelerations_are_initial_frequency_squares():
    model = make_model()

    rhs = model.rhs_free(0.0, [1.0, 0.0, 1.0, 0.0, 1.0, 0.0])

    np.testing.assert_allclose(
        rhs,
        [
            0.0,
            model.omega_x_initial**2,
            0.0,
            model.omega_y_initial**2,
            0.0,
            model.omega_z_initial**2,
        ],
    )


def test_constant_initial_trap_integration_keeps_equilibrium_state():
    model = make_model()
    t_eval = np.linspace(0.0, 1e-4, 8)

    solution = model.integrate(
        t_eval=t_eval,
        omega_x=lambda t: model.omega_x_initial,
        omega_y=lambda t: model.omega_y_initial,
        omega_z=lambda t: model.omega_z_initial,
    )

    assert solution.success
    np.testing.assert_allclose(solution.y[[0, 2, 4]], 1.0, atol=1e-10)
    np.testing.assert_allclose(solution.y[[1, 3, 5]], 0.0, atol=1e-8)


def test_integrate_accepts_explicit_initial_state():
    model = make_model()
    t_eval = np.linspace(0.0, 1e-5, 4)
    initial_state = [1.1, 0.2, 0.95, -0.1, 1.05, 0.0]

    solution = model.integrate(
        t_eval=t_eval,
        omega_x=lambda t: model.omega_x_initial,
        omega_y=lambda t: model.omega_y_initial,
        omega_z=lambda t: model.omega_z_initial,
        initial_state=initial_state,
    )

    np.testing.assert_allclose(solution.y[:, 0], initial_state)


def test_short_free_expansion_grows_from_equilibrium_state():
    model = make_model()
    t_eval = np.linspace(0.0, 1e-5, 8)

    solution = model.integrate_free(t_eval=t_eval)

    assert solution.success
    assert np.all(solution.y[[0, 2, 4], -1] > 1.0)
    assert np.all(solution.y[[1, 3, 5], -1] > 0.0)


def test_integrate_free_accepts_explicit_initial_state():
    model = make_model()
    t_eval = np.linspace(0.0, 1e-5, 4)
    initial_state = [1.1, 0.2, 0.95, -0.1, 1.05, 0.0]

    solution = model.integrate_free(t_eval=t_eval, initial_state=initial_state)

    np.testing.assert_allclose(solution.y[:, 0], initial_state)


def test_characteristic_time_matches_manual_expression():
    model = CastinDumModel(
        omega_x_initial=2 * pi * 500,
        omega_y_initial=2 * pi * 600,
        omega_z_initial=2 * pi * 700,
        omega_x_final=2 * pi * 100,
        omega_y_final=2 * pi * 200,
        omega_z_final=2 * pi * 300,
    )

    expected = max(
        abs(1 / model.omega_x_final - 1 / model.omega_x_initial) / (4 * sqrt(2)),
        abs(1 / model.omega_y_final - 1 / model.omega_y_initial) / (4 * sqrt(2)),
        abs(1 / model.omega_z_final - 1 / model.omega_z_initial) / (4 * sqrt(2)),
    )

    assert isclose(
        model.characteristic_time(),
        expected,
        rel_tol=1e-15,
    )


def test_final_scaling_factors_match_manual_anisotropic_expression():
    model = CastinDumModel(
        omega_x_initial=2 * pi * 500,
        omega_y_initial=2 * pi * 600,
        omega_z_initial=2 * pi * 700,
        omega_x_final=2 * pi * 100,
        omega_y_final=2 * pi * 240,
        omega_z_final=2 * pi * 350,
    )

    ratio_x = model.omega_x_initial / model.omega_x_final
    ratio_y = model.omega_y_initial / model.omega_y_final
    ratio_z = model.omega_z_initial / model.omega_z_final

    expected = (
        ratio_x ** (4 / 5) / (ratio_y * ratio_z) ** (1 / 5),
        ratio_y ** (4 / 5) / (ratio_x * ratio_z) ** (1 / 5),
        ratio_z ** (4 / 5) / (ratio_x * ratio_y) ** (1 / 5),
    )

    np.testing.assert_allclose(model.final_scaling_factors(), expected, rtol=1e-15)


def test_isotropic_final_scaling_factors_are_ratio_to_two_fifths():
    omega_initial = 2 * pi * 500
    ratio = 5.0
    model = CastinDumModel(
        omega_initial,
        omega_initial,
        omega_initial,
        omega_initial / ratio,
        omega_initial / ratio,
        omega_initial / ratio,
    )

    factors = model.final_scaling_factors()

    np.testing.assert_allclose(factors, [ratio ** (2 / 5)] * 3, rtol=1e-15)


def test_integrate_rejects_invalid_initial_state_and_t_eval():
    model = make_model()
    omega = lambda t: model.omega_x_initial

    with pytest.raises(ValueError, match="exactly 6 values"):
        model.integrate([0.0, 1.0], omega, omega, omega, initial_state=[1.0, 0.0])

    with pytest.raises(ValueError, match="must be positive"):
        model.integrate(
            [0.0, 1.0],
            omega,
            omega,
            omega,
            initial_state=[0.0, 0.0, 1.0, 0.0, 1.0, 0.0],
        )

    with pytest.raises(ValueError, match="state values must be finite"):
        model.integrate(
            [0.0, 1.0],
            omega,
            omega,
            omega,
            initial_state=[nan, 0.0, 1.0, 0.0, 1.0, 0.0],
        )

    with pytest.raises(ValueError, match="strictly increasing"):
        model.integrate(
            [0.0, 0.0, 1.0],
            omega,
            omega,
            omega,
            initial_state=[1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
        )

    with pytest.raises(ValueError, match="at least two points"):
        model.integrate(
            [0.0],
            omega,
            omega,
            omega,
            initial_state=[1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
        )
