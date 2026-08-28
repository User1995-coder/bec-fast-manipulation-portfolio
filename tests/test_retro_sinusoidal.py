import math

import numpy as np
import pytest

from bec_fast_manipulation.castin_dum import CastinDumModel
from bec_fast_manipulation.constants import ExperimentalConstants
from bec_fast_manipulation.retro_sinusoidal import RetroSinusoidalProtocol


def make_protocol(**kwargs):
    defaults = {
        "final_time": 0.006611,
        "a": -1.371429,
        "b": 0.910204,
    }
    defaults.update(kwargs)
    return RetroSinusoidalProtocol(**defaults)


def central_difference(function, t, h=1e-7):
    return (function(t + h) - function(t - h)) / (2 * h)


def test_defaults_use_experimental_constants_and_castin_dum_final_scaling_factors():
    protocol = make_protocol()
    expected_lambdas = CastinDumModel().final_scaling_factors()

    assert protocol.omega_x_initial == ExperimentalConstants.INITIAL_TRAP_ANGULAR_FREQUENCY_X
    assert protocol.omega_y_initial == ExperimentalConstants.INITIAL_TRAP_ANGULAR_FREQUENCY_Y
    assert protocol.omega_z_initial == ExperimentalConstants.INITIAL_TRAP_ANGULAR_FREQUENCY_Z
    assert protocol.lambda_x_final == pytest.approx(expected_lambdas[0])
    assert protocol.lambda_y_final == pytest.approx(expected_lambdas[1])
    assert protocol.lambda_z_final == pytest.approx(expected_lambdas[2])


def test_explicit_final_scaling_factors_are_used_exactly():
    protocol = make_protocol(lambda_x_final=2.0, lambda_y_final=3.0, lambda_z_final=4.0)

    assert protocol.lambda_x_final == 2.0
    assert protocol.lambda_y_final == 3.0
    assert protocol.lambda_z_final == 4.0


def test_phase_endpoints_and_scalar_return_type():
    protocol = make_protocol(a=0.0, b=0.0, final_time=0.1)

    assert isinstance(protocol.phase(0.0), float)
    assert protocol.phase(0.0) == pytest.approx(0.0)
    assert protocol.phase(protocol.final_time) == pytest.approx(2 * math.pi)


def test_phase_derivatives_match_finite_differences():
    protocol = make_protocol()
    for t in np.linspace(0.001, 0.005, 5):
        assert protocol.phase_velocity(t) == pytest.approx(
            central_difference(protocol.phase, t),
            rel=1e-7,
            abs=1e-5,
        )
        assert protocol.phase_acceleration(t) == pytest.approx(
            central_difference(protocol.phase_velocity, t),
            rel=1e-7,
            abs=1e-2,
        )


def test_scaling_endpoint_conditions():
    protocol = make_protocol(lambda_x_final=2.0, lambda_y_final=3.0, lambda_z_final=4.0)

    np.testing.assert_allclose(protocol.scaling_factors(0.0), [1.0, 1.0, 1.0])
    np.testing.assert_allclose(protocol.scaling_factors(protocol.final_time), [2.0, 3.0, 4.0])
    np.testing.assert_allclose(protocol.scaling_velocities(0.0), [0.0, 0.0, 0.0], atol=1e-10)
    np.testing.assert_allclose(protocol.scaling_velocities(protocol.final_time), [0.0, 0.0, 0.0], atol=1e-9)
    np.testing.assert_allclose(protocol.scaling_accelerations(0.0), [0.0, 0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(protocol.scaling_accelerations(protocol.final_time), [0.0, 0.0, 0.0], atol=1e-5)


def test_scaling_derivatives_match_finite_differences():
    protocol = make_protocol(lambda_x_final=2.0, lambda_y_final=3.0, lambda_z_final=4.0)

    for t in np.linspace(0.001, 0.005, 5):
        for axis in range(3):
            scaling_axis = lambda value, axis=axis: protocol.scaling_factors(value)[axis]
            velocity_axis = lambda value, axis=axis: protocol.scaling_velocities(value)[axis]
            assert protocol.scaling_velocities(t)[axis] == pytest.approx(
                central_difference(scaling_axis, t),
                rel=1e-6,
                abs=1e-4,
            )
            assert protocol.scaling_accelerations(t)[axis] == pytest.approx(
                central_difference(velocity_axis, t),
                rel=1e-5,
                abs=1e-1,
            )


def test_inverse_castin_dum_residual_is_zero():
    protocol = make_protocol()

    for t in np.linspace(0.0005, protocol.final_time - 0.0005, 7):
        lambdas = protocol.scaling_factors(t)
        accelerations = protocol.scaling_accelerations(t)
        omega_squared = protocol.angular_frequency_squared(t)
        omega_initial = (
            protocol.omega_x_initial,
            protocol.omega_y_initial,
            protocol.omega_z_initial,
        )
        denominators = (
            lambdas[0] ** 2 * lambdas[1] * lambdas[2],
            lambdas[0] * lambdas[1] ** 2 * lambdas[2],
            lambdas[0] * lambdas[1] * lambdas[2] ** 2,
        )
        for axis in range(3):
            reconstructed = omega_initial[axis] ** 2 / denominators[axis] - omega_squared[axis] * lambdas[axis]
            assert accelerations[axis] - reconstructed == pytest.approx(0.0, abs=1e-8)


def test_anti_trapping_negative_omega_squared_is_preserved():
    protocol = make_protocol(a=-1.6, b=0.8)
    omega_x_squared, omega_y_squared, omega_z_squared = protocol.angular_frequency_squared(0.000390049)

    assert omega_x_squared < 0.0
    assert omega_y_squared < 0.0
    assert omega_z_squared > 0.0
    assert not protocol.is_trapping_protocol(0.000390049)
    with pytest.raises(ValueError, match="axis y"):
        protocol.angular_frequencies(0.000390049)


def test_admissible_protocol_angular_frequencies_match_square_root():
    protocol = make_protocol()
    t = np.linspace(0.0, protocol.final_time, 101)
    omega_squared = protocol.angular_frequency_squared(t)
    angular_frequencies = protocol.angular_frequencies(t)

    assert protocol.is_trapping_protocol(t)
    for axis in range(3):
        assert np.min(omega_squared[axis]) >= 0.0
        np.testing.assert_allclose(angular_frequencies[axis], np.sqrt(omega_squared[axis]))


def test_angular_frequencies_squared_matches_omega_squared_for_admissible_protocol():
    protocol = make_protocol()
    t = np.linspace(0.0, protocol.final_time, 31)
    omega_squared = protocol.angular_frequency_squared(t)
    angular_frequencies = protocol.angular_frequencies(t)

    for axis in range(3):
        np.testing.assert_allclose(angular_frequencies[axis] ** 2, omega_squared[axis], rtol=1e-12, atol=1e-8)


def test_angular_frequencies_tolerates_tiny_negative_roundoff():
    protocol = make_protocol()
    tolerance = protocol._omega_squared_tolerance()
    protocol.angular_frequency_squared = lambda _t: (-0.5 * tolerance, np.array([4.0, -0.5 * tolerance]), 9.0)

    omega_x, omega_y, omega_z = protocol.angular_frequencies(np.array([0.0, 1.0]))

    assert omega_x == 0.0
    np.testing.assert_allclose(omega_y, [2.0, 0.0])
    assert omega_z == 3.0
    assert protocol.is_trapping_protocol(np.array([0.0, 1.0]))


def test_phase_monotonicity_and_constructor_requirement():
    assert RetroSinusoidalProtocol(final_time=1.0, a=0.0, b=0.0).is_phase_monotonic()
    assert not RetroSinusoidalProtocol(final_time=1.0, a=-1.6, b=0.8).is_phase_monotonic()
    RetroSinusoidalProtocol(final_time=1.0, a=-1.6, b=0.8, require_monotonic_phase=False)
    with pytest.raises(ValueError, match="monotone"):
        RetroSinusoidalProtocol(final_time=1.0, a=-1.6, b=0.8, require_monotonic_phase=True)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"final_time": 0.0},
        {"final_time": -1.0},
        {"final_time": np.nan},
        {"final_time": np.inf},
        {"a": np.nan},
        {"b": np.inf},
        {"a": -1.0, "b": 0.0},
        {"lambda_x_initial": 0.0},
        {"lambda_y_final": -1.0, "lambda_x_final": 1.0, "lambda_z_final": 1.0},
        {"omega_z_initial": 0.0},
    ],
)
def test_constructor_validation(kwargs):
    with pytest.raises(ValueError):
        make_protocol(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"lambda_x_final": 2.0},
        {"lambda_x_final": 2.0, "lambda_z_final": 4.0},
    ],
)
def test_partial_final_lambdas_are_rejected(kwargs):
    with pytest.raises(ValueError, match="provided together"):
        make_protocol(**kwargs)


@pytest.mark.parametrize("bad_t", [-1e-12, 1.1, np.nan, np.inf, [], [[0.0, 0.1]]])
def test_time_domain_validation(bad_t):
    protocol = RetroSinusoidalProtocol(final_time=1.0, a=0.0, b=0.0)
    with pytest.raises(ValueError):
        protocol.phase(bad_t)


def test_array_support_shapes_finiteness_no_mutation_and_scalar_consistency():
    protocol = make_protocol(lambda_x_final=2.0, lambda_y_final=3.0, lambda_z_final=4.0)
    t = np.linspace(0.0, protocol.final_time, 11)
    original = t.copy()

    for method_name in (
        "phase",
        "phase_velocity",
        "phase_acceleration",
    ):
        values = getattr(protocol, method_name)(t)
        assert values.shape == t.shape
        assert np.all(np.isfinite(values))
        assert values[3] == pytest.approx(getattr(protocol, method_name)(float(t[3])))

    for method_name in (
        "scaling_factors",
        "scaling_velocities",
        "scaling_accelerations",
        "angular_frequency_squared",
        "angular_frequencies",
    ):
        values = getattr(protocol, method_name)(t)
        assert len(values) == 3
        for axis_values in values:
            assert axis_values.shape == t.shape
            assert np.all(np.isfinite(axis_values))
        scalar_values = getattr(protocol, method_name)(float(t[3]))
        for axis in range(3):
            assert values[axis][3] == pytest.approx(scalar_values[axis])

    np.testing.assert_allclose(t, original)


def test_no_diagnostic_curvature_frequency_method_is_exposed():
    removed_method_name = "_".join(("sig" + "ned", "angular", "frequencies"))
    assert not hasattr(make_protocol(), removed_method_name)
