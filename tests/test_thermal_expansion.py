import numpy as np
import pytest

from bec_fast_manipulation.constants import PhysicalConstants
from bec_fast_manipulation.thermal_expansion import ThermalExpansionModel


def expected_temperature(velocity):
    return (
        PhysicalConstants.RUBIDIUM_87_MASS
        / PhysicalConstants.BOLTZMANN_CONSTANT
        * np.asarray(velocity) ** 2
    )


def test_zero_velocity_gives_zero_temperatures():
    model = ThermalExpansionModel()
    temperatures = model.temperatures_from_radius_velocities(0.0, 0.0, 0.0)

    assert temperatures == {"x": 0.0, "y": 0.0, "z": 0.0, "3d": 0.0}


def test_directional_temperatures_match_historical_formula():
    model = ThermalExpansionModel()
    velocities = (0.001, 0.002, 0.003)
    temperatures = model.directional_temperatures(*velocities)

    assert temperatures == pytest.approx(tuple(expected_temperature(velocities)))


def test_directional_temperatures_are_sign_invariant():
    model = ThermalExpansionModel()

    assert model.directional_temperatures(0.1, -0.2, 0.3) == pytest.approx(
        model.directional_temperatures(-0.1, 0.2, -0.3)
    )


def test_temperature_3d_is_directional_average():
    model = ThermalExpansionModel()

    assert model.temperature_3d(1.0, 2.0, 6.0) == pytest.approx(3.0)


def test_arrays_preserve_shapes_values_and_inputs():
    model = ThermalExpansionModel()
    velocity_x = np.array([0.0, 0.001, -0.002])
    velocity_y = np.array([0.003, -0.004, 0.005])
    velocity_z = np.array([-0.006, 0.007, 0.008])
    velocity_x_copy = velocity_x.copy()

    temperatures = model.temperatures_from_radius_velocities(velocity_x, velocity_y, velocity_z)

    assert temperatures["x"].shape == velocity_x.shape
    assert temperatures["3d"].shape == velocity_x.shape
    np.testing.assert_allclose(temperatures["x"], expected_temperature(velocity_x))
    np.testing.assert_allclose(temperatures["y"], expected_temperature(velocity_y))
    np.testing.assert_allclose(temperatures["z"], expected_temperature(velocity_z))
    np.testing.assert_allclose(
        temperatures["3d"],
        (temperatures["x"] + temperatures["y"] + temperatures["z"]) / 3,
    )
    np.testing.assert_array_equal(velocity_x, velocity_x_copy)


def test_historical_factor_three_conventions_match_for_3d_temperature():
    velocities = np.array([0.001, 0.002, 0.003])
    canonical_directional = expected_temperature(velocities)
    canonical_3d = np.sum(canonical_directional) / 3

    alternative_directional = canonical_directional / 3
    alternative_3d = np.sum(alternative_directional)

    assert alternative_3d == pytest.approx(canonical_3d)
    np.testing.assert_allclose(alternative_directional, canonical_directional / 3)


def test_rejects_non_finite_or_incompatible_arrays():
    model = ThermalExpansionModel()

    with pytest.raises(ValueError):
        model.directional_temperatures([0.0, np.nan], [0.0, 0.0], [0.0, 0.0])
    with pytest.raises(ValueError):
        model.temperature_3d([1.0], [1.0, 2.0], [1.0])
