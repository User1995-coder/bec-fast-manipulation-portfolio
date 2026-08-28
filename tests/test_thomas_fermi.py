from math import isclose

import numpy as np
import pytest

from bec_fast_manipulation.constants import ExperimentalConstants, PhysicalConstants
from bec_fast_manipulation.thomas_fermi import ThomasFermiModel


def expected_mu(model):
    omega_bar = (model.omega_x * model.omega_y * model.omega_z) ** (1 / 3)
    hbar = PhysicalConstants.REDUCED_PLANCK_CONSTANT
    mass = PhysicalConstants.RUBIDIUM_87_MASS
    scattering_length = PhysicalConstants.RUBIDIUM_87_SCATTERING_LENGTH
    return (hbar * omega_bar / 2) * (
        15
        * model.atom_number
        * scattering_length
        * np.sqrt(mass * omega_bar / hbar)
    ) ** (2 / 5)


def test_defaults_use_centralized_constants():
    model = ThomasFermiModel()

    assert model.atom_number == ExperimentalConstants.CONDENSATE_ATOM_NUMBER
    assert model.omega_x == ExperimentalConstants.INITIAL_TRAP_ANGULAR_FREQUENCY_X
    assert model.omega_y == ExperimentalConstants.INITIAL_TRAP_ANGULAR_FREQUENCY_Y
    assert model.omega_z == ExperimentalConstants.INITIAL_TRAP_ANGULAR_FREQUENCY_Z


def test_custom_values_override_defaults():
    model = ThomasFermiModel(atom_number=2_000, omega_x=10.0, omega_y=20.0, omega_z=30.0)

    assert model.atom_number == 2_000
    assert model.omega_x == 10.0
    assert model.omega_y == 20.0
    assert model.omega_z == 30.0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"atom_number": 0},
        {"atom_number": -1},
        {"omega_x": 0.0},
        {"omega_y": -1.0},
        {"omega_z": np.nan},
        {"atom_number": np.inf},
    ],
)
def test_invalid_constructor_values_are_rejected(kwargs):
    with pytest.raises(ValueError):
        ThomasFermiModel(**kwargs)


def test_geometric_mean_frequency_matches_independent_calculation():
    model = ThomasFermiModel(omega_x=10.0, omega_y=20.0, omega_z=40.0)

    assert model.geometric_mean_frequency() == pytest.approx((10.0 * 20.0 * 40.0) ** (1 / 3))


def test_chemical_potential_matches_historical_expression():
    model = ThomasFermiModel()

    assert model.chemical_potential() == pytest.approx(expected_mu(model), rel=1e-14)


def test_initial_radii_match_independent_expression():
    model = ThomasFermiModel()
    mu = expected_mu(model)
    mass = PhysicalConstants.RUBIDIUM_87_MASS
    expected = (
        np.sqrt(2 * mu / (mass * model.omega_x**2)),
        np.sqrt(2 * mu / (mass * model.omega_y**2)),
        np.sqrt(2 * mu / (mass * model.omega_z**2)),
    )

    assert model.initial_radii() == pytest.approx(expected, rel=1e-14)


def test_isotropic_initial_radii_are_equal():
    model = ThomasFermiModel(omega_x=100.0, omega_y=100.0, omega_z=100.0)
    radius_x, radius_y, radius_z = model.initial_radii()

    assert radius_x == pytest.approx(radius_y)
    assert radius_y == pytest.approx(radius_z)


def test_scaling_factors_convert_to_radii_without_mutating_inputs():
    model = ThomasFermiModel()
    initial_radii = model.initial_radii()
    radius_x, radius_y, radius_z = model.radii_from_scaling_factors(1.0, 1.0, 1.0)

    assert (radius_x, radius_y, radius_z) == pytest.approx(initial_radii)

    lambda_x = np.array([1.0, 2.0])
    lambda_y = np.array([3.0, 4.0])
    lambda_z = np.array([5.0, 6.0])
    lambda_x_copy = lambda_x.copy()
    radii = model.radii_from_scaling_factors(lambda_x, lambda_y, lambda_z)

    np.testing.assert_allclose(radii[0], initial_radii[0] * lambda_x)
    np.testing.assert_allclose(radii[1], initial_radii[1] * lambda_y)
    np.testing.assert_allclose(radii[2], initial_radii[2] * lambda_z)
    np.testing.assert_array_equal(lambda_x, lambda_x_copy)


def test_scaling_factors_reject_non_positive_or_incompatible_values():
    model = ThomasFermiModel()

    with pytest.raises(ValueError):
        model.radii_from_scaling_factors([1.0, 0.0], [1.0, 1.0], [1.0, 1.0])
    with pytest.raises(ValueError):
        model.radii_from_scaling_factors([1.0], [1.0, 1.0], [1.0])


def test_radius_velocities_from_scaling_velocities():
    model = ThomasFermiModel()
    initial_radii = model.initial_radii()

    assert model.radius_velocities_from_scaling_velocities(0.0, 0.0, 0.0) == pytest.approx((0.0, 0.0, 0.0))

    lambda_x_dot = np.array([-1.0, 2.0])
    lambda_y_dot = np.array([0.0, 3.0])
    lambda_z_dot = np.array([4.0, -5.0])
    velocities = model.radius_velocities_from_scaling_velocities(lambda_x_dot, lambda_y_dot, lambda_z_dot)

    np.testing.assert_allclose(velocities[0], initial_radii[0] * lambda_x_dot)
    np.testing.assert_allclose(velocities[1], initial_radii[1] * lambda_y_dot)
    np.testing.assert_allclose(velocities[2], initial_radii[2] * lambda_z_dot)


def test_radius_velocities_reject_non_finite_values():
    model = ThomasFermiModel()

    with pytest.raises(ValueError):
        model.radius_velocities_from_scaling_velocities([0.0, np.inf], [0.0, 0.0], [0.0, 0.0])
