from math import isclose, pi

from bec_fast_manipulation.constants import PhysicalConstants


def test_fundamental_constants_keep_historical_values():
    assert PhysicalConstants.BOLTZMANN_CONSTANT == 1.380649e-23
    assert PhysicalConstants.REDUCED_PLANCK_CONSTANT == 1.05457180013e-34
    assert PhysicalConstants.SPEED_OF_LIGHT == 2.99792458e8
    assert PhysicalConstants.STANDARD_GRAVITY == 9.80665


def test_rubidium_87_constants_keep_historical_values():
    assert PhysicalConstants.RUBIDIUM_87_MASS == 1.4431606483768263e-25
    assert PhysicalConstants.RUBIDIUM_87_SCATTERING_LENGTH == 100 * 0.529e-10


def test_rubidium_d1_constants_keep_historical_values():
    assert PhysicalConstants.RUBIDIUM_D1_WAVELENGTH == 7.94978851156e-7
    assert isclose(
        PhysicalConstants.RUBIDIUM_D1_ANGULAR_FREQUENCY,
        2 * pi * 3.77107463380e14,
    )
    assert isclose(
        PhysicalConstants.RUBIDIUM_D1_LINEWIDTH,
        2 * pi * 5.7500e6,
    )


def test_rubidium_d2_constants_keep_historical_values():
    assert PhysicalConstants.RUBIDIUM_D2_WAVELENGTH == 7.80241209686e-7
    assert isclose(
        PhysicalConstants.RUBIDIUM_D2_ANGULAR_FREQUENCY,
        2 * pi * 3.842304844685e14,
    )
    assert isclose(
        PhysicalConstants.RUBIDIUM_D2_LINEWIDTH,
        2 * pi * 6.0666e6,
    )


def test_implicit_si_orders_of_magnitude():
    assert 1e-24 < PhysicalConstants.BOLTZMANN_CONSTANT < 1e-22
    assert 1e-35 < PhysicalConstants.REDUCED_PLANCK_CONSTANT < 1e-33
    assert 1e8 < PhysicalConstants.SPEED_OF_LIGHT < 1e9
    assert 1.0 < PhysicalConstants.STANDARD_GRAVITY < 20.0
    assert 1e-26 < PhysicalConstants.RUBIDIUM_87_MASS < 1e-24
    assert 1e-9 < PhysicalConstants.RUBIDIUM_87_SCATTERING_LENGTH < 1e-8
    assert 1e-7 < PhysicalConstants.RUBIDIUM_D1_WAVELENGTH < 1e-6
    assert 1e15 < PhysicalConstants.RUBIDIUM_D1_ANGULAR_FREQUENCY < 1e16
    assert 1e7 < PhysicalConstants.RUBIDIUM_D1_LINEWIDTH < 1e8


def test_experimental_simulation_and_derived_quantities_are_not_included():
    excluded_names = ("N", "lambda_L", "omega_L", "wH", "wV", "U0")

    for name in excluded_names:
        assert not hasattr(PhysicalConstants, name)


def test_import_from_constants_package():
    from bec_fast_manipulation.constants import PhysicalConstants as ImportedConstants

    assert ImportedConstants is PhysicalConstants
