from math import isclose, pi

from bec_fast_manipulation.constants import ExperimentalConstants


def test_experimental_constants_keep_historical_values():
    assert isclose(ExperimentalConstants.LASER_WAVELENGTH, 1.064e-6)
    assert isclose(ExperimentalConstants.HORIZONTAL_BEAM_WAIST, 50e-6)
    assert isclose(ExperimentalConstants.VERTICAL_BEAM_WAIST, 50e-6)


def test_condensate_atom_number_keeps_historical_value():
    assert ExperimentalConstants.CONDENSATE_ATOM_NUMBER == 100_000


def test_nominal_initial_trap_frequencies_keep_historical_values():
    assert isclose(
        ExperimentalConstants.INITIAL_TRAP_ANGULAR_FREQUENCY_X,
        2 * pi * 500,
    )
    assert isclose(
        ExperimentalConstants.INITIAL_TRAP_ANGULAR_FREQUENCY_Y,
        2 * pi * 600,
    )
    assert isclose(
        ExperimentalConstants.INITIAL_TRAP_ANGULAR_FREQUENCY_Z,
        2 * pi * 700,
    )


def test_nominal_final_trap_frequencies_keep_historical_values():
    assert isclose(
        ExperimentalConstants.FINAL_TRAP_ANGULAR_FREQUENCY_X,
        2 * pi * 5,
    )
    assert isclose(
        ExperimentalConstants.FINAL_TRAP_ANGULAR_FREQUENCY_Y,
        2 * pi * 5,
    )
    assert isclose(
        ExperimentalConstants.FINAL_TRAP_ANGULAR_FREQUENCY_Z,
        2 * pi * 60,
    )


def test_nominal_trap_frequencies_are_angular_frequencies():
    assert 2 * pi * 400 < ExperimentalConstants.INITIAL_TRAP_ANGULAR_FREQUENCY_X
    assert ExperimentalConstants.INITIAL_TRAP_ANGULAR_FREQUENCY_X < 2 * pi * 600
    assert 2 * pi * 500 < ExperimentalConstants.INITIAL_TRAP_ANGULAR_FREQUENCY_Y
    assert ExperimentalConstants.INITIAL_TRAP_ANGULAR_FREQUENCY_Y < 2 * pi * 700
    assert 2 * pi * 600 < ExperimentalConstants.INITIAL_TRAP_ANGULAR_FREQUENCY_Z
    assert ExperimentalConstants.INITIAL_TRAP_ANGULAR_FREQUENCY_Z < 2 * pi * 800
    assert 2 * pi * 4 < ExperimentalConstants.FINAL_TRAP_ANGULAR_FREQUENCY_X
    assert ExperimentalConstants.FINAL_TRAP_ANGULAR_FREQUENCY_X < 2 * pi * 6
    assert 2 * pi * 4 < ExperimentalConstants.FINAL_TRAP_ANGULAR_FREQUENCY_Y
    assert ExperimentalConstants.FINAL_TRAP_ANGULAR_FREQUENCY_Y < 2 * pi * 6
    assert 2 * pi * 50 < ExperimentalConstants.FINAL_TRAP_ANGULAR_FREQUENCY_Z
    assert ExperimentalConstants.FINAL_TRAP_ANGULAR_FREQUENCY_Z < 2 * pi * 70


def test_import_from_constants_package():
    from bec_fast_manipulation.constants import (
        ExperimentalConstants as ImportedConstants,
    )

    assert ImportedConstants is ExperimentalConstants


def test_derived_experiment_scenario_and_numeric_values_are_not_included():
    excluded_names = (
        "LASER_ANGULAR_FREQUENCY",
        "U0",
        "ATOM_NUMBER",
        "LASER_POWER",
        "MODULATION_PERIOD",
        "MODULATION_POINTS",
        "MODULATION_AMPLITUDE",
        "CROSSED_BEAM_ANGLE",
        "OMEGA_Z_MIN",
        "NPOINTS",
        "N_SIMULATION",
        "POWER_NOISE_PERCENT",
        "AMPLITUDE_NOISE_PERCENT",
    )

    for name in excluded_names:
        assert not hasattr(ExperimentalConstants, name)
