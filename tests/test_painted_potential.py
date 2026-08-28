from math import pi, sqrt

import numpy as np
import pytest
from scipy.integrate import quad

from bec_fast_manipulation.constants import ExperimentalConstants, PhysicalConstants
from bec_fast_manipulation.painted_potential import (
    CrossedPaintedDipolePotential,
    PaintedPotentialControl,
    harmonic_painting_modulation,
)


def make_potential(order=64):
    return CrossedPaintedDipolePotential(quadrature_order=order)


def test_harmonic_painting_modulation_properties():
    phase = np.linspace(0.0, 1.0, 1001)
    values = harmonic_painting_modulation(phase)

    assert np.all(values <= 1.0 + 1e-14)
    assert np.all(values >= -1.0 - 1e-14)
    assert harmonic_painting_modulation(0.0) == pytest.approx(-1.0)
    assert harmonic_painting_modulation(0.25) == pytest.approx(0.0, abs=1e-14)
    assert harmonic_painting_modulation(0.5) == pytest.approx(1.0)
    assert harmonic_painting_modulation(0.75) == pytest.approx(0.0, abs=1e-14)
    assert harmonic_painting_modulation(1.0) == pytest.approx(-1.0)
    np.testing.assert_allclose(harmonic_painting_modulation(phase + 2.0), values)
    np.testing.assert_allclose(
        harmonic_painting_modulation(phase[:500]),
        -harmonic_painting_modulation((phase[:500] + 0.5) % 1.0),
        atol=1e-13,
    )
    assert np.all(np.diff(harmonic_painting_modulation(np.linspace(0.0, 0.5, 501))) >= -1e-12)
    assert np.all(np.diff(harmonic_painting_modulation(np.linspace(0.5, 1.0, 501))) <= 1e-12)


def test_intensity_is_positive_linear_in_power_and_symmetric():
    potential = make_potential()
    position = np.array([8e-6, -5e-6, 3e-6])
    intensity = potential.averaged_intensity(position, 1.0, 20e-6)

    assert intensity >= 0.0
    assert potential.averaged_intensity(position, 2.0, 20e-6) == pytest.approx(2.0 * intensity)
    assert potential.averaged_intensity([position[1], position[0], position[2]], 1.0, 20e-6) == pytest.approx(
        intensity
    )


def test_public_beam_intensities_sum_to_total_intensity():
    potential = make_potential()
    position = np.array([8e-6, -5e-6, 3e-6])
    beam1, beam2 = potential.averaged_beam_intensities(position, 1.0, 20e-6)

    assert beam1 > 0.0
    assert beam2 > 0.0
    assert potential.averaged_beam1_intensity(position, 1.0, 20e-6) == pytest.approx(beam1)
    assert potential.averaged_beam2_intensity(position, 1.0, 20e-6) == pytest.approx(beam2)
    assert beam1 + beam2 == pytest.approx(potential.averaged_intensity(position, 1.0, 20e-6))


def test_zero_amplitude_recovers_unpainted_crossed_gaussian_center():
    potential = make_potential()
    expected = 4.0 * 1.2 / (
        pi
        * ExperimentalConstants.TRAP_BEAM_HORIZONTAL_WAIST_M
        * ExperimentalConstants.TRAP_BEAM_VERTICAL_WAIST_M
    )

    assert potential.averaged_intensity([0.0, 0.0, 0.0], 1.2, 0.0) == pytest.approx(expected)


def test_quadrature_converges_and_matches_quad():
    position = np.array([7e-6, -3e-6, 2e-6])
    coarse = CrossedPaintedDipolePotential(quadrature_order=32).averaged_intensity(position, 0.8, 25e-6)
    fine_potential = CrossedPaintedDipolePotential(quadrature_order=128)
    fine = fine_potential.averaged_intensity(position, 0.8, 25e-6)

    def integrand(phase):
        return fine_potential.instantaneous_intensity(position, phase, 0.8, 25e-6)

    reference, _ = quad(integrand, 0.0, 1.0, epsabs=0.0, epsrel=1e-10, limit=100)

    assert abs(fine - reference) / reference < 2e-4
    assert abs(fine - reference) < abs(coarse - reference)


def _dirac_profile(x, power_w, amplitude_m):
    return 3.0 * power_w / (4.0 * amplitude_m) * (1.0 - (x / amplitude_m) ** 2)


def _central_gradient(potential, position, power_w, amplitude_m, include_gravity=True, step=1e-8):
    position = np.asarray(position, dtype=float)
    gradient = np.zeros(3, dtype=float)
    basis = np.eye(3)
    for index in range(3):
        offset = step * basis[index]
        gradient[index] = (
            potential.total_potential(position + offset, power_w, amplitude_m, include_gravity=include_gravity)
            - potential.total_potential(position - offset, power_w, amplitude_m, include_gravity=include_gravity)
        ) / (2.0 * step)
    return gradient


def test_exact_painted_profile_converges_toward_dirac_limit():
    waist = 10e-6
    ratios = [4.0, 8.0, 16.0]
    errors = []
    for ratio in ratios:
        amplitude = ratio * waist
        potential = CrossedPaintedDipolePotential(
            horizontal_waist_m=waist,
            vertical_waist_m=50e-6,
            quadrature_order=256,
        )
        x = np.linspace(-0.45 * amplitude, 0.45 * amplitude, 41)
        exact = potential.averaged_intensity(np.vstack([x, np.zeros_like(x), np.zeros_like(x)]), 1.0, amplitude)
        beam1 = np.sum(
            potential._phase_weights.reshape((-1, 1))
            * potential._beam1_intensity(
                x,
                np.zeros_like(x),
                np.zeros_like(x),
                1.0,
                amplitude,
                potential._modulation_values,
            ),
            axis=0,
        )
        one_beam = exact - beam1
        reference = _dirac_profile(x, 1.0, amplitude) * np.sqrt(2.0 / pi) / potential.vertical_waist_m
        errors.append(float(np.sqrt(np.mean(((one_beam - reference) / np.max(reference)) ** 2))))

    assert errors[2] < errors[1] < errors[0]


def test_dipole_coefficient_is_negative_and_rejects_invalid_wavelength():
    potential = make_potential()
    coefficient = potential.dipole_potential_coefficient()

    assert coefficient < 0.0
    assert np.isfinite(coefficient)
    assert potential.optical_potential([0.0, 0.0, 0.0], 1.0, 0.0) < 0.0
    with pytest.raises(ValueError):
        CrossedPaintedDipolePotential(laser_wavelength_m=0.0)


def test_gravity_shifts_minimum_downward_and_stationary_point():
    potential = make_potential()
    no_gravity = potential.find_local_minimum(1.0, 20e-6, include_gravity=False)
    with_gravity = potential.find_local_minimum(1.0, 20e-6, include_gravity=True)
    gradient = _central_gradient(potential, with_gravity.position_m, 1.0, 20e-6)

    assert no_gravity.success
    assert with_gravity.success
    np.testing.assert_allclose(with_gravity.position_m[:2], [0.0, 0.0], atol=0.0)
    assert abs(no_gravity.position_m[2]) < 1e-7
    assert with_gravity.position_m[2] < 0.0
    assert np.linalg.norm(gradient) < 5e-26


def test_find_minimum_alias_keeps_local_minimum_semantics():
    potential = make_potential()

    local_minimum = potential.find_local_minimum(1.0, 20e-6, include_gravity=True)
    alias_minimum = potential.find_minimum(1.0, 20e-6, include_gravity=True)

    np.testing.assert_allclose(alias_minimum.position_m, local_minimum.position_m)
    assert "local" in alias_minimum.message.lower()


def test_hessian_is_symmetric_and_zero_amplitude_matches_gaussian_frequency_scale():
    potential = make_potential()
    minimum = potential.find_minimum(1.0, 0.0, include_gravity=False)
    hessian = potential.hessian_at_minimum(1.0, 0.0, include_gravity=False, minimum=minimum)
    modes = potential.trap_modes(1.0, 0.0, include_gravity=False)
    coefficient = potential.dipole_potential_coefficient()
    intensity_per_beam = 2.0 / (pi * potential.horizontal_waist_m * potential.vertical_waist_m)
    expected_transverse_omega = sqrt(
        (-4.0 * coefficient * intensity_per_beam / potential.horizontal_waist_m**2)
        / PhysicalConstants.RUBIDIUM_87_MASS
    )

    np.testing.assert_allclose(hessian, hessian.T, atol=1e-30)
    assert np.all(np.linalg.eigvalsh(hessian) > 0.0)
    assert modes.horizontal_frequency_hz == pytest.approx(expected_transverse_omega / (2.0 * pi), rel=5e-3)


def test_trap_modes_identify_vertical_mode():
    potential = make_potential()
    modes = potential.trap_modes(1.0, 20e-6, include_gravity=True)
    vertical_index = int(np.argmax(np.abs(modes.eigenvectors[2, :])))

    assert modes.vertical_frequency_hz == pytest.approx(modes.frequencies_hz[vertical_index])
    assert modes.horizontal_frequency_hz < modes.vertical_frequency_hz


def test_inverse_controls_reproduce_target_frequencies():
    potential = CrossedPaintedDipolePotential(quadrature_order=32)
    control = PaintedPotentialControl(potential)
    reference = control.frequencies_from_controls(0.8, 18e-6, include_gravity=False)
    recovered = control.controls_from_frequencies(
        reference["horizontal_frequency_hz"],
        reference["vertical_frequency_hz"],
        initial_guess=(1.0, 22e-6),
        power_bounds=(0.2, 2.0),
        amplitude_bounds=(0.0, 50e-6),
        include_gravity=False,
    )

    assert recovered.success
    assert abs(recovered.residuals[0]) < 1e-5
    assert abs(recovered.residuals[1]) < 1e-5
    assert recovered.horizontal_frequency_hz == pytest.approx(reference["horizontal_frequency_hz"], rel=1e-5)
    assert recovered.vertical_frequency_hz == pytest.approx(reference["vertical_frequency_hz"], rel=1e-5)


@pytest.mark.parametrize(
    "call",
    [
        lambda p: p.averaged_intensity([0.0, 0.0, 0.0], 0.0, 1e-6),
        lambda p: p.averaged_intensity([0.0, 0.0, 0.0], 1.0, -1e-6),
        lambda p: p.averaged_intensity([0.0, np.nan, 0.0], 1.0, 1e-6),
        lambda p: CrossedPaintedDipolePotential(quadrature_order=7),
    ],
)
def test_model_validation_errors(call):
    with pytest.raises(ValueError):
        call(make_potential())


@pytest.mark.parametrize(
    "kwargs",
    [
        {"horizontal_frequency_hz": 0.0},
        {"vertical_frequency_hz": np.inf},
        {"power_bounds": (0.0, 1.0)},
        {"power_bounds": (1.0, 1.0)},
        {"amplitude_bounds": (-1.0, 1.0)},
        {"amplitude_bounds": (1.0, 0.0)},
    ],
)
def test_control_validation_errors(kwargs):
    control = PaintedPotentialControl(make_potential(order=32))
    defaults = {
        "horizontal_frequency_hz": 100.0,
        "vertical_frequency_hz": 200.0,
        "initial_guess": (1.0, 20e-6),
        "power_bounds": (0.1, 2.0),
        "amplitude_bounds": (0.0, 50e-6),
    }
    defaults.update(kwargs)

    with pytest.raises(ValueError):
        control.controls_from_frequencies(**defaults)


def test_unstable_configuration_raises_without_clipping():
    potential = make_potential(order=32)

    with pytest.raises(ValueError, match="not stable"):
        potential.trap_modes(0.05, 20e-6, include_gravity=True)
