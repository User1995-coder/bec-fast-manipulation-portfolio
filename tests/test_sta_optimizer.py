from math import pi

import numpy as np
import pytest

from bec_fast_manipulation.sta_optimizer import (
    STACandidateEvaluation,
    STAOptimizationResult,
    STAOptimizer,
)


def make_optimizer(**overrides):
    defaults = {
        "final_time_bounds_s": (0.0065, 0.0068),
        "a_bounds": (-1.4, -1.3),
        "b_bounds": (0.85, 0.95),
        "minimum_search_samples": 101,
        "maxiter": 2,
        "popsize": 4,
        "polish": False,
        "seed": 123,
    }
    defaults.update(overrides)
    return STAOptimizer(**defaults)


def test_public_exports():
    assert STACandidateEvaluation.__name__ == "STACandidateEvaluation"
    assert STAOptimizationResult.__name__ == "STAOptimizationResult"
    assert STAOptimizer.__name__ == "STAOptimizer"


def test_evaluate_historical_candidate_is_feasible():
    optimizer = make_optimizer()

    evaluation = optimizer.evaluate_candidate(0.006611, -1.371429, 0.910204)

    assert evaluation.phase_monotonic is True
    assert evaluation.trapping_constraints_satisfied is True
    assert evaluation.z_constraint_satisfied is True
    assert evaluation.feasible is True
    assert evaluation.minimum_omega_squared_x == pytest.approx(417.76, rel=2e-2)
    assert evaluation.minimum_omega_squared_y == pytest.approx(776.75, rel=2e-2)
    assert evaluation.minimum_omega_squared_z == pytest.approx(142122.30, rel=2e-2)
    assert evaluation.minimum_frequency_z_hz >= 50.0


def test_evaluate_anti_confining_candidate_keeps_negative_minima_visible():
    optimizer = make_optimizer()

    evaluation = optimizer.evaluate_candidate(0.006611, -1.6, 0.8)

    assert evaluation.feasible is False
    assert evaluation.minimum_omega_squared_x < 0.0
    assert evaluation.minimum_omega_squared_y < 0.0
    assert evaluation.minimum_frequency_x_hz < 0.0
    assert evaluation.minimum_frequency_y_hz < 0.0


def test_z_threshold_is_independent_from_positive_trapping():
    optimizer = make_optimizer(
        final_time_bounds_s=(0.1, 0.3),
        a_bounds=(-0.6, -0.4),
        b_bounds=(-0.3, -0.2),
        require_monotonic_phase=False,
    )

    evaluation = optimizer.evaluate_candidate(0.2, -0.5, -0.25)

    assert evaluation.minimum_omega_squared_x >= 0.0
    assert evaluation.minimum_omega_squared_y >= 0.0
    assert evaluation.minimum_omega_squared_z >= 0.0
    assert evaluation.trapping_constraints_satisfied is True
    assert evaluation.minimum_frequency_z_hz < 50.0
    assert evaluation.z_constraint_satisfied is False
    assert evaluation.feasible is False


def test_phase_monotonicity_option_is_independent_from_trapping_constraints():
    strict_optimizer = make_optimizer(
        final_time_bounds_s=(0.1, 0.3),
        a_bounds=(-0.6, -0.4),
        b_bounds=(-0.3, -0.2),
        minimum_z_frequency_hz=10.0,
        require_monotonic_phase=True,
    )
    relaxed_optimizer = make_optimizer(
        final_time_bounds_s=(0.1, 0.3),
        a_bounds=(-0.6, -0.4),
        b_bounds=(-0.3, -0.2),
        minimum_z_frequency_hz=10.0,
        require_monotonic_phase=False,
    )

    strict = strict_optimizer.evaluate_candidate(0.2, -0.5, -0.25)
    relaxed = relaxed_optimizer.evaluate_candidate(0.2, -0.5, -0.25)

    assert strict.phase_monotonic is False
    assert strict.trapping_constraints_satisfied is True
    assert strict.z_constraint_satisfied is True
    assert strict.feasible is False
    assert relaxed.phase_monotonic is False
    assert relaxed.trapping_constraints_satisfied is True
    assert relaxed.z_constraint_satisfied is True
    assert relaxed.feasible is True


def test_continuous_minimum_refinement_matches_dense_sampling():
    optimizer = make_optimizer(minimum_search_samples=81)
    protocol_evaluation = optimizer.evaluate_candidate(0.006611, -1.371429, 0.910204)
    protocol = optimizer._build_protocol(0.006611, -1.371429, 0.910204)
    assert protocol is not None
    dense_u = np.linspace(0.0, 1.0, 5000)
    dense_minimum = min(
        float(protocol.angular_frequency_squared(dense_u * protocol.final_time)[axis].min())
        for axis in range(3)
    )
    refined_minimum = min(
        protocol_evaluation.minimum_omega_squared_x,
        protocol_evaluation.minimum_omega_squared_y,
        protocol_evaluation.minimum_omega_squared_z,
    )

    assert refined_minimum <= dense_minimum * (1.0 + 1e-8)
    assert refined_minimum == pytest.approx(dense_minimum, rel=2e-3)


def test_optimize_lightweight_reproducible_case():
    optimizer = make_optimizer(
        final_time_bounds_s=(0.00658, 0.00664),
        a_bounds=(-1.38, -1.36),
        b_bounds=(0.90, 0.92),
        maxiter=3,
        popsize=5,
        polish=False,
        seed=321,
    )

    result = optimizer.optimize()
    evaluation = optimizer.evaluate_candidate(result.final_time, result.a, result.b)

    assert result.final_time >= 0.00658
    assert result.final_time <= 0.00664
    assert -1.38 <= result.a <= -1.36
    assert 0.90 <= result.b <= 0.92
    assert result.objective_value == pytest.approx(result.final_time)
    assert result.number_of_function_evaluations is not None
    assert evaluation.feasible is True
    assert result.success in {True, False}


def test_objective_is_final_time_only():
    optimizer = make_optimizer()

    assert optimizer._objective([0.0123, -99.0, 88.0]) == pytest.approx(0.0123)
    assert optimizer._objective([0.0123, 99.0, -88.0]) == pytest.approx(0.0123)


def test_protocol_kwargs_are_forwarded_to_retro_sinusoidal_protocol():
    optimizer = make_optimizer(
        protocol_kwargs={
            "lambda_x_final": 1.2,
            "lambda_y_final": 1.2,
            "lambda_z_final": 0.9,
            "omega_x_initial": 2.0 * pi * 300.0,
            "omega_y_initial": 2.0 * pi * 300.0,
            "omega_z_initial": 2.0 * pi * 500.0,
        }
    )

    protocol = optimizer._build_protocol(0.01, 0.0, 0.0)

    assert protocol.lambda_x_final == pytest.approx(1.2)
    assert protocol.lambda_y_final == pytest.approx(1.2)
    assert protocol.lambda_z_final == pytest.approx(0.9)
    assert protocol.omega_x_initial == pytest.approx(2.0 * pi * 300.0)


def test_frequency_decompression_constraint_accepts_monotone_protocol():
    optimizer = make_optimizer(
        final_time_bounds_s=(0.1, 0.3),
        a_bounds=(-1.0, 1.0),
        b_bounds=(-1.0, 1.0),
        minimum_z_frequency_hz=10.0,
        require_monotonic_phase=True,
        require_frequency_decompression=True,
    )

    evaluation = optimizer.evaluate_candidate(0.2, 0.0, 0.0)

    assert evaluation.frequency_decompression_satisfied is True
    assert evaluation.maximum_omega_squared_increase_horizontal <= 0.0
    assert evaluation.maximum_omega_squared_increase_vertical <= 0.0
    assert evaluation.feasible is True


def test_frequency_decompression_constraint_rejects_temporary_recompression():
    relaxed = make_optimizer(
        final_time_bounds_s=(0.1, 0.3),
        a_bounds=(-1.0, 1.0),
        b_bounds=(-1.0, 1.0),
        minimum_z_frequency_hz=10.0,
        require_monotonic_phase=False,
        require_frequency_decompression=False,
    )
    strict = make_optimizer(
        final_time_bounds_s=(0.1, 0.3),
        a_bounds=(-1.0, 1.0),
        b_bounds=(-1.0, 1.0),
        minimum_z_frequency_hz=10.0,
        require_monotonic_phase=False,
        require_frequency_decompression=True,
    )

    relaxed_evaluation = relaxed.evaluate_candidate(0.2, -0.5, -0.25)
    strict_evaluation = strict.evaluate_candidate(0.2, -0.5, -0.25)

    assert relaxed_evaluation.maximum_omega_squared_increase_horizontal > 0.0
    assert relaxed_evaluation.maximum_omega_squared_increase_vertical > 0.0
    assert strict_evaluation.frequency_decompression_satisfied is False
    assert strict_evaluation.feasible is False


def test_fixed_time_optimization_uses_same_constraints():
    optimizer = make_optimizer(
        final_time_bounds_s=(0.1, 0.3),
        a_bounds=(-0.2, 0.2),
        b_bounds=(-0.2, 0.2),
        minimum_z_frequency_hz=10.0,
        require_monotonic_phase=True,
        require_frequency_decompression=True,
        maxiter=2,
        popsize=4,
        polish=False,
    )

    result = optimizer.optimize_fixed_final_time(0.2)
    evaluation = optimizer.evaluate_candidate(result.final_time, result.a, result.b)

    assert result.final_time == pytest.approx(0.2)
    assert result.number_of_function_evaluations is not None
    assert evaluation.frequency_decompression_satisfied is True
    assert result.success in {True, False}


@pytest.mark.parametrize(
    "kwargs",
    [
        {"final_time_bounds_s": (0.0, 1.0)},
        {"final_time_bounds_s": (1.0, 1.0)},
        {"final_time_bounds_s": (np.nan, 1.0)},
        {"final_time_bounds_s": (1.0, np.inf)},
        {"a_bounds": (1.0, 1.0)},
        {"b_bounds": (2.0, 1.0)},
        {"minimum_z_frequency_hz": 0.0},
        {"minimum_z_frequency_hz": np.nan},
        {"minimum_search_samples": 4},
        {"minimum_search_xatol": 0.0},
        {"maxiter": 0},
        {"popsize": 0},
        {"tolerance": 0.0},
        {"seed": -1},
    ],
)
def test_validation_errors(kwargs):
    defaults = {
        "final_time_bounds_s": (0.1, 0.2),
        "a_bounds": (-1.0, 1.0),
        "b_bounds": (-1.0, 1.0),
    }
    defaults.update(kwargs)

    with pytest.raises(ValueError):
        STAOptimizer(**defaults)


def test_denominator_zero_candidate_is_invalid_not_hidden():
    optimizer = make_optimizer(a_bounds=(-2.0, 2.0), b_bounds=(-2.0, 2.0))

    evaluation = optimizer.evaluate_candidate(0.006611, -1.0, 0.0)

    assert evaluation.feasible is False
    assert "1 + a + b" in evaluation.message
