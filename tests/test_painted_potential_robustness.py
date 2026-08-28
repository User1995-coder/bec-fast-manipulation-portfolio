import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from bec_fast_manipulation.retro_sinusoidal import RetroSinusoidalProtocol


def load_robustness_module():
    path = Path(__file__).resolve().parents[1] / "experiments" / "painted_potential_robustness" / "main.py"
    spec = importlib.util.spec_from_file_location("painted_potential_robustness_main", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def minimal_sta_data():
    return {
        "endpoint_exact_frequencies_hz": {
            "initial": {"horizontal": 200.0184967252074, "vertical": 1183.129415949939},
            "final": {"horizontal": 9.846249838734085, "vertical": 189.84072240879672},
        },
        "characteristic_time_s": 0.0027167599723887516,
        "admissible_protocols": [
            {"factor_Tc": 5.0, "tf_s": 0.013583799861943758, "a": -0.8333333333333333, "b": 0.4444444444444444},
            {"factor_Tc": 6.0, "tf_s": 0.01630055983433251, "a": -0.6666666666666666, "b": 0.3333333333333333},
        ],
    }


def nominal_protocol_and_lambdas(module, n_time_points):
    data = minimal_sta_data()
    selected = module.select_protocol(data, 6)
    frequencies = module.endpoint_angular_frequencies(data)
    model = module.make_castin_dum_from_endpoints(frequencies)
    final_scaling = model.final_scaling_factors()
    nominal = module.reconstruct_nominal_protocol(data, selected, n_time_points=n_time_points)
    protocol = RetroSinusoidalProtocol(
        final_time=float(selected["tf_s"]),
        a=float(selected["a"]),
        b=float(selected["b"]),
        lambda_x_final=final_scaling[0],
        lambda_y_final=final_scaling[1],
        lambda_z_final=final_scaling[2],
        omega_x_initial=frequencies["omega_H_i"],
        omega_y_initial=frequencies["omega_H_i"],
        omega_z_initial=frequencies["omega_V_i"],
        require_monotonic_phase=True,
    )
    analytical_lambdas = protocol.scaling_factors(nominal["time_s"])
    return model, nominal, analytical_lambdas


def rk4_lambda_errors(module, n_time_points):
    model, nominal, analytical_lambdas = nominal_protocol_and_lambdas(module, n_time_points)
    states = module.integrate_castin_dum_fixed_rk4(
        model,
        nominal["time_s"],
        nominal["omega_H_rad_s"],
        nominal["omega_V_rad_s"],
    )
    errors = {
        "x": float(np.max(np.abs(states[0] - analytical_lambdas[0]))),
        "y": float(np.max(np.abs(states[2] - analytical_lambdas[1]))),
        "z": float(np.max(np.abs(states[4] - analytical_lambdas[2]))),
        "final_H": float(abs(0.5 * (states[0, -1] + states[2, -1]) - nominal["lambda_H"][-1])),
        "final_V": float(abs(states[4, -1] - nominal["lambda_V"][-1])),
        "xy_symmetry": float(np.max(np.abs(states[0] - states[2]))),
    }
    return errors, states, nominal, analytical_lambdas


def test_load_sta_results_reads_json(tmp_path):
    module = load_robustness_module()
    path = tmp_path / "sta.json"
    data = minimal_sta_data()
    path.write_text(json.dumps(data), encoding="utf-8")

    loaded = module.load_sta_results(path)

    assert loaded["characteristic_time_s"] == data["characteristic_time_s"]


def test_available_duration_factors_are_extracted_from_json():
    module = load_robustness_module()

    assert module.available_duration_factors(minimal_sta_data()) == [5.0, 6.0]


def test_select_protocol_accepts_available_factor():
    module = load_robustness_module()

    selected = module.select_protocol(minimal_sta_data(), 6)

    assert selected["factor_Tc"] == 6.0
    assert selected["a"] == pytest.approx(-2 / 3)


def test_select_protocol_rejects_missing_factor():
    module = load_robustness_module()

    with pytest.raises(ValueError, match="Available factors"):
        module.select_protocol(minimal_sta_data(), 7)


def test_reconstruct_nominal_protocol_has_expected_shapes():
    module = load_robustness_module()
    data = minimal_sta_data()
    selected = module.select_protocol(data, 6)

    nominal = module.reconstruct_nominal_protocol(data, selected, n_time_points=11)

    for key in ("time_s", "omega_H_rad_s", "omega_V_rad_s", "lambda_H", "lambda_V", "power_w", "amplitude_m"):
        assert nominal[key].shape == (11,)
        assert np.all(np.isfinite(nominal[key]))


def test_nominal_controls_reproduce_nominal_omega_with_relative_formula():
    module = load_robustness_module()
    data = minimal_sta_data()
    selected = module.select_protocol(data, 6)
    nominal = module.reconstruct_nominal_protocol(data, selected, n_time_points=9)

    omega_H, omega_V = module.controls_to_omega_relative(
        nominal["power_w"].reshape(1, -1),
        nominal["amplitude_m"].reshape(1, -1),
        nominal,
    )

    np.testing.assert_allclose(omega_H[0], nominal["omega_H_rad_s"])
    np.testing.assert_allclose(omega_V[0], nominal["omega_V_rad_s"])


def test_build_scenarios_power_only_keeps_nominal_amplitude():
    module = load_robustness_module()
    nominal = {"power_w": np.array([1.0, 2.0]), "amplitude_m": np.array([3.0, 4.0])}
    power_noisy = np.array([[1.1, 2.2], [0.9, 1.8]])
    amplitude_noisy = np.array([[3.3, 4.4], [2.7, 3.6]])

    scenarios = module.build_scenarios(nominal, power_noisy, amplitude_noisy)

    np.testing.assert_allclose(scenarios["power_only"]["power_w"], power_noisy)
    np.testing.assert_allclose(scenarios["power_only"]["amplitude_m"], [[3.0, 4.0], [3.0, 4.0]])


def test_build_scenarios_amplitude_only_keeps_nominal_power():
    module = load_robustness_module()
    nominal = {"power_w": np.array([1.0, 2.0]), "amplitude_m": np.array([3.0, 4.0])}
    power_noisy = np.array([[1.1, 2.2], [0.9, 1.8]])
    amplitude_noisy = np.array([[3.3, 4.4], [2.7, 3.6]])

    scenarios = module.build_scenarios(nominal, power_noisy, amplitude_noisy)

    np.testing.assert_allclose(scenarios["amplitude_only"]["power_w"], [[1.0, 2.0], [1.0, 2.0]])
    np.testing.assert_allclose(scenarios["amplitude_only"]["amplitude_m"], amplitude_noisy)


def test_build_scenarios_combined_uses_both_noisy_controls():
    module = load_robustness_module()
    nominal = {"power_w": np.array([1.0, 2.0]), "amplitude_m": np.array([3.0, 4.0])}
    power_noisy = np.array([[1.1, 2.2], [0.9, 1.8]])
    amplitude_noisy = np.array([[3.3, 4.4], [2.7, 3.6]])

    scenarios = module.build_scenarios(nominal, power_noisy, amplitude_noisy)

    np.testing.assert_allclose(scenarios["combined"]["power_w"], power_noisy)
    np.testing.assert_allclose(scenarios["combined"]["amplitude_m"], amplitude_noisy)


def test_deterministic_control_bounds_contain_nominal_controls():
    module = load_robustness_module()
    data = minimal_sta_data()
    selected = module.select_protocol(data, 6)
    nominal = module.reconstruct_nominal_protocol(data, selected, n_time_points=17)

    bounds = module.deterministic_control_bounds(nominal)

    assert np.all(bounds["power_w"]["lower"] <= nominal["power_w"])
    assert np.all(nominal["power_w"] <= bounds["power_w"]["upper"])
    assert np.all(bounds["amplitude_m"]["lower"] <= nominal["amplitude_m"])
    assert np.all(nominal["amplitude_m"] <= bounds["amplitude_m"]["upper"])
    np.testing.assert_allclose(bounds["power_w"]["lower"], nominal["power_w"] * 0.95)
    np.testing.assert_allclose(bounds["amplitude_m"]["upper"], nominal["amplitude_m"] * 1.05)


def test_deterministic_frequency_bound_factors_are_correct_for_all_scenarios():
    module = load_robustness_module()
    eps_p = module.POWER_RELATIVE_NOISE
    eps_h = module.AMPLITUDE_RELATIVE_NOISE

    factors = module.deterministic_frequency_bound_factors()

    assert factors["power_only"]["omega_H"] == pytest.approx((np.sqrt(1 - eps_p), np.sqrt(1 + eps_p)))
    assert factors["power_only"]["omega_V"] == pytest.approx((np.sqrt(1 - eps_p), np.sqrt(1 + eps_p)))
    assert factors["amplitude_only"]["omega_H"] == pytest.approx(((1 + eps_h) ** -1.5, (1 - eps_h) ** -1.5))
    assert factors["amplitude_only"]["omega_V"] == pytest.approx(((1 + eps_h) ** -0.5, (1 - eps_h) ** -0.5))
    assert factors["combined"]["omega_H"] == pytest.approx(
        (np.sqrt((1 - eps_p) / (1 + eps_h) ** 3), np.sqrt((1 + eps_p) / (1 - eps_h) ** 3))
    )
    assert factors["combined"]["omega_V"] == pytest.approx(
        (np.sqrt((1 - eps_p) / (1 + eps_h)), np.sqrt((1 + eps_p) / (1 - eps_h)))
    )


def test_deterministic_frequency_bounds_contain_nominal_and_match_relative_mapping():
    module = load_robustness_module()
    data = minimal_sta_data()
    selected = module.select_protocol(data, 6)
    nominal = module.reconstruct_nominal_protocol(data, selected, n_time_points=13)

    bounds = module.deterministic_frequency_bounds(nominal)
    control_bounds = module.deterministic_control_bounds(nominal)

    for scenario in module.SCENARIOS:
        for omega_name, nominal_key in [("omega_H", "omega_H_rad_s"), ("omega_V", "omega_V_rad_s")]:
            assert np.all(bounds[scenario][omega_name]["lower"] <= nominal[nominal_key])
            assert np.all(nominal[nominal_key] <= bounds[scenario][omega_name]["upper"])

    power_lower_omega_H, power_lower_omega_V = module.controls_to_omega_relative(
        control_bounds["power_w"]["lower"].reshape(1, -1),
        nominal["amplitude_m"].reshape(1, -1),
        nominal,
    )
    np.testing.assert_allclose(power_lower_omega_H[0], bounds["power_only"]["omega_H"]["lower"])
    np.testing.assert_allclose(power_lower_omega_V[0], bounds["power_only"]["omega_V"]["lower"])

    combined_upper_omega_H, combined_upper_omega_V = module.controls_to_omega_relative(
        control_bounds["power_w"]["upper"].reshape(1, -1),
        control_bounds["amplitude_m"]["lower"].reshape(1, -1),
        nominal,
    )
    np.testing.assert_allclose(combined_upper_omega_H[0], bounds["combined"]["omega_H"]["upper"])
    np.testing.assert_allclose(combined_upper_omega_V[0], bounds["combined"]["omega_V"]["upper"])


def test_current_500_simulation_control_ensembles_have_expected_shapes():
    module = load_robustness_module()
    data = minimal_sta_data()
    selected = module.select_protocol(data, 6)
    nominal = module.reconstruct_nominal_protocol(data, selected, n_time_points=23)

    power_noisy, amplitude_noisy = module.generate_noisy_controls(nominal)
    scenarios = module.build_scenarios(nominal, power_noisy, amplitude_noisy)

    assert power_noisy.shape == (module.N_SIMULATIONS, 23)
    assert amplitude_noisy.shape == (module.N_SIMULATIONS, 23)
    assert module.N_SIMULATIONS == 500
    for scenario in scenarios.values():
        assert scenario["power_w"].shape == (500, 23)
        assert scenario["amplitude_m"].shape == (500, 23)


def test_final_statistics_are_structured():
    module = load_robustness_module()
    samples = np.array([[1.0, 2.0], [1.0, 4.0], [1.0, 6.0]])

    stats = module.final_statistics(samples, nominal_final=4.0)

    assert set(stats) == {"mean", "std", "min", "max", "median", "q05", "q95", "nominal", "bias", "relative_bias"}
    assert stats["mean"] == pytest.approx(4.0)
    assert stats["bias"] == pytest.approx(0.0)


def test_final_extreme_realizations_select_argmin_and_argmax():
    module = load_robustness_module()
    samples = np.array(
        [
            [1.0, 3.0],
            [1.0, 2.0],
            [1.0, 5.0],
        ]
    )
    valid_indices = np.array([10, 11, 12])

    extremes = module.final_extreme_realizations(samples, valid_indices)

    assert extremes["lowest_local_index"] == 1
    assert extremes["highest_local_index"] == 2
    assert extremes["lowest_final_realization_index"] == 11
    assert extremes["highest_final_realization_index"] == 12
    np.testing.assert_allclose(extremes["lowest_trajectory"], samples[1])
    np.testing.assert_allclose(extremes["highest_trajectory"], samples[2])


def test_final_extreme_realizations_are_stable_for_ties():
    module = load_robustness_module()
    samples = np.array(
        [
            [1.0, 2.0],
            [1.0, 2.0],
            [1.0, 4.0],
            [1.0, 4.0],
        ]
    )
    valid_indices = np.array([20, 21, 22, 23])

    extremes = module.final_extreme_realizations(samples, valid_indices)

    assert extremes["lowest_final_realization_index"] == 20
    assert extremes["highest_final_realization_index"] == 22


def test_small_castin_dum_smoke_produces_valid_samples():
    module = load_robustness_module()
    data = minimal_sta_data()
    selected = module.select_protocol(data, 6)
    nominal = module.reconstruct_nominal_protocol(data, selected, n_time_points=101)
    omega_H = np.tile(nominal["omega_H_rad_s"], (2, 1))
    omega_V = np.tile(nominal["omega_V_rad_s"], (2, 1))

    result = module.propagate_castin_dum_ensemble(nominal, omega_H, omega_V)

    assert result["lambda_H"].shape == (2, 101)
    assert result["lambda_V"].shape == (2, 101)
    assert result["rejection_reasons"] == {}


def test_rk4_nominal_trajectory_matches_retro_sinusoidal_lambdas():
    module = load_robustness_module()

    errors, states, _, analytical_lambdas = rk4_lambda_errors(module, 1001)

    assert errors["x"] < 3.0e-5
    assert errors["y"] < 3.0e-5
    assert errors["z"] < 5.0e-6
    assert errors["final_H"] < 1.5e-5
    assert errors["final_V"] < 3.0e-6
    assert errors["xy_symmetry"] == pytest.approx(0.0, abs=1e-14)
    np.testing.assert_allclose(states[0], states[2], atol=1e-14, rtol=0.0)
    np.testing.assert_allclose(states[0], analytical_lambdas[0], atol=3.0e-5, rtol=0.0)
    np.testing.assert_allclose(states[2], analytical_lambdas[1], atol=3.0e-5, rtol=0.0)
    np.testing.assert_allclose(states[4], analytical_lambdas[2], atol=5.0e-6, rtol=0.0)


def test_rk4_nominal_trajectory_temporal_convergence():
    module = load_robustness_module()

    errors_1001, _, _, _ = rk4_lambda_errors(module, 1001)
    errors_2001, _, _, _ = rk4_lambda_errors(module, 2001)

    assert max(errors_2001["x"], errors_2001["y"]) < 0.35 * max(errors_1001["x"], errors_1001["y"])
    assert errors_2001["z"] < 0.35 * errors_1001["z"]
    assert errors_2001["final_H"] < 0.35 * errors_1001["final_H"]
    assert errors_2001["final_V"] < 0.35 * errors_1001["final_V"]


def test_zero_noise_controls_recover_nominal_frequency_and_lambda_trajectory():
    module = load_robustness_module()
    model, nominal, analytical_lambdas = nominal_protocol_and_lambdas(module, 1001)
    power_noisy = nominal["power_w"].reshape(1, -1)
    amplitude_noisy = nominal["amplitude_m"].reshape(1, -1)

    np.testing.assert_array_equal(power_noisy[0], nominal["power_w"])
    np.testing.assert_array_equal(amplitude_noisy[0], nominal["amplitude_m"])

    omega_H, omega_V = module.controls_to_omega_relative(power_noisy, amplitude_noisy, nominal)

    np.testing.assert_allclose(omega_H[0], nominal["omega_H_rad_s"], atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(omega_V[0], nominal["omega_V_rad_s"], atol=1e-12, rtol=1e-12)

    states = module.integrate_castin_dum_fixed_rk4(model, nominal["time_s"], omega_H[0], omega_V[0])
    np.testing.assert_allclose(states[0], analytical_lambdas[0], atol=3.0e-5, rtol=0.0)
    np.testing.assert_allclose(states[2], analytical_lambdas[1], atol=3.0e-5, rtol=0.0)
    np.testing.assert_allclose(states[4], analytical_lambdas[2], atol=5.0e-6, rtol=0.0)
