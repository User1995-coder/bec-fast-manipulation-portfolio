import numpy as np
import pytest

from bec_fast_manipulation.monte_carlo import MonteCarlo


def test_invalid_n_simulations_raises():
    with pytest.raises(ValueError, match="n_simulations"):
        MonteCarlo(0, 0.05)


def test_negative_relative_noise_raises():
    with pytest.raises(ValueError, match="relative_noise"):
        MonteCarlo(10, -0.01)


def test_unknown_distribution_raises():
    with pytest.raises(ValueError, match="distribution"):
        MonteCarlo(10, 0.05, distribution="laplace")


def test_generate_vector_shape():
    signal = np.array([1.0, 2.0, 3.0])
    ensemble = MonteCarlo(7, 0.05, seed=1).generate(signal)

    assert ensemble.shape == (7, 3)


def test_zero_signal_remains_zero_with_noise():
    signal = np.zeros(5)
    ensemble = MonteCarlo(8, 0.5, seed=2).generate(signal)

    np.testing.assert_array_equal(ensemble, np.zeros((8, 5)))


def test_zero_relative_noise_reproduces_signal():
    signal = np.array([1.0, -2.0, 3.5])
    ensemble = MonteCarlo(4, 0.0, seed=3).generate(signal)

    np.testing.assert_array_equal(ensemble, np.tile(signal, (4, 1)))


def test_uniform_noise_is_bounded():
    signal = np.ones(100)
    relative_noise = 0.05
    ensemble = MonteCarlo(20, relative_noise, distribution="uniform", seed=4).generate(signal)
    epsilon = ensemble - 1.0

    assert np.all(epsilon >= -relative_noise)
    assert np.all(epsilon <= relative_noise)


def test_gaussian_noise_is_not_artificially_bounded():
    signal = np.ones(200)
    relative_noise = 0.05
    ensemble = MonteCarlo(30, relative_noise, distribution="gaussian", seed=5).generate(signal)
    epsilon = ensemble - 1.0

    assert np.any(np.abs(epsilon) > relative_noise)


def test_same_seed_reproduces_first_ensemble():
    signal = np.array([1.0, 2.0, 4.0])
    first = MonteCarlo(5, 0.1, seed=6).generate(signal)
    second = MonteCarlo(5, 0.1, seed=6).generate(signal)

    np.testing.assert_array_equal(first, second)


def test_different_seeds_produce_different_ensembles():
    signal = np.array([1.0, 2.0, 4.0])
    first = MonteCarlo(5, 0.1, seed=7).generate(signal)
    second = MonteCarlo(5, 0.1, seed=8).generate(signal)

    assert not np.array_equal(first, second)


def test_same_instance_rng_advances_between_calls():
    signal = np.array([1.0, 2.0, 4.0])
    monte_carlo = MonteCarlo(5, 0.1, seed=9)

    first = monte_carlo.generate(signal)
    second = monte_carlo.generate(signal)

    assert not np.array_equal(first, second)


def test_correlated_noise_has_constant_ratio_in_time():
    signal = np.array([1.0, 2.0, 4.0, 8.0])
    ensemble = MonteCarlo(6, 0.1, seed=10).generate(signal, independent_points=False)
    ratios = ensemble / signal.reshape(1, -1)

    np.testing.assert_allclose(ratios, np.repeat(ratios[:, [0]], signal.size, axis=1))


def test_independent_points_noise_generally_varies_in_time():
    signal = np.array([1.0, 2.0, 4.0, 8.0])
    ensemble = MonteCarlo(6, 0.1, seed=11).generate(signal, independent_points=True)
    ratios = ensemble / signal.reshape(1, -1)

    assert np.any(~np.isclose(ratios, ratios[:, [0]]))


def test_scalar_signal_is_supported():
    ensemble = MonteCarlo(3, 0.0, seed=12).generate(2.5)

    np.testing.assert_array_equal(ensemble, np.array([2.5, 2.5, 2.5]))


def test_statistics_have_expected_shapes():
    ensemble = np.array(
        [
            [1.0, 2.0, 3.0],
            [2.0, 3.0, 4.0],
            [3.0, 4.0, 5.0],
        ]
    )
    stats = MonteCarlo.statistics(ensemble)

    assert set(stats) == {"mean", "std", "min", "max", "median", "q05", "q95"}
    assert all(value.shape == (3,) for value in stats.values())


def test_statistics_ordering():
    ensemble = np.array(
        [
            [1.0, 5.0],
            [2.0, 6.0],
            [3.0, 7.0],
        ]
    )
    stats = MonteCarlo.statistics(ensemble)

    assert np.all(stats["min"] <= stats["median"])
    assert np.all(stats["median"] <= stats["max"])
    assert np.all(stats["q05"] <= stats["q95"])


@pytest.mark.parametrize("bad_signal", [[], [1.0, np.nan], [1.0, np.inf]])
def test_invalid_signal_raises(bad_signal):
    with pytest.raises(ValueError, match="signal"):
        MonteCarlo(3, 0.1).generate(bad_signal)
