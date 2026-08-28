from math import pi

import matplotlib
import numpy as np
import pytest

from bec_fast_manipulation.analysis import ConsoleReporter, Plotter, StatisticalAnalysis
from bec_fast_manipulation.analysis.plotter import X_COLOR, Y_COLOR, Z_COLOR, BORDEAUX, ROYAL_BLUE


def make_plotter(tmp_path):
    return Plotter(figures_dir=tmp_path, show=False, save=False)


def test_plotter_radii_converts_units_and_uses_axis_colors_with_references(tmp_path):
    plotter = make_plotter(tmp_path)
    time_s = np.array([0.0, 1e-3, 2e-3])

    _, ax = plotter.plot_thomas_fermi_radii(
        time_s,
        [1e-6, 2e-6, 3e-6],
        [4e-6, 5e-6, 6e-6],
        [7e-6, 8e-6, 9e-6],
        reference_radius_x_m=[2e-6, 3e-6, 4e-6],
        reference_radius_y_m=[5e-6, 6e-6, 7e-6],
        reference_radius_z_m=[8e-6, 9e-6, 10e-6],
    )

    lines = ax.get_lines()
    assert len(lines) == 6
    np.testing.assert_allclose(lines[0].get_xdata(), [0.0, 1.0, 2.0])
    np.testing.assert_allclose(lines[0].get_ydata(), [1.0, 2.0, 3.0])
    assert lines[0].get_color() == X_COLOR
    assert lines[1].get_color() == Y_COLOR
    assert lines[2].get_color() == Z_COLOR
    assert [line.get_linestyle() for line in lines[3:]] == ["--", "--", "--"]


def test_plotter_rejects_partial_reference_triplet(tmp_path):
    plotter = make_plotter(tmp_path)
    with pytest.raises(ValueError, match="Reference curves"):
        plotter.plot_thomas_fermi_radii([0.0], [1e-6], [1e-6], [1e-6], reference_radius_x_m=[1e-6])


def test_plotter_velocity_converts_m_per_s_to_mm_per_s(tmp_path):
    plotter = make_plotter(tmp_path)
    _, ax = plotter.plot_thomas_fermi_radius_velocities(
        [0.0, 1e-3],
        [1e-3, 2e-3],
        [3e-3, 4e-3],
        [5e-3, 6e-3],
    )

    np.testing.assert_allclose(ax.get_lines()[0].get_ydata(), [1.0, 2.0])
    assert ax.get_ylabel() == "Thomas-Fermi radius velocity [mm/s]"


def test_plotter_trap_frequencies_converts_rad_per_s_to_hz_with_reference(tmp_path):
    plotter = make_plotter(tmp_path)
    omega_x = np.array([2 * pi, 4 * pi])
    omega_y = np.array([6 * pi, 8 * pi])
    omega_z = np.array([10 * pi, 12 * pi])
    original_inputs = [array.copy() for array in (omega_x, omega_y, omega_z)]

    _, ax = plotter.plot_trap_frequencies(
        [0.0, 1e-3],
        omega_x,
        omega_y,
        omega_z,
        reference_frequency_hz=50.0,
    )

    lines = ax.get_lines()
    assert len(lines) == 4
    np.testing.assert_allclose(lines[0].get_xdata(), [0.0, 1.0])
    np.testing.assert_allclose(lines[0].get_ydata(), [1.0, 2.0])
    np.testing.assert_allclose(lines[1].get_ydata(), [3.0, 4.0])
    np.testing.assert_allclose(lines[2].get_ydata(), [5.0, 6.0])
    np.testing.assert_allclose(lines[3].get_ydata(), [50.0, 50.0])
    assert [line.get_color() for line in lines[:3]] == [X_COLOR, Y_COLOR, Z_COLOR]
    assert lines[3].get_color() == "black"
    assert lines[3].get_linestyle() == "--"
    assert lines[3].get_label() == "50 Hz threshold"
    assert ax.get_ylabel() == "Frequency [Hz]"
    for original, current in zip(original_inputs, (omega_x, omega_y, omega_z)):
        np.testing.assert_allclose(current, original)


def test_plotter_trap_frequencies_accepts_no_reference(tmp_path):
    plotter = make_plotter(tmp_path)
    _, ax = plotter.plot_trap_frequencies(
        [0.0, 1e-3],
        [2 * pi, 4 * pi],
        [6 * pi, 8 * pi],
        [10 * pi, 12 * pi],
    )

    lines = ax.get_lines()
    assert len(lines) == 3
    np.testing.assert_allclose(lines[2].get_ydata(), [5.0, 6.0])
    assert ax.get_xlabel() == "Time [ms]"
    assert ax.get_ylabel() == "Frequency [Hz]"


def test_plotter_expansion_temperatures_converts_kelvin_to_nanokelvin(tmp_path):
    plotter = make_plotter(tmp_path)
    _, ax = plotter.plot_expansion_temperatures(
        [0.0, 1e-3],
        [1e-9, 2e-9],
        [3e-9, 4e-9],
        [5e-9, 6e-9],
        reference_temperature_x_K=[2e-9, 3e-9],
        reference_temperature_y_K=[4e-9, 5e-9],
        reference_temperature_z_K=[6e-9, 7e-9],
    )

    lines = ax.get_lines()
    assert len(lines) == 6
    np.testing.assert_allclose(lines[0].get_ydata(), [1.0, 2.0])
    assert [lines[index].get_color() for index in range(3)] == [X_COLOR, Y_COLOR, Z_COLOR]
    assert [line.get_linestyle() for line in lines[3:]] == ["--", "--", "--"]


def test_plotter_temperature_3d_with_and_without_reference(tmp_path):
    plotter = make_plotter(tmp_path)
    _, ax = plotter.plot_expansion_temperature_3d([0.0, 1e-3], [1e-9, 2e-9])
    assert len(ax.get_lines()) == 1
    assert ax.get_lines()[0].get_color() == BORDEAUX

    _, ax_ref = plotter.plot_expansion_temperature_3d(
        [0.0, 1e-3],
        [1e-9, 2e-9],
        reference_temperature_3d_K=[3e-9, 4e-9],
    )
    lines = ax_ref.get_lines()
    assert len(lines) == 2
    np.testing.assert_allclose(lines[0].get_ydata(), [1.0, 2.0])
    np.testing.assert_allclose(lines[1].get_ydata(), [3.0, 4.0])
    assert lines[1].get_color() == ROYAL_BLUE
    assert lines[1].get_linestyle() == "--"


def test_plotter_curve_with_bounds_returns_fig_ax_and_filled_envelope(tmp_path):
    plotter = make_plotter(tmp_path)
    x = np.array([0.0, 1.0, 2.0])

    fig, ax = plotter.plot_curve_with_bounds(
        x,
        nominal=[1.0, 2.0, 3.0],
        lower=[0.8, 1.8, 2.8],
        upper=[1.2, 2.2, 3.2],
        sample=[0.9, 2.1, 2.9],
        xlabel="x",
        ylabel="y",
        show=False,
    )

    assert fig is ax.figure
    assert len(ax.collections) == 1
    assert len(ax.get_lines()) == 2
    assert ax.get_xlabel() == "x"
    assert ax.get_ylabel() == "y"


def test_plotter_curve_with_bounds_rejects_bad_shapes_and_bounds(tmp_path):
    plotter = make_plotter(tmp_path)

    with pytest.raises(ValueError, match="same shape"):
        plotter.plot_curve_with_bounds([0.0, 1.0], [1.0], [0.0], [2.0])
    with pytest.raises(ValueError, match="finite"):
        plotter.plot_curve_with_bounds([0.0, 1.0], [1.0, np.nan], [0.0, 0.0], [2.0, 2.0])
    with pytest.raises(ValueError, match="lower"):
        plotter.plot_curve_with_bounds([0.0, 1.0], [1.0, 1.0], [2.0, 0.0], [1.0, 2.0])
    with pytest.raises(ValueError, match="sample"):
        plotter.plot_curve_with_bounds([0.0, 1.0], [1.0, 1.0], [0.0, 0.0], [2.0, 2.0], sample=[1.0])


def test_plotter_ensemble_envelope_returns_fig_ax_and_optional_curves(tmp_path):
    plotter = make_plotter(tmp_path)
    x = np.array([0.0, 1.0, 2.0])
    ensemble = np.array(
        [
            [1.0, 2.0, 3.0],
            [1.2, 2.2, 3.2],
            [0.8, 1.8, 2.8],
        ]
    )

    fig, ax = plotter.plot_ensemble_envelope(
        x,
        ensemble,
        nominal=[1.0, 2.0, 3.0],
        center="mean",
        sample_indices=[0, 2],
        show=False,
    )

    assert fig is ax.figure
    assert len(ax.collections) == 1
    assert len(ax.get_lines()) == 4


def test_plotter_ensemble_envelope_rejects_invalid_inputs(tmp_path):
    plotter = make_plotter(tmp_path)
    ensemble = np.ones((3, 4))

    with pytest.raises(ValueError, match="2D"):
        plotter.plot_ensemble_envelope([0.0, 1.0], [1.0, 2.0])
    with pytest.raises(ValueError, match="len\\(x\\)"):
        plotter.plot_ensemble_envelope([0.0, 1.0, 2.0], ensemble)
    with pytest.raises(ValueError, match="quantiles"):
        plotter.plot_ensemble_envelope([0.0, 1.0, 2.0, 3.0], ensemble, lower_quantile=0.9, upper_quantile=0.1)
    with pytest.raises(ValueError, match="center"):
        plotter.plot_ensemble_envelope([0.0, 1.0, 2.0, 3.0], ensemble, center="mode")
    with pytest.raises(ValueError, match="nominal"):
        plotter.plot_ensemble_envelope([0.0, 1.0, 2.0, 3.0], ensemble, nominal=[1.0, 2.0])
    with pytest.raises(ValueError, match="sample_indices"):
        plotter.plot_ensemble_envelope([0.0, 1.0, 2.0, 3.0], ensemble, sample_indices=[3])
    with pytest.raises(ValueError, match="finite"):
        plotter.plot_ensemble_envelope([0.0, 1.0], [[1.0, np.inf]])


def test_plotter_histogram_with_reference_returns_fig_ax(tmp_path):
    plotter = make_plotter(tmp_path)

    fig, ax = plotter.plot_histogram_with_reference(
        [1.0, 2.0, 3.0, 4.0],
        reference=2.5,
        bins=2,
        xlabel="final value",
        show=False,
    )

    assert fig is ax.figure
    assert ax.get_xlabel() == "final value"
    assert ax.get_ylabel() == "Count"
    assert len(ax.get_lines()) == 2
    assert len(ax.patches) >= 2


def test_plotter_histogram_rejects_invalid_values(tmp_path):
    plotter = make_plotter(tmp_path)

    with pytest.raises(ValueError, match="empty"):
        plotter.plot_histogram_with_reference([])
    with pytest.raises(ValueError, match="finite"):
        plotter.plot_histogram_with_reference([1.0, np.nan])
    with pytest.raises(ValueError, match="reference"):
        plotter.plot_histogram_with_reference([1.0, 2.0], reference=np.inf)


def test_plotter_generic_monte_carlo_primitives_work_without_interactive_show(tmp_path):
    assert matplotlib.get_backend().lower() == "agg"
    plotter = Plotter(figures_dir=tmp_path, show=False, save=True)
    x = np.array([0.0, 1.0, 2.0])

    plotter.plot_curve_with_bounds(x, [1.0, 2.0, 3.0], [0.9, 1.9, 2.9], [1.1, 2.1, 3.1], filename="bounds.png")
    plotter.plot_ensemble_envelope(x, np.array([[1.0, 2.0, 3.0], [1.1, 2.1, 3.1]]), filename="envelope.png")
    plotter.plot_histogram_with_reference([1.0, 2.0, 3.0], filename="histogram.png")

    assert (tmp_path / "bounds.png").exists()
    assert (tmp_path / "envelope.png").exists()
    assert (tmp_path / "histogram.png").exists()


def test_plotter_highlights_interval_in_milliseconds(tmp_path):
    plotter = make_plotter(tmp_path)
    _, ax = plotter.plot_expansion_temperature_3d(
        [0.0, 1e-3, 2e-3],
        [1e-9, 2e-9, 3e-9],
        highlight_interval_s=(0.5e-3, 1.5e-3),
        xlim_s=(0.0, 3e-3),
    )

    assert len(ax.patches) == 1
    assert ax.patches[0].get_label() == "Delta kick"
    np.testing.assert_allclose(ax.get_xlim(), [0.0, 3.0])
    legend_labels = [text.get_text() for text in ax.get_legend().get_texts()]
    assert "Delta kick" in legend_labels


def test_statistical_analysis_safe_ratio():
    analysis = StatisticalAnalysis()
    assert analysis.safe_ratio(6.0, 2.0) == pytest.approx(3.0)
    assert np.isnan(analysis.safe_ratio(1.0, 0.0))
    assert np.isnan(analysis.safe_ratio(1.0, 1e-15, atol=1e-12))


def test_statistical_analysis_reductions():
    analysis = StatisticalAnalysis()
    assert analysis.relative_change(8.0, 10.0) == pytest.approx(-0.2)
    assert analysis.reduction_fraction(8.0, 10.0) == pytest.approx(0.2)
    assert analysis.reduction_fraction(10.0, 10.0) == pytest.approx(0.0)
    assert analysis.reduction_fraction(12.0, 10.0) == pytest.approx(-0.2)
    assert analysis.reduction_percent(8.0, 10.0) == pytest.approx(20.0)


def test_statistical_analysis_compare_axis_and_scalar_values():
    analysis = StatisticalAnalysis()
    comparison = analysis.compare_axis_values([5.0, 8.0, 9.0], [10.0, 10.0, 12.0])
    assert set(comparison) == {"x", "y", "z"}
    assert comparison["x"]["ratio"] == pytest.approx(0.5)
    assert comparison["y"]["reduction_percent"] == pytest.approx(20.0)

    scalar = analysis.compare_scalar_values(3.0, 4.0)
    assert scalar["value"] == pytest.approx(3.0)
    assert scalar["reduction_fraction"] == pytest.approx(0.25)


def test_statistical_analysis_final_axis_values_and_rms():
    analysis = StatisticalAnalysis()
    assert analysis.final_axis_values([1, 2], [3, 4], [5, 6]) == {"x": 2.0, "y": 4.0, "z": 6.0}
    assert analysis.rms([3.0, 4.0]) == pytest.approx(np.sqrt(12.5))


def test_console_reporter_header_and_section(capsys):
    reporter = ConsoleReporter(width=20)
    reporter.header("Analysis")
    reporter.section("Temperatures")

    output = capsys.readouterr().out
    assert "Analysis" in output
    assert "Temperatures" in output


def test_console_reporter_parameters_and_axis_table(capsys):
    reporter = ConsoleReporter()
    reporter.parameters(
        [
            ("Atom number", np.int64(100000), "-"),
            ("omega_x / 2pi", np.float64(500.0), "Hz"),
            ("missing", np.nan, "-"),
        ]
    )
    reporter.axis_table(
        {
            "Radius [um]": [np.float64(1.2), 2.3, np.nan],
            "Velocity [mm/s]": [4.5, 5.6, 6.7],
            "Temperature [nK]": [10.0, 20.0, 30.0],
            "Reduction [%]": [50.0, 0.0, -10.0],
        }
    )

    output = capsys.readouterr().out
    assert "Atom number" in output
    assert "Radius [um]" in output
    assert "Velocity [mm/s]" in output
    assert "Temperature [nK]" in output
    assert "Reduction [%]" in output
    assert "nan" in output


def test_console_reporter_comparison_tables(capsys):
    reporter = ConsoleReporter()
    comparison = StatisticalAnalysis.compare_axis_values([5.0, 8.0, 12.0], [10.0, 10.0, 10.0])
    scalar = StatisticalAnalysis.compare_scalar_values(2.0, 4.0)

    reporter.comparison_table(comparison)
    reporter.scalar_comparison("T_3D", scalar)

    output = capsys.readouterr().out
    assert "Axis" in output
    assert "Value" in output
    assert "Reference" in output
    assert "Ratio" in output
    assert "Reduction [%]" in output
    assert "T_3D" in output


def test_console_reporter_comparison_tables_can_customize_columns(capsys):
    reporter = ConsoleReporter()
    comparison = StatisticalAnalysis.compare_axis_values([5.0, 8.0, 12.0], [10.0, 10.0, 10.0])
    scalar = StatisticalAnalysis.compare_scalar_values(2.0, 4.0)

    reporter.comparison_table(
        comparison,
        value_label="Delta Kick [um]",
        reference_label="Free Expansion [um]",
        show_ratio=False,
    )
    reporter.scalar_comparison(
        "T_3D",
        scalar,
        value_label="Delta Kick [nK]",
        reference_label="Free Expansion [nK]",
        show_ratio=False,
    )

    output = capsys.readouterr().out
    assert "Delta Kick [um]" in output
    assert "Free Expansion [um]" in output
    assert "Delta Kick [nK]" in output
    assert "Free Expansion [nK]" in output
    assert "Ratio" not in output
