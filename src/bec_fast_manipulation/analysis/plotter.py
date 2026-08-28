"""Plotting helpers for scientific analysis outputs."""

from __future__ import annotations

from math import pi
from pathlib import Path
from typing import Iterable

import numpy as np
import matplotlib.pyplot as plt


BORDEAUX = (128 / 255, 0, 32 / 255)
ROYAL_BLUE = (65 / 255, 105 / 255, 225 / 255)
IMPERIAL_GREEN = (0 / 255, 120 / 255, 60 / 255)
DEEP_TEAL = (0 / 255, 128 / 255, 128 / 255)
VIVID_ORANGE = (255 / 255, 140 / 255, 0 / 255)
AMETHYST_PURPLE = (153 / 255, 102 / 255, 204 / 255)

CUSTOM_COLORS = [
    BORDEAUX,
    ROYAL_BLUE,
    IMPERIAL_GREEN,
    DEEP_TEAL,
    VIVID_ORANGE,
    AMETHYST_PURPLE,
]

X_COLOR = BORDEAUX
Y_COLOR = ROYAL_BLUE
Z_COLOR = IMPERIAL_GREEN


class Plotter:
    """Centralized matplotlib interface for project analysis figures."""

    DEFAULT_FIGSIZE = (7, 4.5)
    DEFAULT_LINEWIDTH = 2.0
    DEFAULT_GRID_ALPHA = 0.3

    def __init__(
        self,
        figures_dir: str | Path = "figures",
        dpi: int = 300,
        show: bool = True,
        save: bool = False,
    ) -> None:
        self.figures_dir = Path(figures_dir)
        self.dpi = dpi
        self.show = show
        self.save = save
        self.set_style()

    @staticmethod
    def set_style() -> None:
        """Apply the shared scientific matplotlib style."""
        plt.rcParams.update(
            {
                "font.family": "serif",
                "font.size": 13,
                "axes.labelsize": 14,
                "axes.titlesize": 12,
                "xtick.labelsize": 12,
                "ytick.labelsize": 12,
                "legend.fontsize": 12,
                "mathtext.fontset": "cm",
            }
        )

    @staticmethod
    def _as_1d_array(values, name: str) -> np.ndarray:
        array = np.asarray(values, dtype=float)
        if array.size == 0:
            raise ValueError(f"{name} must not be empty.")
        if array.ndim != 1:
            raise ValueError(f"{name} must be a 1D array.")
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must contain only finite values.")
        return array

    @classmethod
    def _validate_compatible_1d_arrays(cls, x, ys: Iterable, names: Iterable[str]) -> tuple[np.ndarray, list[np.ndarray]]:
        x_array = cls._as_1d_array(x, "x")
        y_arrays = [cls._as_1d_array(y, name) for y, name in zip(ys, names)]
        for y_array, name in zip(y_arrays, names):
            if y_array.shape != x_array.shape:
                raise ValueError(f"{name} must have the same shape as x.")
        return x_array, y_arrays

    @staticmethod
    def _validate_reference_triplet(references: tuple[object | None, object | None, object | None]) -> bool:
        present = [reference is not None for reference in references]
        if any(present) and not all(present):
            raise ValueError("Reference curves must be provided for x, y, and z together.")
        return all(present)

    @staticmethod
    def _validate_highlight_interval(interval: tuple[float, float] | None) -> tuple[float, float] | None:
        if interval is None:
            return None
        if len(interval) != 2:
            raise ValueError("highlight_interval must contain exactly two values.")
        start, end = (float(interval[0]), float(interval[1]))
        if not np.isfinite(start) or not np.isfinite(end):
            raise ValueError("highlight_interval must contain only finite values.")
        if end < start:
            raise ValueError("highlight_interval end must be greater than or equal to start.")
        return start, end

    def _finalize(
        self,
        fig,
        filename: str,
        *,
        save: bool | None = None,
        show: bool | None = None,
    ) -> None:
        fig.tight_layout()
        should_save = self.save if save is None else save
        should_show = self.show if show is None else show
        if should_save:
            self.figures_dir.mkdir(parents=True, exist_ok=True)
            fig.savefig(self.figures_dir / filename, dpi=self.dpi, bbox_inches="tight")
        if should_show:
            plt.show()
        plt.close(fig)

    def plot_curves(
        self,
        x,
        ys,
        xlabel: str = "",
        ylabel: str = "",
        title: str | None = None,
        labels: list[str] | tuple[str, ...] | None = None,
        xlim: tuple[float, float] | None = None,
        ylim: tuple[float, float] | None = None,
        grid: bool = True,
        figsize: tuple[float, float] = DEFAULT_FIGSIZE,
        linewidth: float = DEFAULT_LINEWIDTH,
        colors: list | tuple | None = None,
        linestyles: list[str] | tuple[str, ...] | None = None,
        highlight_interval: tuple[float, float] | None = None,
        highlight_label: str = "Delta kick",
        filename: str = "figure.png",
        save: bool | None = None,
        show: bool | None = None,
    ):
        """Plot generic one-dimensional curves."""
        y_values = list(ys)
        if not y_values:
            raise ValueError("ys must contain at least one curve.")
        if labels is not None and len(labels) != len(y_values):
            raise ValueError("labels must have the same length as ys.")

        names = [f"ys[{index}]" for index in range(len(y_values))]
        x_array, y_arrays = self._validate_compatible_1d_arrays(x, y_values, names)

        colors = colors or CUSTOM_COLORS
        linestyles = linestyles or ["-"] * len(y_arrays)
        if len(linestyles) < len(y_arrays):
            raise ValueError("linestyles must provide one style per curve.")

        fig, ax = plt.subplots(figsize=figsize)
        highlight = self._validate_highlight_interval(highlight_interval)
        if highlight is not None:
            ax.axvspan(
                highlight[0],
                highlight[1],
                color="0.85",
                alpha=0.35,
                label=highlight_label,
                zorder=0,
            )
        for index, y_array in enumerate(y_arrays):
            label = None if labels is None else labels[index]
            ax.plot(
                x_array,
                y_array,
                label=label,
                color=colors[index % len(colors)],
                linestyle=linestyles[index],
                linewidth=linewidth,
            )
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if title is not None:
            ax.set_title(title)
        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)
        if grid:
            ax.grid(True, alpha=self.DEFAULT_GRID_ALPHA)
        if labels is not None or highlight is not None:
            ax.legend()
        self._finalize(fig, filename, save=save, show=show)
        return fig, ax

    def plot_curve_series(
        self,
        xs,
        ys,
        *,
        xlabel: str = "",
        ylabel: str = "",
        title: str | None = None,
        labels: list[str] | tuple[str, ...] | None = None,
        colors: list | tuple | None = None,
        linestyles: list[str] | tuple[str, ...] | None = None,
        filename: str = "figure.png",
        save: bool | None = None,
        show: bool | None = None,
    ):
        """Plot generic curves that may each have their own x coordinates."""
        x_values = list(xs)
        y_values = list(ys)
        if not x_values or len(x_values) != len(y_values):
            raise ValueError("xs and ys must contain the same non-zero number of curves.")
        if labels is not None and len(labels) != len(y_values):
            raise ValueError("labels must have the same length as ys.")
        colors = colors or CUSTOM_COLORS
        linestyles = linestyles or ["-"] * len(y_values)
        if len(linestyles) < len(y_values):
            raise ValueError("linestyles must provide one style per curve.")

        fig, ax = plt.subplots(figsize=self.DEFAULT_FIGSIZE)
        for index, (x, y) in enumerate(zip(x_values, y_values)):
            x_array, [y_array] = self._validate_compatible_1d_arrays(x, [y], [f"ys[{index}]"])
            ax.plot(
                x_array,
                y_array,
                label=None if labels is None else labels[index],
                color=colors[index % len(colors)],
                linestyle=linestyles[index],
                linewidth=self.DEFAULT_LINEWIDTH,
            )
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if title is not None:
            ax.set_title(title)
        ax.grid(True, alpha=self.DEFAULT_GRID_ALPHA)
        if labels is not None:
            ax.legend()
        self._finalize(fig, filename, save=save, show=show)
        return fig, ax

    @staticmethod
    def _as_2d_array(values, name: str) -> np.ndarray:
        array = np.asarray(values, dtype=float)
        if array.size == 0:
            raise ValueError(f"{name} must not be empty.")
        if array.ndim != 2:
            raise ValueError(f"{name} must be a 2D array.")
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must contain only finite values.")
        return array

    @staticmethod
    def _validate_quantiles(lower_quantile: float, upper_quantile: float) -> tuple[float, float]:
        lower = float(lower_quantile)
        upper = float(upper_quantile)
        if not np.isfinite(lower) or not np.isfinite(upper):
            raise ValueError("quantiles must be finite.")
        if not 0.0 <= lower < upper <= 1.0:
            raise ValueError("quantiles must satisfy 0 <= lower_quantile < upper_quantile <= 1.")
        return lower, upper

    @staticmethod
    def _validate_sample_indices(sample_indices, n_simulations: int) -> list[int]:
        if sample_indices is None:
            return []
        indices = []
        for index in sample_indices:
            if isinstance(index, bool):
                raise ValueError("sample_indices must contain integer indices.")
            integer_index = int(index)
            if integer_index != index:
                raise ValueError("sample_indices must contain integer indices.")
            if integer_index < 0 or integer_index >= n_simulations:
                raise ValueError("sample_indices entries must be valid ensemble indices.")
            indices.append(integer_index)
        return indices

    def plot_curve_with_bounds(
        self,
        x,
        nominal,
        lower,
        upper,
        *,
        sample=None,
        xlabel: str | None = None,
        ylabel: str | None = None,
        title: str | None = None,
        nominal_label: str = "Nominal",
        bounds_label: str | None = None,
        sample_label: str = "Sample",
        filename: str = "curve_with_bounds.png",
        save: bool | None = None,
        show: bool | None = None,
    ):
        """Plot a nominal curve with a filled lower/upper envelope."""
        x_array, [nominal_array, lower_array, upper_array] = self._validate_compatible_1d_arrays(
            x,
            [nominal, lower, upper],
            ["nominal", "lower", "upper"],
        )
        if np.any(lower_array > upper_array):
            raise ValueError("lower must be less than or equal to upper everywhere.")
        sample_array = None
        if sample is not None:
            sample_array = self._as_1d_array(sample, "sample")
            if sample_array.shape != x_array.shape:
                raise ValueError("sample must have the same shape as x.")

        fig, ax = plt.subplots(figsize=self.DEFAULT_FIGSIZE)
        label = bounds_label or "Bounds"
        ax.fill_between(x_array, lower_array, upper_array, color=ROYAL_BLUE, alpha=0.18, label=label)
        if sample_array is not None:
            ax.plot(x_array, sample_array, color="0.35", linewidth=1.0, alpha=0.55, label=sample_label)
        ax.plot(x_array, nominal_array, color=BORDEAUX, linewidth=self.DEFAULT_LINEWIDTH, label=nominal_label)
        ax.set_xlabel("" if xlabel is None else xlabel)
        ax.set_ylabel("" if ylabel is None else ylabel)
        if title is not None:
            ax.set_title(title)
        ax.grid(True, alpha=self.DEFAULT_GRID_ALPHA)
        ax.legend()
        self._finalize(fig, filename, save=save, show=show)
        return fig, ax

    def plot_ensemble_envelope(
        self,
        x,
        ensemble,
        *,
        nominal=None,
        lower_quantile: float = 0.05,
        upper_quantile: float = 0.95,
        center: str = "median",
        sample_indices=None,
        xlabel: str | None = None,
        ylabel: str | None = None,
        title: str | None = None,
        filename: str = "ensemble_envelope.png",
        save: bool | None = None,
        show: bool | None = None,
    ):
        """Plot an ensemble quantile envelope and optional selected samples."""
        x_array = self._as_1d_array(x, "x")
        ensemble_array = self._as_2d_array(ensemble, "ensemble")
        if ensemble_array.shape[1] != x_array.size:
            raise ValueError("ensemble must have shape (n_simulations, len(x)).")
        lower, upper = self._validate_quantiles(lower_quantile, upper_quantile)
        if center == "median":
            center_array = np.median(ensemble_array, axis=0)
            center_label = "Median"
        elif center == "mean":
            center_array = np.mean(ensemble_array, axis=0)
            center_label = "Mean"
        else:
            raise ValueError("center must be 'median' or 'mean'.")

        nominal_array = None
        if nominal is not None:
            nominal_array = self._as_1d_array(nominal, "nominal")
            if nominal_array.shape != x_array.shape:
                raise ValueError("nominal must have the same shape as x.")
        indices = self._validate_sample_indices(sample_indices, ensemble_array.shape[0])

        q_low = np.quantile(ensemble_array, lower, axis=0)
        q_high = np.quantile(ensemble_array, upper, axis=0)
        envelope_label = f"{lower:g}-{upper:g} quantile envelope"

        fig, ax = plt.subplots(figsize=self.DEFAULT_FIGSIZE)
        ax.fill_between(x_array, q_low, q_high, color=ROYAL_BLUE, alpha=0.18, label=envelope_label)
        for plot_index, sample_index in enumerate(indices):
            ax.plot(
                x_array,
                ensemble_array[sample_index],
                color="0.45",
                linewidth=0.8,
                alpha=0.35,
                label="Samples" if plot_index == 0 else None,
            )
        ax.plot(x_array, center_array, color=BORDEAUX, linewidth=self.DEFAULT_LINEWIDTH, label=center_label)
        if nominal_array is not None:
            ax.plot(x_array, nominal_array, color=IMPERIAL_GREEN, linewidth=1.6, linestyle="--", label="Nominal")
        ax.set_xlabel("" if xlabel is None else xlabel)
        ax.set_ylabel("" if ylabel is None else ylabel)
        if title is not None:
            ax.set_title(title)
        ax.grid(True, alpha=self.DEFAULT_GRID_ALPHA)
        ax.legend()
        self._finalize(fig, filename, save=save, show=show)
        return fig, ax

    def plot_histogram_with_reference(
        self,
        values,
        *,
        reference=None,
        bins: int = 30,
        xlabel: str | None = None,
        ylabel: str = "Count",
        title: str | None = None,
        reference_label: str = "Nominal",
        show_mean: bool = True,
        show_std: bool = True,
        filename: str = "histogram_with_reference.png",
        save: bool | None = None,
        show: bool | None = None,
    ):
        """Plot a one-dimensional histogram with optional reference statistics."""
        value_array = self._as_1d_array(values, "values")
        if reference is not None:
            reference = float(reference)
            if not np.isfinite(reference):
                raise ValueError("reference must be finite.")

        fig, ax = plt.subplots(figsize=self.DEFAULT_FIGSIZE)
        ax.hist(value_array, bins=bins, color=ROYAL_BLUE, alpha=0.68, edgecolor="white")
        if reference is not None:
            ax.axvline(reference, color=BORDEAUX, linewidth=self.DEFAULT_LINEWIDTH, label=reference_label)
        if show_mean or show_std:
            mean = float(np.mean(value_array))
            std = float(np.std(value_array))
        if show_mean:
            ax.axvline(mean, color="black", linestyle="--", linewidth=1.5, label="Mean")
        if show_std:
            ax.axvspan(mean - std, mean + std, color="0.2", alpha=0.08, label="Mean +/- std")
        ax.set_xlabel("" if xlabel is None else xlabel)
        ax.set_ylabel(ylabel)
        if title is not None:
            ax.set_title(title)
        ax.grid(True, axis="y", alpha=self.DEFAULT_GRID_ALPHA)
        if reference is not None or show_mean or show_std:
            ax.legend()
        self._finalize(fig, filename, save=save, show=show)
        return fig, ax

    def plot_heatmap(
        self,
        x,
        y,
        values,
        *,
        xlabel: str = "",
        ylabel: str = "",
        colorbar_label: str = "",
        title: str | None = None,
        cmap: str = "viridis",
        marker_points: list[tuple[float, float, str]] | None = None,
        filename: str = "heatmap.png",
        save: bool | None = None,
        show: bool | None = None,
    ):
        """Plot a generic 2D heatmap from already computed data."""
        x_array = self._as_1d_array(x, "x")
        y_array = self._as_1d_array(y, "y")
        value_array = np.asarray(values, dtype=float)
        if value_array.shape != (y_array.size, x_array.size):
            raise ValueError("values must have shape (len(y), len(x)).")

        fig, ax = plt.subplots(figsize=self.DEFAULT_FIGSIZE)
        mesh = ax.pcolormesh(x_array, y_array, value_array, shading="auto", cmap=cmap)
        colorbar = fig.colorbar(mesh, ax=ax)
        colorbar.set_label(colorbar_label)
        if marker_points is not None:
            for x_value, y_value, label in marker_points:
                ax.plot(float(x_value), float(y_value), "x", color="black", markersize=7, markeredgewidth=1.5)
                if label:
                    ax.annotate(label, (float(x_value), float(y_value)), xytext=(5, 5), textcoords="offset points")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if title is not None:
            ax.set_title(title)
        self._finalize(fig, filename, save=save, show=show)
        return fig, ax

    def plot_surface_3d(
        self,
        x,
        y,
        z,
        *,
        xlabel: str = "",
        ylabel: str = "",
        zlabel: str = "",
        title: str | None = None,
        cmap: str = "viridis",
        filename: str = "surface.png",
        save: bool | None = None,
        show: bool | None = None,
    ):
        """Plot a generic 3D surface from already computed data."""
        x_array = self._as_1d_array(x, "x")
        y_array = self._as_1d_array(y, "y")
        z_array = np.asarray(z, dtype=float)
        if z_array.shape != (y_array.size, x_array.size):
            raise ValueError("z must have shape (len(y), len(x)).")

        x_grid, y_grid = np.meshgrid(x_array, y_array)
        fig = plt.figure(figsize=(7, 5.2))
        ax = fig.add_subplot(111, projection="3d")
        surface = ax.plot_surface(x_grid, y_grid, z_array, cmap=cmap, linewidth=0, antialiased=True)
        colorbar = fig.colorbar(surface, ax=ax, shrink=0.72, pad=0.12)
        colorbar.set_label(zlabel)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_zlabel(zlabel)
        if title is not None:
            ax.set_title(title)
        self._finalize(fig, filename, save=save, show=show)
        return fig, ax

    def plot_thomas_fermi_radii(
        self,
        time_s,
        radius_x_m,
        radius_y_m,
        radius_z_m,
        *,
        reference_radius_x_m=None,
        reference_radius_y_m=None,
        reference_radius_z_m=None,
        highlight_interval_s: tuple[float, float] | None = None,
        xlim_s: tuple[float, float] | None = None,
        title: str | None = None,
        filename: str = "thomas_fermi_radii.png",
        save: bool | None = None,
        show: bool | None = None,
    ):
        references = (reference_radius_x_m, reference_radius_y_m, reference_radius_z_m)
        has_reference = self._validate_reference_triplet(references)
        time_ms = self._as_1d_array(time_s, "time_s") * 1e3
        curves = [np.asarray(radius_x_m) * 1e6, np.asarray(radius_y_m) * 1e6, np.asarray(radius_z_m) * 1e6]
        labels = [r"$R_x$", r"$R_y$", r"$R_z$"]
        colors = [X_COLOR, Y_COLOR, Z_COLOR]
        linestyles = ["-", "-", "-"]
        if has_reference:
            curves.extend([np.asarray(reference) * 1e6 for reference in references])
            labels.extend([r"$R_x$ reference", r"$R_y$ reference", r"$R_z$ reference"])
            colors.extend([X_COLOR, Y_COLOR, Z_COLOR])
            linestyles.extend(["--", "--", "--"])
        return self.plot_curves(
            time_ms,
            curves,
            xlabel="Time [ms]",
            ylabel=r"Thomas-Fermi radius [$\mu$m]",
            title=title,
            labels=labels,
            colors=colors,
            linestyles=linestyles,
            highlight_interval=None
            if highlight_interval_s is None
            else (highlight_interval_s[0] * 1e3, highlight_interval_s[1] * 1e3),
            xlim=None if xlim_s is None else (xlim_s[0] * 1e3, xlim_s[1] * 1e3),
            filename=filename,
            save=save,
            show=show,
        )

    def plot_trap_frequencies(
        self,
        time_s,
        omega_x_rad_s,
        omega_y_rad_s,
        omega_z_rad_s,
        *,
        reference_frequency_hz: float | None = None,
        title: str | None = None,
        filename: str = "angular_frequencies.png",
        save: bool | None = None,
        show: bool | None = None,
    ):
        """Plot axis-resolved trap frequencies in Hz from angular frequencies."""
        time_ms = self._as_1d_array(time_s, "time_s") * 1e3
        omega_arrays = [
            self._as_1d_array(omega_x_rad_s, "omega_x_rad_s"),
            self._as_1d_array(omega_y_rad_s, "omega_y_rad_s"),
            self._as_1d_array(omega_z_rad_s, "omega_z_rad_s"),
        ]
        frequency_arrays = [omega_array / (2 * pi) for omega_array in omega_arrays]

        if any(frequency_array.shape != time_ms.shape for frequency_array in frequency_arrays):
            raise ValueError("omega arrays must have the same shape as time_s.")

        fig, ax = plt.subplots(figsize=self.DEFAULT_FIGSIZE)
        for frequency_array, label, color in zip(
            frequency_arrays,
            [r"$f_x$", r"$f_y$", r"$f_z$"],
            [X_COLOR, Y_COLOR, Z_COLOR],
        ):
            ax.plot(
                time_ms,
                frequency_array,
                label=label,
                color=color,
                linewidth=self.DEFAULT_LINEWIDTH,
            )
        if reference_frequency_hz is not None:
            reference_frequency_hz = float(reference_frequency_hz)
            if not np.isfinite(reference_frequency_hz):
                raise ValueError("reference_frequency_hz must be finite.")
            ax.axhline(
                reference_frequency_hz,
                color="black",
                linestyle="--",
                linewidth=self.DEFAULT_LINEWIDTH,
                label=f"{reference_frequency_hz:g} Hz threshold",
            )
        ax.set_xlabel("Time [ms]")
        ax.set_ylabel("Frequency [Hz]")
        if title is not None:
            ax.set_title(title)
        ax.grid(True, alpha=self.DEFAULT_GRID_ALPHA)
        ax.legend()
        self._finalize(fig, filename, save=save, show=show)
        return fig, ax

    def plot_thomas_fermi_radius_velocities(
        self,
        time_s,
        velocity_x_m_s,
        velocity_y_m_s,
        velocity_z_m_s,
        *,
        reference_velocity_x_m_s=None,
        reference_velocity_y_m_s=None,
        reference_velocity_z_m_s=None,
        highlight_interval_s: tuple[float, float] | None = None,
        xlim_s: tuple[float, float] | None = None,
        title: str | None = None,
        filename: str = "thomas_fermi_radius_velocities.png",
        save: bool | None = None,
        show: bool | None = None,
    ):
        references = (reference_velocity_x_m_s, reference_velocity_y_m_s, reference_velocity_z_m_s)
        has_reference = self._validate_reference_triplet(references)
        time_ms = self._as_1d_array(time_s, "time_s") * 1e3
        curves = [np.asarray(velocity_x_m_s) * 1e3, np.asarray(velocity_y_m_s) * 1e3, np.asarray(velocity_z_m_s) * 1e3]
        labels = [r"$\dot R_x$", r"$\dot R_y$", r"$\dot R_z$"]
        colors = [X_COLOR, Y_COLOR, Z_COLOR]
        linestyles = ["-", "-", "-"]
        if has_reference:
            curves.extend([np.asarray(reference) * 1e3 for reference in references])
            labels.extend([r"$\dot R_x$ reference", r"$\dot R_y$ reference", r"$\dot R_z$ reference"])
            colors.extend([X_COLOR, Y_COLOR, Z_COLOR])
            linestyles.extend(["--", "--", "--"])
        return self.plot_curves(
            time_ms,
            curves,
            xlabel="Time [ms]",
            ylabel="Thomas-Fermi radius velocity [mm/s]",
            title=title,
            labels=labels,
            colors=colors,
            linestyles=linestyles,
            highlight_interval=None
            if highlight_interval_s is None
            else (highlight_interval_s[0] * 1e3, highlight_interval_s[1] * 1e3),
            xlim=None if xlim_s is None else (xlim_s[0] * 1e3, xlim_s[1] * 1e3),
            filename=filename,
            save=save,
            show=show,
        )

    def plot_expansion_temperatures(
        self,
        time_s,
        temperature_x_K,
        temperature_y_K,
        temperature_z_K,
        *,
        reference_temperature_x_K=None,
        reference_temperature_y_K=None,
        reference_temperature_z_K=None,
        highlight_interval_s: tuple[float, float] | None = None,
        xlim_s: tuple[float, float] | None = None,
        title: str | None = None,
        filename: str = "expansion_temperatures.png",
        save: bool | None = None,
        show: bool | None = None,
    ):
        references = (reference_temperature_x_K, reference_temperature_y_K, reference_temperature_z_K)
        has_reference = self._validate_reference_triplet(references)
        time_ms = self._as_1d_array(time_s, "time_s") * 1e3
        curves = [np.asarray(temperature_x_K) * 1e9, np.asarray(temperature_y_K) * 1e9, np.asarray(temperature_z_K) * 1e9]
        labels = [r"$T_x$", r"$T_y$", r"$T_z$"]
        colors = [X_COLOR, Y_COLOR, Z_COLOR]
        linestyles = ["-", "-", "-"]
        if has_reference:
            curves.extend([np.asarray(reference) * 1e9 for reference in references])
            labels.extend([r"$T_x$ reference", r"$T_y$ reference", r"$T_z$ reference"])
            colors.extend([X_COLOR, Y_COLOR, Z_COLOR])
            linestyles.extend(["--", "--", "--"])
        return self.plot_curves(
            time_ms,
            curves,
            xlabel="Time [ms]",
            ylabel="Expansion temperature [nK]",
            title=title,
            labels=labels,
            colors=colors,
            linestyles=linestyles,
            highlight_interval=None
            if highlight_interval_s is None
            else (highlight_interval_s[0] * 1e3, highlight_interval_s[1] * 1e3),
            xlim=None if xlim_s is None else (xlim_s[0] * 1e3, xlim_s[1] * 1e3),
            filename=filename,
            save=save,
            show=show,
        )

    def plot_expansion_temperature_3d(
        self,
        time_s,
        temperature_3d_K,
        *,
        reference_temperature_3d_K=None,
        highlight_interval_s: tuple[float, float] | None = None,
        xlim_s: tuple[float, float] | None = None,
        title: str | None = None,
        filename: str = "expansion_temperature_3d.png",
        save: bool | None = None,
        show: bool | None = None,
    ):
        """Plot scalar 3D temperature; colors encode scenario, not spatial axis."""
        time_ms = self._as_1d_array(time_s, "time_s") * 1e3
        curves = [np.asarray(temperature_3d_K) * 1e9]
        labels = [r"$T_{3D}$"]
        colors = [BORDEAUX]
        linestyles = ["-"]
        if reference_temperature_3d_K is not None:
            curves.append(np.asarray(reference_temperature_3d_K) * 1e9)
            labels.append(r"$T_{3D}$ reference")
            colors.append(ROYAL_BLUE)
            linestyles.append("--")
        return self.plot_curves(
            time_ms,
            curves,
            xlabel="Time [ms]",
            ylabel="Expansion temperature [nK]",
            title=title,
            labels=labels,
            colors=colors,
            linestyles=linestyles,
            highlight_interval=None
            if highlight_interval_s is None
            else (highlight_interval_s[0] * 1e3, highlight_interval_s[1] * 1e3),
            xlim=None if xlim_s is None else (xlim_s[0] * 1e3, xlim_s[1] * 1e3),
            filename=filename,
            save=save,
            show=show,
        )
