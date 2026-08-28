# Plotter

`Plotter` centralizes matplotlib usage for scientific figures. Experiment code
should normally use this class instead of importing `matplotlib` or
`matplotlib.pyplot` directly.

It performs no Castin-Dum, Thomas-Fermi, or temperature calculation. All
physical quantities must be computed before they are passed to the plotting API.

## Palette

The shared colors are:

- `BORDEAUX = (128 / 255, 0, 32 / 255)`
- `ROYAL_BLUE = (65 / 255, 105 / 255, 225 / 255)`
- `IMPERIAL_GREEN = (0 / 255, 120 / 255, 60 / 255)`
- `DEEP_TEAL = (0 / 255, 128 / 255, 128 / 255)`
- `VIVID_ORANGE = (255 / 255, 140 / 255, 0 / 255)`
- `AMETHYST_PURPLE = (153 / 255, 102 / 255, 204 / 255)`

Permanent spatial convention:

- `x`: `BORDEAUX`
- `y`: `ROYAL_BLUE`
- `z`: `IMPERIAL_GREEN`

The general convention is color equals spatial axis and linestyle equals
scenario. A main scenario uses solid lines; a reference scenario uses dashed
lines.

## Style

`set_style()` applies a serif font, Computer Modern math text, 13 pt base font,
14 pt axis labels, 12 pt titles, 12 pt ticks, and 12 pt legends. Defaults are
`figsize=(7, 4.5)`, `linewidth=2.0`, grid alpha `0.3`, and `dpi=300`.

## Units

Inputs are SI values. Plotter performs display conversions only:

- time: `s` to `ms`
- radii: `m` to `um`
- radius velocities: `m/s` to `mm/s`
- temperatures: `K` to `nK`

## Public Plots

- `plot_curves(...)`: generic one-dimensional curves.
- `plot_thomas_fermi_radii(...)`: plots `R_x`, `R_y`, `R_z` together.
- `plot_thomas_fermi_radius_velocities(...)`: plots `dR_x/dt`, `dR_y/dt`,
  `dR_z/dt` together.
- `plot_expansion_temperatures(...)`: plots already-computed `T_x`, `T_y`,
  `T_z` together.
- `plot_expansion_temperature_3d(...)`: plots already-computed scalar `T_3D`.

For `T_3D`, color no longer represents a spatial axis because the quantity is
scalar. The main curve is bordeaux and the optional reference curve is royal
blue dashed.

There is intentionally no dedicated lambda or scaling-factor plotting method in
the experiment-facing API.
