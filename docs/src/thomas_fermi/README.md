# Thomas-Fermi Model

## Purpose

`ThomasFermiModel` represents the initial Thomas-Fermi state of the condensate
and converts Castin-Dum scaling outputs into physical radii and radius
velocities.

## Required Physical Parameters

The model uses the nominal atom number from `ExperimentalConstants`:

- `ExperimentalConstants.CONDENSATE_ATOM_NUMBER` as `N`

It uses the following physical constants from `PhysicalConstants`:

- `RUBIDIUM_87_SCATTERING_LENGTH` as `a_s`
- `RUBIDIUM_87_MASS` as `m`
- `REDUCED_PLANCK_CONSTANT` as `hbar`

## Trap Parameters

The default trap angular frequencies are:

- `ExperimentalConstants.INITIAL_TRAP_ANGULAR_FREQUENCY_X`
- `ExperimentalConstants.INITIAL_TRAP_ANGULAR_FREQUENCY_Y`
- `ExperimentalConstants.INITIAL_TRAP_ANGULAR_FREQUENCY_Z`

They are pulsations in rad/s, not frequencies in Hz.

## Geometric Mean Frequency

```text
omega_bar = (omega_x * omega_y * omega_z)^(1/3)
```

`omega_bar` is a geometric mean angular frequency in rad/s.

## Chemical Potential

The historical expression is:

```text
mu = (hbar * omega_bar / 2)
     * (15 * N * a_s * sqrt(m * omega_bar / hbar))^(2/5)
```

The result is in J.

## Initial Thomas-Fermi Radii

```text
R_i0 = sqrt(2 * mu / (m * omega_i^2))
```

for `i = x, y, z`. Radii are returned in m.

## Castin-Dum Scaling

Castin-Dum scaling factors are converted with:

```text
R_i(t) = R_i0 * lambda_i(t)
```

The model accepts scalars or compatible one-dimensional NumPy arrays.

## Radius Velocities

Castin-Dum scaling velocities are converted with:

```text
R_dot_i(t) = R_i0 * lambda_dot_i(t)
```

The result is in m/s.

## Units

All internal quantities use SI units: atom number is dimensionless, angular
frequencies are in rad/s, radii are in m, velocities are in m/s, and energy is
in J.

## Usage

```python
from bec_fast_manipulation.thomas_fermi import ThomasFermiModel

model = ThomasFermiModel()
rx0, ry0, rz0 = model.initial_radii()
rx, ry, rz = model.radii_from_scaling_factors(lambda_x, lambda_y, lambda_z)
```

## Scope

`ThomasFermiModel` does not integrate Castin-Dum equations and does not compute
expansion temperatures.
