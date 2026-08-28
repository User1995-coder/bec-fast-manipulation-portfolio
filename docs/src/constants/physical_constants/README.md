# Physical Constants

## Purpose

`PhysicalConstants` is the single source for fundamental constants and
rubidium-87 atomic data used by the Python models. Keeping these values in one
small namespace avoids silent drift between simulations and makes unit
conventions explicit before the dynamical and experimental models are migrated.

`PhysicalConstants` contains fundamental constants and independent rubidium-87
atomic data. It does not contain laser setup parameters, condensate preparation
parameters, simulation choices, or values derived by a physical model.

## Usage

```python
from bec_fast_manipulation.constants import PhysicalConstants

m = PhysicalConstants.RUBIDIUM_87_MASS
k_B = PhysicalConstants.BOLTZMANN_CONSTANT
```

No instantiation is required.

## Fundamental Constants

| Name | Symbol | Value | Unit | Meaning |
| --- | --- | --- | --- | --- |
| `BOLTZMANN_CONSTANT` | `k_B` | `1.380649e-23` | J/K | Converts temperature to thermal energy. |
| `REDUCED_PLANCK_CONSTANT` | `hbar` | `1.05457180013e-34` | J.s | Quantum of angular action. |
| `SPEED_OF_LIGHT` | `c` | `2.99792458e8` | m/s | Speed of light in vacuum. |
| `STANDARD_GRAVITY` | `g` | `9.80665` | m/s^2 | Standard gravitational acceleration. |

## Rubidium-87 Properties

| Name | Symbol | Value | Unit | Meaning |
| --- | --- | --- | --- | --- |
| `RUBIDIUM_87_MASS` | `m_Rb87` | `1.4431606483768263e-25` | kg | Atomic mass used for rubidium-87. |
| `RUBIDIUM_87_SCATTERING_LENGTH` | `a_s` | `100 * 0.529e-10` | m | Historical s-wave scattering length used by the project. |

## D1 Transition

| Name | Symbol | Value | Unit | Meaning |
| --- | --- | --- | --- | --- |
| `RUBIDIUM_D1_WAVELENGTH` | `lambda_D1` | `7.94978851156e-7` | m | D1 transition wavelength. |
| `RUBIDIUM_D1_ANGULAR_FREQUENCY` | `omega_D1` | `2 * pi * 3.77107463380e14` | rad/s | D1 transition angular frequency. |
| `RUBIDIUM_D1_LINEWIDTH` | `Gamma_D1` | `2 * pi * 5.7500e6` | rad/s | D1 angular linewidth. |

## D2 Transition

| Name | Symbol | Value | Unit | Meaning |
| --- | --- | --- | --- | --- |
| `RUBIDIUM_D2_WAVELENGTH` | `lambda_D2` | `7.80241209686e-7` | m | D2 transition wavelength. |
| `RUBIDIUM_D2_ANGULAR_FREQUENCY` | `omega_D2` | `2 * pi * 3.842304844685e14` | rad/s | D2 transition angular frequency. |
| `RUBIDIUM_D2_LINEWIDTH` | `Gamma_D2` | `2 * pi * 6.0666e6` | rad/s | D2 angular linewidth. |

## Frequency versus Angular Frequency

The historical code stores transition rates as angular frequencies.

```text
f      in Hz
omega  in rad/s
```

A frequency `f` in hertz is converted to an angular frequency with

```text
omega = 2*pi*f
```

This means `RUBIDIUM_D1_LINEWIDTH` and `RUBIDIUM_D2_LINEWIDTH` are in rad/s,
not Hz. The same convention applies to the D1 and D2 transition angular
frequencies.

## Units Convention

All quantities are expressed in SI units.

| Quantity | Unit |
| --- | --- |
| energy per temperature | J/K |
| action | J.s |
| speed | m/s |
| acceleration | m/s^2 |
| mass | kg |
| length | m |
| angular frequency | rad/s |

## What Is Not Included

`PhysicalConstants` deliberately excludes quantities that are not fundamental
physical constants or independent rubidium atomic data.

Condensate preparation parameters such as the nominal atom number, and laser
parameters such as the trapping laser wavelength and beam waists, belong to
`ExperimentalConstants` because they describe the historical setup rather than
universal physics.

The optical potential scale `U0` is excluded because it depends on the laser
and atomic data. It belongs to a derived experimental model rather than to this
constants namespace.

The chemical potential and Thomas-Fermi radii are also excluded. They depend on
the atom number and trap frequencies, so they belong to `ThomasFermiModel`
rather than to this constants module.
