# Castin-Dum Model

## Purpose

The Castin-Dum model is the central dynamical model for the condensate scaling
factors. In this project, `CastinDumModel` represents a complete trap
transformation from initial angular frequencies to final angular frequencies.
It describes the time evolution of

```text
lambda_x(t)
lambda_y(t)
lambda_z(t)
```

where each `lambda` is a dimensionless scale factor for the condensate size
along one spatial direction.

## State Convention

The state vector is always ordered as

```text
[lambda_x, lambda_x_dot,
 lambda_y, lambda_y_dot,
 lambda_z, lambda_z_dot]
```

This convention is used by both the direct right-hand side evaluation and the
numerical integration methods.

## Initial State

The canonical equilibrium initial state is

```text
[1, 0, 1, 0, 1, 0]
```

using the same state ordering:

```text
[lambda_x, lambda_x_dot,
 lambda_y, lambda_y_dot,
 lambda_z, lambda_z_dot]
```

Physically, this means

```text
lambda_x(0) = lambda_y(0) = lambda_z(0) = 1
lambda_x_dot(0) = lambda_y_dot(0) = lambda_z_dot(0) = 0
```

The unit scaling factors express that the lambdas are defined relative to the
initial condensate size. The zero derivatives express that the condensate
starts in equilibrium in the initial trap, with no initial expansion or
compression velocity.

This state is part of the Castin-Dum state convention, not an experimental
parameter, so it is not stored in `ExperimentalConstants`.

`integrate()` and `integrate_free()` use this equilibrium state automatically
when no `initial_state` is supplied.

```python
solution = model.integrate(
    t_eval=t,
    omega_x=omega_x,
    omega_y=omega_y,
    omega_z=omega_z,
)
```

A custom initial state can still be provided explicitly:

```python
solution = model.integrate(
    t_eval=t,
    omega_x=omega_x,
    omega_y=omega_y,
    omega_z=omega_z,
    initial_state=[
        1.1, 0.2,
        0.95, -0.1,
        1.05, 0.0,
    ],
)
```

## Trap Frequencies

`CastinDumModel` keeps three distinct frequency concepts separate.

### Initial angular frequencies

`omega_x_initial`, `omega_y_initial`, and `omega_z_initial` are the initial
trap angular frequencies. They appear in the interaction term of the
Castin-Dum equations.

### Instantaneous angular frequencies

`omega_x(t)`, `omega_y(t)`, and `omega_z(t)` are the instantaneous trap angular
frequencies used by `rhs(...)` during the evolution. They may vary in time and
are supplied to `integrate(...)` as callables.

### Final angular frequencies

`omega_x_final`, `omega_y_final`, and `omega_z_final` are the target trap
angular frequencies of the transformation. They are used by
`characteristic_time()` and `final_scaling_factors()`, and they define the
final conditions that will later feed inverse-engineering protocols.

## Castin-Dum Equations

For initial angular trapping frequencies
`\omega_{xi}`, `\omega_{yi}`, and `\omega_{zi}`, and instantaneous trap
frequencies `\omega_x(t)`, `\omega_y(t)`, and `\omega_z(t)`, the implemented
equations are the following.

Calling `CastinDumModel()` without arguments uses the nominal initial trap
angular frequencies defined in `ExperimentalConstants`. Explicit constructor
arguments always override these nominal defaults.

```math
\ddot{\lambda}_x =
\frac{\omega_{xi}^2}{\lambda_x^2 \lambda_y \lambda_z}
- \omega_x(t)^2 \lambda_x
```

```math
\ddot{\lambda}_y =
\frac{\omega_{yi}^2}{\lambda_x \lambda_y^2 \lambda_z}
- \omega_y(t)^2 \lambda_y
```

```math
\ddot{\lambda}_z =
\frac{\omega_{zi}^2}{\lambda_x \lambda_y \lambda_z^2}
- \omega_z(t)^2 \lambda_z
```

## Free Expansion

When the trap is switched off,
`\omega_x(t) = \omega_y(t) = \omega_z(t) = 0`, so the equations become

```math
\ddot{\lambda}_x =
\frac{\omega_{xi}^2}{\lambda_x^2 \lambda_y \lambda_z}
```

```math
\ddot{\lambda}_y =
\frac{\omega_{yi}^2}{\lambda_x \lambda_y^2 \lambda_z}
```

```math
\ddot{\lambda}_z =
\frac{\omega_{zi}^2}{\lambda_x \lambda_y \lambda_z^2}
```

## Equilibrium Check

At the initial equilibrium state,

```text
lambda_x = lambda_y = lambda_z = 1
lambda_x_dot = lambda_y_dot = lambda_z_dot = 0
omega_x(t) = omega_x_initial
omega_y(t) = omega_y_initial
omega_z(t) = omega_z_initial
```

the interaction and trap terms cancel on each axis, giving
`\ddot{\lambda}_x = \ddot{\lambda}_y = \ddot{\lambda}_z = 0`. This is an
important physical check of the implementation.

## Characteristic Time

The historical project used the following characteristic time for a change from
initial to final trap frequencies:

```math
T_{c,i} =
\frac{\left|1/\omega_{if} - 1/\omega_{ii}\right|}{4\sqrt{2}}
```

with

```math
T_c = \max(T_{c,x}, T_{c,y}, T_{c,z})
```

This module preserves that definition. No stronger physical interpretation is
assumed here.

## Analytical Final Scaling Factors

For final angular frequencies `\omega_{xf}`, `\omega_{yf}`, and `\omega_{zf}`,
the analytical final scaling factors are

```math
\lambda_{xf} =
\frac{(\omega_{xi}/\omega_{xf})^{4/5}}
{[(\omega_{yi}/\omega_{yf})(\omega_{zi}/\omega_{zf})]^{1/5}}
```

```math
\lambda_{yf} =
\frac{(\omega_{yi}/\omega_{yf})^{4/5}}
{[(\omega_{xi}/\omega_{xf})(\omega_{zi}/\omega_{zf})]^{1/5}}
```

```math
\lambda_{zf} =
\frac{(\omega_{zi}/\omega_{zf})^{4/5}}
{[(\omega_{xi}/\omega_{xf})(\omega_{yi}/\omega_{yf})]^{1/5}}
```

In the isotropic case, if the same ratio
`r = \omega_initial / \omega_final` applies on all axes, each final scaling
factor is

```math
\lambda_f = r^{2/5}
```

## Numerical Integration

`CastinDumModel.integrate` and `CastinDumModel.integrate_free` use
`scipy.integrate.solve_ivp`. The default method is `RK45`.

The `t_eval` argument is the one-dimensional array of times, in seconds, where
the solution is requested. It must contain at least two strictly increasing
values. The numerical tolerances `rtol` and `atol` are integration settings, not
part of the physical state of the model. The implementation checks
`solution.success` and raises `RuntimeError` with the solver message if the
integration fails.

## Minimal Example

```python
from bec_fast_manipulation.castin_dum import CastinDumModel

model = CastinDumModel()

characteristic_time = model.characteristic_time()

lambda_x_final, lambda_y_final, lambda_z_final = (
    model.final_scaling_factors()
)
```

This uses the nominal historical transformation:

```text
omega_x_initial = 2*pi*500 rad/s
omega_y_initial = 2*pi*600 rad/s
omega_z_initial = 2*pi*700 rad/s
omega_x_final = 2*pi*5 rad/s
omega_y_final = 2*pi*5 rad/s
omega_z_final = 2*pi*60 rad/s
```

For another experimental configuration, pass the initial angular frequencies
and final angular frequencies explicitly:

```python
from math import pi

import numpy as np

from bec_fast_manipulation.castin_dum import CastinDumModel

model = CastinDumModel(
    omega_x_initial=2 * pi * 500,
    omega_y_initial=2 * pi * 600,
    omega_z_initial=2 * pi * 700,
    omega_x_final=2 * pi * 5,
    omega_y_final=2 * pi * 5,
    omega_z_final=2 * pi * 60,
)

t_eval = np.linspace(0.0, 1e-4, 100)

solution = model.integrate(
    t_eval=t_eval,
    omega_x=lambda t: model.omega_x_initial,
    omega_y=lambda t: model.omega_y_initial,
    omega_z=lambda t: model.omega_z_initial,
)
```

## Units

| Quantity | Unit |
| --- | --- |
| time | s |
| omega | rad/s |
| lambda | dimensionless |
| lambda_dot | s^-1 |

## What This Module Does Not Treat

This module does not build retro-sinusoidal profiles, invert scale factors into
trap frequencies, model the optical potential, compute temperatures, or perform
STA optimization. Those responsibilities belong to future dedicated modules.

## Historical Denominator Correction

An old exploratory implementation used an incorrect denominator in the
interaction term. The canonical implementation here uses

```text
omega_xi^2 / (lambda_x^2 lambda_y lambda_z)
```

for the x equation, and the corresponding cyclic expressions for y and z. That
older exploratory expression is not used as a physical reference.
