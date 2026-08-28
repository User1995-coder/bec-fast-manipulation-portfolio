# Retro-Sinusoidal Inverse Engineering

## Purpose

The direct Castin-Dum model maps trap angular frequencies to scaling factors:

```text
omega(t) -> lambda(t)
```

The retro-sinusoidal protocol is the inverse-engineering block:

```text
imposed lambda(t) -> omega_squared(t)
```

It is analytical and does not integrate differential equations.

## Phase Profile

With `u = t / tf`,

```text
phi(t) = 2*pi*u*(1 + a*u + b*u**2)/(1 + a + b)
```

## Phase Derivatives

```text
phi_dot(t) = (2*pi/tf)*(1 + 2*a*u + 3*b*u**2)/(1 + a + b)
```

```text
phi_ddot(t) = (2*pi/tf**2)*(2*a + 6*b*u)/(1 + a + b)
```

## Scaling Profile

```text
lambda(t) = lambda_initial
          + (lambda_final - lambda_initial)/(12*pi)
            * (6*phi - 8*sin(phi) + sin(2*phi))
```

The same phase is used on every axis; each axis has its own initial and final
scaling factor.

## Scaling Derivatives

```text
lambda_dot(t) = (lambda_final - lambda_initial)/(12*pi)
              * (6 - 8*cos(phi) + 2*cos(2*phi))
              * phi_dot
```

```text
lambda_ddot(t) = (lambda_final - lambda_initial)/(12*pi)
               * (
                   6*phi_ddot
                   - 8*(phi_ddot*cos(phi) - phi_dot**2*sin(phi))
                   + 2*(phi_ddot*cos(2*phi) - 2*phi_dot**2*sin(2*phi))
                 )
```

## Boundary Conditions

The profile gives, without special endpoint cases:

```text
lambda(0) = lambda_initial
lambda(tf) = lambda_final
lambda_dot(0) = 0
lambda_dot(tf) = 0
lambda_ddot(0) = 0
lambda_ddot(tf) = 0
```

## Phase Monotonicity

`1 + a + b != 0` does not guarantee that the phase is monotone. The relevant
quantity is:

```text
(1 + 2*a*u + 3*b*u**2)/(1 + a + b)
```

on `0 <= u <= 1`.

`require_monotonic_phase=False` is the default because historical searches
explored non-monotone regions. When `require_monotonic_phase=True`, construction
fails unless `phi_dot(t) >= 0` on the full interval.

## Inverse Castin-Dum Equations

```text
omega_x_squared =
    omega_x_initial**2/(lambda_x**3 * lambda_y * lambda_z)
    - lambda_x_ddot/lambda_x
```

```text
omega_y_squared =
    omega_y_initial**2/(lambda_x * lambda_y**3 * lambda_z)
    - lambda_y_ddot/lambda_y
```

```text
omega_z_squared =
    omega_z_initial**2/(lambda_x * lambda_y * lambda_z**3)
    - lambda_z_ddot/lambda_z
```

## Physical Admissibility

`angular_frequency_squared` is the direct result of inverse Castin-Dum
algebra. It is left unclipped so that inadmissible regions remain visible for
diagnostics.

Mathematically, a retro-sinusoidal profile can produce:

```text
omega_squared < 0
```

In this project, this is not interpreted as a negative angular frequency. It
means the protocol would require anti-confining curvature, which is outside the
experimental model retained here.

`angular_frequencies` therefore refuses any requested time grid containing a
genuinely negative `omega_squared` and raises `ValueError`. Tiny negative values
compatible with floating-point roundoff near zero are set to zero only inside
`angular_frequencies` before the square root. `angular_frequency_squared` itself
is never modified.

`is_trapping_protocol` checks the same admissibility rule and returns `True` or
`False` without raising an error for anti-confining profiles.

No silent clipping is allowed:

```text
omega_squared < 0 -> rejected as non-realizable here
```

## Units

```text
phi              rad / dimensionless angle
phi_dot          1/s
phi_ddot         1/s^2
lambda           dimensionless
lambda_dot       1/s
lambda_ddot      1/s^2
omega_squared    1/s^2, equivalently (rad/s)^2
omega            rad/s
```

## Relationship with CastinDumModel

When final scaling factors are omitted, `RetroSinusoidalProtocol` obtains the
nominal final scaling factors from `CastinDumModel().final_scaling_factors()`.
That keeps the equilibrium relation between nominal initial and final trap
frequencies in one place.

## Scope

This block does not perform numerical integration. It does not compute
Thomas-Fermi radii, temperatures, chemical potential, optical power, optical
potential, or any complete experiment.
