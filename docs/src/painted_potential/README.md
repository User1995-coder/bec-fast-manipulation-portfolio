# Painted Dipole Potential

## Purpose

This module models crossed painted dipole traps from experimental controls. It
starts from the true Gaussian beam intensity, averages that intensity over a
painting cycle, builds the optical and gravitational potential, finds the local
metastable minimum connected to the trap center, and extracts local trap
frequencies from the Hessian.

## Physical Geometry

The implemented nominal geometry uses two orthogonal Gaussian beams.

Beam 1 propagates along `x`, has horizontal transverse coordinate `y`, vertical
coordinate `z`, and is painted along `y`. Beam 2 propagates along `y`, has
horizontal transverse coordinate `x`, vertical coordinate `z`, and is painted
along `x`.

The Gaussian waists evolve along propagation according to:

```text
w(q) = w0 * sqrt(1 + (q / z_R)^2)
z_R = pi*w0^2/lambda_L
```

## Harmonic Painting Modulation

The normalized modulation is the real branch of:

```text
f - f^3/3 = target
```

on `f in [-1, 1]`. It is extended periodically over a full cycle: the first
half cycle moves from `-1` to `+1`, and the second half returns from `+1` to
`-1`.

## Exact Painted Intensity

The painted intensity is the cycle average of the displaced Gaussian beams:

```text
I_bar = integral_0^1 [I1(r, phase) + I2(r, phase)] dphase
```

The model does not make the Dirac-brush approximation `h >> w0`; the exact
Gaussian profile is integrated numerically.

## Dipole Potential

The optical potential is:

```text
U_dip = C_dip * I_bar
```

`C_dip` is computed from the Rb87 D1/D2 transition data and the laser wavelength.
For the nominal 1064 nm trap laser it is negative, giving an attractive optical
potential at high intensity.

## Gravity

The convention is `z` positive upward:

```text
U_g = m*g*z
```

Gravity has zero Hessian, but it shifts the equilibrium position. In an
anharmonic optical potential this can change the optical Hessian evaluated at
the local total-potential minimum. With gravity, the total potential is not
globally confining because `m*g*z` is unbounded below for `z -> -infinity`.
Positive local curvatures establish local harmonic stability; global or
metastable confinement additionally depends on the finite escape barrier of the
optical potential.

## Trap Minimum and Hessian

The model finds the local minimum near the crossed-beam center. For the nominal
symmetric crossed trap with gravity, the equilibrium search is reduced to one
dimension with `x = y = 0`, and only the vertical equilibrium is solved
numerically. The Hessian is computed by central finite differences with a step
tied to the beam waist scale and symmetrized before diagonalization.

## Trap Frequencies

The local modes are obtained from eigenvalues of `H/m`. Stable modes require
strictly positive eigenvalues. The vertical mode is identified by the largest
projection of its eigenvector on the `z` axis.

## Direct Problem

```text
P, h -> frequencies
```

`PaintedPotentialControl.frequencies_from_controls()` delegates to the direct
potential model.

## Inverse Problem

```text
target frequencies -> P, h
```

`PaintedPotentialControl.controls_from_frequencies()` solves a bounded
least-squares problem with normalized frequency residuals.

## Relation to the Dirac Approximation

The historical analytical Dirac expression is used only as an asymptotic test
reference when `h/w0` is large. It is not the source of truth for this model.

## Scope

The module does not perform:

- Castin-Dum dynamics
- STA optimization
- Thomas-Fermi dynamics
- plotting
- JSON
- experiment orchestration

## Units

All quantities are SI unless a method or field is explicitly named `*_hz`.
