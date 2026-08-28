# Shortcut-to-Adiabaticity Optimizer

## Purpose

`STAOptimizer` searches for the fastest feasible retro-sinusoidal protocol for
opening the trap. It optimizes only the protocol parameters and does not run a
direct Castin-Dum integration.

## Optimization Variables

The optimized variables are:

- `final_time`
- `a`
- `b`

All bounds are supplied explicitly by the caller.

## Fixed Boundary Conditions

The initial and final trap pulsations are fixed by the validated model
configuration. The final scaling factors are provided by `CastinDumModel` through
`RetroSinusoidalProtocol`; the optimizer does not duplicate those formulas.

## Objective

The objective is exactly:

```text
minimize final_time
```

No historical composite score or power heuristic is used.

## Constraints

The feasibility constraints are checked over the full normalized interval
`u = t / final_time`:

```text
f_x > 0
f_y > 0
f_z >= minimum_z_frequency_hz
```

The nominal vertical threshold is `50 Hz`. Internally the z constraint is applied
to angular-frequency squared values using:

```text
omega_z_squared >= (2*pi*minimum_z_frequency_hz)^2
```

## Phase Monotonicity

When `require_monotonic_phase=True`, candidates for which
`RetroSinusoidalProtocol.is_phase_monotonic()` is false are not feasible. When
the option is false, phase monotonicity is reported but does not reject a
candidate by itself.

## Continuous Constraint Checking

A fixed grid of 300 points is not used as the sole feasibility guarantee. The
optimizer works in normalized time, samples the interval to locate local minimum
candidates, includes both endpoints, and refines candidate minima with
`scipy.optimize.minimize_scalar`.

## Optimization Method

`STAOptimizer.optimize()` uses `scipy.optimize.differential_evolution` with
nonlinear constraints. This replaces the historical grid search because the
problem has only three variables, may be non-convex, and does not require an
analytical gradient.

## Candidate Evaluation

`evaluate_candidate(final_time, a, b)` evaluates one triplet without launching
the optimizer. It returns minimum angular-frequency squared values, minimum
frequencies in Hz, phase monotonicity, individual constraint flags, and a global
feasibility flag.

## Result

`STAOptimizationResult` stores the optimized triplet, minimum frequencies,
minimum angular-frequency squared values, phase monotonicity, success state,
message, objective value, and function-evaluation count. It does not store full
time trajectories.

## Scope

The class does not perform:

- Castin-Dum direct integration
- Thomas-Fermi calculations
- temperature calculations
- plotting
- console output
- JSON persistence
- experimental power modelling

## Units

Times are in seconds. Internal pulsations are in rad/s. `omega_squared` is in
s^-2. The experimental z threshold is supplied in Hz.
