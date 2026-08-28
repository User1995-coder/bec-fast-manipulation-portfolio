# Analysis

This package contains the shared analysis infrastructure for the project:

- `Plotter`: scientific visualization only.
- `StatisticalAnalysis`: generic numerical analysis only.
- `ConsoleReporter`: console presentation only.
- `ResultWriter`: generic JSON serialization only.

These tools do not implement Castin-Dum dynamics, Thomas-Fermi radii, thermal
expansion formulas, or experiment orchestration. Physical models compute
quantities and pass their results to this layer.

The current analysis layer is ready to receive and present:

- Thomas-Fermi radii `R_x(t)`, `R_y(t)`, `R_z(t)`.
- Radius velocities `dR_x/dt`, `dR_y/dt`, `dR_z/dt`.
- Directional expansion temperatures `T_x`, `T_y`, `T_z`.
- Scalar expansion temperature `T_3D`.

`ResultWriter` writes nested numerical dictionaries to readable JSON. It has no
knowledge of any experiment or physical model.
