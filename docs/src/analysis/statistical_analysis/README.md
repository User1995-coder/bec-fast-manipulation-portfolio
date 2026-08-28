# StatisticalAnalysis

`StatisticalAnalysis` provides generic numerical helpers. It does not import
matplotlib, does not print, does not use pandas for scientific logic, and does
not implement Castin-Dum, Thomas-Fermi, or temperature formulas.

The methods can be applied to radii, velocities, temperatures, or other scalar
analysis outputs that have already been computed by physical models.

## API

- `safe_ratio(numerator, denominator, *, atol=1e-12)`: returns
  `numerator / denominator`, using `nan` when the denominator is zero or too
  close to zero.
- `relative_change(value, reference)`: returns `(value - reference) / reference`.
- `reduction_fraction(value, reference)`: returns `1 - value / reference`.
- `reduction_percent(value, reference)`: returns
  `100 * reduction_fraction(value, reference)`.
- `compare_axis_values(values, references, axis_names=("x", "y", "z"))`:
  returns a dictionary keyed by axis with value, reference, ratio, reduction
  fraction, and reduction percent.
- `compare_scalar_values(value, reference)`: returns the same comparison fields
  for one scalar, for example a `T_3D` comparison.
- `final_axis_values(x_values, y_values, z_values)`: returns the final x, y, z
  values from three one-dimensional arrays.
- `rms(values)`: returns `sqrt(mean(values**2))`.
