# ConsoleReporter

`ConsoleReporter` organizes, formats, and prints analysis results in the
console. It uses `pandas.DataFrame` for structured tables.

It performs no physical calculation and no numerical comparison. Callers should
compute values with models and, when useful, compute comparison dictionaries
with `StatisticalAnalysis` before passing them to the reporter.

## API

- `header(title)`: prints a top-level title.
- `section(title)`: prints a section title.
- `parameters(...)`: prints a generic parameter table with parameter, value,
  and unit columns.
- `axis_table(...)`: prints a generic axis table. The caller defines the
  columns, so the same method can display radii, velocities, directional
  temperatures, ratios, and reductions.
- `comparison_table(...)`: prints a table directly compatible with
  `StatisticalAnalysis.compare_axis_values(...)`.
- `scalar_comparison(...)`: prints one scalar comparison such as `T_3D` versus
  a reference.

Numeric formatting handles Python and NumPy integers, Python and NumPy floats,
and `nan`. Temperature columns such as `Temperature [nK]` or `Reduction [%]`
are presentation choices made by the caller, not hard-coded physical
conventions.
