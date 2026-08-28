# ResultWriter

## Purpose

`ResultWriter` serializes generic numerical results to JSON. It is part of the
analysis infrastructure, but it contains no physics and knows nothing about
Delta Kick, Castin-Dum, Thomas-Fermi radii, or temperature conventions.

## API

```python
from bec_fast_manipulation.analysis import ResultWriter

writer = ResultWriter(data_dir)
path = writer.write_json("results.json", data)
```

`write_json` creates `data_dir` when needed, writes UTF-8 JSON with indentation,
and returns the `Path` of the created file.

## Supported Data

The writer accepts nested dictionaries containing:

- dictionaries;
- lists and tuples;
- `numpy.ndarray`;
- NumPy floats and integers;
- Python `float`, `int`, `str`, `bool`, and `None`.

Values are converted recursively to native JSON-compatible types.

## Finite Values

`NaN` and infinity are rejected. The writer intentionally avoids producing
non-standard JSON numeric values.
