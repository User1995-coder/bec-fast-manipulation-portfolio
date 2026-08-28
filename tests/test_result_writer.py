import json

import numpy as np
import pytest

from bec_fast_manipulation.analysis import ResultWriter


def test_write_json_creates_directory_and_returns_path(tmp_path):
    data_dir = tmp_path / "nested" / "data"
    writer = ResultWriter(data_dir)

    output_path = writer.write_json("results.json", {"value": 1})

    assert output_path == data_dir / "results.json"
    assert output_path.exists()
    with output_path.open(encoding="utf-8") as file:
        assert json.load(file) == {"value": 1}


def test_write_json_serializes_nested_numpy_and_sequence_values(tmp_path):
    writer = ResultWriter(tmp_path)
    data = {
        "metadata": {"name": "demo", "ok": True, "none": None},
        "array": np.array([1.0, 2.0, 3.0]),
        "matrix": np.array([[1, 2], [3, 4]], dtype=np.int64),
        "float": np.float64(1.5),
        "int": np.int64(7),
        "tuple": (np.float32(2.5), [np.int32(3), "x"]),
    }

    output_path = writer.write_json("data.json", data)

    with output_path.open(encoding="utf-8") as file:
        loaded = json.load(file)
    assert loaded == {
        "metadata": {"name": "demo", "ok": True, "none": None},
        "array": [1.0, 2.0, 3.0],
        "matrix": [[1, 2], [3, 4]],
        "float": 1.5,
        "int": 7,
        "tuple": [2.5, [3, "x"]],
    }


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), -float("inf")])
def test_write_json_rejects_non_finite_scalars(tmp_path, bad_value):
    writer = ResultWriter(tmp_path)

    with pytest.raises(ValueError, match="NaN or infinity"):
        writer.write_json("bad.json", {"bad": bad_value})


def test_write_json_rejects_non_finite_arrays(tmp_path):
    writer = ResultWriter(tmp_path)

    with pytest.raises(ValueError, match="NaN or infinity"):
        writer.write_json("bad.json", {"bad": np.array([1.0, np.nan])})


def test_write_json_requires_dictionary_data(tmp_path):
    writer = ResultWriter(tmp_path)

    with pytest.raises(TypeError, match="dictionary"):
        writer.write_json("bad.json", [1, 2, 3])
