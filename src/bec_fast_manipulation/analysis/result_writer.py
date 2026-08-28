"""Generic JSON result writer for numerical outputs."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np


class ResultWriter:
    """Serialize nested numerical results to readable standard JSON."""

    def __init__(self, data_dir: str | Path) -> None:
        self.data_dir = Path(data_dir)

    def write_json(self, filename: str | Path, data: dict[str, Any]) -> Path:
        """Write data to UTF-8 JSON and return the created path."""
        if not isinstance(data, dict):
            raise TypeError("data must be a dictionary.")
        output_path = self.data_dir / filename
        self.data_dir.mkdir(parents=True, exist_ok=True)
        serializable_data = self._to_json_native(data)
        with output_path.open("w", encoding="utf-8") as file:
            json.dump(
                serializable_data,
                file,
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            )
            file.write("\n")
        return output_path

    @classmethod
    def _to_json_native(cls, value):
        if isinstance(value, dict):
            return {str(key): cls._to_json_native(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [cls._to_json_native(item) for item in value]
        if isinstance(value, np.ndarray):
            if not np.all(np.isfinite(value)):
                raise ValueError("JSON data must not contain NaN or infinity.")
            return cls._to_json_native(value.tolist())
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            return cls._finite_float(float(value))
        if isinstance(value, float):
            return cls._finite_float(value)
        if isinstance(value, (int, str, bool)) or value is None:
            return value
        raise TypeError(f"Unsupported JSON value type: {type(value).__name__}.")

    @staticmethod
    def _finite_float(value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("JSON data must not contain NaN or infinity.")
        return value
