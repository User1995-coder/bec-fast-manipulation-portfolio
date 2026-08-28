"""Console presentation helpers for analysis results."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd


class ConsoleReporter:
    """Format and print analysis tables without computing physical results."""

    def __init__(self, width: int = 72) -> None:
        self.width = width

    def header(self, title: str) -> None:
        print("=" * self.width)
        print(title)
        print("=" * self.width)

    def section(self, title: str) -> None:
        print()
        print(title)
        print("-" * min(self.width, max(len(title), 12)))

    def parameters(self, parameters, *, columns: tuple[str, str, str] = ("Parameter", "Value", "Unit")) -> pd.DataFrame:
        rows = []
        if isinstance(parameters, Mapping):
            for name, value in parameters.items():
                if isinstance(value, tuple) and len(value) == 2:
                    rows.append({columns[0]: name, columns[1]: value[0], columns[2]: value[1]})
                else:
                    rows.append({columns[0]: name, columns[1]: value, columns[2]: "-"})
        else:
            for item in parameters:
                if isinstance(item, Mapping):
                    rows.append(dict(item))
                else:
                    name, value, *rest = item
                    rows.append({columns[0]: name, columns[1]: value, columns[2]: rest[0] if rest else "-"})
        return self._print_dataframe(pd.DataFrame(rows, columns=columns))

    def axis_table(
        self,
        data,
        *,
        axis_names: tuple[str, ...] = ("x", "y", "z"),
    ) -> pd.DataFrame:
        if isinstance(data, Mapping):
            rows = {"Axis": list(axis_names)}
            for column, values in data.items():
                value_list = list(values)
                if len(value_list) != len(axis_names):
                    raise ValueError(f"{column} must contain one value per axis.")
                rows[column] = value_list
            dataframe = pd.DataFrame(rows)
        else:
            dataframe = pd.DataFrame(data)
        return self._print_dataframe(dataframe)

    def comparison_table(
        self,
        comparison: Mapping[str, Mapping[str, float]],
        *,
        value_label: str = "Value",
        reference_label: str = "Reference",
        show_ratio: bool = True,
    ) -> pd.DataFrame:
        rows = []
        for axis, values in comparison.items():
            row = {
                "Axis": axis,
                value_label: values["value"],
                reference_label: values["reference"],
            }
            if show_ratio:
                row["Ratio"] = values["ratio"]
            row["Reduction [%]"] = values["reduction_percent"]
            rows.append(row)
        return self._print_dataframe(pd.DataFrame(rows))

    def scalar_comparison(
        self,
        quantity: str,
        comparison: Mapping[str, float],
        *,
        value_label: str = "Value",
        reference_label: str = "Reference",
        show_ratio: bool = True,
    ) -> pd.DataFrame:
        row = {
            "Quantity": quantity,
            value_label: comparison["value"],
            reference_label: comparison["reference"],
        }
        if show_ratio:
            row["Ratio"] = comparison["ratio"]
        row["Reduction [%]"] = comparison["reduction_percent"]
        dataframe = pd.DataFrame([row])
        return self._print_dataframe(dataframe)

    def _print_dataframe(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        formatted = dataframe.astype("object").apply(lambda column: column.map(self._format_value))
        print(formatted.to_string(index=False))
        return dataframe

    @staticmethod
    def _format_value(value) -> str:
        if isinstance(value, str):
            return value
        if value is None:
            return "-"
        if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
            return str(int(value))
        if isinstance(value, (np.floating, float)):
            value_float = float(value)
            if np.isnan(value_float):
                return "nan"
            if value_float == 0.0:
                return "0"
            if abs(value_float) >= 1e4 or abs(value_float) < 1e-3:
                return f"{value_float:.4g}"
            return f"{value_float:.4g}"
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            return "[" + ", ".join(ConsoleReporter._format_value(item) for item in value) + "]"
        return str(value)
