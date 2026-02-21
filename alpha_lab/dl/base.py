from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import pandas as pd


@dataclass(slots=True)
class DLContext:
    frame: pd.DataFrame
    factor_frame: pd.DataFrame
    feature_factors: list[str]
    params: dict[str, object]
    output_name: str


class DLPlugin(Protocol):
    name: str

    def generate(self, context: DLContext) -> pd.Series:
        """Generate DL factor values on MultiIndex(datetime, symbol)."""
