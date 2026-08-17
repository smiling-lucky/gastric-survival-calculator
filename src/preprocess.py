from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def apply_preprocessor(data: pd.DataFrame, state: dict[str, Any], scale_continuous: bool) -> pd.DataFrame:
    result = pd.DataFrame(index=data.index)
    continuous = set(state["continuous_features"])
    logged = set(state["log1p_features"])
    for feature in state["predictors"]:
        values = pd.to_numeric(data[feature], errors="coerce").astype(float)
        values = values.fillna(float(state["medians"][feature]))
        if feature in logged:
            values = np.log1p(values.clip(lower=0.0))
        if scale_continuous and feature in continuous:
            values = (values - float(state["means"][feature])) / float(state["scales"][feature])
        result[feature] = values
    return result


def features_to_frame(features: dict[str, float], predictors: list[str]) -> pd.DataFrame:
    return pd.DataFrame([{name: features[name] for name in predictors}])
