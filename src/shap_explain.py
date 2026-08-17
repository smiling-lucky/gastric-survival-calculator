from __future__ import annotations

from typing import Any

import numpy as np
import shap

from .inference import display_names, load_bundle, transform


def explain_case(features: dict[str, float], bundle: dict[str, Any] | None = None) -> shap.Explanation:
    bundle = bundle or load_bundle()
    x = transform(features, bundle)
    explained = bundle["explainer"](x, max_evals=int(bundle["config"]["shap"]["max_evals"]))
    values = np.asarray(explained.values, dtype=float)
    if values.ndim > 2:
        values = np.squeeze(values)
    base_values = np.asarray(explained.base_values, dtype=float).reshape(-1)
    raw = np.array([[features[name] for name in bundle["features"]]], dtype=float)
    names = display_names(bundle["features"], bundle["display_name_map"])
    return shap.Explanation(
        values=values,
        base_values=base_values,
        data=raw,
        feature_names=names,
    )
