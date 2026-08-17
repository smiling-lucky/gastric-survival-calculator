from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from .preprocess import apply_preprocessor, features_to_frame

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "artifacts"
HORIZONS = (365.0, 1095.0, 1825.0)
HORIZON_LABELS = ("1-year", "3-year", "5-year")


@lru_cache(maxsize=1)
def load_bundle() -> dict[str, Any]:
    config = json.loads((ARTIFACTS / "app_config.json").read_text(encoding="utf-8"))
    preprocessor = json.loads((ARTIFACTS / "final_preprocessor.json").read_text(encoding="utf-8"))
    km = pd.read_csv(ARTIFACTS / "km_curve_coordinates.csv")
    landmarks = pd.read_csv(ARTIFACTS / "survival_1_3_5_year_summary.csv")
    model = joblib.load(ARTIFACTS / "final_model.joblib")
    calibrator = joblib.load(ARTIFACTS / "calibrator.joblib")
    demo_cases = json.loads((ARTIFACTS / "demo_cases.json").read_text(encoding="utf-8"))
    features = list(preprocessor["predictors"])
    return {
        "config": config,
        "preprocessor": preprocessor,
        "km": km,
        "landmarks": landmarks,
        "model": model,
        "demo_cases": demo_cases,
        "calibrator": calibrator,
        "center": float(config["calibrator"]["center"]),
        "scale": float(config["calibrator"]["scale"]),
        "features": features,
        "scale_continuous": bool(preprocessor["scale_continuous"]),
        "display_name_map": dict(config["display_name_map"]),
        "cutoff": float(config["risk_grouping"]["cohort_risk_raw_median"]),
    }


def display_names(features: list[str], mapping: dict[str, str]) -> list[str]:
    return [mapping.get(name, name) for name in features]


def transform(features: dict[str, float], bundle: dict[str, Any] | None = None) -> np.ndarray:
    bundle = bundle or load_bundle()
    frame = features_to_frame(features, bundle["features"])
    processed = apply_preprocessor(frame, bundle["preprocessor"], bundle["scale_continuous"])
    return processed[bundle["features"]].to_numpy(float)


def predict_risk(features: dict[str, float], bundle: dict[str, Any] | None = None) -> float:
    bundle = bundle or load_bundle()
    x = transform(features, bundle)
    return float(np.asarray(bundle["model"].predict(x), dtype=float).reshape(-1)[0])


def assign_risk_group(risk: float, bundle: dict[str, Any] | None = None) -> str:
    bundle = bundle or load_bundle()
    return "High risk" if risk >= bundle["cutoff"] else "Low risk"


def calibrated_event_probabilities(risk: float, bundle: dict[str, Any] | None = None) -> dict[str, float]:
    bundle = bundle or load_bundle()
    x = np.array([[(risk - bundle["center"]) / bundle["scale"]]], dtype=float)
    survival_fn = bundle["calibrator"].predict_survival_function(x)[0]
    probs = {}
    for label, horizon in zip(HORIZON_LABELS, HORIZONS):
        probs[label] = float(np.clip(1.0 - float(survival_fn(horizon)), 0.0, 1.0))
    return probs


def individual_survival_curve(
    features: dict[str, float],
    bundle: dict[str, Any] | None = None,
    n_points: int = 366,
) -> pd.DataFrame:
    bundle = bundle or load_bundle()
    x = transform(features, bundle)
    survival_fn = bundle["model"].predict_survival_function(x)[0]
    times = np.unique(np.concatenate([np.linspace(0.0, 1825.0, n_points), np.array(HORIZONS)]))
    times.sort()
    survival = np.clip([float(survival_fn(t)) for t in times], 0.0, 1.0)
    return pd.DataFrame({"Time_days": times, "Years": times / 365.0, "Survival": survival})


def predict_case(features: dict[str, float], bundle: dict[str, Any] | None = None) -> dict[str, Any]:
    bundle = bundle or load_bundle()
    risk = predict_risk(features, bundle)
    group = assign_risk_group(risk, bundle)
    probabilities = calibrated_event_probabilities(risk, bundle)
    curve = individual_survival_curve(features, bundle)
    native_survival = {}
    for label, horizon in zip(HORIZON_LABELS, HORIZONS):
        native_survival[label] = float(curve.loc[(curve["Time_days"] - horizon).abs().idxmin(), "Survival"])
    return {
        "risk_raw": risk,
        "risk_group": group,
        "cutoff": bundle["cutoff"],
        "event_probability": probabilities,
        "survival_probability_calibrated": {k: 1.0 - v for k, v in probabilities.items()},
        "survival_probability_native": native_survival,
        "survival_curve": curve,
        "processed": transform(features, bundle),
        "raw": np.array([[features[name] for name in bundle["features"]]], dtype=float),
    }
