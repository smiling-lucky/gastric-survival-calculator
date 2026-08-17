from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.inference import assign_risk_group, load_bundle, predict_case, transform
from src.preprocess import apply_preprocessor, features_to_frame


class PreprocessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.bundle = load_bundle()
        cls.state = cls.bundle["preprocessor"]

    def test_log1p_ldh_and_lymphocyte_without_scaling(self) -> None:
        raw = {
            "RS": 0.0,
            "IHC": 1.0,
            "BMI": 21.0,
            "Albumin": 37.0,
            "pTNM": 3,
            "PLT": 214.5,
            "LDH": 191.0,
            "Size": 4.0,
            "Lymphocyte": 1.5,
            "Hb": 129.0,
        }
        processed = apply_preprocessor(
            features_to_frame(raw, self.state["predictors"]),
            self.state,
            scale_continuous=False,
        )
        self.assertAlmostEqual(float(processed.loc[0, "LDH"]), float(np.log1p(191.0)))
        self.assertAlmostEqual(float(processed.loc[0, "Lymphocyte"]), float(np.log1p(1.5)))
        self.assertAlmostEqual(float(processed.loc[0, "BMI"]), 21.0)
        self.assertFalse(self.state["scale_continuous"])

    def test_missing_filled_with_training_median(self) -> None:
        raw = {name: np.nan for name in self.state["predictors"]}
        processed = apply_preprocessor(
            features_to_frame(raw, self.state["predictors"]),
            self.state,
            scale_continuous=False,
        )
        self.assertAlmostEqual(float(processed.loc[0, "Albumin"]), float(self.state["medians"]["Albumin"]))
        self.assertAlmostEqual(float(processed.loc[0, "pTNM"]), float(self.state["medians"]["pTNM"]))


class ReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.bundle = load_bundle()
        cls.demo = json.loads((ROOT / "artifacts" / "demo_cases.json").read_text(encoding="utf-8"))

    def test_replay_three_training_patients(self) -> None:
        for case in self.demo:
            result = predict_case(case["features"], self.bundle)
            self.assertAlmostEqual(result["risk_raw"], case["expected_risk_raw"], places=8)
            for label in ("1-year", "3-year", "5-year"):
                self.assertAlmostEqual(
                    result["event_probability"][label],
                    case["expected_probability"][label],
                    places=8,
                    msg=f"{case['label']} {label}",
                )

    def test_risk_group_uses_final_model_median(self) -> None:
        self.assertEqual(assign_risk_group(1.1467812496738907, self.bundle), "Low risk")
        self.assertEqual(assign_risk_group(26.09812029898042, self.bundle), "High risk")
        self.assertEqual(assign_risk_group(71.49442586818286, self.bundle), "High risk")

    def test_processed_feature_order_matches_model(self) -> None:
        case = self.demo[0]["features"]
        x = transform(case, self.bundle)
        self.assertEqual(x.shape, (1, 10))
        self.assertEqual(self.bundle["features"], list(self.bundle["preprocessor"]["predictors"]))


class SurvivalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.bundle = load_bundle()

    def test_km_coordinates_monotone_and_bounded(self) -> None:
        km = self.bundle["km"]
        for group_name, subset in km.groupby("Risk_group"):
            ordered = subset.sort_values("Time")
            survival = ordered["Survival"].to_numpy(float)
            self.assertTrue(np.all(survival >= 0.0) and np.all(survival <= 1.0), group_name)
            self.assertTrue(np.all(np.diff(survival) <= 1e-12), group_name)
            self.assertAlmostEqual(float(survival[0]), 1.0)

    def test_individual_curve_monotone_and_bounded(self) -> None:
        demo = json.loads((ROOT / "artifacts" / "demo_cases.json").read_text(encoding="utf-8"))
        curve = predict_case(demo[0]["features"], self.bundle)["survival_curve"]
        survival = curve["Survival"].to_numpy(float)
        self.assertTrue(np.all(survival >= 0.0) and np.all(survival <= 1.0))
        self.assertTrue(np.all(np.diff(survival) <= 1e-12))


if __name__ == "__main__":
    unittest.main()
