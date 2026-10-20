"""Meaningful model, arithmetic, segmentation and interface integration checks."""

import unittest
import hashlib
import json
import numpy as np
from PIL import Image
from src.data import ROOT, SPECS, load_data
from src.ml import load_model, predict
from src.advisory import (
    fertilizer_quantity, farm_economics, nutrient_comparison, rainfall_scenarios,
)
from src.vision import SAMPLE_CROPS, analyze_leaf

CROP = dict(N=85, P=45, K=40, temperature=25, humidity=82, ph=6.5, rainfall=220)
YIELD = dict(Crop="Rice", Season="Kharif", State="Punjab", Irrigation_Type="Canal",
             Area_ha=2, Rainfall_mm=800, Temperature_c=27, Fertilizer_kg_ha=150)


class ProjectTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.models = {kind: load_model(kind) for kind in SPECS}

    def test_data_and_no_yield_target_leakage(self):
        for kind in SPECS:
            frame = load_data(kind)
            self.assertGreater(len(frame), 100)
            self.assertNotIn(SPECS[kind]["target"], self.models[kind]["features"])
        self.assertNotIn("Production_tons", self.models["yield"]["features"])

    def test_reference_asset_hashes(self):
        provenance = json.loads((ROOT / "data/provenance.json").read_text(encoding="utf-8"))
        for path, expected in provenance["files_sha256"].items():
            with self.subTest(path=path):
                self.assertEqual(hashlib.sha256((ROOT / path).read_bytes()).hexdigest(), expected)

    def test_classifier_predictions_and_probabilities(self):
        inputs = {
            "crop": CROP,
            "fertilizer": dict(Temperature=28, Humidity=65, Moisture=40, Soil_Type="Loamy",
                               Crop_Type="Rice", Nitrogen=15, Phosphorous=30, Potassium=30),
        }
        for kind, values in inputs.items():
            with self.subTest(kind=kind):
                output = predict(kind, values, self.models[kind])
                self.assertAlmostEqual(output["ranking"]["Model score (%)"].sum(), 100)
                self.assertEqual(output["prediction"], output["ranking"].iloc[0]["Label"])
        self.assertEqual(predict("fertilizer", inputs["fertilizer"], self.models["fertilizer"])["prediction"], "Urea")

    def test_invalid_and_nonfinite_inputs(self):
        for change in [dict(humidity=120), dict(N=-1), dict(N=float("nan")), dict(ph=15), dict(N="abc")]:
            with self.subTest(change=change), self.assertRaises(ValueError):
                predict("crop", {**CROP, **change}, self.models["crop"])
        with self.assertRaises(ValueError):
            predict("crop", {"N": 10}, self.models["crop"])

    def test_unknown_category_and_zero_area(self):
        for change in [dict(Crop="Unknown crop"), dict(Area_ha=0)]:
            with self.subTest(change=change), self.assertRaises(ValueError):
                predict("yield", {**YIELD, **change}, self.models["yield"])

    def test_training_range_warning(self):
        output = predict("crop", {**CROP, "N": 250}, self.models["crop"])
        self.assertTrue(any("N=250" in warning for warning in output["warnings"]))

    def test_model_artifacts_and_reports(self):
        for kind, bundle in self.models.items():
            metrics = bundle["metrics"]
            frame = load_data(kind)
            self.assertEqual(metrics["test_rows"] + metrics["train_rows"], len(frame))
            self.assertEqual(len(metrics["comparison"]), 3)
            self.assertTrue((ROOT / "reports" / f"{kind}_metrics.json").exists())

    def test_economics_units_profit_and_zero_revenue(self):
        result = farm_economics(4, 2, 2000, 30000)
        self.assertEqual(result["production_tons"], 8)
        self.assertEqual(result["revenue"], 160000)
        self.assertEqual(result["cost"], 60000)
        self.assertEqual(result["profit"], 100000)
        self.assertAlmostEqual(result["margin_pct"], 62.5)
        zero = farm_economics(4, 2, 0, 30000)
        self.assertIsNone(zero["margin_pct"])
        self.assertEqual(zero["profit"], -60000)
        with self.assertRaises(ValueError):
            farm_economics(4, 0, 2000, 30000)

    def test_fertilizer_quantities_and_reference_gaps(self):
        result = fertilizer_quantity(125, 1.5)
        self.assertEqual(result["total_kg"], 187.5)
        self.assertEqual(result["bags_equivalent"], 3.75)
        self.assertEqual(result["whole_bags"], 4)
        gaps = nutrient_comparison("Rice", 15, 40, 30)
        self.assertEqual(gaps["Gap"].tolist(), [25, 0, 0])
        with self.assertRaises(ValueError):
            fertilizer_quantity(100, 2, bag_kg=0)

    def test_rainfall_scenario_isolation(self):
        original = YIELD.copy()
        table = rainfall_scenarios(YIELD, self.models["yield"])
        self.assertEqual(YIELD, original)
        self.assertEqual(table["Rainfall (mm)"].tolist(), [600, 800, 1000])
        self.assertTrue((table["Yield (tons/ha)"] > 0).all())

    def test_all_leaf_samples_masks_and_determinism(self):
        percentages = {}
        for filename, crop in SAMPLE_CROPS.items():
            with self.subTest(sample=filename):
                with Image.open(ROOT / "samples" / filename) as picture:
                    result = analyze_leaf(picture, crop)
                    repeat = analyze_leaf(picture, crop)
                self.assertTrue(np.array_equal(result["affected_mask"], repeat["affected_mask"]))
                marked = result["affected_mask"] > 0
                leaf = result["leaf_mask"] > 0
                self.assertFalse((marked & ~leaf).any())
                self.assertTrue(np.array_equal(result["original"][~marked], result["overlay"][~marked]))
                self.assertAlmostEqual(result["affected_percent"], marked.sum() / leaf.sum() * 100)
                self.assertTrue(0 <= result["affected_percent"] <= 100)
                percentages[filename] = result["affected_percent"]
        healthy = percentages.pop("healthy_leaf.png")
        self.assertLess(healthy, 2.5)
        self.assertTrue(all(value > healthy for value in percentages.values()))

    def test_blank_and_tiny_images_rejected(self):
        for picture in [Image.new("RGB", (100, 100), "white"), Image.new("RGB", (8, 8), "green")]:
            with self.assertRaises(ValueError):
                analyze_leaf(picture)


class AppTests(unittest.TestCase):
    def test_all_pages_and_default_submissions(self):
        from streamlit.testing.v1 import AppTest
        app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.error), 0)
        for page in ["Crop recommendation", "Fertilizer advice", "Yield and economics", "Leaf image analysis"]:
            with self.subTest(page=page):
                app.sidebar.radio[0].set_value(page).run()
                self.assertEqual(len(app.exception), 0)
                app.button[0].click().run()
                self.assertEqual(len(app.exception), 0)
                self.assertEqual(len(app.error), 0)
                if page == "Crop recommendation":
                    self.assertEqual(app.session_state["recommended_crop"], "rice")
        sample_widget = next(widget for widget in app.selectbox if widget.label == "Sample image")
        sample_widget.set_value("tomato_early_blight.png").run()
        app.button[0].click().run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.error), 0)
        self.assertGreater(float(app.metric[0].value.rstrip("%")), 2.5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
