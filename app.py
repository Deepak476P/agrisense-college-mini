"""A single college-demo interface for all four project modules."""

import json
import pandas as pd
import streamlit as st
from PIL import Image, UnidentifiedImageError
from src.data import ROOT, load_data, reference_metadata
from src.ml import load_model, predict, feature_importance
from src.advisory import (
    ACRES_PER_HECTARE, nutrient_comparison, fertilizer_quantity,
    farm_economics, rainfall_scenarios,
)
from src.vision import SAMPLE_CROPS, ILLUSTRATIVE_MATCHES, analyze_leaf

st.set_page_config(page_title="AgriSense Mini", page_icon="🌱", layout="wide")


@st.cache_resource
def models():
    return {kind: load_model(kind) for kind in ("crop", "fertilizer", "yield")}


@st.cache_data
def dataset(kind):
    return load_data(kind)


def show_warnings(result):
    for warning in result["warnings"]:
        st.warning(warning)


def evaluation(kind):
    with st.expander("How this model was evaluated"):
        metrics = MODELS[kind]["metrics"]
        st.write(f"Selected model: **{metrics['selected_model']}**.")
        st.caption(
            f"80/20 split: {metrics['train_rows']} training rows and {metrics['test_rows']} test rows. "
            "Three-fold validation on training data selected the model. All data is synthetic."
        )
        if kind == "yield":
            st.write(f"Held-out MAE: **{metrics['mae']:.3f} tons/ha** · R²: **{metrics['r2']:.3f}**")
        else:
            st.write(f"Held-out accuracy: **{metrics['accuracy']:.2%}** · Macro F1: **{metrics['macro_f1']:.3f}**")
        st.dataframe(pd.DataFrame(metrics["comparison"]), hide_index=True)
        st.caption(metrics["cv_metric"])
        st.bar_chart(feature_importance(MODELS[kind]).head(10).set_index("Feature"))
        st.caption("Feature importance describes this model's splits; it does not establish cause and effect.")


def crop_default(choices, fallback="Rice"):
    recommended = st.session_state.get("recommended_crop", "").title()
    return choices.index(recommended if recommended in choices else fallback)


def area_inputs(key):
    left, right = st.columns(2)
    area = left.number_input("Farm area", min_value=0.01, max_value=10000.0, value=2.0, key=f"{key}_area")
    unit = right.selectbox("Area unit", ["Hectares", "Acres"], key=f"{key}_unit")
    return area if unit == "Hectares" else area / ACRES_PER_HECTARE


def overview():
    st.title("AgriSense Mini")
    st.write("A college project connecting machine learning, image processing and a simple farming dashboard.")
    a, b, c, d = st.columns(4)
    a.metric("Features", 4)
    b.metric("Trained models", 3)
    c.metric("Synthetic records", "5,900")
    d.metric("Learning notebooks", 5)
    st.info("Educational demonstration. Results show what the supplied synthetic data and image rules produce.")
    st.subheader("Explore the complete mini project")
    st.dataframe(pd.DataFrame([
        ["Crop recommendation", "7 soil/climate inputs", "Crop rankings, scores and reference crop information"],
        ["Fertilizer advice", "Soil, crop and nutrient inputs", "Product ranking, reference nutrient gaps, quantity arithmetic"],
        ["Yield and economics", "Crop, season, state, irrigation and inputs", "Yield, production, sample profit and rainfall scenarios"],
        ["Leaf image analysis", "An uploaded image or an illustrated sample", "Leaf mask, colour-change mask, overlay and affected-area estimate"],
    ], columns=["Module", "Input", "Output"]), hide_index=True, width="stretch")
    st.subheader("How to study it")
    st.write(
        "Start with `notebooks/00_project_guide.ipynb`, then open notebooks 01–04 in order. "
        "Each includes what/why explanations, executable examples, outputs, limitations and viva questions."
    )
    st.caption(
        "This partial implementation covers every main module. Field validation, real disease classification, "
        "live weather, current prices, accounts and production deployment remain future work."
    )
    st.subheader("What was improved from the reference")
    st.write(
        "Measured metrics replace fallback scores. Leaf results are deterministic. Invalid inputs are rejected, "
        "and inputs beyond training ranges are flagged. Finance uses editable assumptions. "
        "Nutrient numbers are not silently transferred between incompatible datasets."
    )


def crop_page():
    st.title("Crop recommendation")
    st.caption("22 crop labels · 7 numerical features · scores describe the fitted model, not guaranteed crop suitability.")
    with st.form("crop_form"):
        left, right = st.columns(2)
        n = left.number_input("Nitrogen (dataset units)", 0.0, 300.0, 85.0)
        p = left.number_input("Phosphorus (dataset units)", 0.0, 300.0, 45.0)
        k = left.number_input("Potassium (dataset units)", 0.0, 300.0, 40.0)
        ph = left.number_input("Soil pH", 0.0, 14.0, 6.5)
        temp = right.number_input("Temperature (°C)", -50.0, 60.0, 25.0)
        humidity = right.number_input("Humidity (%)", 0.0, 100.0, 82.0)
        rain = right.number_input("Rainfall (reference mm; period unspecified)", 0.0, 3000.0, 220.0)
        submitted = st.form_submit_button("Recommend crop", type="primary")
    if submitted:
        inputs = dict(N=n, P=p, K=k, ph=ph, temperature=temp, humidity=humidity, rainfall=rain)
        result = predict("crop", inputs, MODELS["crop"])
        st.session_state["recommended_crop"] = result["prediction"]
        show_warnings(result)
        st.success(f"Highest-ranked crop: {result['prediction'].title()}")
        top = result["ranking"].head(3)
        st.dataframe(top, hide_index=True)
        st.bar_chart(top.set_index("Label"))
        st.caption("Scores are uncalibrated class probabilities. An unfamiliar input can still receive a high score.")
        meta = reference_metadata()["CROP_METADATA"].get(result["prediction"], {})
        with st.expander("Crop information from the supplied reference"):
            st.caption("Illustrative source text; regional suitability and values have not been independently verified.")
            for field in ("category", "season", "water_need", "growth_duration", "optimal_soil", "description"):
                st.write(f"**{field.replace('_', ' ').title()}:** {meta.get(field, 'Not provided')}")
        payload = {"inputs": inputs, "top_three": top.to_dict(orient="records"), "warnings": result["warnings"]}
        st.download_button("Download crop result", json.dumps(payload, indent=2), "crop_result.json", "application/json")
    evaluation("crop")


def fertilizer_page():
    st.title("Fertilizer advice")
    st.caption("Synthetic rule-labelled data · product classification and transparent quantity arithmetic.")
    frame = dataset("fertilizer")
    crops = sorted(frame["Crop_Type"].unique())
    st.info(
        "Crop selection can carry over when supported. Enter nutrient values for this dataset separately. "
        "Reference nutrient gaps do not determine a field application dose."
    )
    with st.form("fertilizer_form"):
        left, right = st.columns(2)
        crop = left.selectbox("Crop", crops, index=crop_default(crops), key="fert_crop")
        soil = left.selectbox("Soil type", sorted(frame["Soil_Type"].unique()))
        temp = left.number_input("Temperature (°C)", -50.0, 60.0, 28.0, key="fert_temp")
        humidity = left.number_input("Humidity (%)", 0.0, 100.0, 65.0, key="fert_humidity")
        moisture = left.number_input("Moisture (reference scale, 0–100)", 0.0, 100.0, 40.0)
        n = right.number_input("Nitrogen (fertilizer dataset units)", 0.0, 200.0, 15.0)
        p = right.number_input("Phosphorus (fertilizer dataset units)", 0.0, 200.0, 30.0)
        k = right.number_input("Potassium (fertilizer dataset units)", 0.0, 200.0, 30.0)
        rate = right.number_input("Assumed application rate (kg/ha)", 0.0, 1000.0, 100.0)
        st.caption("The rate above is an editable arithmetic example, not a model recommendation.")
        area = area_inputs("fert")
        submitted = st.form_submit_button("Explore fertilizer advice", type="primary")
    if submitted:
        values = dict(Crop_Type=crop, Soil_Type=soil, Temperature=temp, Humidity=humidity,
                      Moisture=moisture, Nitrogen=n, Phosphorous=p, Potassium=k)
        result = predict("fertilizer", values, MODELS["fertilizer"])
        show_warnings(result)
        st.success(f"Predicted dataset label: {result['prediction']}")
        st.dataframe(result["ranking"].head(3), hide_index=True)
        st.caption("These model scores are uncalibrated. The selected tree may produce 100% scores.")
        nutrients = nutrient_comparison(crop, n, p, k)
        st.subheader("Illustrative nutrient comparison")
        st.dataframe(nutrients, hide_index=True)
        st.bar_chart(nutrients.set_index("Nutrient")[["Input", "Illustrative target"]])
        quantity = fertilizer_quantity(rate, area)
        a, b, c = st.columns(3)
        a.metric("Quantity at your assumed rate", f"{quantity['total_kg']:.1f} kg")
        b.metric("50 kg bag equivalents", f"{quantity['bags_equivalent']:.2f}")
        c.metric("Whole bags rounded up", quantity["whole_bags"])
        with st.expander("Fertilizer reference notes"):
            st.caption("Copied educational notes from the supplied project, including unverified application examples.")
            guide = reference_metadata()["FERTILIZER_GUIDE"].get(result["prediction"], {})
            for key, value in guide.items():
                st.write(f"**{key.replace('_', ' ').title()}:** {value}")
    evaluation("fertilizer")


def yield_page():
    st.title("Yield and economics")
    st.caption("Forecast a synthetic dataset value, then explore editable financial assumptions.")
    frame = dataset("yield")
    crops = sorted(frame["Crop"].unique())
    with st.form("yield_form"):
        left, right = st.columns(2)
        crop = left.selectbox("Crop", crops, index=crop_default(crops), key="yield_crop")
        season = left.selectbox("Season", sorted(frame["Season"].unique()))
        state = left.selectbox("State", sorted(frame["State"].unique()))
        irrigation = left.selectbox("Irrigation", sorted(frame["Irrigation_Type"].unique()))
        rain = right.number_input("Rainfall (mm, yield dataset)", 0.0, 5000.0, 800.0)
        temp = right.number_input("Temperature (°C)", -50.0, 60.0, 27.0, key="yield_temp")
        fertilizer = right.number_input("Fertilizer input (kg/ha)", 0.0, 1000.0, 150.0)
        area = area_inputs("yield")
        st.markdown("**Sample financial assumptions — change these for your scenario**")
        a, b = st.columns(2)
        price = a.number_input("Assumed selling price (₹/quintal)", 0.0, 100000.0, 2000.0)
        cost = b.number_input("Assumed cultivation cost (₹/ha)", 0.0, 1000000.0, 30000.0)
        submitted = st.form_submit_button("Estimate yield", type="primary")
    if submitted:
        values = dict(Crop=crop, Season=season, State=state, Irrigation_Type=irrigation,
                      Area_ha=area, Rainfall_mm=rain, Temperature_c=temp, Fertilizer_kg_ha=fertilizer)
        result = predict("yield", values, MODELS["yield"])
        show_warnings(result)
        economics = farm_economics(result["prediction"], area, price, cost)
        a, b, c = st.columns(3)
        a.metric("Estimated yield", f"{result['prediction']:.2f} tons/ha")
        b.metric("Estimated production", f"{economics['production_tons']:.2f} tons")
        c.metric("Yield per acre", f"{economics['yield_quintals_acre']:.2f} quintals")
        a, b, c = st.columns(3)
        a.metric("Sample revenue", f"₹{economics['revenue']:,.0f}")
        b.metric("Sample cost", f"₹{economics['cost']:,.0f}")
        c.metric("Sample net profit / loss", f"₹{economics['profit']:,.0f}")
        margin = economics["margin_pct"]
        st.write("Profit margin: undefined at zero revenue." if margin is None else f"Sample profit margin: {margin:.1f}%")
        st.caption("Prices and costs are user assumptions, not current market quotations. Forecast errors affect every total.")
        scenarios = rainfall_scenarios(values, MODELS["yield"])
        st.subheader("Rainfall sensitivity")
        st.dataframe(scenarios, hide_index=True)
        st.bar_chart(scenarios.set_index("Scenario")[["Yield (tons/ha)"]])
        st.caption(
            "Only rainfall changes in this model exercise; these are not causal climate forecasts. "
            "The table flags scenarios containing values beyond training ranges."
        )
    evaluation("yield")


def leaf_page():
    st.title("Leaf image analysis")
    st.caption("Image-processing demonstration · colour rules, masks and an illustrative crop-specific match.")
    st.info(
        "This module measures colour changes. It cannot confirm a disease or distinguish disease from lighting, "
        "ageing or nutrient stress. No diagnostic confidence percentage is generated."
    )
    mode = st.radio("Image source", ["Included illustration", "Upload a leaf photo"], horizontal=True)
    source = None
    if mode == "Included illustration":
        filename = st.selectbox("Sample image", list(SAMPLE_CROPS))
        source = ROOT / "samples" / filename
        default_crop = SAMPLE_CROPS[filename]
    else:
        source = st.file_uploader("Leaf image (PNG or JPEG)", type=["png", "jpg", "jpeg"])
        default_crop = "Unspecified"
    crops = ["Unspecified", *ILLUSTRATIVE_MATCHES]
    crop = st.selectbox("Crop hint", crops, index=crops.index(default_crop), key=f"leaf_hint_{default_crop}")
    st.caption("Use one visible green leaf on a plain light background. Included samples are drawn illustrations.")
    if st.button("Analyze leaf", type="primary"):
        if source is None:
            st.warning("Choose an image first.")
            return
        try:
            with Image.open(source) as picture:
                result = analyze_leaf(picture, crop)
        except (UnidentifiedImageError, ValueError, OSError, Image.DecompressionBombError) as error:
            st.error(str(error))
            return
        a, b, c = st.columns(3)
        a.metric("Marked leaf area", f"{result['affected_percent']:.2f}%")
        b.metric("Illustrative severity band", result["severity_band"])
        c.metric("Connected marked regions", result["regions"])
        st.write(f"**Rule-based comparison:** {result['illustrative_match']}")
        a, b = st.columns(2)
        a.image(result["original"], caption="Input image", width="stretch")
        b.image(result["overlay"], caption="Marked colour changes", width="stretch")
        with st.expander("See the masks and calculation"):
            a, b = st.columns(2)
            a.image(result["leaf_mask"], caption="Leaf mask", width="stretch")
            b.image(result["affected_mask"], caption="Marked-area mask", width="stretch")
            st.code(
                f"marked pixels / leaf pixels × 100\n"
                f"{result['affected_pixels']} / {result['leaf_pixels']} × 100 = {result['affected_percent']:.2f}%"
            )
        st.write(
            "For the project report, compare the overlay with the source illustration and discuss missed or "
            "incorrectly marked areas. For an actual plant, diagnosis and treatment require qualified local advice."
        )


with st.sidebar:
    st.title("🌱 AgriSense Mini")
    st.caption("College learning project")
    page = st.radio("Open module", [
        "Project overview", "Crop recommendation", "Fertilizer advice",
        "Yield and economics", "Leaf image analysis",
    ])
    if "recommended_crop" in st.session_state:
        st.caption(f"Last crop result: {st.session_state['recommended_crop'].title()}")
    st.divider()
    st.caption("Three ML models · one image-processing module · five explained notebooks")

try:
    MODELS = models()
except FileNotFoundError as error:
    st.error(str(error))
    st.stop()

PAGES = {
    "Project overview": overview,
    "Crop recommendation": crop_page,
    "Fertilizer advice": fertilizer_page,
    "Yield and economics": yield_page,
    "Leaf image analysis": leaf_page,
}
try:
    PAGES[page]()
except ValueError as error:
    st.error(str(error))
