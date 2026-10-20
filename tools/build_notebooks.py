"""Build the five teaching notebooks from explicit Markdown and Python cells."""

from pathlib import Path
import textwrap
import nbformat as nb

ROOT = Path(__file__).resolve().parents[1]


def md(text):
    return nb.v4.new_markdown_cell(textwrap.dedent(text).strip())


def code(text):
    return nb.v4.new_code_cell(textwrap.dedent(text).strip())


def lesson(title, what, why, python, interpretation):
    return [
        md(f"## {title}\n\n**What we do:** {what}\n\n**Why we do it:** {why}"),
        code(python),
        md(f"**Read the output:** {interpretation}"),
    ]


SETUP = """
from pathlib import Path
import sys
ROOT = Path.cwd()
if not (ROOT / "src").is_dir():
    ROOT = ROOT.parent
if not (ROOT / "src").is_dir():
    raise RuntimeError("Open this notebook from the project folder or its notebooks folder.")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
from src.data import SPECS, load_data
plt.rcParams.update({"figure.figsize": (9, 4), "axes.spines.top": False, "axes.spines.right": False})
print("Project:", ROOT.name)
"""


def setup():
    return lesson(
        "Environment and imports",
        "Locate the project and import the libraries used in this notebook.",
        "Relative paths let this project work after it is moved or cloned. pandas handles tables, "
        "NumPy handles arrays, and Matplotlib draws graphs. The src package contains shared, readable code.",
        SETUP,
        "The printed folder should be agrisense-college-mini. In VS Code select the project's .venv Python "
        "environment as the notebook kernel. Run cells from top to bottom.",
    )


def write(name, cells):
    notebook = nb.v4.new_notebook(cells=cells)
    notebook.metadata = {
        "kernelspec": {"display_name": "Python 3 (AgriSense)", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    }
    nb.validate(notebook)
    nb.write(notebook, ROOT / "notebooks" / name)


def guide():
    cells = [md("""
    # AgriSense Mini: complete college-project guide

    **Project type:** a partial educational implementation of the supplied AgriSense AI project.
    **Coverage:** all four major features, simplified so a student can understand and present them.
    **Audience:** a beginner who knows Python variables, functions, lists and basic tables.

    ## Abstract
    Agriculture applications often combine soil data, weather inputs and images. This project demonstrates
    how three supervised machine-learning models and one image-processing algorithm can be presented
    in a single Streamlit application. It uses the supplied project's synthetic datasets and illustrated
    sample leaves. Its purpose is to teach the development process, not to establish field-ready accuracy.

    ## Problem statement and objectives
    Build a small application that accepts inputs, produces explainable demonstration outputs, and
    documents the complete path from raw data to evaluation and interface. We will learn classification,
    regression, categorical encoding, honest evaluation, image masks and basic financial arithmetic.

    ## All features and what they mean
    | Feature | Implemented output | Method and boundary |
    |---|---|---|
    | Crop recommendation | Top crop and three ranked choices | Classifier on seven soil/climate values |
    | Crop information | Season, growth duration, water and soil notes | Unverified reference metadata, not model-generated knowledge |
    | Fertilizer recommendation | Predicted product and ranked labels | Classifier trained on synthetic rule-generated labels |
    | Nutrient comparison | N/P/K input, reference target, gap and status | Explicit arithmetic in undocumented dataset units |
    | Fertilizer quantity | Total kg, 50 kg bag equivalents and rounded whole bags | User-specified rate; no automatic field dosage |
    | Fertilizer reference notes | Formulation, application examples and organic alternatives | Source notes, not independently validated advice |
    | Yield prediction | Tons/ha, quintals/acre and total production | Regression plus unit conversion |
    | Farm economics | Revenue, cost, profit/loss and margin | Editable assumed price and cost |
    | Rainfall scenarios | Outputs at -25%, entered, +25% rainfall | Model sensitivity, not a causal climate forecast |
    | Leaf input | Five illustrations or PNG/JPEG upload | PIL image reading and orientation handling |
    | Leaf segmentation | Leaf mask, marked mask and colour overlay | Deterministic OpenCV thresholds |
    | Leaf area estimate | Marked pixels / leaf pixels, severity band and region count | Estimated colour change, not confirmed infection |
    | Leaf comparison | Crop-specific example description | Explicit rule-based illustration, no diagnostic confidence |
    | Evaluation | Classification reports, confusion matrices, regression errors | Actual held-out results on synthetic data |
    | Learning support | Five notebooks, source comments and viva answers | What, why, code, output interpretation |

    ## Deliberate simplifications
    Real disease classification and treatment prescriptions require labelled photographs, validation and
    domain expertise; they are future extensions. The source's randomly generated disease-confidence
    values are not reproduced. No live weather API, live market feed, login, database or cloud deployment
    is needed. Unsupported crop categories are not silently mapped to another crop. Crop choices can
    carry forward where supported, but nutrient numbers stay separate because their dataset scales differ.

    ## Read these notebooks in order
    1. **00_project_guide.ipynb** — architecture, scope, workflow and project presentation.
    2. **01_crop_recommendation.ipynb** — classification from seven numerical inputs.
    3. **02_fertilizer_advice.ipynb** — category encoding, classifier and transparent reference calculations.
    4. **03_yield_and_economics.ipynb** — regression, leakage prevention, unit conversion and assumptions.
    5. **04_leaf_image_analysis.ipynb** — pixels, colour spaces, masks, severity and failure cases.
    """)]
    cells += setup()
    cells += lesson(
        "Inspect the three datasets",
        "Load every dataset and count its rows, columns and targets.",
        "An inventory establishes exactly what data is available before any model is built.",
        """
        inventory = []
        for kind, spec in SPECS.items():
            df = load_data(kind)
            inventory.append({
                "Module": kind, "Rows": len(df), "Columns": len(df.columns),
                "Task": spec["task"], "Target": spec["target"],
            })
        display(pd.DataFrame(inventory))
        """,
        "There are 2,200 crop, 1,200 fertilizer and 2,500 yield rows: 5,900 records in total. "
        "A CSV row is a generated example, not a verified farm observation.",
    )
    cells += [md("""
    ## Architecture and file responsibilities
    ```text
    CSV data -> inspect/check -> split -> encode -> compare -> train -> evaluate -> saved model
                                                                              |
    user inputs -> validate ------------------------------------------------> prediction -> app result

    leaf image -> resize -> HSV colour values -> leaf mask -> marked mask -> overlay + area estimate
    ```

    | File/folder | What it does | Why it exists |
    |---|---|---|
    | `app.py` | Five-page Streamlit interface | Connects all four modules for demonstration |
    | `src/data.py` | Dataset schemas, paths and input validation | Keeps feature definitions consistent |
    | `src/ml.py` | Pipelines, selection, training, prediction, importance | Shares model logic between notebooks and app |
    | `src/advisory.py` | Reference gaps, quantities, conversions and economics | Separates arithmetic assumptions from ML |
    | `src/vision.py` | Deterministic leaf segmentation | Shows what image processing can and cannot measure |
    | `train_models.py` | Retrains the three models | Reproduces model artifacts and metric reports |
    | `models/` | Locally trained joblib bundles | Allows predictions without retraining |
    | `data/` | CSVs, reference text and provenance | Preserves input data and its origin |
    | `samples/` | Five illustrated leaves | Gives a repeatable image demonstration |
    | `reports/` | Measured metrics and verification records | Keeps evidence separate from claims |
    | `tests/` | Integration and mathematical checks | Detects regressions and incorrect edge cases |
    | `tools/` | Builds/executes notebooks; prepares selected reference assets | Makes setup and documentation reproducible |
    | `.vscode/` | Interpreter, debugger and extension suggestions | Makes the project easy to open in VS Code |

    ## Stepwise implementation
    1. Establish scope; copy only the three datasets, sample images and selected reference text.
    2. Inspect data quality and declare inputs/targets; document synthetic-data limitations.
    3. Split data 80/20. Compare candidates by three-fold validation on training data only.
    4. Fit the selected candidate on training data; evaluate once on held-out data.
    5. Implement explicit arithmetic and a deterministic image-processing module.
    6. Connect the modules to forms, validation, charts and clear result labels.
    7. Execute the notebooks, run integration checks and demonstrate the complete workflow.

    ## Essential concepts
    **Feature:** an input column. **Target/label:** the answer a supervised model learns to predict.
    **Classification:** selecting a category. **Regression:** estimating a number.
    **Training:** fitting the model from examples. **Inference:** using a fitted model on a new input.
    **Validation:** choosing among candidates without using the final test data.
    **Test set:** held-out examples used to measure the selected model's performance.
    **Overfitting:** learning details that do not generalize. **Leakage:** using information unavailable
    at prediction time, such as the answer or a direct derivative of it.
    **Random seed:** makes random splits and model construction reproducible within the tested environment.
    """)]
    cells += lesson(
        "Read measured model results",
        "Load the locally trained bundles and summarize the saved test metrics.",
        "The report must contain measured values. High scores on synthetic data do not prove field reliability.",
        """
        from src.ml import load_model
        summary = []
        for kind in SPECS:
            metrics = load_model(kind)["metrics"]
            summary.append({
                "Module": kind, "Selected model": metrics["selected_model"],
                "Test rows": metrics["test_rows"],
                "Accuracy": metrics.get("accuracy", np.nan),
                "Macro F1": metrics.get("macro_f1", np.nan),
                "MAE (tons/ha)": metrics.get("mae", np.nan),
                "R2": metrics.get("r2", np.nan),
            })
        display(pd.DataFrame(summary).round(4))
        """,
        "Blank entries mean a metric is not applicable. Accuracy is for classification; MAE and R² are for regression. "
        "The leaf module has no measured disease-classification accuracy because it is not a trained disease model.",
    )
    cells += [md("""
    ## How to run in VS Code
    Open this project folder, select `.venv\\Scripts\\python.exe` as the Python interpreter, and choose it
    as the notebook kernel. Start here, then use **Run All** in each notebook. The Python and Jupyter
    extensions are recommended in `.vscode/extensions.json`.

    In the terminal:
    ```powershell
    .\\.venv\\Scripts\\python.exe train_models.py
    .\\.venv\\Scripts\\python.exe -m streamlit run app.py --server.address 127.0.0.1
    ```
    Alternatively use the VS Code **Run AgriSense app** debug configuration. The README contains clean-machine setup.

    ## Suggested demonstration sequence
    1. Introduce the problem and explain why the dataset is synthetic.
    2. Show a crop dataset row, its input features, and the train/test split.
    3. Run a crop prediction and explain its model score.
    4. Open fertilizer advice, compare reference gaps and edit the assumed rate.
    5. Open yield, change assumed price/cost and explain a negative-profit result.
    6. Compare a healthy leaf illustration with a spotted one; show both masks.
    7. Finish with measured results, limitations and future work.

    ## Viva questions with short answers
    **Why use Python?** Its table, ML, plotting and image-processing libraries fit this small project.

    **Why Streamlit?** It creates forms, charts and result displays directly from Python.

    **Why not train on all rows before measuring accuracy?** That would remove an independent test set.

    **Why might the fertilizer model reach 100%?** Its synthetic labels follow simple rules that a tree can reproduce.

    **Why not use deep learning for the leaf module?** The supplied archive has illustrated samples and no
    labelled photographic training dataset. A small thresholding demonstration is explainable with this evidence.

    **What would make the project ready for real use?** Verified data with units and provenance, geographic and
    temporal evaluation, labelled image training data, calibrated uncertainty and expert agronomic validation.

    ## Source and attribution
    Reference: user-supplied `agri-ai-suite.zip` (AgriSense AI). Dataset and sample-image hashes are in
    `data/provenance.json`. Source crop/fertilizer descriptions are clearly marked. The original serialized
    models were not reused. The archive did not contain a license file; confirm redistribution rights before
    publishing its data/assets publicly.
    """)]
    write("00_project_guide.ipynb", cells)


def data_lessons(kind):
    spec = {"crop": "crop", "fertilizer": "fertilizer", "yield": "yield"}[kind]
    cells = setup()
    cells += lesson(
        "Load and inspect data",
        "Read the CSV, preview its rows, check column types and summarize numerical values.",
        "Inspecting real file contents avoids guessing the schema. load_data also rejects missing values, "
        "duplicate rows and non-finite numerical values instead of silently changing them.",
        f"""
        KIND = "{spec}"
        df = load_data(KIND)
        spec = SPECS[KIND]
        display(df.head())
        display(df.dtypes.to_frame("Data type"))
        display(df.describe().round(2))
        print("Shape:", df.shape)
        print("Missing values:", int(df.isna().sum().sum()))
        print("Duplicate rows:", int(df.duplicated().sum()))
        """,
        "Check that the printed columns match the data dictionary. If missing values or duplicates are introduced "
        "later, inspect their cause and document a cleaning strategy before training.",
    )
    return cells


def training_lessons(kind):
    return lesson(
        "Features, target and independent split",
        "Select only the declared input columns, then create an 80/20 train/test split.",
        "The test labels must not influence fitting, preprocessing or model selection. Classification is stratified "
        "to preserve class proportions. Seed 42 makes the split reproducible.",
        """
        from src.ml import split_data, make_pipeline, train_model, predict, feature_importance
        X_train, X_test, y_train, y_test = split_data(KIND, df)
        print("Input columns:", X_train.columns.tolist())
        print("Training shape:", X_train.shape, "Test shape:", X_test.shape)
        print("Index overlap:", len(set(X_train.index) & set(X_test.index)))
        display(X_train.head())
        """,
        "The overlap should be zero. X contains inputs and y contains the target. Data splitting is performed before "
        "fitting the categorical encoder. No scaling is needed for these threshold-based tree models.",
    ) + lesson(
        "Train and choose a model",
        "Compare a dummy baseline, a Decision Tree and a Random Forest on three training-data folds; "
        "fit the best non-dummy candidate and evaluate it on the held-out test set.",
        "A baseline gives context. A Decision Tree is easy to explain; a Random Forest combines multiple trees. "
        "We choose between the two fitted approaches by validation results rather than assuming one will always win.",
        """
        # Read src/ml.py for the shared training implementation.
        # save=False reproduces results without overwriting the app's saved models.
        bundle = train_model(KIND, save=False)
        metrics = bundle["metrics"]
        display(pd.DataFrame(metrics["comparison"]).round(4))
        print("Selection metric:", metrics["cv_metric"])
        print("Selected model:", metrics["selected_model"])
        display(bundle["pipeline"])
        """,
        "For classifiers, higher macro F1 is better. For yield regression, lower mean absolute error is better. "
        "The dummy baseline predicts the most frequent class or median yield. It is a reference, not a candidate "
        "for the final app. Cross-validation scores are separate from the held-out metrics shown next.",
    )


def classification_evaluation():
    return lesson(
        "Measure classification performance",
        "Calculate accuracy and per-class precision, recall and F1; draw the confusion matrix.",
        "A single percentage can hide weak classes. The confusion matrix shows which true labels are mistaken "
        "for which predicted labels. Row labels are actual classes; columns are predictions.",
        """
        print(f"Test accuracy: {metrics['accuracy']:.2%}")
        print(f"Macro F1: {metrics['macro_f1']:.4f}")
        display(pd.DataFrame(metrics["classification_report"]).T.round(3))
        from sklearn.metrics import ConfusionMatrixDisplay
        fig, ax = plt.subplots(figsize=(11, 9))
        ConfusionMatrixDisplay(
            np.array(metrics["confusion_matrix"]), display_labels=metrics["labels"]
        ).plot(ax=ax, cmap="Greens", xticks_rotation=90, colorbar=False)
        ax.set_title("Held-out confusion matrix")
        plt.tight_layout()
        plt.show()
        """,
        "Accuracy = correct predictions / all predictions. Precision = TP/(TP+FP), recall = TP/(TP+FN), "
        "and F1 is their harmonic mean. Macro F1 averages class F1 values equally; support is the number of "
        "test examples for each class. Strong diagonal counts mean more correct classifications. All scores "
        "describe this synthetic test set.",
    )


def importance_lesson():
    return lesson(
        "Inspect model feature importance",
        "Rank inputs by their contribution to tree impurity reduction.",
        "This gives a starting point for discussing what the model uses. It is not a causal explanation and "
        "can be misleading for correlated or high-cardinality inputs.",
        """
        importance = feature_importance(bundle)
        display(importance.head(12))
        importance.head(12).sort_values("Importance").plot.barh(x="Feature", y="Importance", legend=False)
        plt.title("Selected model: impurity-based feature importance")
        plt.tight_layout()
        plt.show()
        """,
        "The prefix numeric__ identifies a numerical feature; category__ identifies an encoded category. "
        "Importance values are model-specific, not biological rules.",
    )


def crop():
    cells = [md("""
    # 01 — Crop recommendation

    ## Goal and learning outcomes
    Predict one of 22 crop labels from seven numerical features. Learn data inspection, exploration,
    stratified splitting, model comparison, classification metrics and ranked predictions.

    ## Data dictionary
    | Column | Meaning | Unit or limitation |
    |---|---|---|
    | N, P, K | Nitrogen, phosphorus and potassium inputs | Reference dataset values; measurement units undocumented |
    | temperature | Temperature | Degrees Celsius in the reference |
    | humidity | Relative humidity input | Percentage |
    | ph | Soil acidity/alkalinity | pH scale |
    | rainfall | Rainfall input | Reference labels say mm; accumulation period is undocumented |
    | label | Crop to predict | 22 categories |

    The source generates 100 examples per crop using predefined profiles and random noise. Therefore,
    recognizable profile boundaries can make this task easy. Do not describe these rows as measured field data.
    """)]
    cells += data_lessons("crop")
    cells += lesson(
        "Explore class balance and numerical relationships",
        "Count examples per crop and plot correlations between numerical inputs.",
        "Class balance affects how accuracy should be interpreted. Correlation reveals associations within this "
        "dataset, including associations caused by its generation rules.",
        """
        df["label"].value_counts().sort_index().plot.bar(color="#2b7a45")
        plt.ylabel("Generated examples")
        plt.title("Examples per crop")
        plt.tight_layout()
        plt.show()
        corr = df[spec["numeric"]].corr()
        fig, ax = plt.subplots(figsize=(7, 6))
        image = ax.imshow(corr, cmap="BrBG", vmin=-1, vmax=1)
        ax.set_xticks(range(len(corr)), corr.columns, rotation=60, ha="right")
        ax.set_yticks(range(len(corr)), corr.columns)
        fig.colorbar(image, ax=ax, label="Correlation")
        plt.tight_layout()
        plt.show()
        """,
        "Each crop has 100 examples. Correlation ranges from -1 to +1 and describes linear association. "
        "It does not tell us whether a feature causes crop success.",
    )
    cells += training_lessons("crop") + classification_evaluation() + importance_lesson()
    cells += lesson(
        "Follow one prediction from inputs to output",
        "Construct a named input dictionary, validate it and rank the model's class probabilities.",
        "Named columns avoid accidentally swapping nitrogen with phosphorus or temperature with rainfall.",
        """
        example = {"N": 85, "P": 45, "K": 40, "temperature": 25,
                   "humidity": 82, "ph": 6.5, "rainfall": 220}
        result = predict(KIND, example, bundle)
        print("Top crop:", result["prediction"])
        print("Range warnings:", result["warnings"])
        display(result["ranking"].head(3).round(2))
        result["ranking"].head(3).set_index("Label").plot.bar(legend=False)
        plt.ylabel("Uncalibrated model score (%)")
        plt.tight_layout()
        plt.show()
        """,
        "Scores come from predict_proba; they are not guaranteed suitability or field success. A model can "
        "be confident on unfamiliar inputs. Validation rejects impossible values and reports training-range warnings.",
    )
    cells += lesson(
        "Check invalid inputs and the saved model",
        "Try an impossible humidity and compare the fresh model with the locally saved artifact.",
        "The interface must reject invalid data, and reloading should preserve predictions.",
        """
        from src.ml import load_model
        try:
            predict(KIND, {**example, "humidity": 120}, bundle)
        except ValueError as error:
            print("Expected validation error:", error)
        saved = load_model(KIND)
        print("Saved-model result:", predict(KIND, example, saved)["prediction"])
        print("Same prediction:", predict(KIND, example, saved)["prediction"] == result["prediction"])
        """,
        "The error is expected, and the two predictions should match. train_models.py writes locally trained "
        "joblib bundles; never load a model file from an untrusted source.",
    )
    cells += [md("""
    ## Application connection
    `app.py` collects these seven values in a form. `src/data.py` validates them; `src/ml.py` predicts.
    The interface shows the top three scores and source crop notes. Only a compatible crop name can carry
    forward to the other modules; numerical nutrient values are not copied across their different scales.

    ## Results, limitations and future work
    Report the accuracy and macro F1 printed above, not a promised score. The current data has fixed profiles,
    equal class counts and no regional/seasonal field variation. The rainfall period and N/P/K units need
    documentation before meaningful real-world input mapping. Future work needs verified observations,
    region-aware evaluation, probability calibration and expert review.

    ## Viva questions
    **Why classification?** The output is one crop category.

    **Why stratify?** It preserves the proportion of each crop in the training and test sets.

    **Why not standardize the inputs?** Tree splits compare each input with thresholds; this pipeline does
    not depend on distance between differently scaled features.

    **Why compare a tree and forest?** A tree is a simple baseline; a forest can reduce variance by combining trees.

    **Does 99% test accuracy mean 99% farm success?** No. It measures label agreement on this synthetic split.

    **Exercise:** change rainfall while keeping other values fixed. Record rankings and warnings. Explain
    why this is a model response rather than proof that changing rainfall would cause a particular outcome.
    """)]
    write("01_crop_recommendation.ipynb", cells)


def fertilizer():
    cells = [md("""
    # 02 — Fertilizer advice and quantity arithmetic

    ## Goal
    Predict a fertilizer label, inspect illustrative N/P/K gaps, and calculate quantities from a user-entered rate.
    These are three separate operations: learned classification, reference comparison and unit arithmetic.

    ## Data dictionary
    | Columns | Meaning |
    |---|---|
    | Temperature, Humidity | Temperature in °C and humidity in % |
    | Moisture | Reference moisture value; measurement basis is undocumented |
    | Soil_Type, Crop_Type | Categorical inputs |
    | Nitrogen, Phosphorous, Potassium | Numeric reference values, not verified soil-test units |
    | Fertilizer_Name | Target product label |

    `Phosphorous` preserves the source CSV's spelling so feature names stay consistent.
    The source generates the target with if/elif nutrient rules. For example, nitrogen below 20 with
    phosphorus and potassium at least 20 receives the Urea label. This creates a learnable classification
    exercise; it does not prove an agronomic prescription. Crop and soil values are generated independently.
    """)]
    cells += data_lessons("fertilizer")
    cells += lesson(
        "Inspect fertilizer labels and categories",
        "List soil/crop values and plot the distribution of fertilizer labels.",
        "Category coverage defines which choices the interface supports. Unequal class counts make macro F1 useful.",
        """
        for name in spec["categorical"]:
            print(name, sorted(df[name].unique()))
        df[spec["target"]].value_counts().plot.bar(color="#2b7a45")
        plt.ylabel("Examples")
        plt.title("Synthetic fertilizer label distribution")
        plt.tight_layout()
        plt.show()
        """,
        "The module supports the categories shown here. Unsupported categories are rejected rather than silently "
        "treated as a known crop. Some source crop labels have no counterpart in this dataset.",
    )
    cells += lesson(
        "Understand one-hot encoding",
        "Turn soil and crop strings into binary columns, fitting the encoder on training data only.",
        "A tree cannot directly consume these strings. One-hot encoding avoids inventing a numerical order "
        "such as Sandy=1 and Loamy=2. A Pipeline keeps preprocessing inside each validation fold.",
        """
        from sklearn.preprocessing import OneHotEncoder
        from src.ml import split_data
        X_train, X_test, y_train, y_test = split_data(KIND, df)
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        encoded = encoder.fit_transform(X_train[spec["categorical"]])
        display(pd.DataFrame(encoded[:5], columns=encoder.get_feature_names_out()))
        print("Encoded shape:", encoded.shape)
        """,
        "Each category becomes a 0/1 column. The app rejects unknown categories explicitly; the encoder's "
        "ignore option is an additional technical safeguard, not a claim that unseen crops are supported.",
    )
    cells += training_lessons("fertilizer") + classification_evaluation() + importance_lesson()
    cells += lesson(
        "Predict a fertilizer label",
        "Create a sample row and examine the top-ranked product labels.",
        "This connects the encoded training workflow to a normal form submission.",
        """
        example = {"Temperature": 28, "Humidity": 65, "Moisture": 40, "Soil_Type": "Loamy",
                   "Crop_Type": "Rice", "Nitrogen": 15, "Phosphorous": 30, "Potassium": 30}
        result = predict(KIND, example, bundle)
        print("Predicted label:", result["prediction"])
        display(result["ranking"].head(3))
        """,
        "A Decision Tree can return 100% probability for a pure leaf. That is a property of the learned tree, "
        "not verified confidence that applying the product will help a farm.",
    )
    cells += lesson(
        "Compare illustrative nutrient gaps",
        "Subtract each supplied value from the reference crop target and clamp negative gaps to zero.",
        "The formula makes the calculation visible: gap = max(target - input, 0). The original project's "
        "targets are retained only as an educational comparison.",
        """
        from src.advisory import nutrient_comparison
        nutrients = nutrient_comparison("Rice", nitrogen=15, phosphorus=30, potassium=30)
        display(nutrients)
        nutrients.set_index("Nutrient")[["Input", "Illustrative target"]].plot.bar()
        plt.ylabel("Reference dataset units")
        plt.tight_layout()
        plt.show()
        """,
        "A gap of 25 means 25 reference units below the illustrative target. It does not mean 25 kg of fertilizer. "
        "Converting a soil-test value into a dose requires measurement units and validated agronomic assumptions.",
    )
    cells += lesson(
        "Calculate an assumed quantity and bag count",
        "Multiply a user-entered rate by area, then divide by bag mass.",
        "This keeps useful unit arithmetic without pretending to derive a field dose from undocumented nutrient values.",
        """
        from src.advisory import fertilizer_quantity, ACRES_PER_HECTARE
        rate_kg_ha = 100  # Assumed for this arithmetic example, not prescribed.
        area_ha = 2
        quantity = fertilizer_quantity(rate_kg_ha, area_ha)
        display(pd.Series(quantity, name="Quantity"))
        print("Area in acres:", round(area_ha * ACRES_PER_HECTARE, 3))
        """,
        "100 kg/ha × 2 ha = 200 kg. Dividing by 50 kg per bag gives 4 bag equivalents. "
        "Whole bags are rounded upward when the equivalent count is fractional.",
    )
    cells += [md("""
    ## Application connection and reference notes
    The fertilizer page combines product rankings, the nutrient table, a bar chart and the quantity calculator.
    Its optional reference panel reproduces the source's product descriptions, application examples and
    organic alternatives with a clear provenance label. They are not generated or validated by the classifier.

    ## Limitations and extensions
    Excellent test scores are expected when labels come from deterministic rules. The model mainly recovers
    those rules. It does not establish whether the rules are correct for a crop, soil, location or season.
    Future work would use measured soil tests with units, agronomic labels and independent regional evaluation.

    ## Viva questions
    **Why one-hot encoding?** It represents categories without implying an ordering.

    **Why a pipeline?** It fits preprocessing and the model together, including inside cross-validation.

    **Why might soil have little importance?** The source label-generation rules use nutrient inputs, not soil category.

    **Is the gap a fertilizer dose?** No. It is a difference between two reference values.

    **Why keep quantity arithmetic separate from ML?** Its assumptions and units can be inspected independently.

    **Exercise:** enter 125 kg/ha and 1.5 ha, then explain bag equivalents versus rounded whole bags.
    """)]
    write("02_fertilizer_advice.ipynb", cells)


def yield_notebook():
    cells = [md("""
    # 03 — Yield prediction, economics and rainfall scenarios

    ## Goal
    Predict a continuous yield, convert it into total production, and calculate sample economics with explicit
    assumptions. Learn regression metrics and why information leakage can invalidate an apparently strong result.

    ## Data dictionary
    | Column | Meaning |
    |---|---|
    | Crop, Season, State, Irrigation_Type | Categorical context |
    | Area_ha | Area in hectares |
    | Rainfall_mm | Reference rainfall input; period is not documented |
    | Temperature_c | Temperature in °C |
    | Fertilizer_kg_ha | Fertilizer input in kg/ha |
    | Yield_tons_per_ha | Target: yield per unit area |
    | Production_tons | Derived output: yield × area; excluded from features |

    The source generates yield using a crop baseline, rainfall/fertilizer/irrigation factors and noise.
    Season, state and temperature are present as inputs but do not enter that generation formula.
    A model trained here cannot demonstrate genuine effects of those variables.
    """)]
    cells += data_lessons("yield")
    cells += lesson(
        "Inspect yield scale and prevent leakage",
        "Plot mean yield by crop and show how production is derived from the target.",
        "Different crops have very different yield scales. Including Production_tons would expose information "
        "derived from the answer, creating target leakage.",
        """
        df.groupby("Crop")["Yield_tons_per_ha"].mean().sort_values().plot.barh(color="#2b7a45")
        plt.xlabel("Generated mean yield (tons/ha)")
        plt.tight_layout()
        plt.show()
        production_error = (df["Production_tons"] - df["Yield_tons_per_ha"] * df["Area_ha"]).abs()
        print("Largest production difference after rounding:", production_error.max())
        print("Actual input columns:", spec["numeric"] + spec["categorical"])
        assert "Production_tons" not in spec["numeric"] + spec["categorical"]
        """,
        "Production differs from yield × area only by source rounding. It is excluded from X. "
        "Large between-crop differences can make aggregate R² look strong even when some per-crop errors matter.",
    )
    cells += training_lessons("yield")
    cells += lesson(
        "Evaluate regression in meaningful units",
        "Report MAE, RMSE and R²; plot actual versus predicted yield and errors by crop.",
        "MAE is the average absolute error in tons/ha. RMSE emphasizes larger errors. R² compares squared "
        "error with a constant-mean baseline and can be negative for poor predictions.",
        """
        print(f"MAE: {metrics['mae']:.3f} tons/ha")
        print(f"RMSE: {metrics['rmse']:.3f} tons/ha")
        print(f"R2: {metrics['r2']:.4f}")
        predicted = bundle["pipeline"].predict(X_test)
        fig, ax = plt.subplots()
        ax.scatter(y_test, predicted, s=15, alpha=0.5, color="#2b7a45")
        limit = max(float(y_test.max()), float(predicted.max()))
        ax.plot([0, limit], [0, limit], "--", color="black")
        ax.set(xlabel="Actual synthetic yield (tons/ha)", ylabel="Predicted yield (tons/ha)")
        plt.tight_layout()
        plt.show()
        display(pd.Series(metrics["mae_by_crop"], name="MAE (tons/ha)").sort_values().to_frame())
        """,
        "Points near the diagonal are closer predictions. R² is not a percentage accuracy. Review per-crop "
        "MAE because an error of one ton/ha means different things for low-yield and high-yield crops.",
    )
    cells += importance_lesson()
    cells += lesson(
        "Predict an example farm's yield",
        "Supply the categorical and numerical features expected by the saved pipeline.",
        "Input validation checks categories, positive area, finite numbers and training-range coverage.",
        """
        example = {"Crop": "Rice", "Season": "Kharif", "State": "Punjab", "Irrigation_Type": "Canal",
                   "Area_ha": 2.0, "Rainfall_mm": 800.0, "Temperature_c": 27.0, "Fertilizer_kg_ha": 150.0}
        result = predict(KIND, example, bundle)
        print(f"Predicted yield: {result['prediction']:.3f} tons/ha")
        print("Range warnings:", result["warnings"])
        """,
        "The returned number is yield per hectare, not total production. It is a model estimate with the "
        "test-error limitations discussed above.",
    )
    cells += lesson(
        "Convert units and calculate sample economics",
        "Convert yield to production and quintals/acre, then calculate revenue, cost, profit and margin.",
        "Explicit arithmetic makes the financial assumptions auditable. The supplied app uses editable "
        "assumptions instead of describing old static prices as current market data.",
        """
        from src.advisory import farm_economics
        economics = farm_economics(
            yield_tons_ha=result["prediction"], area_ha=2.0,
            price_per_quintal=2000.0, cost_per_ha=30000.0
        )
        display(pd.Series(economics, name="Scenario result"))
        pd.Series({key: economics[key] for key in ["revenue", "cost", "profit"]}).plot.bar(color="#2b7a45")
        plt.ylabel("Assumed scenario amount (INR)")
        plt.tight_layout()
        plt.show()
        """,
        "Production = tons/ha × ha. One metric ton is 10 quintals, and one hectare is approximately "
        "2.47105 acres. Revenue = production × 10 × price/quintal. Cost = area × cost/ha. "
        "Profit = revenue - cost; margin = profit/revenue × 100. Zero revenue has an undefined margin.",
    )
    cells += lesson(
        "Run rainfall scenarios",
        "Predict three scenarios while changing only rainfall by -25%, 0% and +25%.",
        "This illustrates sensitivity of the fitted model. It does not establish a causal effect or predict weather.",
        """
        from src.advisory import rainfall_scenarios
        scenarios = rainfall_scenarios(example, bundle)
        display(scenarios)
        scenarios.set_index("Scenario")["Yield (tons/ha)"].plot.bar(color="#2b7a45")
        plt.ylabel("Model output (tons/ha)")
        plt.tight_layout()
        plt.show()
        """,
        "More rain need not produce more yield in this model. The generated relationship has a preferred "
        "rainfall range. The last column flags inputs outside the model's training ranges.",
    )
    cells += [md("""
    ## Application connection
    The app accepts hectares or acres, converts to hectares, predicts yield and displays production and
    sample economics. Price and cultivation cost are visibly editable. Negative profit is kept as a loss;
    it is not clamped to zero. A zero selling price produces an undefined margin rather than a division error.

    ## Limitations and future work
    This dataset has random noise and hand-set yield relationships. Its state/season/temperature fields
    do not justify regional predictions. A real extension needs observed yields across multiple seasons
    and locations, geographic/time-based holdouts, documented input periods and uncertainty estimates.
    Money outputs inherit forecast error and omitted costs as well as the user's assumed prices.

    ## Viva questions
    **Why regression?** Yield is a continuous number.

    **Why exclude production?** It is derived directly from the target and area, so using it leaks the answer.

    **Why MAE as the selection metric?** It is interpretable in the target unit, tons/ha.

    **Why show per-crop errors?** Aggregate metrics can hide scale-related differences.

    **Is R² an accuracy percentage?** No. It compares squared prediction error with target variability.

    **Exercise:** keep yield fixed, lower the assumed selling price, and find when profit becomes negative.
    """)]
    write("03_yield_and_economics.ipynb", cells)


def leaf():
    cells = [md("""
    # 04 — Leaf image analysis with colour segmentation

    ## Goal and honest scope
    Find a visible leaf, mark selected colour changes, calculate the marked-area percentage and display
    an overlay. This is an explainable image-processing exercise, not a trained disease classifier.
    It does not generate a diagnostic confidence score or a treatment prescription.

    ## Image concepts
    A colour image is an array of height × width × 3 values. In RGB, the channels are red, green and blue.
    HSV separates hue (colour), saturation (colour strength) and value (brightness). OpenCV's 8-bit HSV
    hue range is 0–179; saturation and value range from 0–255. A binary mask stores 255 for included
    pixels and 0 for excluded pixels. A contour describes a connected region's boundary.

    ## Processing sequence
    1. Correct orientation, convert to RGB and resize while preserving aspect ratio.
    2. Convert RGB to HSV and check that visible green tissue exists.
    3. Find the largest coloured region and fill its outline as a leaf mask.
    4. Select brown/rust, yellow and dark pixels within that mask.
    5. Remove tiny specks with morphological opening.
    6. Divide marked pixels by leaf pixels and blend red only over marked pixels.

    The thresholds are heuristic design choices documented in `src/vision.py`. The included samples are
    drawings from the source project. Their filenames are illustrative labels, not ground-truth photographic evidence.
    """)]
    cells += setup()
    cells += lesson(
        "Open and inspect a sample",
        "Read the tomato illustration and display its shape and pixels.",
        "Looking at the actual image is necessary before reasoning about segmentation. The image is RGB, "
        "which avoids confusing OpenCV's common BGR input convention.",
        """
        from PIL import Image
        from src.vision import analyze_leaf, SAMPLE_CROPS
        picture = Image.open(ROOT / "samples" / "tomato_early_blight.png").convert("RGB")
        array = np.asarray(picture)
        print("Shape (height, width, channels):", array.shape)
        print("Data type:", array.dtype, "Value range:", array.min(), array.max())
        plt.imshow(array)
        plt.axis("off")
        plt.title("Source illustration: not a labelled field photograph")
        plt.show()
        """,
        "There are three channels per pixel. A light background helps the chosen colour-based approach, "
        "but a cluttered scene can violate this assumption.",
    )
    cells += lesson(
        "Examine the HSV channels",
        "Convert RGB to HSV and display the hue, saturation and brightness channels.",
        "Separating colour from brightness makes simple threshold rules easier to explain.",
        """
        import cv2
        hsv = cv2.cvtColor(array, cv2.COLOR_RGB2HSV)
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        for index, label in enumerate(["Hue", "Saturation", "Value"]):
            axes[index].imshow(hsv[:, :, index], cmap="gray")
            axes[index].set_title(label)
            axes[index].axis("off")
        plt.tight_layout()
        plt.show()
        """,
        "The channels encode different aspects of the same image. Thresholds can still fail under shadows, "
        "unusual lighting, yellow leaves, coloured backgrounds or entirely dead leaves.",
    )
    cells += lesson(
        "Generate masks and an overlay",
        "Run the deterministic algorithm and display the original, leaf mask, marked mask and overlay.",
        "Showing intermediate masks makes the result inspectable. Filling the leaf boundary preserves dark "
        "internal patches; opening removes tiny marked specks. Unmarked pixels remain unchanged.",
        """
        result = analyze_leaf(picture, crop="Tomato")
        fig, axes = plt.subplots(1, 4, figsize=(14, 4))
        for ax, name, title in zip(
            axes, ["original", "leaf_mask", "affected_mask", "overlay"],
            ["Original", "Leaf mask", "Marked colour changes", "Overlay"]
        ):
            ax.imshow(result[name], cmap="gray" if "mask" in name else None)
            ax.set_title(title)
            ax.axis("off")
        plt.tight_layout()
        plt.show()
        """,
        "White mask pixels are included. The marked mask must stay inside the leaf mask. A visually sensible "
        "overlay shows agreement with this illustration, not validated disease-recognition ability.",
    )
    cells += lesson(
        "Calculate affected-area percentage",
        "Count marked and leaf pixels, then apply the ratio formula.",
        "Using leaf area as the denominator avoids diluting the estimate with background pixels.",
        """
        manual_percentage = result["affected_pixels"] / result["leaf_pixels"] * 100
        display(pd.Series({
            "Leaf pixels": result["leaf_pixels"],
            "Marked pixels": result["affected_pixels"],
            "Marked percentage": manual_percentage,
            "Severity band": result["severity_band"],
            "Connected regions": result["regions"],
            "Illustrative match": result["illustrative_match"],
        }, name="Image result"))
        print("Ratio matches algorithm:", np.isclose(manual_percentage, result["affected_percent"]))
        print("Marked pixels outside leaf:", np.count_nonzero((result["affected_mask"] > 0) & (result["leaf_mask"] == 0)))
        """,
        "The band thresholds are educational: below 2.5% low colour change, below 10% mild, up to 25% "
        "moderate, above 25% high. These are not validated clinical/pathology severity categories. Connected "
        "regions are colour blobs, not necessarily individual lesions.",
    )
    cells += lesson(
        "Compare every included sample",
        "Process all five illustrations with explicit crop hints and compare marked percentages.",
        "A reproducible sample table is more informative than one selected image. The crop hint controls "
        "the descriptive example; it does not identify the crop from pixels.",
        """
        rows = []
        fig, axes = plt.subplots(1, 5, figsize=(15, 4))
        for ax, (filename, crop_hint) in zip(axes, SAMPLE_CROPS.items()):
            with Image.open(ROOT / "samples" / filename) as sample:
                output = analyze_leaf(sample, crop_hint)
            rows.append({
                "Sample": filename, "Crop hint": crop_hint,
                "Marked area (%)": round(output["affected_percent"], 2),
                "Band": output["severity_band"],
            })
            ax.imshow(output["overlay"])
            ax.set_title(filename.replace("_", "\\n").replace(".png", ""), fontsize=9)
            ax.axis("off")
        plt.tight_layout()
        plt.show()
        display(pd.DataFrame(rows))
        """,
        "The healthy illustration should have little marked area, while spotted samples should have more. "
        "Differences reflect colour rules and drawing design. This table cannot measure disease-classification accuracy.",
    )
    cells += lesson(
        "Test repeatability and reject an empty image",
        "Run the same input twice and try a plain white image.",
        "A deterministic algorithm should return the same result for the same input. A missing leaf should "
        "produce an explanation instead of inventing a leaf boundary or diagnosis.",
        """
        repeated = analyze_leaf(picture, "Tomato")
        print("Same percentage:", repeated["affected_percent"] == result["affected_percent"])
        print("Same mask:", np.array_equal(repeated["affected_mask"], result["affected_mask"]))
        unchanged = result["affected_mask"] == 0
        print("Unmarked pixels preserved:", np.array_equal(result["original"][unchanged], result["overlay"][unchanged]))
        try:
            analyze_leaf(Image.new("RGB", (200, 200), "white"))
        except ValueError as error:
            print("Expected rejection:", error)
        """,
        "The repeatability checks should be True, and the blank image should be rejected. The reference "
        "project used random confidence percentages; this project deliberately has no such output.",
    )
    cells += [md("""
    ## Application connection
    The interface accepts one of the included illustrations or an uploaded PNG/JPEG. It offers an explicit
    crop hint and displays masks, marked area, an overlay and an illustrative comparison. It does not
    automatically identify a causal pathogen. Real treatment decisions need an actual diagnosis.

    ## Known failure cases
    Shadows can look like dark patches; yellowing may be normal ageing or stress; a colourful background
    can become the largest contour. Multiple leaves, very small leaves and completely brown leaves may be
    rejected or segmented incorrectly. Resizing also changes small details. An apparently clear match can be wrong.

    ## Future extension
    Collect appropriately licensed, expert-labelled leaf photographs. Split by plant/location rather than
    near-duplicate image. Train and evaluate a real image classifier, report per-class metrics, and use
    pixel-level annotations if validating segmentation. Validate under lighting/background shifts.

    ## Viva questions
    **Is this deep learning?** No; it is deterministic image processing.

    **Why HSV?** It separates colour, saturation and brightness for interpretable thresholds.

    **Why a mask?** It selects the pixels used in a calculation or visualization.

    **What does morphological opening do?** Erosion followed by dilation removes small isolated regions.

    **Why preserve aspect ratio?** Stretching changes shapes and can distort visual interpretation.

    **Does a brown patch confirm disease?** No. The algorithm measures appearance, not cause.

    **Exercise:** compare the healthy sample with a darker version using a photo editor. Explain why a
    lighting change could alter the measured result without changing the underlying plant condition.
    """)]
    write("04_leaf_image_analysis.ipynb", cells)


if __name__ == "__main__":
    (ROOT / "notebooks").mkdir(exist_ok=True)
    guide()
    crop()
    fertilizer()
    yield_notebook()
    leaf()
    print("Built five detailed notebooks.")
