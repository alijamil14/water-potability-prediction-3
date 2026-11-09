from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "water_potability.csv"
MODEL_PATH = ROOT / "models" / "water_potability_model.pkl"
METRICS_PATH = ROOT / "models" / "evaluation_metrics.json"

FEATURES = [
    "ph", "Hardness", "Solids", "Chloramines", "Sulfate",
    "Conductivity", "Organic_carbon", "Trihalomethanes", "Turbidity"
]
FRIENDLY_NAMES = {
    "ph": "pH",
    "Hardness": "Hardness",
    "Solids": "Total dissolved solids",
    "Chloramines": "Chloramines",
    "Sulfate": "Sulfate",
    "Conductivity": "Conductivity",
    "Organic_carbon": "Organic carbon",
    "Trihalomethanes": "Trihalomethanes",
    "Turbidity": "Turbidity"
}

st.set_page_config(
    page_title="Water Potability | ML Dashboard",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styling: colors are also defined in .streamlit/config.toml.
st.markdown("""
    /* Predict Potability Button */
    div.stButton > button[kind="primary"] {
        background-color: #000000 !important;
        color: #FFFFFF !important;
        border: 1px solid #000000 !important;
        border-radius: 10px;
        font-weight: 700;
        min-height: 3rem;
    }

    div.stButton > button[kind="primary"]:hover {
        background-color: #333333 !important;
        color: #FFFFFF !important;
        border: 1px solid #333333 !important;
    }

    /* Number Input Fields */
    [data-testid="stNumberInput"] input {
        background-color: #FFFFFF !important;
        color: #000000 !important;
        border: 1px solid #000000 !important;
    }

    /* Number Input Plus and Minus Buttons */
    [data-testid="stNumberInput"] button {
        background-color: #000000 !important;
        color: #FFFFFF !important;
        border: 1px solid #000000 !important;
    }

    [data-testid="stNumberInput"] button:hover {
        background-color: #333333 !important;
        color: #FFFFFF !important;
    }

    /* Success Alert */
    div[data-testid="stAlert"]:has([data-testid="stAlertContentSuccess"]) {
        background-color: #D1FAE5 !important;
        border: 1px solid #16A34A !important;
        color: #14532D !important;
    }

    /* Error Alert */
    div[data-testid="stAlert"]:has([data-testid="stAlertContentError"]) {
        background-color: #FEE2E2 !important;
        border: 1px solid #DC2626 !important;
        color: #991B1B !important;
    }
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

@st.cache_resource
def load_model():
    # Only load trusted model files that you created.
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_metrics():
    return json.loads(METRICS_PATH.read_text(encoding="utf-8"))

st.markdown("""
<div class="hero">
    <h1 style="text-align: center;">UI Created By Ali Jamil</h1>
    <h1 style="text-align: center;">💧 Water Potability Prediction</h1>
    <p style="text-align: center;">Machine-learning dashboard • Random Forest binary classification • Prediction classes: 0 and 1</p>
</div>
""", unsafe_allow_html=True)

missing_files = [
    path.name for path in [DATA_PATH, MODEL_PATH, METRICS_PATH]
    if not path.exists()
]
if missing_files:
    st.error(
        "Required project files are missing: "
        + ", ".join(missing_files)
        + ". Check the data/ and models/ folders."
    )
    st.stop()

df = load_data()
model = load_model()
metrics = load_metrics()

st.sidebar.header("Water-quality inputs")
st.sidebar.caption("Enter measurements, then run the prediction.")

values = {}
for feature in FEATURES:
    series = df[feature].dropna()
    default_value = float(series.median())
    observed_min = float(series.min())
    observed_max = float(series.max())
    values[feature] = st.sidebar.number_input(
        FRIENDLY_NAMES[feature],
        value=default_value,
        step=max((observed_max - observed_min) / 100, 0.01),
        format="%.4f",
        help=f"Observed dataset range: {observed_min:.3f} to {observed_max:.3f}"
    )

input_df = pd.DataFrame([values], columns=FEATURES)

st.subheader("Water sample prediction")
predict_col, input_col = st.columns([1.05, 1])

with predict_col:

    st.write("Run the trained model on the values entered in the sidebar.")

    if st.button("Predict potability", type="primary", use_container_width=True):

        predicted_class = int(model.predict(input_df)[0])
        probabilities = model.predict_proba(input_df)[0]

        class_to_probability = {
            int(label): float(probability)
            for label, probability in zip(model.classes_, probabilities)
        }

        potable_probability = class_to_probability.get(1, 0.0)
        not_potable_probability = class_to_probability.get(0, 0.0)

        st.session_state["last_prediction"] = predicted_class
        st.session_state["last_probabilities"] = {
            "potable": potable_probability,
            "not_potable": not_potable_probability
        }
    # Display prediction result
    if "last_prediction" in st.session_state:

        predicted_class = st.session_state["last_prediction"]
        probs = st.session_state["last_probabilities"]

        if predicted_class == 1:

            st.success(
                "Water is drinkable and healthy for people",
                icon="✅"
            )

        else:

            st.error(
                "Water is not drinkable and may affect people's health",
                icon="⚠️"
            )

        prob1, prob2 = st.columns(2)

        prob1.metric(
            "Estimated class 1 probability",
            f"{probs['potable'] * 100:.1f}%"
        )

        prob2.metric(
            "Estimated class 0 probability",
            f"{probs['not_potable'] * 100:.1f}%"
        )

        st.caption(
            "These are model estimates, not a laboratory result "
            "or a guarantee of drinking-water safety."
        )

    else:

        st.info(
            "Enter the sample measurements in the sidebar "
            "and click **Predict potability**."
        )


with input_col:

    st.markdown("**Current input measurements**")

    display_input = input_df.rename(columns=FRIENDLY_NAMES).T
    display_input.columns = ["Entered value"]

    st.dataframe(display_input, use_container_width=True)

st.divider()
st.header("Model evaluation on the held-out test set")
metric_cols = st.columns(4)
metric_cols[0].metric("Accuracy", f"{metrics['accuracy'] * 100:.2f}%")
metric_cols[1].metric("Precision", f"{metrics['precision'] * 100:.2f}%")
metric_cols[2].metric("Recall", f"{metrics['recall'] * 100:.2f}%")
metric_cols[3].metric("F1-score", f"{metrics['f1_score'] * 100:.2f}%")

cm = np.asarray(metrics["confusion_matrix"], dtype=int)
fig, ax = plt.subplots(figsize=(6.5, 4.5))
ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Not potable (0)", "Potable (1)"]
).plot(ax=ax, cmap="GnBu", colorbar=False, values_format="d")
ax.set_title("Random Forest confusion matrix")
fig.tight_layout()
st.pyplot(fig, use_container_width=True)
plt.close(fig)

with st.expander("About this project and its limitations"):
    st.write(
        "This educational application uses a Random Forest classifier. "
        "The target is Potability: 0 means not potable and 1 means potable. "
        "Missing feature values are imputed using medians learned from the training data. "
        "A model prediction must not replace accredited laboratory water testing "
        "or public-health guidance."
    )
