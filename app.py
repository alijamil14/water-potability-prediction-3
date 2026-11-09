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
<style>

/* Main application background */
.stApp {
    background: linear-gradient(135deg, #F4F8FB 0%, #E8EEF3 100%);
    color: #FFFFFF;
}

/* Main heading */
.hero {
    padding: 35px 25px;
    border-radius: 50px;
    background: linear-gradient(135deg, #111827 0%, #374151 50%, #111827 100%);
    color: white;
    margin-bottom: 30px;
    box-shadow: 0 8px 25px rgba(0,0,0,0.20);
}

.hero h1 {
    color: #FFFFFF !important;
    font-size: 2.1rem;
    margin-bottom: 12px;
}

.hero p {
    color: #D1D5DB !important;
    font-size: 1rem;
}

/* Sidebar - same gradient as main heading */
[data-testid="stSidebar"] {
    background: linear-gradient(135deg, #111827 0%, #374151 50%, #111827 100%);
}

[data-testid="stSidebar"] * {
    color: #FFFFFF !important;
}

[data-testid="stSidebar"] input {
    background-color: #FFFFFF !important;
    color: #111827 !important;
    border-radius: 10px;
}

/* Sidebar number input buttons */
[data-testid="stSidebar"] [data-testid="stNumberInput"] button {
    background-color: #000000 !important;
    color: #FFFFFF !important;
}

/* Main headings */
h1, h2, h3 {
    color: #FFFFFF !important;
}

/* Main sections */
.section-box {
    background: linear-gradient(135deg, #111827 0%, #374151 100%);
    padding: 30px;
    border-radius: 50px;
    margin: 20px 0;
    color: #FFFFFF;
    box-shadow: 0 8px 20px rgba(0,0,0,0.15);
}

.section-box h2,
.section-box h3,
.section-box p {
    color: #FFFFFF !important;
}

/* Prediction success */
.prediction-success {
    background: linear-gradient(135deg, #064E3B, #16A34A);
    color: #FFFFFF;
    padding: 25px;
    border-radius: 30px;
    margin: 20px 0;
    border: 1px solid #22C55E;
    font-size: 18px;
    font-weight: 700;
}

/* Prediction error */
.prediction-error {
    background: linear-gradient(135deg, #7F1D1D, #DC2626);
    color: #FFFFFF;
    padding: 25px;
    border-radius: 30px;
    margin: 20px 0;
    border: 1px solid #EF4444;
    font-size: 18px;
    font-weight: 700;
}

/* Probability cards */
.probability-card {
    background: linear-gradient(135deg, #111827, #374151);
    padding: 25px;
    border-radius: 30px;
    margin: 10px 0;
    text-align: center;
    color: #FFFFFF;
}

.probability-card h4 {
    color: #D1D5DB !important;
    font-size: 15px;
}

.probability-card h2 {
    color: #FFFFFF !important;
    font-size: 30px;
}

/* Evaluation metric cards */
div[data-testid="stMetric"] {
    background: linear-gradient(135deg, #111827, #374151);
    padding: 25px;
    border-radius: 30px;
    border: 1px solid #4B5563;
    box-shadow: 0 5px 15px rgba(0,0,0,0.15);
}

div[data-testid="stMetric"] label,
div[data-testid="stMetric"] [data-testid="stMetricValue"] {
    color: #FFFFFF !important;
}

/* Predict button */
div.stButton > button[kind="primary"] {
    background-color: #000000 !important;
    color: #FFFFFF !important;
    border: 1px solid #000000 !important;
    border-radius: 15px;
    font-weight: 700;
    min-height: 3rem;
}

div.stButton > button[kind="primary"]:hover {
    background-color: #374151 !important;
    color: #FFFFFF !important;
}

/* Number input */
[data-testid="stNumberInput"] input {
    background-color: #FFFFFF !important;
    color: #000000 !important;
    border: 1px solid #000000 !important;
}

/* Plus and minus buttons */
[data-testid="stNumberInput"] button {
    background-color: #000000 !important;
    color: #FFFFFF !important;
}

/* Dataframe */
div[data-testid="stDataFrame"] {
    border-radius: 20px;
    overflow: hidden;
}

/* Caption */
.small-note {
    color: #D1D5DB;
    font-size: 0.9rem;
}

</style>
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

    <h1 style="text-align: center;">
        UI Created By Ali Jamil
    </h1>

    <h1 style="text-align: center;">
        💧 Water Potability Prediction
    </h1>

    <p style="text-align: center;">
        Machine-learning dashboard • Random Forest binary classification
        • Prediction classes: 0 and 1
    </p>

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

    # Successful prediction
    if predicted_class == 1:

        st.markdown("""
        <div class="prediction-success">
            ✅ WATER IS POTABLE
            <br><br>
            Water is classified as potable by the machine-learning model.
        </div>
        """, unsafe_allow_html=True)

    # Unsuccessful prediction
    else:

        st.markdown("""
        <div class="prediction-error">
            ⚠️ WATER IS NOT POTABLE
            <br><br>
            Water is classified as not potable by the machine-learning model.
        </div>
        """, unsafe_allow_html=True)

    # Probability cards
    prob1, prob2 = st.columns(2)

    with prob1:
        st.markdown(f"""
        <div class="probability-card">
            <h4>Estimated class 1 probability</h4>
            <h2>{probs['potable'] * 100:.1f}%</h2>
            <p>Potable</p>
        </div>
        """, unsafe_allow_html=True)

    with prob2:
        st.markdown(f"""
        <div class="probability-card">
            <h4>Estimated class 0 probability</h4>
            <h2>{probs['not_potable'] * 100:.1f}%</h2>
            <p>Not Potable</p>
        </div>
        """, unsafe_allow_html=True)

    st.caption(
        "Model estimates only. These results do not guarantee drinking-water safety."
    )

    else:

    st.info(
        "Enter the sample measurements in the sidebar and click Predict Potability."
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
