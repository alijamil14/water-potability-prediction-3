from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay


# ---------------------------------------------------------
# PROJECT PATHS
# ---------------------------------------------------------
ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "water_potability.csv"
MODEL_PATH = ROOT / "models" / "water_potability_model.pkl"
METRICS_PATH = ROOT / "models" / "evaluation_metrics.json"

FEATURES = [
    "ph",
    "Hardness",
    "Solids",
    "Chloramines",
    "Sulfate",
    "Conductivity",
    "Organic_carbon",
    "Trihalomethanes",
    "Turbidity",
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
    "Turbidity": "Turbidity",
}


# ---------------------------------------------------------
# STREAMLIT PAGE CONFIGURATION
# ---------------------------------------------------------
st.set_page_config(
    page_title="Water Potability | ML Dashboard",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------
# CUSTOM CSS
# ---------------------------------------------------------
st.markdown(
    """
    <style>
    /* Main page background and default text */
    .stApp {
        background: linear-gradient(
            135deg,
            #F0FAFC 0%,
            #E1F5F8 50%,
            #F4FBFC 100%
        );
        color: #111827;
    }

    /* Main content headings and text */
    .stApp h1,
    .stApp h2,
    .stApp h3,
    .stApp h4,
    .stApp p,
    .stApp label {
        color: #111827;
    }

    /* Water-gradient hero heading */
    .hero {
        padding: 35px 25px;
        border-radius: 50px;
        background: linear-gradient(
            120deg,
            #075985 0%,
            #0891B2 45%,
            #2DD4BF 100%
        );
        margin-bottom: 28px;
        box-shadow: 0 10px 30px rgba(8, 145, 178, 0.20);
    }

    .hero h1 {
        color: #FFFFFF !important;
        font-size: 2.1rem;
        margin: 8px 0;
    }

    .hero p {
        color: #E0F7FA !important;
        font-size: 1rem;
        margin-bottom: 0;
    }

    /* Sidebar water gradient */
    [data-testid="stSidebar"] {
        background: linear-gradient(
            160deg,
            #075985 0%,
            #0891B2 50%,
            #2DD4BF 100%
        );
    }

    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] span {
        color: #FFFFFF !important;
    }

    /* Complete sidebar input group */
    .st-key-water_input_group {
        background: linear-gradient(
            145deg,
            #075985 0%,
            #0891B2 55%,
            #2DD4BF 100%
        );
        padding: 24px 18px;
        border-radius: 100px;
        border: 1px solid rgba(255, 255, 255, 0.35);
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15);
    }

    /* Number input fields */
    [data-testid="stNumberInput"] input {
        background-color: #FFFFFF !important;
        color: #111827 !important;
        border: 1px solid #111827 !important;
        border-radius: 10px !important;
    }

    /* Plus and minus controls */
    [data-testid="stNumberInput"] button {
        background-color: #111827 !important;
        color: #FFFFFF !important;
        border: 1px solid #111827 !important;
    }

    /* Prediction button */
    div.stButton > button[kind="primary"] {
        background-color: #111827 !important;
        color: #FFFFFF !important;
        border: 1px solid #111827 !important;
        border-radius: 15px !important;
        font-weight: 700 !important;
        min-height: 3rem;
    }

    div.stButton > button[kind="primary"]:hover {
        background-color: #374151 !important;
        color: #FFFFFF !important;
    }

    /* Prediction result panels */
    .prediction-success {
        background: linear-gradient(
            120deg,
            #047857,
            #0D9488,
            #2DD4BF
        );
        color: #FFFFFF !important;
        padding: 25px;
        border-radius: 30px;
        margin: 18px 0;
        border: 1px solid #5EEAD4;
        font-size: 18px;
        font-weight: 700;
    }

    .prediction-error {
        background: linear-gradient(
            120deg,
            #B91C1C,
            #DC2626,
            #F87171
        );
        color: #FFFFFF !important;
        padding: 25px;
        border-radius: 30px;
        margin: 18px 0;
        border: 1px solid #FCA5A5;
        font-size: 18px;
        font-weight: 700;
    }

    /* Probability cards */
    .probability-card {
        background: linear-gradient(
            135deg,
            #075985,
            #0891B2,
            #164E63
        );
        padding: 22px 14px;
        border-radius: 30px;
        margin: 10px 0;
        text-align: center;
        color: #FFFFFF !important;
        min-height: 145px;
    }

    .probability-card h4,
    .probability-card h2,
    .probability-card p {
        color: #FFFFFF !important;
    }

    .probability-card h2 {
        font-size: 30px;
        margin: 8px 0;
    }

    /* Charcoal panel around the current measurements table */
    .st-key-input_measurements {
        background: linear-gradient(
            135deg,
            #111827 0%,
            #374151 55%,
            #1F2937 100%
        );
        padding: 24px;
        border-radius: 30px;
        margin-bottom: 20px;
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.15);
    }

    .st-key-input_measurements h1,
    .st-key-input_measurements h2,
    .st-key-input_measurements h3,
    .st-key-input_measurements p {
        color: #FFFFFF !important;
    }

    /* Evaluation metric cards */
    div[data-testid="stMetric"] {
        background: linear-gradient(
            135deg,
            #111827,
            #374151,
            #1F2937
        );
        padding: 20px;
        border-radius: 25px;
        border: 1px solid #4B5563;
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.15);
    }

    div[data-testid="stMetric"] label,
    div[data-testid="stMetric"] [data-testid="stMetricValue"],
    div[data-testid="stMetric"] [data-testid="stMetricDelta"] {
        color: #FFFFFF !important;
    }

    /* Dataframe frame */
    div[data-testid="stDataFrame"] {
        border-radius: 14px;
        overflow: hidden;
    }

    /* Expanders and captions */
    .stApp [data-testid="stCaptionContainer"] {
        color: #374151;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# LOAD PROJECT DATA
# ---------------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


@st.cache_resource
def load_model():
    # Load only the trusted model file created for this project.
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_metrics():
    return json.loads(METRICS_PATH.read_text(encoding="utf-8"))


# ---------------------------------------------------------
# PAGE HEADER
# ---------------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <h1 style="text-align: center;">UI Created By Ali Jamil</h1>
        <h1 style="text-align: center;">💧 Water Potability Prediction</h1>
        <p style="text-align: center;">
            Machine-learning dashboard • Random Forest binary classification
            • Prediction classes: 0 and 1
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# CHECK REQUIRED FILES
# ---------------------------------------------------------
missing_files = [
    path.name
    for path in [DATA_PATH, MODEL_PATH, METRICS_PATH]
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


# ---------------------------------------------------------
# SIDEBAR INPUTS
# ---------------------------------------------------------
values = {}

with st.sidebar.container(key="water_input_group"):
    st.header("💧 Water-quality inputs")
    st.caption("Enter measurements, then run the prediction.")

    for feature in FEATURES:
        series = df[feature].dropna()

        default_value = float(series.median())
        observed_min = float(series.min())
        observed_max = float(series.max())

        values[feature] = st.number_input(
            FRIENDLY_NAMES[feature],
            value=default_value,
            step=max((observed_max - observed_min) / 100, 0.01),
            format="%.4f",
            help=(
                f"Observed dataset range: "
                f"{observed_min:.3f} to {observed_max:.3f}"
            ),
        )


input_df = pd.DataFrame([values], columns=FEATURES)


# ---------------------------------------------------------
# PREDICTION AND INPUT DISPLAY
# ---------------------------------------------------------
st.subheader("Water sample prediction")
predict_col, input_col = st.columns([1.05, 1])

with predict_col:
    st.write("Run the trained model on the values entered in the sidebar.")

    if st.button(
        "Predict potability",
        type="primary",
        use_container_width=True,
    ):
        predicted_class = int(model.predict(input_df)[0])
        probabilities = model.predict_proba(input_df)[0]

        class_to_probability = {
            int(label): float(probability)
            for label, probability in zip(model.classes_, probabilities)
        }

        st.session_state["last_prediction"] = predicted_class
        st.session_state["last_probabilities"] = {
            "potable": class_to_probability.get(1, 0.0),
            "not_potable": class_to_probability.get(0, 0.0),
        }

    # Show the last prediction after the button is clicked.
    if "last_prediction" in st.session_state:
        predicted_class = st.session_state["last_prediction"]
        probs = st.session_state["last_probabilities"]

        if predicted_class == 1:
            st.markdown(
                """
                <div class="prediction-success">
                    ✅ WATER IS POTABLE
                    <br><br>
                    Water is classified as potable by the machine-learning model.
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
                <div class="prediction-error">
                    ⚠️ WATER IS NOT POTABLE
                    <br><br>
                    Water is classified as not potable by the machine-learning model.
                </div>
                """,
                unsafe_allow_html=True,
            )

        prob1, prob2 = st.columns(2)

        with prob1:
            st.markdown(
                f"""
                <div class="probability-card">
                    <h4>Estimated class 1 probability</h4>
                    <h2>{probs['potable'] * 100:.1f}%</h2>
                    <p>Potable</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with prob2:
            st.markdown(
                f"""
                <div class="probability-card">
                    <h4>Estimated class 0 probability</h4>
                    <h2>{probs['not_potable'] * 100:.1f}%</h2>
                    <p>Not Potable</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.caption(
            "Model estimates only. These results do not guarantee "
            "drinking-water safety."
        )
    else:
        st.info(
            "Enter the sample measurements in the sidebar "
            "and click Predict Potability."
        )


with input_col:
    with st.container(key="input_measurements"):
        st.markdown("### Current input measurements")

        display_input = input_df.rename(columns=FRIENDLY_NAMES).T
        display_input.columns = ["Entered value"]

        st.dataframe(
            display_input,
            use_container_width=True,
        )


# ---------------------------------------------------------
# MODEL EVALUATION
# ---------------------------------------------------------
st.divider()

st.markdown(
    """
    <div class="hero">
        <h1 style="text-align: center;">📊 Model Evaluation</h1>
        <p style="text-align: center;">
            Performance on the held-out test dataset
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

metric_cols = st.columns(4)

metric_cols[0].metric(
    "Accuracy",
    f"{metrics['accuracy'] * 100:.2f}%",
)
metric_cols[1].metric(
    "Precision",
    f"{metrics['precision'] * 100:.2f}%",
)
metric_cols[2].metric(
    "Recall",
    f"{metrics['recall'] * 100:.2f}%",
)
metric_cols[3].metric(
    "F1-score",
    f"{metrics['f1_score'] * 100:.2f}%",
)


# ---------------------------------------------------------
# CONFUSION MATRIX
# ---------------------------------------------------------
cm = np.asarray(metrics["confusion_matrix"], dtype=int)

fig, ax = plt.subplots(figsize=(6.5, 4.5))

ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Not potable (0)", "Potable (1)"],
).plot(
    ax=ax,
    cmap="GnBu",
    colorbar=False,
    values_format="d",
)

ax.set_title("Random Forest confusion matrix")
fig.tight_layout()

st.pyplot(fig, use_container_width=True)
plt.close(fig)


# ---------------------------------------------------------
# PROJECT INFORMATION AND LIMITATIONS
# ---------------------------------------------------------
with st.expander("About this project and its limitations"):
    st.write(
        "This educational application uses a Random Forest classifier. "
        "The target is Potability: 0 means not potable and 1 means potable. "
        "Missing feature values are imputed using medians learned from the "
        "training data. A model prediction must not replace accredited "
        "laboratory water testing or public-health guidance."
    )
