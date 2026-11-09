from pathlib import Path
from textwrap import dedent
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


# ---------------------------------------------------------
# DATASET FEATURES
# ---------------------------------------------------------

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
# STREAMLIT CONFIGURATION
# ---------------------------------------------------------

st.set_page_config(
    page_title="Water Potability | ML Dashboard",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------
# HTML RENDERING HELPER
# ---------------------------------------------------------

def render_html(content):
    st.markdown(
        dedent(content).strip(),
        unsafe_allow_html=True
    )


# ---------------------------------------------------------
# CUSTOM CSS
# ---------------------------------------------------------

render_html("""
<style>

/* -----------------------------------------------------
   MAIN APPLICATION
----------------------------------------------------- */

.stApp {
    background: linear-gradient(
        135deg,
        #F0FAFC 0%,
        #E1F5F8 50%,
        #F4FBFC 100%
    );

    color: #111827;
}

.stApp h1,
.stApp h2,
.stApp h3,
.stApp h4,
.stApp p,
.stApp label {
    color: #111827;
}


/* -----------------------------------------------------
   HERO SECTION
----------------------------------------------------- */

.hero {
    padding: 35px 25px;

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
    font-weight: 750;
    margin: 8px 0;
}

.hero p {
    color: #E0F7FA !important;
    font-size: 1rem;
    margin-bottom: 0;
}


/* -----------------------------------------------------
   SIDEBAR
----------------------------------------------------- */

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

/* Input fields */

[data-testid="stSidebar"] [data-testid="stNumberInput"] input {
    background: #FFFFFF !important;
    color: #111827 !important;
    border: 1px solid #111827 !important;
    border-radius: 5px !important;
}

/* Plus and minus buttons */

[data-testid="stSidebar"] [data-testid="stNumberInput"] button {
    background: #111827 !important;
    color: #FFFFFF !important;
    border: 1px solid #111827 !important;
}


/* -----------------------------------------------------
   PREDICTION BUTTON
----------------------------------------------------- */

div.stButton > button[kind="primary"] {
    background: #000000 !important;
    color: #FFFFFF !important;

    border: 1px solid #000000 !important;
    border-radius: 5px !important;

    font-weight: 700 !important;
    min-height: 3rem;
}

div.stButton > button[kind="primary"]:hover {
    background: #252525 !important;
    color: #FFFFFF !important;
    border-color: #252525 !important;
}


/* -----------------------------------------------------
   PREDICTION RESULT
----------------------------------------------------- */

.prediction-panel {
    background: linear-gradient(
        135deg,
        #050505 0%,
        #252525 55%,
        #101010 100%
    );

    padding: 26px;
    margin-top: 18px;
    margin-bottom: 18px;

    color: #FFFFFF !important;

    box-shadow: 0 8px 22px rgba(0, 0, 0, 0.20);
}

.prediction-panel h3 {
    color: #FFFFFF !important;
    margin-top: 0;
}

.prediction-panel p {
    color: #E5E5E5 !important;
    margin-bottom: 0;
}

.status-potable {
    display: inline-block;

    background: linear-gradient(
        120deg,
        #15803D,
        #16A34A
    );

    color: #FFFFFF !important;

    padding: 11px 18px;
    margin-bottom: 16px;

    font-size: 18px;
    font-weight: 750;
}

.status-not-potable {
    display: inline-block;

    background: linear-gradient(
        120deg,
        #B91C1C,
        #EF4444
    );

    color: #FFFFFF !important;

    padding: 11px 18px;
    margin-bottom: 16px;

    font-size: 18px;
    font-weight: 750;
}


/* -----------------------------------------------------
   CURRENT INPUT MEASUREMENTS
----------------------------------------------------- */

.measurement-panel {
    background: linear-gradient(
        135deg,
        #080808 0%,
        #292929 55%,
        #111111 100%
    );

    padding: 24px;

    box-shadow: 0 8px 20px rgba(0, 0, 0, 0.20);
}

.measurement-panel h3 {
    color: #FFFFFF !important;
    margin-top: 0;
    margin-bottom: 18px;
}

.measurement-panel table {
    width: 100%;
    border-collapse: collapse;
    color: #FFFFFF !important;
}

.measurement-panel th {
    background: #303030;
    color: #FFFFFF !important;

    text-align: left;
    padding: 12px;

    border-bottom: 1px solid #555555;
}

.measurement-panel td {
    background: #171717;
    color: #FFFFFF !important;

    padding: 11px 12px;

    border-bottom: 1px solid #383838;
}


/* -----------------------------------------------------
   MODEL EVALUATION HEADER
----------------------------------------------------- */

.evaluation-header {
    background: linear-gradient(
        135deg,
        #050505 0%,
        #292929 55%,
        #101010 100%
    );

    padding: 27px;

    margin-top: 25px;
    margin-bottom: 22px;

    box-shadow: 0 8px 20px rgba(0, 0, 0, 0.20);
}

.evaluation-header h2 {
    color: #FFFFFF !important;
    margin: 0;
    text-align: center;
    font-size: 1.8rem;
}

.evaluation-header p {
    color: #D4D4D4 !important;
    text-align: center;
    margin-top: 8px;
    margin-bottom: 0;
}


/* -----------------------------------------------------
   METRIC CARDS
----------------------------------------------------- */

div[data-testid="stMetric"] {
    background: linear-gradient(
        135deg,
        #000000 0%,
        #171717 55%,
        #080808 100%
    );

    padding: 20px;

    border: 1px solid #303030;

    box-shadow: 0 5px 15px rgba(0, 0, 0, 0.20);
}

/* Force all metric text to white */

div[data-testid="stMetric"] * {
    color: #FFFFFF !important;
}

div[data-testid="stMetric"] label {
    color: #D4D4D4 !important;
}

div[data-testid="stMetricValue"] {
    color: #FFFFFF !important;
}


/* -----------------------------------------------------
   CONFUSION MATRIX
----------------------------------------------------- */

.confusion-title {
    color: #111827 !important;

    text-align: center;

    font-size: 22px;
    font-weight: 750;

    margin-top: 28px;
    margin-bottom: 15px;
}


/* -----------------------------------------------------
   GENERAL
----------------------------------------------------- */

.stApp [data-testid="stCaptionContainer"] {
    color: #374151;
}

</style>
""")


# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_metrics():
    return json.loads(
        METRICS_PATH.read_text(encoding="utf-8")
    )


# ---------------------------------------------------------
# PAGE HEADER
# ---------------------------------------------------------

render_html("""
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
""")


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
        + ". Check your data/ and models/ folders."
    )

    st.stop()


# ---------------------------------------------------------
# LOAD PROJECT RESOURCES
# ---------------------------------------------------------

df = load_data()
model = load_model()
metrics = load_metrics()


# ---------------------------------------------------------
# SIDEBAR INPUTS
# ---------------------------------------------------------

values = {}

with st.sidebar:

    st.header("💧 Water-quality inputs")

    st.caption(
        "Enter the water measurements below."
    )

    st.divider()

    for feature in FEATURES:

        series = df[feature].dropna()

        default_value = float(series.median())

        observed_min = float(series.min())
        observed_max = float(series.max())

        values[feature] = st.number_input(
            FRIENDLY_NAMES[feature],

            value=default_value,

            step=max(
                (observed_max - observed_min) / 100,
                0.01
            ),

            format="%.4f",

            help=(
                f"Observed dataset range: "
                f"{observed_min:.3f} to {observed_max:.3f}"
            )
        )


# ---------------------------------------------------------
# PREPARE INPUT DATA
# ---------------------------------------------------------

input_df = pd.DataFrame(
    [values],
    columns=FEATURES
)


# ---------------------------------------------------------
# PREDICTION AND MEASUREMENTS
# ---------------------------------------------------------

st.subheader("Water sample prediction")

predict_col, input_col = st.columns([1.05, 1])


# ---------------------------------------------------------
# PREDICTION SECTION
# ---------------------------------------------------------

with predict_col:

    st.write(
        "Click the button below to predict water potability."
    )

    predict_clicked = st.button(
        "Predict Potability",
        type="primary",
        use_container_width=True
    )

    if predict_clicked:

        predicted_class = int(
            model.predict(input_df)[0]
        )

        st.session_state["last_prediction"] = predicted_class

    # Display the latest prediction below the button.

    if "last_prediction" in st.session_state:

        predicted_class = st.session_state["last_prediction"]

        if predicted_class == 1:

            render_html("""
            <div class="prediction-panel">

                <div class="status-potable">
                    ✓ WATER IS POTABLE
                </div>

                <h3>Prediction Result</h3>

                <p>
                    The machine-learning model classifies
                    this water sample as potable.
                </p>

            </div>
            """)

        else:

            render_html("""
            <div class="prediction-panel">

                <div class="status-not-potable">
                    ✕ WATER IS NOT POTABLE
                </div>

                <h3>Prediction Result</h3>

                <p>
                    The machine-learning model classifies
                    this water sample as not potable.
                </p>

            </div>
            """)

        st.caption(
            "Model prediction only. This result does not "
            "guarantee drinking-water safety."
        )


# ---------------------------------------------------------
# CURRENT INPUT MEASUREMENTS
# ---------------------------------------------------------

with input_col:

    measurement_rows = ""

    for feature in FEATURES:

        label = FRIENDLY_NAMES[feature]
        value = values[feature]

        measurement_rows += (
            f"<tr>"
            f"<td>{label}</td>"
            f"<td>{value:,.4f}</td>"
            f"</tr>"
        )

    render_html(f"""
    <div class="measurement-panel">

        <h3>Current input measurements</h3>

        <table>

            <thead>
                <tr>
                    <th>Measurement</th>
                    <th>Entered value</th>
                </tr>
            </thead>

            <tbody>
                {measurement_rows}
            </tbody>

        </table>

    </div>
    """)


st.divider()


# ---------------------------------------------------------
# MODEL EVALUATION
# ---------------------------------------------------------

render_html("""
<div class="evaluation-header">

    <h2>📊 Model Evaluation</h2>

    <p>
        Performance on the held-out test dataset
    </p>

</div>
""")


# ---------------------------------------------------------
# EVALUATION METRICS
# ---------------------------------------------------------

metric_cols = st.columns(4)

metric_cols[0].metric(
    "Accuracy",
    f"{metrics['accuracy'] * 100:.2f}%"
)

metric_cols[1].metric(
    "Precision",
    f"{metrics['precision'] * 100:.2f}%"
)

metric_cols[2].metric(
    "Recall",
    f"{metrics['recall'] * 100:.2f}%"
)

metric_cols[3].metric(
    "F1-score",
    f"{metrics['f1_score'] * 100:.2f}%"
)


# ---------------------------------------------------------
# RANDOM FOREST CONFUSION MATRIX
# ---------------------------------------------------------

render_html("""
<div class="confusion-title">
    Random Forest Confusion Matrix
</div>
""")

cm = np.asarray(
    metrics["confusion_matrix"],
    dtype=int
)

fig, ax = plt.subplots(
    figsize=(4.5, 3.4),
    dpi=120
)

ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=[
        "Not potable (0)",
        "Potable (1)"
    ]
).plot(
    ax=ax,
    cmap="GnBu",
    colorbar=False,
    values_format="d"
)

ax.set_title(
    "Random Forest",
    fontsize=11,
    pad=12
)

ax.tick_params(
    axis="both",
    labelsize=8
)

fig.tight_layout()

left_col, center_col, right_col = st.columns(
    [1, 1.5, 1]
)

with center_col:

    st.pyplot(
        fig,
        use_container_width=True
    )

plt.close(fig)


# ---------------------------------------------------------
# PROJECT INFORMATION AND LIMITATIONS
# ---------------------------------------------------------

with st.expander(
    "About this project and its limitations"
):

    st.write(
        "This educational application uses a Random Forest classifier. "
        "The target is Potability: 0 means not potable and 1 means potable. "
        "Missing feature values are imputed using medians learned from the "
        "training data. A model prediction must not replace accredited "
        "laboratory water testing or public-health guidance."
    )