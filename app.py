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

st.set_page_config(
    page_title="Water Potability | ML Dashboard",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded",
)

def render_html(content):
    html = "\n".join(
        line.lstrip()
        for line in content.splitlines()
    ).strip()

    st.markdown(
        html,
        unsafe_allow_html=True
    )

render_html("""
<style>

/* Main application background */
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
.stApp p {
    color: #111827;
}

/* Hero section */
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
/* Sidebar */
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
/* Rounded number input group */
[data-testid="stSidebar"] [data-testid="stNumberInput"] > div {
    border: 1px solid #111827 !important;
    border-radius: 12px !important;
    overflow: hidden !important;
    background: #FFFFFF !important;
}
/* Number input text field */
[data-testid="stSidebar"] [data-testid="stNumberInput"] input {
    background: #FFFFFF !important;
    color: #111827 !important;
    border: none !important;
    border-radius: 0 !important;
    box-shadow: none !important;
}
/* Increment/decrement controls */
[data-testid="stSidebar"] [data-testid="stNumberInput"] button {
    background: #111827 !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 0 !important;
}
/* Observed dataset range */
.range-info {
    background: #FFFFFF !important;
    color: #111827 !important;
    padding: 7px 10px;
    margin-top: 6px;
    margin-bottom: 13px;
    font-size: 12px;
    font-weight: 600;
    border-radius: 7px;
    border-left: 3px solid #075985;
}
.range-info span {
    color: #111827 !important;
}
/* Predict button */
/* Predict Potability button */
div.stButton > button[kind="primary"],
div.stButton > button[data-testid="stBaseButton-primary"] {
    background-color: #000000 !important;
    color: #F4B21A !important;
    border: 1px solid #000000 !important;
    border-radius: 9px !important;
    font-weight: 700 !important;
    min-height: 3rem;
    transition: color 0.25s ease, border-color 0.25s ease;
}
div.stButton > button[kind="primary"] *,
div.stButton > button[data-testid="stBaseButton-primary"] * {
    color: #F4B21A !important;
}
div.stButton > button[kind="primary"]:hover,
div.stButton > button[data-testid="stBaseButton-primary"]:hover {
    background-color: #000000 !important;
    color: #F4B21A !important;
    border: 1px solid #F4B21A !important;
    box-shadow: 0 0 10px rgba(244, 178, 26, 0.25);
}
div.stButton > button[kind="primary"]:hover *,
div.stButton > button[data-testid="stBaseButton-primary"]:hover * {
    color: #F4B21A !important;
}
/* Prediction result */
.prediction-panel {
    padding: 22px;
    margin-top: 18px;
    margin-bottom: 14px;
    border-radius: 14px;
    color: #FFFFFF !important;
    box-shadow: 0 8px 22px rgba(0, 0, 0, 0.18);
}
.prediction-success {
    background: linear-gradient(
        120deg,
        #14532D 0%,
        #15803D 50%,
        #22C55E 100%
    );
}
.prediction-error {
    background: linear-gradient(
        120deg,
        #7F1D1D 0%,
        #B91C1C 50%,
        #EF4444 100%
    );
}
.prediction-panel h2 {
    color: #FFFFFF !important;
    font-size: 21px;
    margin-top: 0;
    margin-bottom: 9px;
}
.prediction-panel p {
    color: #FFFFFF !important;
    font-size: 14px;
    margin-bottom: 0;
}
/* Probability cards */
.probability-card {
    background: linear-gradient(
        135deg,
        #000000 0%,
        #202020 55%,
        #101010 100%
    );
    padding: 17px 12px;
    border-radius: 12px;
    text-align: center;
    border: 1px solid #383838;
    box-shadow: 0 6px 18px rgba(0, 0, 0, 0.20);
    min-height: 112px;
}
.probability-card h4 {
    color: #D4D4D4 !important;
    font-size: 13px;
    margin: 0;
}
.probability-card h2 {
    color: #FFFFFF !important;
    font-size: 25px;
    margin: 10px 0;
}
.probability-card p {
    color: #AAAAAA !important;
    font-size: 11px;
    margin: 0;
}
/* Current input measurements */
.measurement-panel {
    background: linear-gradient(
        135deg,
        #080808 0%,
        #292929 55%,
        #111111 100%
    );
    padding: 24px;
    border-radius: 12px;
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
/* Model evaluation header */
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
    border-radius: 14px;
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
/* Evaluation metric cards */
div[data-testid="stMetric"] {
    background: linear-gradient(
        135deg,
        #000000 0%,
        #171717 55%,
        #080808 100%
    ) !important;
    padding: 22px !important;
    border-radius: 15px !important;
    border: 1px solid #303030 !important;
    box-shadow: 0 5px 15px rgba(0, 0, 0, 0.20);
}
/* Metric labels and values: force gold text inside each card */
div[data-testid="stMetric"] label,
div[data-testid="stMetric"] [data-testid="stMetricLabel"],
div[data-testid="stMetric"] [data-testid="stMetricLabel"] *,
div[data-testid="stMetric"] [data-testid="stMetricValue"],
div[data-testid="stMetric"] [data-testid="stMetricValue"] *,
div[data-testid="stMetric"] [data-testid="stMetricDelta"],
div[data-testid="stMetric"] [data-testid="stMetricDelta"] * {
    color: #F4B21A !important;
}
div[data-testid="stMetric"] [data-testid="stMetricValue"] {
    font-weight: 700 !important;
}
/* Confusion matrix heading */
.confusion-title {
    color: #111827 !important;
    text-align: center;
    font-size: 22px;
    font-weight: 750;
    margin-top: 28px;
    margin-bottom: 15px;
}
/* Captions */
.stApp [data-testid="stCaptionContainer"] {
    color: #374151;
}
</style>
""")

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
render_html("""
<div class="hero">
    <h1 style="text-align: center;">UI Created By Ali Jamil</h1>



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



    # The divider has been removed to avoid the unwanted

    # horizontal line in the sidebar.



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

            format="%.4f"

        )



        st.markdown(

            f"""

            <div class="range-info">

                Observed range:

                <span>{observed_min:.3f}</span>

                to

                <span>{observed_max:.3f}</span>

            </div>

            """,

            unsafe_allow_html=True

        )





# ---------------------------------------------------------

# PREPARE INPUT DATA

# ---------------------------------------------------------



input_df = pd.DataFrame(

    [values],

    columns=FEATURES

)





# ---------------------------------------------------------

# WATER SAMPLE PREDICTION AND MEASUREMENTS

# ---------------------------------------------------------



st.subheader("Water sample prediction")



predict_col, input_col = st.columns([1.05, 1])





# ---------------------------------------------------------

# PREDICTION BUTTON AND RESULT

# ---------------------------------------------------------



with predict_col:

    st.write(

        "Enter the measurements and click the button to predict water potability."

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



        probabilities = model.predict_proba(input_df)[0]



        class_probabilities = {

            int(label): float(probability)

            for label, probability in zip(

                model.classes_,

                probabilities

            )

        }



        st.session_state["last_prediction"] = predicted_class



        st.session_state["last_probabilities"] = {

            "potable": class_probabilities.get(1, 0.0),

            "not_potable": class_probabilities.get(0, 0.0)

        }



        # Keep the result linked to the values that produced it.

        st.session_state["last_input_values"] = values.copy()



    # Show the prediction and probability cards directly below

    # the button, in the empty area marked in the reference image.



    if "last_prediction" in st.session_state:

        predicted_class = st.session_state["last_prediction"]

        probabilities = st.session_state["last_probabilities"]



        if "last_input_values" in st.session_state:

            same_input = all(

                np.isclose(

                    values[feature],

                    st.session_state["last_input_values"][feature]

                )

                for feature in FEATURES

            )

        else:

            same_input = False



        if same_input:

            if predicted_class == 1:

                render_html("""

                <div class="prediction-panel prediction-success">

                    <h2>✓ Water is good for health and drinkable</h2>

                    <p>

                        The Random Forest model classifies this sample

                        as potable water.

                    </p>

                </div>

                """)

            else:

                render_html("""

                <div class="prediction-panel prediction-error">

                    <h2>✕ Water is not drinkable and dangerous for health</h2>

                    <p>

                        The Random Forest model classifies this sample

                        as non-potable water.

                    </p>

                </div>

                """)



            probability_col1, probability_col2 = st.columns(2)



            with probability_col1:

                render_html(f"""

                <div class="probability-card">

                    <h4>Potable Probability</h4>

                    <h2>{probabilities['potable'] * 100:.2f}%</h2>

                    <p>Class 1 - Potable</p>

                </div>

                """)



            with probability_col2:

                render_html(f"""

                <div class="probability-card">

                    <h4>Non-Potable Probability</h4>

                    <h2>{probabilities['not_potable'] * 100:.2f}%</h2>

                    <p>Class 0 - Not potable</p>

                </div>

                """)



            st.caption(

                "These percentages are model estimates, not laboratory "

                "measurements. A prediction alone cannot establish "

                "drinking-water safety."

            )

        else:

            st.info(

                "The input values have changed. Click Predict Potability "

                "again to calculate a result for the updated measurements."

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





# ---------------------------------------------------------

# MODEL EVALUATION

# ---------------------------------------------------------



st.divider()



render_html("""

<div class="evaluation-header">

    <h2>📊 Model Evaluation</h2>

    <p>

        Performance on the held-out test dataset

    </p>

</div>

""")





# ---------------------------------------------------------

# MODEL EVALUATION METRICS

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
