from pathlib import Path
import json
import joblib
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
    page_title="Water Potability ML Dashboard",
    page_icon="💧",
    layout="wide"
)

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_metrics():
    return json.loads(METRICS_PATH.read_text(encoding="utf-8"))

st.title("💧 Water Potability Prediction")
st.caption("Interactive machine-learning demonstration using the Water Potability dataset.")

if not DATA_PATH.exists():
    st.error("Dataset not found. Place water_potability.csv inside the data folder.")
    st.stop()

if not MODEL_PATH.exists() or not METRICS_PATH.exists():
    st.warning("The trained model has not been created yet. Run `python train_model.py` first.")
    st.code("python train_model.py")
    st.stop()

df = load_data()
model = load_model()
metrics = load_metrics()

st.sidebar.header("Enter water-quality measurements")
st.sidebar.caption("Change the values below, then click Predict. Defaults use the dataset medians.")

values = {}
for feature in FEATURES:
    series = df[feature].dropna()
    min_value = float(series.min())
    max_value = float(series.max())
    default_value = float(series.median())
    # Number inputs permit values outside the observed dataset range too.
    values[feature] = st.sidebar.number_input(
        FRIENDLY_NAMES[feature],
        min_value=None,
        max_value=None,
        value=default_value,
        step=max((max_value - min_value) / 100, 0.01),
        format="%.4f",
        help=f"Observed dataset range: {min_value:.3f} to {max_value:.3f}"
    )

input_df = pd.DataFrame([values], columns=FEATURES)

left, right = st.columns([1.15, 1])
with left:
    st.subheader("Prediction")
    if st.button("Predict water potability", type="primary", use_container_width=True):
        prediction = int(model.predict(input_df)[0])
        probabilities = model.predict_proba(input_df)[0]
        potable_probability = float(probabilities[list(model.classes_).index(1)])
        not_potable_probability = float(probabilities[list(model.classes_).index(0)])

        if prediction == 1:
            st.success("Model prediction: POTABLE (class 1)")
        else:
            st.error("Model prediction: NOT POTABLE (class 0)")

        c1, c2 = st.columns(2)
        c1.metric("Estimated potable probability", f"{potable_probability * 100:.1f}%")
        c2.metric("Estimated not-potable probability", f"{not_potable_probability * 100:.1f}%")
        st.caption(
            "These probabilities are model estimates, not a certified laboratory "
            "assessment or a guarantee that water is safe to drink."
        )
    else:
        st.info("Adjust the measurements in the sidebar and click **Predict water potability**.")

with right:
    st.subheader("Input values")
    st.dataframe(
        input_df.rename(columns=FRIENDLY_NAMES).T.rename(columns={0: "Entered value"}),
        use_container_width=True
    )

st.divider()
st.header("Model evaluation")

m1, m2, m3, m4 = st.columns(4)
m1.metric("Accuracy", f"{metrics['accuracy'] * 100:.2f}%")
m2.metric("Precision", f"{metrics['precision'] * 100:.2f}%")
m3.metric("Recall", f"{metrics['recall'] * 100:.2f}%")
m4.metric("F1-score", f"{metrics['f1_score'] * 100:.2f}%")

# Confusion Matrix

cm = np.array(metrics["confusion_matrix"])

fig, ax = plt.subplots(figsize=(7, 5))

ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=[
        "Not Potable (0)",
        "Potable (1)"
    ]
).plot(
    ax=ax,
    cmap="Blues",
    colorbar=False
)

ax.set_title("Random Forest Confusion Matrix")

ax.set_xlabel("Predicted Label")

ax.set_ylabel("Actual Label")

st.pyplot(fig)

plt.close(fig)
st.divider()
st.header("Dataset visualizations")

chart_feature = st.selectbox(
    "Choose a feature to explore",
    FEATURES,
    format_func=lambda x: FRIENDLY_NAMES[x]
)
fig2, ax2 = plt.subplots(figsize=(9, 4))
ax2.hist(df[chart_feature].dropna(), bins=30, edgecolor="black", alpha=0.75)
ax2.axvline(
    values[chart_feature],
    linestyle="--",
    linewidth=2,
    label=f"Current input: {values[chart_feature]:.3f}"
)
ax2.set_xlabel(FRIENDLY_NAMES[chart_feature])
ax2.set_ylabel("Number of dataset records")
ax2.set_title(f"Dataset distribution: {FRIENDLY_NAMES[chart_feature]}")
ax2.legend()
st.pyplot(fig2)
plt.close(fig2)

st.subheader("Potability class distribution")
class_counts = df["Potability"].value_counts().sort_index()
class_chart = pd.DataFrame({
    "Water class": ["Not potable (0)", "Potable (1)"],
    "Records": [int(class_counts.get(0, 0)), int(class_counts.get(1, 0))]
}).set_index("Water class")
st.bar_chart(class_chart)

with st.expander("About this project and its limitations"):
    st.write(
        "This application uses a Random Forest classification model. The target "
        "column `Potability` has two classes: 0 (not potable) and 1 (potable). "
        "Missing feature values are imputed with training-set medians inside the "
        "machine-learning pipeline. The model is intended for educational "
        "demonstration only; it must not replace accredited water testing or "
        "public-health guidance."
    )
