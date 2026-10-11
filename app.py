
import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Email Spam Classifier",
    page_icon="✉️",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "spam_classifier.pkl"
METRICS_PATH = BASE_DIR / "models" / "metrics.json"
KEYWORDS_PATH = BASE_DIR / "models" / "spam_keywords.csv"
COMPARISON_PATH = BASE_DIR / "models" / "model_comparison.csv"

st.markdown("""
<style>
.stApp {
    background: #ffffff;
    color: #172033;
}
header[data-testid="stHeader"] {
    background: #ffffff;
}
.block-container {
    max-width: 1150px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}
h1, h2, h3, p, label {
    color: #172033;
}
h1 {
    font-weight: 750;
    letter-spacing: -1px;
}
div[data-testid="stMetric"] {
    background: #f7f9fc;
    border: 1px solid #e5eaf2;
    border-radius: 12px;
    padding: 16px;
}
.stTextArea textarea {
    background: #ffffff;
    color: #172033;
    border: 1px solid #d5dce7;
    border-radius: 10px;
}
.stButton button {
    background: #2457d6;
    color: #ffffff;
    border: 0;
    border-radius: 9px;
    padding: 0.6rem 1.2rem;
    font-weight: 600;
}
.stButton button:hover {
    background: #1744b5;
    color: #ffffff;
}
hr {
    border-color: #e5eaf2;
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_metrics():
    if METRICS_PATH.exists():
        with open(METRICS_PATH, "r", encoding="utf-8") as file:
            return json.load(file)
    return {}


st.title("✉️ Email Spam Classifier")
st.caption("MESSAGE CLASSIFICATION / MACHINE LEARNING")
st.divider()

if not MODEL_PATH.exists():
    st.error("Model file not found. Run `python train_model.py` first.")
    st.stop()

model = load_model()
metrics = load_metrics()

if "analyzed_message" not in st.session_state:
    st.session_state.analyzed_message = None
    st.session_state.prediction = None

st.subheader("Message Analyzer")

message = st.text_area(
    "Enter your message",
    height=170,
    placeholder="Paste the message you want to classify..."
)

if st.button("Analyze Message", use_container_width=True):
    if not message.strip():
        st.warning("Enter a message first.")
    else:
        st.session_state.analyzed_message = message
        st.session_state.prediction = model.predict([message])[0]

if st.session_state.analyzed_message is not None:
    st.divider()
    st.subheader("Classification Result")

    prediction = st.session_state.prediction

    if prediction == "spam":
        st.error("SPAM DETECTED")
    else:
        st.success("LEGITIMATE MESSAGE")

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(
            [st.session_state.analyzed_message]
        )[0]
        classes = list(model.classes_)
        spam_index = classes.index("spam")

        st.metric(
            "Estimated Spam Score",
            f"{probabilities[spam_index]:.1%}"
        )

    st.divider()

    performance_tab, keywords_tab = st.tabs(
        ["Model Performance", "Spam Keywords"]
    )

    with performance_tab:
        st.subheader("Evaluation Results")

        results = metrics.get("results", [])
        best_name = metrics.get("best_model")

        if results:
            best_result = next(
                (item for item in results if item["model"] == best_name),
                results[0]
            )

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Selected Model", best_name or "N/A")
            c2.metric("Accuracy", f'{best_result["accuracy"]:.1%}')
            c3.metric("Precision", f'{best_result["precision"]:.1%}')
            c4.metric("F1 Score", f'{best_result["f1_score"]:.1%}')

            st.caption(
                f'Training samples: {metrics.get("training_samples", "N/A")} '
                f'| Test samples: {metrics.get("test_samples", "N/A")}'
            )
        else:
            st.info("Evaluation metrics are unavailable.")

        if COMPARISON_PATH.exists():
            comparison = pd.read_csv(COMPARISON_PATH)
            st.subheader("Model Comparison")

            columns = [
                "model", "accuracy", "precision",
                "recall", "f1_score", "false_positive_rate"
            ]
            columns = [c for c in columns if c in comparison.columns]

            st.dataframe(
                comparison[columns].style.format({
                    c: "{:.1%}" for c in columns
                    if c != "model"
                }),
                use_container_width=True,
                hide_index=True
            )

            if {"model", "accuracy", "f1_score"}.issubset(comparison.columns):
                st.bar_chart(
                    comparison.set_index("model")[["accuracy", "f1_score"]]
                )

    with keywords_tab:
        st.subheader("Spam-Associated Keywords")

        if KEYWORDS_PATH.exists():
            keywords = pd.read_csv(KEYWORDS_PATH)

            if not keywords.empty and {
                "keyword", "spam_score"
            }.issubset(keywords.columns):
                st.bar_chart(
                    keywords.head(15).set_index("keyword")[["spam_score"]]
                )
                st.dataframe(
                    keywords.head(20),
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.info("No keyword data available.")
        else:
            st.info("Keyword data not found. Run the training script.")

st.divider()
st.caption("Email Spam Classifier · Machine Learning Project")
