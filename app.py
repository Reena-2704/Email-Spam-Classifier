
import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns


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


# -------------------- Dashboard Styling --------------------

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


# -------------------- Load Saved Files --------------------

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_metrics():
    if METRICS_PATH.exists():
        with open(METRICS_PATH, "r", encoding="utf-8") as file:
            return json.load(file)
    return {}


@st.cache_data
def load_comparison():
    if COMPARISON_PATH.exists():
        return pd.read_csv(COMPARISON_PATH)
    return pd.DataFrame()


@st.cache_data
def load_keywords():
    if KEYWORDS_PATH.exists():
        return pd.read_csv(KEYWORDS_PATH)
    return pd.DataFrame()


# -------------------- Page Header --------------------

st.title("✉️ Email Spam Classifier")
st.caption("MESSAGE CLASSIFICATION / MACHINE LEARNING")
st.divider()


# -------------------- Check Model --------------------

if not MODEL_PATH.exists():
    st.error("Model file not found. Run `python train_model.py` first.")
    st.stop()

model = load_model()
metrics = load_metrics()
comparison = load_comparison()
keywords = load_keywords()


# -------------------- Message Analyzer --------------------

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
        st.session_state.analyzed_message = message.strip()
        st.session_state.prediction = model.predict(
            [message.strip()]
        )[0]


if st.session_state.get("analyzed_message") is not None:
    st.divider()
    st.subheader("Classification Result")

    prediction = st.session_state.prediction

    if str(prediction).lower() == "spam":
        st.error("🚨 SPAM DETECTED")
    else:
        st.success("✅ LEGITIMATE MESSAGE")

    # Show spam probability only if supported by the model
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(
            [st.session_state.analyzed_message]
        )[0]

        classes = list(model.classes_)

        if "spam" in classes:
            spam_index = classes.index("spam")
            spam_score = probabilities[spam_index]

            st.metric(
                "Estimated Spam Score",
                f"{spam_score:.1%}"
            )

    st.divider()

    # -------------------- Dashboard Tabs --------------------

    performance_tab, keywords_tab = st.tabs(
        ["Model Performance", "Spam Keywords"]
    )

    # -------------------- Model Performance --------------------

    with performance_tab:
        st.subheader("Evaluation Results")

        results = metrics.get("results", [])
        best_name = metrics.get("best_model")

        best_result = None

        if results:
            best_result = next(
                (
                    item for item in results
                    if item.get("model") == best_name
                ),
                results[0]
            )

            c1, c2, c3, c4 = st.columns(4)

            c1.metric(
                "Selected Model",
                best_name or "N/A"
            )
            c2.metric(
                "Accuracy",
                f'{best_result.get("accuracy", 0):.1%}'
            )
            c3.metric(
                "Precision",
                f'{best_result.get("precision", 0):.1%}'
            )
            c4.metric(
                "F1 Score",
                f'{best_result.get("f1_score", 0):.1%}'
            )

            st.caption(
                f'Training samples: {metrics.get("training_samples", "N/A")} '
                f'| Test samples: {metrics.get("test_samples", "N/A")}'
            )
        else:
            st.info("Evaluation metrics are unavailable.")

        # -------------------- Model Comparison --------------------

        st.divider()
        st.subheader("Model Comparison")

        if not comparison.empty:
            display_columns = [
                "model",
                "accuracy",
                "precision",
                "recall",
                "f1_score",
                "false_positive_rate"
            ]

            display_columns = [
                col for col in display_columns
                if col in comparison.columns
            ]

            formatted_comparison = comparison[display_columns].copy()

            percentage_columns = [
                col for col in display_columns
                if col != "model"
            ]

            st.dataframe(
                formatted_comparison.style.format({
                    col: "{:.1%}"
                    for col in percentage_columns
                }),
                use_container_width=True,
                hide_index=True
            )

            if {"model", "accuracy", "f1_score"}.issubset(
                comparison.columns
            ):
                chart_data = comparison.set_index("model")[
                    ["accuracy", "f1_score"]
                ]

                st.bar_chart(chart_data)

        else:
            st.info(
                "Model comparison file not found. "
                "Check models/model_comparison.csv."
            )

        # -------------------- Confusion Matrix --------------------

        st.divider()
        st.subheader("Confusion Matrix")

        if best_result is not None:
            # Support common key naming conventions
            tn = best_result.get("tn", best_result.get("TN"))
            fp = best_result.get("fp", best_result.get("FP"))
            fn = best_result.get("fn", best_result.get("FN"))
            tp = best_result.get("tp", best_result.get("TP"))

            # Also support a nested confusion matrix, if present
            matrix = best_result.get("confusion_matrix")

            if matrix is not None:
                try:
                    if len(matrix) == 2 and all(
                        len(row) == 2 for row in matrix
                    ):
                        tn, fp = matrix[0]
                        fn, tp = matrix[1]
                except (TypeError, ValueError):
                    pass

            if all(value is not None for value in [tn, fp, fn, tp]):
                confusion_data = [
                    [int(tn), int(fp)],
                    [int(fn), int(tp)]
                ]

                fig, ax = plt.subplots(figsize=(7, 5))

                sns.heatmap(
                    confusion_data,
                    annot=True,
                    fmt="d",
                    cmap="Blues",
                    xticklabels=["Legitimate", "Spam"],
                    yticklabels=["Legitimate", "Spam"],
                    linewidths=1,
                    linecolor="white",
                    cbar=False,
                    ax=ax
                )

                ax.set_xlabel("Predicted Label")
                ax.set_ylabel("Actual Label")
                ax.set_title(f"Confusion Matrix - {best_name}")

                st.pyplot(fig)
                plt.close(fig)

                st.markdown("""
                **How to read the confusion matrix**
                - **TN:** Legitimate messages correctly classified.
                - **FP:** Legitimate messages incorrectly marked as spam.
                - **FN:** Spam messages incorrectly classified as legitimate.
                - **TP:** Spam messages correctly detected.
                """)

            else:
                st.warning(
                    "Confusion matrix counts are not available in "
                    "models/metrics.json. Update train_model.py to save "
                    "TN, FP, FN and TP, then retrain the model."
                )
        else:
            st.info(
                "Confusion matrix unavailable because evaluation "
                "results were not found."
            )

    # -------------------- Spam Keywords --------------------

    with keywords_tab:
        st.subheader("Spam-Associated Keywords")

        if not keywords.empty:
            if {
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
                st.warning(
                    "The keyword CSV must contain 'keyword' and "
                    "'spam_score' columns."
                )
        else:
            st.info(
                "Keyword data not found. "
                "Check models/spam_keywords.csv."
            )


# -------------------- Footer --------------------

st.divider()
st.caption("Email Spam Classifier · Machine Learning Project")
