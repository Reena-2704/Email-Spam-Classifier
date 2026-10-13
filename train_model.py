

import os
import json
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


# -------------------- File Paths --------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(
    BASE_DIR, "data", "SMSSpamCollection.tsv"
)

MODEL_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(MODEL_DIR, exist_ok=True)


# -------------------- Load and Clean Dataset --------------------

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH}\n"
        "Check that data/SMSSpamCollection.tsv exists."
    )

data = pd.read_csv(
    DATA_PATH,
    sep="\t",
    header=None,
    names=["label", "message"],
    encoding="latin-1"
)

data = data.dropna(subset=["label", "message"])
data["label"] = data["label"].astype(str).str.strip().str.lower()
data = data[data["label"].isin(["ham", "spam"])]
data = data.drop_duplicates(subset=["label", "message"])

X = data["message"].astype(str)
y = data["label"]

if data.empty:
    raise ValueError("The dataset contains no valid messages.")

print("Dataset loaded:", len(data), "messages")


# -------------------- Split Dataset --------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# -------------------- Define Models --------------------

classifiers = {
    "Naive Bayes": MultinomialNB(),

    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    ),

    "SVM": LinearSVC(
        class_weight="balanced"
    )
}


# -------------------- Train and Evaluate Models --------------------

results = []
best_model = None
best_name = None
best_f1 = -1

for name, classifier in classifiers.items():

    print("\nTraining:", name)

    model = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                lowercase=True,
                stop_words="english",
                ngram_range=(1, 2),
                max_features=10000
            )
        ),
        ("classifier", classifier)
    ])

    # Train model
    model.fit(X_train, y_train)

    # Predict test messages
    predictions = model.predict(X_test)

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, predictions)

    precision = precision_score(
        y_test,
        predictions,
        pos_label="spam",
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        pos_label="spam",
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        pos_label="spam",
        zero_division=0
    )

    # Confusion matrix:
    # Rows = actual labels; columns = predicted labels
    # Order: ham, spam
    cm = confusion_matrix(
        y_test,
        predictions,
        labels=["ham", "spam"]
    )

    tn, fp, fn, tp = cm.ravel()

    # False-positive rate
    fpr = fp / (fp + tn) if (fp + tn) else 0

    # Save all evaluation metrics and confusion matrix counts
    results.append({
        "model": name,
        "accuracy": round(float(accuracy), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1_score": round(float(f1), 4),
        "false_positive_rate": round(float(fpr), 4),

        # Short names used by the dashboard
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),

        # Descriptive names retained for clarity
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),

        # Full confusion matrix
        "confusion_matrix": cm.tolist()
    })

    print(
        f"Accuracy: {accuracy:.2%} | "
        f"Precision: {precision:.2%} | "
        f"Recall: {recall:.2%} | "
        f"F1 Score: {f1:.2%}"
    )

    print("Confusion Matrix:")
    print(cm)
    print(f"TN={tn}, FP={fp}, FN={fn}, TP={tp}")

    # Select the model with the highest F1 score
    if f1 > best_f1:
        best_f1 = f1
        best_model = model
        best_name = name


# -------------------- Save Best Model --------------------

model_path = os.path.join(
    MODEL_DIR,
    "spam_classifier.pkl"
)

joblib.dump(best_model, model_path)


# -------------------- Save Model Comparison --------------------

results_df = pd.DataFrame(results)

comparison_path = os.path.join(
    MODEL_DIR,
    "model_comparison.csv"
)

results_df.to_csv(
    comparison_path,
    index=False
)


# -------------------- Save Evaluation Metrics --------------------

metadata = {
    "best_model": best_name,
    "test_samples": int(len(y_test)),
    "training_samples": int(len(y_train)),
    "results": results
}

metrics_path = os.path.join(
    MODEL_DIR,
    "metrics.json"
)

with open(metrics_path, "w", encoding="utf-8") as file:
    json.dump(metadata, file, indent=4)


# -------------------- Extract Spam Keywords --------------------

vectorizer = best_model.named_steps["tfidf"]
classifier = best_model.named_steps["classifier"]

terms = vectorizer.get_feature_names_out()

if hasattr(classifier, "coef_"):

    # Linear models: coefficient scores
    scores = classifier.coef_[0]

    classes = list(classifier.classes_)

    if classes[1] != "spam":
        scores = -scores

elif hasattr(classifier, "feature_log_prob_"):

    # Naive Bayes: compare spam and ham feature probabilities
    classes = list(classifier.classes_)

    spam_index = classes.index("spam")
    ham_index = classes.index("ham")

    scores = (
        classifier.feature_log_prob_[spam_index]
        - classifier.feature_log_prob_[ham_index]
    )

else:
    scores = None


if scores is not None:

    keywords = pd.DataFrame({
        "keyword": terms,
        "spam_score": scores
    })

    keywords = keywords.sort_values(
        "spam_score",
        ascending=False
    ).head(20)

    keywords_path = os.path.join(
        MODEL_DIR,
        "spam_keywords.csv"
    )

    keywords.to_csv(
        keywords_path,
        index=False
    )


# -------------------- Training Summary --------------------

print("\nTraining completed successfully!")
print("Selected model:", best_name)
print(f"Best F1 Score: {best_f1:.2%}")
print("Saved model:", model_path)
print("Saved metrics:", metrics_path)
print("Saved comparison:", comparison_path)
print("Saved keyword information in:", MODEL_DIR)
