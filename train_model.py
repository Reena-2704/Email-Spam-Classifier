
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

DATA_PATH = "data/SMSSpamCollection.tsv"
MODEL_DIR = "models"

os.makedirs(MODEL_DIR, exist_ok=True)

# Load and clean dataset
data = pd.read_csv(
    DATA_PATH,
    sep="\t",
    header=None,
    names=["label", "message"],
    encoding="latin-1"
)

data = data.dropna(subset=["label", "message"])
data["label"] = data["label"].str.strip().str.lower()
data = data[data["label"].isin(["ham", "spam"])]
data = data.drop_duplicates(subset=["label", "message"])

X = data["message"].astype(str)
y = data["label"]

print("Dataset loaded:", len(data), "messages")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

classifiers = {
    "Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    ),
    "SVM": LinearSVC(class_weight="balanced")
}

results = []
best_model = None
best_name = None
best_f1 = -1

for name, classifier in classifiers.items():
    print("Training:", name)

    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            max_features=10000
        )),
        ("classifier", classifier)
    ])

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(
        y_test, predictions, pos_label="spam", zero_division=0
    )
    recall = recall_score(
        y_test, predictions, pos_label="spam", zero_division=0
    )
    f1 = f1_score(
        y_test, predictions, pos_label="spam", zero_division=0
    )

    # Ham is the negative class; spam is the positive class
    tn, fp, fn, tp = confusion_matrix(
        y_test, predictions, labels=["ham", "spam"]
    ).ravel()

    fpr = fp / (fp + tn) if (fp + tn) else 0

    results.append({
        "model": name,
        "accuracy": round(float(accuracy), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1_score": round(float(f1), 4),
        "false_positive_rate": round(float(fpr), 4),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp)
    })

    print(
        f"Accuracy: {accuracy:.2%} | "
        f"Precision: {precision:.2%} | "
        f"Recall: {recall:.2%} | F1: {f1:.4f}"
    )

    # Select using F1, which balances precision and recall
    if f1 > best_f1:
        best_f1 = f1
        best_model = model
        best_name = name

# Save the selected model
joblib.dump(best_model, os.path.join(MODEL_DIR, "spam_classifier.pkl"))

# Save measured evaluation results
results_df = pd.DataFrame(results)
results_df.to_csv(
    os.path.join(MODEL_DIR, "model_comparison.csv"),
    index=False
)

metadata = {
    "best_model": best_name,
    "test_samples": int(len(y_test)),
    "training_samples": int(len(y_train)),
    "results": results
}

with open(os.path.join(MODEL_DIR, "metrics.json"), "w") as file:
    json.dump(metadata, file, indent=4)

# Extract features associated with spam
vectorizer = best_model.named_steps["tfidf"]
classifier = best_model.named_steps["classifier"]
terms = vectorizer.get_feature_names_out()

if hasattr(classifier, "coef_"):
    # Positive scores indicate association with spam
    scores = classifier.coef_[0]

    if list(classifier.classes_)[1] != "spam":
        scores = -scores

elif hasattr(classifier, "feature_log_prob_"):
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
        "spam_score", ascending=False
    ).head(20)

    keywords.to_csv(
        os.path.join(MODEL_DIR, "spam_keywords.csv"),
        index=False
    )

print("\nTraining completed!")
print("Selected model:", best_name)
print("Selection metric: F1 score")
print("Saved model and evaluation files in:", MODEL_DIR)
