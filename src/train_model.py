# =============================================================================
# src/train_model.py
# -----------------------------------------------------------------------------
# PURPOSE : Feature extraction with TF-IDF, training two classifiers
#           (Logistic Regression + Passive-Aggressive), evaluating both, and
#           saving the best model + vectorizer as .pkl files.
#
# WORKFLOW:
#   preprocess()  →  TF-IDF  →  train LR & PAC
#                             →  evaluate  →  save best model
# =============================================================================

import os
import sys
import logging
import joblib

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend (safe in all envs)
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix,
)

# allow running as a standalone script from the project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.preprocess import preprocess

# ── logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# ── constants ────────────────────────────────────────────────────────────────
MODEL_DIR      = "models"
LR_MODEL_PATH  = os.path.join(MODEL_DIR, "logistic_regression.pkl")
PAC_MODEL_PATH = os.path.join(MODEL_DIR, "passive_aggressive.pkl")
BEST_MODEL_PATH= os.path.join(MODEL_DIR, "best_model.pkl")
VECTORIZER_PATH= os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")


# ── evaluation helper ─────────────────────────────────────────────────────────
def evaluate_model(model, X_test, y_test, model_name: str) -> dict:
    """
    Print and return a dict of evaluation metrics for a trained model.

    Metrics: Accuracy, Precision, Recall, F1-score (macro avg)
    """
    y_pred = model.predict(X_test)

    metrics = {
        "accuracy" : accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="macro", zero_division=0),
        "recall"   : recall_score(y_test, y_pred, average="macro", zero_division=0),
        "f1"       : f1_score(y_test, y_pred, average="macro", zero_division=0),
    }

    log.info("─" * 55)
    log.info("  Model  : %s", model_name)
    log.info("  Accuracy  : %.4f", metrics["accuracy"])
    log.info("  Precision : %.4f", metrics["precision"])
    log.info("  Recall    : %.4f", metrics["recall"])
    log.info("  F1-Score  : %.4f", metrics["f1"])
    log.info("\n%s", classification_report(y_test, y_pred,
                                           target_names=["FAKE", "REAL"]))

    # Save confusion matrix plot
    cm = confusion_matrix(y_test, y_pred)
    _save_confusion_matrix(cm, model_name)

    return metrics


def _save_confusion_matrix(cm: np.ndarray, model_name: str) -> None:
    """Save a labelled heat-map PNG of the confusion matrix."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["FAKE", "REAL"],
                yticklabels=["FAKE", "REAL"], ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix – {model_name}")
    safe_name = model_name.lower().replace(" ", "_")
    fig.tight_layout()
    fig.savefig(os.path.join(MODEL_DIR, f"cm_{safe_name}.png"), dpi=120)
    plt.close(fig)
    log.info("Confusion matrix saved → models/cm_%s.png", safe_name)


# ── main training pipeline ────────────────────────────────────────────────────
def train(fake_path: str = "data/Fake.csv",
          true_path: str = "data/True.csv") -> None:
    """
    End-to-end training pipeline:

      1. Preprocess raw data
      2. Train / test split  (80 / 20, stratified)
      3. TF-IDF vectorisation  (top 50 000 uni+bi-grams, sub-linear TF)
      4. Train Logistic Regression
      5. Train Passive-Aggressive Classifier
      6. Evaluate both
      7. Save every model + vectorizer as .pkl
      8. Save the better model as  best_model.pkl
    """
    # ── 1. preprocess ─────────────────────────────────────────────────────────
    df = preprocess(fake_path, true_path)

    X = df["cleaned_text"].values
    y = df["label"].values

    # ── 2. split ──────────────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y,          # keep class balance in both splits
    )
    log.info("Train size: %d   Test size: %d", len(X_train), len(X_test))

    # ── 3. TF-IDF vectorisation ───────────────────────────────────────────────
    log.info("Fitting TF-IDF vectorizer …")
    vectorizer = TfidfVectorizer(
        max_features=50_000,   # vocabulary cap
        ngram_range=(1, 2),    # uni-grams + bi-grams
        sublinear_tf=True,     # apply log(1+tf) to dampen high-frequency terms
        min_df=2,              # ignore terms appearing in < 2 documents
    )

    X_train_tfidf = vectorizer.fit_transform(X_train)   # fit on train ONLY
    X_test_tfidf  = vectorizer.transform(X_test)        # transform test

    # ── 4. Logistic Regression ────────────────────────────────────────────────
    log.info("Training Logistic Regression …")
    lr_model = LogisticRegression(
        max_iter=1000,
        C=1.0,
        solver="lbfgs",
        n_jobs=-1,
        random_state=42,
    )
    lr_model.fit(X_train_tfidf, y_train)

    # ── 5. Passive-Aggressive Classifier ─────────────────────────────────────
    log.info("Training Passive-Aggressive Classifier …")
    pac_model = PassiveAggressiveClassifier(
        max_iter=1000,
        C=1.0,
        random_state=42,
    )
    pac_model.fit(X_train_tfidf, y_train)

    # ── 6. evaluate ───────────────────────────────────────────────────────────
    lr_metrics  = evaluate_model(lr_model,  X_test_tfidf, y_test, "Logistic Regression")
    pac_metrics = evaluate_model(pac_model, X_test_tfidf, y_test, "Passive Aggressive")

    # ── 7. save models & vectorizer ───────────────────────────────────────────
    os.makedirs(MODEL_DIR, exist_ok=True)

    joblib.dump(vectorizer, VECTORIZER_PATH)
    log.info("Vectorizer saved → %s", VECTORIZER_PATH)

    joblib.dump(lr_model,  LR_MODEL_PATH)
    log.info("Logistic Regression saved → %s", LR_MODEL_PATH)

    joblib.dump(pac_model, PAC_MODEL_PATH)
    log.info("Passive Aggressive saved → %s", PAC_MODEL_PATH)

    # ── 8. save the better model as best_model.pkl ────────────────────────────
    if lr_metrics["f1"] >= pac_metrics["f1"]:
        best_model, best_name = lr_model,  "Logistic Regression"
    else:
        best_model, best_name = pac_model, "Passive Aggressive"

    joblib.dump(best_model, BEST_MODEL_PATH)
    log.info("Best model (%s) saved → %s", best_name, BEST_MODEL_PATH)

    # ── summary table ─────────────────────────────────────────────────────────
    summary = pd.DataFrame(
        {"Logistic Regression": lr_metrics,
         "Passive Aggressive" : pac_metrics}
    ).T
    log.info("\n── Comparison Summary ──\n%s", summary.to_string())


# ── entry point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    train()
