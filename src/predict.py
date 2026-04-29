# =============================================================================
# src/predict.py
# -----------------------------------------------------------------------------
# PURPOSE : Load a saved model + TF-IDF vectorizer and classify raw news text
#           as FAKE or REAL.
#
# USAGE (CLI):
#   python src/predict.py "Paste your news article here …"
#
# USAGE (import):
#   from src.predict import predict
#   result = predict("Your article text …")
# =============================================================================

import os
import sys
import logging

import joblib

# allow running as a standalone script from the project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.preprocess import clean_text, download_nltk_resources

# ── logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# ── default paths ─────────────────────────────────────────────────────────────
BEST_MODEL_PATH = os.path.join("models", "best_model.pkl")
VECTORIZER_PATH = os.path.join("models", "tfidf_vectorizer.pkl")


# ── loader (cached at module level) ──────────────────────────────────────────
_model      = None
_vectorizer = None


def _load_artifacts(model_path: str = BEST_MODEL_PATH,
                    vectorizer_path: str = VECTORIZER_PATH):
    """
    Load model and vectorizer from disk only once, then cache them.
    Raises FileNotFoundError with a helpful message if files are missing.
    """
    global _model, _vectorizer

    if _model is None or _vectorizer is None:
        for path in (model_path, vectorizer_path):
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"Required file not found: {path}\n"
                    "Run  python src/train_model.py  first to generate .pkl files."
                )
        _model      = joblib.load(model_path)
        _vectorizer = joblib.load(vectorizer_path)
        log.info("Model and vectorizer loaded successfully.")

    return _model, _vectorizer


# ── prediction function ───────────────────────────────────────────────────────
def predict(text: str,
            model_path: str = BEST_MODEL_PATH,
            vectorizer_path: str = VECTORIZER_PATH) -> dict:
    """
    Classify a news article as FAKE or REAL.

    Parameters
    ----------
    text           : str  – raw article text (title + body, any length)
    model_path     : str  – path to the saved classifier .pkl
    vectorizer_path: str  – path to the saved TF-IDF vectorizer .pkl

    Returns
    -------
    dict with keys:
        label       : "FAKE" or "REAL"
        confidence  : float (0.0 – 1.0), probability of the predicted class
        fake_prob   : float – probability of FAKE
        real_prob   : float – probability of REAL
        cleaned_text: str   – the cleaned version that was actually classified
    """
    if not text or not text.strip():
        return {
            "label"       : "UNKNOWN",
            "confidence"  : 0.0,
            "fake_prob"   : 0.0,
            "real_prob"   : 0.0,
            "cleaned_text": "",
        }

    # ── ensure NLTK data is present ───────────────────────────────────────────
    download_nltk_resources()

    # ── preprocess ────────────────────────────────────────────────────────────
    cleaned = clean_text(text)

    # ── load artifacts ────────────────────────────────────────────────────────
    model, vectorizer = _load_artifacts(model_path, vectorizer_path)

    # ── vectorise & predict ───────────────────────────────────────────────────
    X = vectorizer.transform([cleaned])

    prediction = model.predict(X)[0]               # 0 = FAKE, 1 = REAL
    label = "REAL" if prediction == 1 else "FAKE"

    # Confidence / probability (not all models expose predict_proba)
    fake_prob = real_prob = confidence = 0.5
    if hasattr(model, "predict_proba"):
        proba     = model.predict_proba(X)[0]      # [P(FAKE), P(REAL)]
        fake_prob = float(proba[0])
        real_prob = float(proba[1])
        confidence = real_prob if prediction == 1 else fake_prob
    elif hasattr(model, "decision_function"):
        # For Passive-Aggressive we can use the decision score as a proxy
        score      = float(model.decision_function(X)[0])
        # Map score to 0-1 range via sigmoid
        import math
        real_prob  = 1 / (1 + math.exp(-score))
        fake_prob  = 1 - real_prob
        confidence = real_prob if prediction == 1 else fake_prob

    return {
        "label"       : label,
        "confidence"  : round(confidence, 4),
        "fake_prob"   : round(fake_prob, 4),
        "real_prob"   : round(real_prob, 4),
        "cleaned_text": cleaned,
    }


# ── CLI entry-point ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/predict.py \"<news text>\"")
        sys.exit(1)

    input_text = " ".join(sys.argv[1:])
    result = predict(input_text)

    print("\n" + "=" * 55)
    print(f"  Prediction : {result['label']}")
    print(f"  Confidence : {result['confidence'] * 100:.2f} %")
    print(f"  FAKE prob  : {result['fake_prob'] * 100:.2f} %")
    print(f"  REAL prob  : {result['real_prob'] * 100:.2f} %")
    print("=" * 55)
