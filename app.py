# =============================================================================
# app.py  –  Streamlit UI for Fake News Detection
# -----------------------------------------------------------------------------
# Run with:   streamlit run app.py
# =============================================================================

import os
import sys
import time
import requests

import streamlit as st

from dotenv import load_dotenv
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

def google_fact_check(query: str) -> dict:
    try:
        url = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
        params = {
            "query": query[:200],   # max 200 chars
            "key": GOOGLE_API_KEY,
            "pageSize": 3,
        }
        response = requests.get(url, params=params, timeout=5)
        data = response.json()

        claims = data.get("claims", [])
        if not claims:
            return {"found": False}

        results = []
        for claim in claims[:3]:
            review = claim.get("claimReview", [{}])[0]
            results.append({
                "claim"    : claim.get("text", ""),
                "rating"   : review.get("textualRating", "Unknown"),
                "publisher": review.get("publisher", {}).get("name", "Unknown"),
                "url"      : review.get("url", ""),
            })
        return {"found": True, "results": results}

    except Exception:
        return {"found": False}


from typing import Optional
def verify_fact(text: str) -> Optional[dict]:
    import requests

    query = text.strip()

    try:
        # Use Wikipedia search instead of hardcoded rules
        search_url = "https://en.wikipedia.org/w/api.php"
        params = {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "format": "json"
        }

        res = requests.get(search_url, params=params, timeout=5)
        data = res.json()

        results = data.get("query", {}).get("search", [])

        if not results:
            return None

        top_result = results[0]["title"]

        # Get summary of top result
        summary_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{top_result}"
        summary_res = requests.get(summary_url, timeout=5)
        summary_data = summary_res.json()

        extract = summary_data.get("extract", "").lower()

        # Simple contradiction check
        if any(word in text.lower() for word in ["not", "fake", "wrong"]):
            return None

        # If important keywords mismatch → likely fake
        words = text.lower().split()
        mismatch = sum(1 for w in words if w not in extract)

        if mismatch > len(words) * 0.5:
            return {
                "label": "FAKE",
                "confidence": 0.85,
                "reason": "Claim does not match known information from Wikipedia"
            }

        return {
            "label": "REAL",
            "confidence": 0.7,
            "reason": summary_data.get("extract", "")[:200]
        }

    except Exception:
        return None

# ── make sure  src/  is importable ───────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── custom CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* ---- Global ---- */
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=Inter:wght@400;500;600&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    h1, h2, h3 { font-family: 'Syne', sans-serif; }

    /* ---- Header banner ---- */
    .hero-banner {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        border-radius: 16px;
        padding: 2.5rem 2rem;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
        box-shadow: 0 8px 32px rgba(0,0,0,0.35);
    }
    .hero-banner h1 { font-size: 2.8rem; font-weight: 800; margin: 0; letter-spacing: -1px; }
    .hero-banner p  { font-size: 1.05rem; opacity: 0.75; margin-top: 0.5rem; }

    /* ---- Result cards ---- */
    .result-card {
        border-radius: 14px;
        padding: 1.8rem;
        text-align: center;
        color: white;
        box-shadow: 0 6px 24px rgba(0,0,0,0.25);
        margin-top: 1rem;
    }
    .result-fake { background: linear-gradient(135deg, #e52d27, #b31217); }
    .result-real { background: linear-gradient(135deg, #11998e, #38ef7d); color: #111; }
    .result-card h2 { font-family: 'Syne', sans-serif; font-size: 2.2rem; margin: 0; }
    .result-card p  { font-size: 1rem; margin-top: 0.4rem; opacity: 0.85; }

    /* ---- Metric tiles ---- */
    .metric-tile {
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }

    /* ---- Sidebar ---- */
    [data-testid="stSidebar"] { background: #0d0d1a; color: white; }

    /* ---- Textarea ---- */
    textarea { font-size: 0.95rem !important; }

    /* ---- Spinner text ---- */
    .stSpinner > div > div { border-top-color: #7c6af7 !important; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── hero banner ───────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="hero-banner">
        <h1>🔍 Fake News Detector</h1>
    </div>
    """,
    unsafe_allow_html=True,
)


# ── sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    model_choice = st.selectbox(
        "Choose Model",
        ["Best Model (auto)", "Logistic Regression", "Passive Aggressive"],
    )

    st.markdown("---")
    st.markdown("### 📖 How it works")
    st.markdown(
        """
1. **Preprocess** – clean text, remove stop-words  
2. **TF-IDF** – convert to numeric features  
3. **Classify** – ML model predicts FAKE or REAL  
4. **Confidence** – probability score (LR) or decision score (PAC)
        """
    )

    st.markdown("---")
    st.markdown("### 🚀 Train Models")
    if st.button("🔧 Train Now", use_container_width=True):
        with st.spinner("Training… this takes 1-3 minutes"):
            try:
                from src.train_model import train
                train()
                st.success("Training complete! Models saved to  models/")
            except FileNotFoundError as exc:
                st.error(str(exc))
            except Exception as exc:
                st.error(f"Training error: {exc}")


# ── model path mapping ────────────────────────────────────────────────────────
MODEL_MAP = {
    "Best Model (auto)": os.path.join("models", "best_model.pkl"),
    "Logistic Regression": os.path.join("models", "logistic_regression.pkl"),
    "Passive Aggressive": os.path.join("models", "passive_aggressive.pkl"),
}
VECTORIZER_PATH = os.path.join("models", "tfidf_vectorizer.pkl")


# ── main content ──────────────────────────────────────────────────────────────
col_left, col_right = st.columns([3, 2], gap="large")

with col_left:
    st.markdown("### 📰 Enter News Article")
    news_text = st.text_area(
        label="Paste a news headline or full article below:",
        height=260,
        placeholder="e.g.  'Scientists discover water on Mars in groundbreaking study…'",
    )

    example_col1, example_col2 = st.columns(2)
    with example_col1:
        if st.button("📌 Load REAL example"):
            news_text = (
                "The Federal Reserve raised interest rates by 25 basis points on "
                "Wednesday, citing persistent inflation above its 2% target. "
                "Chair Jerome Powell said policymakers remain committed to restoring "
                "price stability even at the cost of slower economic growth."
            )
            st.session_state["example_text"] = news_text

    with example_col2:
        if st.button("⚠️ Load FAKE example"):
            news_text = (
                "SHOCKING: Government secretly puts mind-control chemicals in tap water! "
                "Leaked documents prove 5G towers activate nanobots injected via vaccines. "
                "Share this before it gets DELETED!!!"
            )
            st.session_state["example_text"] = news_text

    # Overwrite textarea value when example is loaded
    if "example_text" in st.session_state and not news_text:
        news_text = st.session_state.pop("example_text")

    predict_btn = st.button("🔍 Detect", type="primary", use_container_width=True)


# ── prediction ────────────────────────────────────────────────────────────────
with col_right:
    st.markdown("### 📊 Result")

    if predict_btn:
        if not news_text.strip():
            st.warning("Please enter some text first.")
        else:
            model_path = MODEL_MAP[model_choice]
            models_missing = (
                not os.path.exists(model_path) or
                not os.path.exists(VECTORIZER_PATH)
            )

            if models_missing:
                st.error(
                    "⚠️ Trained models not found.\n\n"
                    "Click **🔧 Train Now** in the sidebar, or run:\n"
                    "```\npython src/train_model.py\n```"
                )
            else:
                with st.spinner("Analysing article …"):
                    time.sleep(0.4)

                    # 1) Rule-based / fact verification
                    fact = verify_fact(news_text)

                    if fact:
                        label = fact["label"]
                        confidence = fact["confidence"] * 100
                        fake_prob = 1 - fact["confidence"]
                        real_prob = fact["confidence"]
                        result = {
                            "label": label,
                            "confidence": fact["confidence"],
                            "fake_prob": fake_prob,
                            "real_prob": real_prob,
                            "cleaned_text": news_text,
                        }
                        st.warning(f"🧠 Reason: {fact['reason']}")
                    else:
                        try:
                            from src.predict import predict
                            result = predict(
                                news_text,
                                model_path=model_path,
                                vectorizer_path=VECTORIZER_PATH,
                            )
                        except Exception as exc:
                            st.error(f"Prediction failed: {exc}")
                            st.stop()

                label = result["label"]
                confidence = result["confidence"] * 100
                fake_prob = result["fake_prob"] * 100
                real_prob = result["real_prob"] * 100
                card_class = "result-real" if label == "REAL" else "result-fake"
                icon       = "✅" if label == "REAL" else "🚫"

                # Result card
                st.markdown(
                    f"""
                    <div class="result-card {card_class}">
                        <h2>{icon} {label}</h2>
                        <p>Confidence: <strong>{confidence:.1f}%</strong></p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.markdown("<br>", unsafe_allow_html=True)

                # Probability bar
                st.markdown("**Probability Breakdown**")
                st.progress(real_prob / 100, text=f"REAL: {real_prob:.1f}%")

                # Metric tiles
                m1, m2 = st.columns(2)
                m1.metric("FAKE Probability", f"{fake_prob:.1f}%")
                m2.metric("REAL Probability", f"{real_prob:.1f}%")

                # Cleaned text expander
                with st.expander("🔤 Cleaned text used for prediction"):
                    st.caption(result["cleaned_text"][:800] + ("…" if len(result["cleaned_text"]) > 800 else ""))

                # ── Google Fact Check ─────────────────────────────────────
                st.markdown("---")
                
                with st.spinner("Searching Google Fact Check…"):
                    fc = google_fact_check(news_text[:200])

                if fc["found"]:
                    for r in fc["results"]:
                        rating = r["rating"].lower()
                        if any(w in rating for w in ["false","fake","wrong","misleading","incorrect"]):
                            icon = "🚫"
                            color = "#e52d27"
                        elif any(w in rating for w in ["true","correct","accurate","verified"]):
                            icon = "✅"
                            color = "#11998e"
                        else:
                            icon = "⚠️"
                            color = "#f5a623"

                        st.markdown(
                            f"""
                            <div style="background:{color}22;border-left:4px solid {color};
                                        padding:0.8rem 1rem;border-radius:8px;margin-bottom:0.5rem">
                                <b>{icon} {r['rating']}</b><br>
                                <small>📰 {r['publisher']}</small><br>
                                <small>🔍 Claim: {r['claim'][:120]}…</small>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

    else:
        st.info("Enter an article and click **Detect** to see the prediction.")


# ── footer section: model metrics (show if confusion matrix images exist) ─────
st.markdown("---")
st.markdown("### 📈 Model Evaluation")

cm_lr  = os.path.join("models", "cm_logistic_regression.png")
cm_pac = os.path.join("models", "cm_passive_aggressive.png")

if os.path.exists(cm_lr) and os.path.exists(cm_pac):
    fc1, fc2 = st.columns(2)
    with fc1:
        st.image(cm_lr, caption="Logistic Regression – Confusion Matrix", width=600)
    with fc2:
        st.image(cm_pac, caption="Passive Aggressive – Confusion Matrix", width=600)
else:
    st.info("Train the models first to see confusion matrices here.")

# ── about section ─────────────────────────────────────────────────────────────
with st.expander("ℹ️ About this project"):
    st.markdown(
        """
**Fake News Detection** using classical Machine Learning.

| Component | Detail |
|-----------|--------|
| Dataset   | Fake.csv + True.csv (Kaggle) |
| Features  | TF-IDF (50K features, uni+bi-grams) |
| Models    | Logistic Regression, Passive-Aggressive Classifier |
| Metrics   | Accuracy, Precision, Recall, F1-score |
| UI        | Streamlit |

**Project Structure**
```
fake-news-detection/
├── data/          ← place Fake.csv & True.csv here
├── models/        ← trained .pkl files saved here
├── src/
│   ├── preprocess.py   clean & prepare text
│   ├── train_model.py  TF-IDF + train + evaluate
│   └── predict.py      load model & classify
├── app.py         ← this file (Streamlit UI)
├── requirements.txt
└── README.md
```
        """
    )
