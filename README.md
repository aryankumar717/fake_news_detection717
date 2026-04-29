# 🔍 Fake News Detection — ML Project

A complete machine-learning pipeline that classifies news articles as **FAKE** or **REAL**,
with a Streamlit web UI for interactive predictions.

---

## 📁 Project Structure

```
fake-news-detection/
├── data/
│   ├── Fake.csv          ← download from Kaggle
│   └── True.csv          ← download from Kaggle
├── models/               ← auto-created after training
│   ├── tfidf_vectorizer.pkl
│   ├── logistic_regression.pkl
│   ├── passive_aggressive.pkl
│   └── best_model.pkl
├── src/
│   ├── __init__.py
│   ├── preprocess.py     ← text cleaning & data loading
│   ├── train_model.py    ← TF-IDF + model training + evaluation
│   └── predict.py        ← load model & classify new text
├── app.py                ← Streamlit web UI
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone / download the project

```bash
git clone <your-repo-url>
cd fake-news-detection
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Get the dataset

Download from Kaggle → [Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)

Place `Fake.csv` and `True.csv` inside the `data/` folder.

### 4. Train the models

```bash
python src/train_model.py
```

This will:
- Preprocess the data (clean text, remove stop-words)
- Fit a TF-IDF vectorizer (50 000 features)
- Train Logistic Regression and Passive-Aggressive Classifier
- Print accuracy, precision, recall, F1-score for both
- Save all `.pkl` files to `models/`

### 5. Run the Streamlit app

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`

---

## 🧪 Predict from the command line

```bash
python src/predict.py "Breaking news: Scientists discover water on Mars."
```

---

## 📊 How it works

```
Raw CSV  →  preprocess.py  →  cleaned text
                          ↓
              TF-IDF vectorizer (fit on train set)
                          ↓
          Logistic Regression  /  Passive-Aggressive
                          ↓
               FAKE (0)  or  REAL (1)
```

### Key Preprocessing Steps
| Step | What it does |
|------|-------------|
| Lower-case | Normalise capitalisation |
| Remove URLs | Strip http/www links |
| Strip HTML | Remove `<tags>` |
| Remove punctuation | Strip `.,!?…` etc. |
| Remove digits | Strip numbers |
| Stop-word removal | Drop "the", "is", "at" … |

### TF-IDF Settings
| Parameter | Value | Reason |
|-----------|-------|--------|
| `max_features` | 50 000 | cap vocabulary size |
| `ngram_range` | (1, 2) | uni + bi-grams |
| `sublinear_tf` | True | log-dampen high-freq terms |
| `min_df` | 2 | ignore extremely rare terms |

---

## 📈 Typical Results

| Model | Accuracy | F1-score |
|-------|----------|----------|
| Logistic Regression | ~98–99 % | ~0.98–0.99 |
| Passive-Aggressive | ~97–99 % | ~0.97–0.99 |

*(exact numbers depend on random seed & dataset split)*

---

## 💡 Optional Improvements with Deep Learning / BERT

### Option A — LSTM with word embeddings

```python
from tensorflow.keras.layers import Embedding, LSTM, Dense
# Use pre-trained GloVe embeddings (glove.6B.100d.txt)
```

### Option B — Fine-tuned BERT (best accuracy)

```python
from transformers import BertTokenizer, BertForSequenceClassification
# Requires ~4 GB VRAM; fine-tune for 2-3 epochs
# Typical accuracy: 99.5 %+
```

**Why BERT is better:**
- Understands **context** (not just word frequency)
- Handles sarcasm, irony, subtle misinformation
- Sub-word tokenization handles out-of-vocabulary words

**Why TF-IDF + ML is still useful:**
- Trains in seconds on a CPU
- Highly interpretable (you can inspect top features)
- No GPU needed
- Great baseline for college projects

---

## 📝 Viva / Interview Tips

1. **Why TF-IDF?** — It converts text to numbers while down-weighting common words, giving more importance to rare but meaningful terms.
2. **Why Passive-Aggressive?** — It's an online learning algorithm: updates weights only when it makes a mistake, making it very fast.
3. **How do you prevent data leakage?** — The TF-IDF vectorizer is fit **only on training data**, then applied to test data.
4. **What is stratified split?** — It ensures both FAKE and REAL classes are proportionally represented in train and test sets.
5. **What would improve this further?** — BERT, data augmentation, cross-validation, ensemble methods.

---

## 🛠 Dependencies

| Library | Purpose |
|---------|---------|
| pandas / numpy | Data handling |
| scikit-learn | TF-IDF, models, metrics |
| nltk | Stop-words, tokenization |
| joblib | Save / load .pkl files |
| streamlit | Web UI |
| matplotlib / seaborn | Confusion matrix plots |

---

*Built for college submission — Fake News Detection using Machine Learning*
