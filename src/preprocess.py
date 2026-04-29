# =============================================================================
# src/preprocess.py
# -----------------------------------------------------------------------------
# PURPOSE : Load raw CSV data, clean article text, and return a ready-to-train
#           DataFrame.
# STEPS   : 1) Load Fake.csv / True.csv
#           2) Add 'label' column  (0 = FAKE, 1 = REAL)
#           3) Combine into one DataFrame
#           4) Clean text  ── lowercase, remove URLs/HTML, punctuation,
#                             digits, extra spaces, stopwords
# =============================================================================

import re
import string
import logging

import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# ── logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# ── download required NLTK data (safe to call multiple times) ────────────────
def download_nltk_resources() -> None:
    """Download punkt and stopwords if not already present."""
    for resource in ["punkt", "stopwords", "punkt_tab"]:
        try:
            nltk.download(resource, quiet=True)
        except Exception as exc:
            log.warning("Could not download NLTK resource '%s': %s", resource, exc)


# ── text cleaning ─────────────────────────────────────────────────────────────
def clean_text(text: str) -> str:
    """
    Clean a single article string.

    Operations (in order):
        1. Lower-case everything
        2. Remove URLs  (http/https/www)
        3. Strip HTML tags
        4. Remove punctuation
        5. Remove digits
        6. Collapse multiple spaces → single space
        7. Remove English stop-words
        8. Return stripped string
    """
    if not isinstance(text, str):
        return ""

    # 1. lower-case
    text = text.lower()

    # 2. remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)

    # 3. strip HTML / XML tags
    text = re.sub(r"<[^>]+>", " ", text)

    # 4. remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))

    # 5. remove digits
    text = re.sub(r"\d+", " ", text)

    # 6. collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # 7. remove stop-words
    stop_words = set(stopwords.words("english"))
    tokens = word_tokenize(text)
    tokens = [w for w in tokens if w not in stop_words and len(w) > 1]

    return " ".join(tokens)


# ── data loading & labelling ──────────────────────────────────────────────────
def load_data(fake_path: str, true_path: str) -> pd.DataFrame:
    """
    Load Fake.csv and True.csv, attach labels, and merge into one DataFrame.

    Parameters
    ----------
    fake_path : str   – path to Fake.csv
    true_path : str   – path to True.csv

    Returns
    -------
    pd.DataFrame with columns: title, text, combined_text, label
        label: 0 = FAKE, 1 = REAL
    """
    log.info("Loading dataset …")

    try:
        fake_df = pd.read_csv(fake_path)
        true_df = pd.read_csv(true_path)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Dataset file not found: {exc.filename}\n"
            "Place Fake.csv and True.csv inside the  data/  folder."
        ) from exc

    # Attach integer labels
    fake_df["label"] = 0   # 0 → FAKE
    true_df["label"] = 1   # 1 → REAL

    # Combine
    df = pd.concat([fake_df, true_df], ignore_index=True)
    log.info("Raw dataset shape: %s", df.shape)

    # Keep only the columns we need; handle datasets that may lack 'title'
    if "title" in df.columns and "text" in df.columns:
        df["combined_text"] = df["title"].fillna("") + " " + df["text"].fillna("")
    elif "text" in df.columns:
        df["combined_text"] = df["text"].fillna("")
    else:
        raise ValueError("Dataset must contain at least a 'text' column.")

    # Drop rows where combined_text is empty
    df = df[df["combined_text"].str.strip() != ""].reset_index(drop=True)
    log.info("After dropping empty rows: %s", df.shape)

    return df[["combined_text", "label"]]


# ── main preprocessing pipeline ───────────────────────────────────────────────
def preprocess(fake_path: str = "data/Fake.csv",
               true_path: str = "data/True.csv") -> pd.DataFrame:
    """
    Full preprocessing pipeline.

    1. Download NLTK resources
    2. Load & merge raw CSVs
    3. Clean text
    4. Return final DataFrame  (columns: cleaned_text, label)
    """
    download_nltk_resources()

    df = load_data(fake_path, true_path)

    log.info("Cleaning text – this may take a minute …")
    df["cleaned_text"] = df["combined_text"].apply(clean_text)

    # Drop rows that became empty after cleaning
    df = df[df["cleaned_text"].str.strip() != ""].reset_index(drop=True)
    log.info("Final dataset shape: %s", df.shape)
    log.info("Label distribution:\n%s", df["label"].value_counts())

    return df[["cleaned_text", "label"]]


# ── quick smoke-test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    sample = "Breaking NEWS!!! President signs NEW bill – visit http://example.com for details."
    print("Original :", sample)
    download_nltk_resources()
    print("Cleaned  :", clean_text(sample))
