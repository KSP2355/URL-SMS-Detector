"""Train and export an SMS spam/phishing detection pipeline."""

from __future__ import annotations

import argparse
import io
import re
import string
import urllib.request
import zipfile
from pathlib import Path
from urllib.error import URLError

import joblib
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

DEFAULT_DATA_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
)
DEFAULT_MODEL_PATH = "spam_classifier.pkl"


def ensure_nltk_resources() -> None:
    """Download required NLTK corpora if they are not present."""
    try:
        stopwords.words("english")
    except LookupError:
        nltk.download("stopwords", quiet=True)


def load_dataset(data_path: str | None = None) -> pd.DataFrame:
    """Load SMS dataset from local file or download from UCI."""
    if data_path:
        local_file = Path(data_path)
        if not local_file.exists():
            raise FileNotFoundError(f"Dataset not found: {local_file}")
        return parse_dataset(local_file.read_text(encoding="utf-8", errors="ignore"))

    for candidate in ("spam.csv", "SMSSpamCollection"):
        candidate_path = Path(candidate)
        if candidate_path.exists():
            return parse_dataset(
                candidate_path.read_text(encoding="utf-8", errors="ignore")
            )

    try:
        with urllib.request.urlopen(DEFAULT_DATA_URL, timeout=30) as response:
            zip_bytes = response.read()
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            with zf.open("SMSSpamCollection") as dataset_file:
                text_data = dataset_file.read().decode("utf-8", errors="ignore")
        return parse_dataset(text_data)
    except URLError as exc:
        raise RuntimeError(
            "Unable to download the dataset. Please pass a local file with "
            "--data-path (CSV with v1/v2 columns or TSV label/text format)."
        ) from exc


def parse_dataset(raw_text: str) -> pd.DataFrame:
    """Parse dataset text in either CSV(v1/v2) or TSV(label/text) format."""
    csv_df = pd.read_csv(io.StringIO(raw_text), on_bad_lines="skip")
    if {"v1", "v2"}.issubset(csv_df.columns):
        df = csv_df[["v1", "v2"]].rename(columns={"v1": "label", "v2": "text"})
        return normalize_dataset(df)

    tsv_df = pd.read_csv(io.StringIO(raw_text), sep="\t", header=None, names=["label", "text"])
    return normalize_dataset(tsv_df)


def normalize_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize expected columns and values."""
    normalized = df.copy()
    normalized["label"] = normalized["label"].astype(str).str.strip().str.lower()
    normalized["text"] = normalized["text"].astype(str).str.strip()
    normalized = normalized[normalized["text"] != ""]
    return normalized


def text_analyzer(text: str) -> list[str]:
    """Tokenize and normalize text with stemming/stopword removal."""
    stemmer = PorterStemmer()
    stop_words = set(stopwords.words("english"))

    text = text.lower().translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    tokens = re.findall(r"\b[a-z0-9]+\b", text)
    return [stemmer.stem(token) for token in tokens if token not in stop_words]


def preprocess_text(text: str) -> str:
    """Convert text into a cleaned, stemmed whitespace-separated string."""
    return " ".join(text_analyzer(text))


def build_pipeline() -> Pipeline:
    """Create end-to-end sklearn pipeline."""
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    max_features=3000,
                ),
            ),
            (
                "classifier",
                CalibratedClassifierCV(
                    estimator=LinearSVC(),
                    method="sigmoid",
                    cv=5,
                ),
            ),
        ]
    )


def train_and_evaluate(data_path: str | None, model_path: str) -> None:
    """Train, evaluate, and persist the model."""
    ensure_nltk_resources()
    data = load_dataset(data_path)
    data["processed_text"] = data["text"].apply(preprocess_text)

    x_train, x_test, y_train, y_test = train_test_split(
        data["processed_text"],
        data["label"],
        test_size=0.2,
        random_state=42,
        stratify=data["label"],
    )

    pipeline = build_pipeline()
    pipeline.fit(x_train, y_train)

    predictions = pipeline.predict(x_test)
    print(f"Accuracy: {accuracy_score(y_test, predictions):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, predictions, zero_division=0))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, predictions))

    joblib.dump(pipeline, model_path)
    print(f"\nSaved model to: {model_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SMS spam/phishing detector")
    parser.add_argument(
        "--data-path",
        default=None,
        help="Optional local dataset path (CSV with v1/v2 or TSV label/text).",
    )
    parser.add_argument(
        "--model-path",
        default=DEFAULT_MODEL_PATH,
        help=f"Output model path (default: {DEFAULT_MODEL_PATH}).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_and_evaluate(args.data_path, args.model_path)
