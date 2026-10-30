"""Streamlit app for SMS spam/phishing detection."""

from __future__ import annotations

from pathlib import Path

import joblib
import streamlit as st
from train import ensure_nltk_resources, preprocess_text

MODEL_PATH = Path("spam_classifier.pkl")
SAMPLE_MESSAGES = {
    "Spam / Phishing": "Congratulations! You've won a $1000 gift card. Click this link now to claim your prize.",
    "Legitimate": "Hi, just checking if we're still on for dinner tonight at 8 PM.",
    "Banking Scam": "Your bank account is locked. Verify your identity immediately at http://secure-bank-update.example.",
}


@st.cache_resource
def load_model(model_path: Path):
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found at '{model_path}'. Run train.py first."
        )
    return joblib.load(model_path)


def classify_message(model, message: str) -> tuple[str, float | None]:
    processed_message = preprocess_text(message)
    predicted_label = str(model.predict([processed_message])[0]).lower()
    is_spam = predicted_label in {"spam", "phishing", "1"}
    confidence = None

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba([processed_message])[0]
        class_labels = [str(cls).lower() for cls in model.classes_]
        if "spam" in class_labels:
            confidence = float(probabilities[class_labels.index("spam")])
        else:
            confidence = float(max(probabilities))
    elif hasattr(model, "decision_function"):
        score = model.decision_function([processed_message])[0]
        confidence = float(abs(score))

    return ("Spam / Phishing" if is_spam else "Ham / Legitimate"), confidence


def main() -> None:
    st.set_page_config(page_title="SMS & Phishing Detector", page_icon="🛡️")
    st.title("🛡️ SMS & Phishing Text Detector")
    st.caption("Paste a suspicious SMS or message and scan it instantly.")

    try:
        ensure_nltk_resources()
        model = load_model(MODEL_PATH)
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()

    if "message_input" not in st.session_state:
        st.session_state.message_input = ""

    sample_choice = st.selectbox("Try a sample message", options=list(SAMPLE_MESSAGES))
    if st.button("Load sample message"):
        st.session_state.message_input = SAMPLE_MESSAGES[sample_choice]

    st.text_area(
        "Message text",
        key="message_input",
        placeholder="Paste SMS or suspicious text here...",
        height=180,
    )

    if st.button("Scan Message", type="primary"):
        message = st.session_state.message_input.strip()
        if not message:
            st.warning("Please enter a message before scanning.")
            st.stop()

        label, confidence = classify_message(model, message)
        if label == "Spam / Phishing":
            st.error(f"Prediction: **{label}**")
        else:
            st.success(f"Prediction: **{label}**")

        if confidence is not None:
            if confidence <= 1:
                st.info(f"Confidence score: **{confidence:.2%}**")
            else:
                st.info(f"Decision score: **{confidence:.4f}**")


if __name__ == "__main__":
    main()
