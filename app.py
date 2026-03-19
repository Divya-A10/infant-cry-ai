"""
app.py — Streamlit dashboard for Infant Cry Classification.

Run with:
    streamlit run app.py

Features
--------
* Audio upload and in-browser playback
* Mel spectrogram visualisation
* CNN+CRNN prediction with confidence score
* Uncertainty-aware output ("Uncertain" when confidence < threshold)
* Full class probability distribution bar chart
* Toggle to show/hide raw technical details
* CSV prediction log (timestamp, filename, class, confidence)
* Model limitations and uncertainty reasoning section
"""

import csv
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from model import CRY_CLASSES, UNCERTAINTY_THRESHOLD, load_model, predict
from utils import (
    compute_features,
    load_audio_from_bytes,
    plot_probability_bar,
    plot_spectrogram,
)

# ─── Page Configuration ───────────────────────────────────────────────────────

st.set_page_config(
    page_title="Infant Cry Classifier",
    page_icon="👶",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── Constants ────────────────────────────────────────────────────────────────

LOG_FILE = "predictions_log.csv"
LOG_COLUMNS = ["timestamp", "filename", "predicted_class", "confidence"]

# Brief description shown beneath each class name
CLASS_DESCRIPTIONS: dict[str, str] = {
    "hunger":      "Rhythmic, repetitive cry — the baby needs feeding.",
    "pain":        "Sudden, high-pitched cry — possible discomfort or injury.",
    "discomfort":  "Continuous, whiny cry — nappy, temperature, or position.",
    "tiredness":   "Weak, slow cry — the baby needs rest.",
    "belly_pain":  "High-pitched, intermittent cry — possible gas or colic.",
    "burping":     "Short, intermittent cry — the baby needs to be burped.",
}


# ─── Helpers ──────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading model…")
def get_model():
    """Load (or initialise) the CNN+CRNN model once per session."""
    return load_model()


def ensure_log_file() -> None:
    """Create the CSV log file with a header row if it does not exist."""
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, "w", newline="", encoding="utf-8") as fh:
            csv.writer(fh).writerow(LOG_COLUMNS)


def log_prediction(filename: str, predicted_class: str, confidence: float) -> None:
    """Append one row to the prediction log CSV."""
    ensure_log_file()
    with open(LOG_FILE, "a", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            filename,
            predicted_class,
            f"{confidence:.4f}",
        ])


# ─── UI Sections ──────────────────────────────────────────────────────────────

def render_header() -> None:
    st.title("👶 Infant Cry Classification")
    st.markdown(
        "Upload a short recording of an infant's cry to classify its type "
        "and understand what the baby might need."
    )
    st.divider()


def render_upload_section() -> st.runtime.uploaded_file_manager.UploadedFile | None:
    st.header("📁 Upload Audio")
    uploaded = st.file_uploader(
        "Choose a WAV or MP3 file",
        type=["wav", "mp3"],
        help="For best results upload a clear 2–5 second recording.",
    )
    return uploaded


def render_audio_and_spectrogram(audio_bytes: bytes, mel_db: np.ndarray) -> None:
    """Show the audio player and mel spectrogram side by side."""
    st.header("🎧 Audio & Spectrogram")
    col_audio, col_spec = st.columns([1, 2], gap="large")

    with col_audio:
        st.subheader("Playback")
        st.audio(audio_bytes)

    with col_spec:
        st.subheader("Mel Spectrogram")
        fig = plot_spectrogram(mel_db)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    st.divider()


def render_prediction_results(
    predicted_class: str,
    confidence: float,
    probabilities: np.ndarray,
) -> None:
    """Display prediction metrics, bar chart, and technical details toggle."""
    st.header("🔍 Prediction Results")

    predicted_idx = int(np.argmax(probabilities))

    # ── Top metrics ──────────────────────────────────────────────────────────
    col1, col2, col3 = st.columns(3, gap="medium")

    display_class = (
        "Uncertain"
        if predicted_class == "Uncertain"
        else predicted_class.replace("_", " ").title()
    )
    with col1:
        st.metric("Predicted Class", display_class)

    with col2:
        st.metric("Confidence", f"{confidence:.1%}")

    with col3:
        if predicted_class == "Uncertain":
            st.metric("Status", "⚠️ Uncertain")
        else:
            st.metric("Status", "✅ Confident")

    # ── Uncertainty warning ───────────────────────────────────────────────────
    if predicted_class == "Uncertain":
        st.warning(
            "⚠️ **Uncertain prediction.** The model's confidence is below "
            f"{UNCERTAINTY_THRESHOLD:.0%}. The audio may be noisy, too short, "
            "or not a recognisable cry type. See the *Model Limitations* section "
            "for guidance."
        )
    else:
        description = CLASS_DESCRIPTIONS.get(CRY_CLASSES[predicted_idx], "")
        if description:
            st.info(f"💡 **{display_class}** — {description}")

    st.divider()

    # ── Probability bar chart ─────────────────────────────────────────────────
    st.subheader("📊 Class Probability Distribution")
    fig = plot_probability_bar(probabilities, CRY_CLASSES, predicted_idx)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.divider()

    # ── Technical details toggle ──────────────────────────────────────────────
    with st.expander("🔧 Technical Details — Raw Probabilities", expanded=False):
        st.caption(
            "Raw softmax probabilities produced by the model for every class."
        )
        prob_df = pd.DataFrame({
            "Class": [c.replace("_", " ").title() for c in CRY_CLASSES],
            "Probability": probabilities.round(6),
            "Percentage": [f"{p:.2%}" for p in probabilities],
        })
        st.dataframe(prob_df, use_container_width=True, hide_index=True)


def render_prediction_log() -> None:
    """Show the last 10 logged predictions."""
    with st.expander("📋 Prediction History (last 10)", expanded=False):
        if os.path.exists(LOG_FILE):
            log_df = pd.read_csv(LOG_FILE)
            if log_df.empty:
                st.info("No predictions have been logged yet.")
            else:
                st.dataframe(
                    log_df.tail(10).reset_index(drop=True),
                    use_container_width=True,
                    hide_index=True,
                )
        else:
            st.info("No predictions have been logged yet.")


def render_limitations_section() -> None:
    """Explain model architecture, limitations, and uncertainty reasoning."""
    st.header("ℹ️ Model Limitations & Uncertainty")
    with st.expander("About the Model & Uncertainty Reasoning", expanded=False):
        st.markdown(f"""
### Model Architecture
This classifier uses a **CNN + CRNN** (Convolutional Neural Network + Convolutional
Recurrent Neural Network) architecture.

* The **CNN** extracts local spectro-temporal features from the mel spectrogram.
* The **GRU** layers capture long-range temporal patterns in the cry sequence.
* A fully-connected **classification head** maps features to six cry classes.

---

### Supported Cry Types

| Class | Description |
|---|---|
| **Hunger** | Rhythmic, repetitive cry |
| **Pain** | Sudden, high-pitched cry |
| **Discomfort** | Continuous, whiny cry |
| **Tiredness** | Weak, slow cry |
| **Belly Pain** | High-pitched, intermittent cry |
| **Burping** | Short, intermittent cry |

---

### When to Expect "Uncertain" Predictions

A prediction is reported as **Uncertain** when the model's top-1 confidence
falls below **{UNCERTAINTY_THRESHOLD:.0%}**. This can happen when:

* **Background noise** drowns out the cry.
* The recording is **too short** (< 2 s) to capture sufficient features.
* The sound is **not a cry** (babbling, laughter, coughing).
* The infant's cry does not match patterns in the training data.

---

### Known Limitations

* The model is trained on a limited dataset and may not generalise to all infants
  or recording environments.
* **Audio quality** significantly affects accuracy — use a close-mic recording
  in a quiet room for best results.
* Individual and cultural variations in infant cries may reduce accuracy.
* This tool is for **informational and research purposes only** and must not
  replace professional medical advice or diagnosis.

---

### Uncertainty Reasoning

The uncertainty mechanism works by examining the **softmax probability** of the
top predicted class.  If no single class captures ≥ {UNCERTAINTY_THRESHOLD:.0%}
of the probability mass, the model is insufficiently confident and reports
"Uncertain" rather than risk a misleading classification.  The full probability
distribution is always displayed so you can inspect the model's "second opinion".
""")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    render_header()

    model = get_model()

    uploaded_file = render_upload_section()

    if uploaded_file is not None:
        st.divider()

        # Read bytes once; reuse for both audio player and feature extraction
        audio_bytes = uploaded_file.getvalue()

        # Compute features ────────────────────────────────────────────────────
        with st.spinner("Analysing audio…"):
            audio, sr = load_audio_from_bytes(audio_bytes)
            mel_db, mel_norm = compute_features(audio, sr)
            predicted_class, confidence, probabilities = predict(model, mel_norm)

        # Log the prediction ──────────────────────────────────────────────────
        log_prediction(uploaded_file.name, predicted_class, confidence)

        # Render sections ─────────────────────────────────────────────────────
        render_audio_and_spectrogram(audio_bytes, mel_db)
        render_prediction_results(predicted_class, confidence, probabilities)
        render_prediction_log()

    st.divider()
    render_limitations_section()


if __name__ == "__main__":
    main()
