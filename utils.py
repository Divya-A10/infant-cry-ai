"""
utils.py — Audio processing and visualisation utilities.

Provides helpers for loading audio, computing mel spectrograms, and
generating matplotlib figures used by the Streamlit dashboard.
"""

import io
from typing import Union

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

# ─── Audio Parameters ─────────────────────────────────────────────────────────

TARGET_SR: int = 22_050   # sample rate
DURATION: float = 3.0     # seconds — clips are padded / truncated to this
N_MELS: int = 128         # mel filterbank bins
N_FFT: int = 2048         # FFT window size
HOP_LENGTH: int = 512     # STFT hop length


# ─── Audio Loading ────────────────────────────────────────────────────────────

def load_audio(
    source: Union[str, bytes, io.IOBase],
    sr: int = TARGET_SR,
    duration: float = DURATION,
) -> tuple[np.ndarray, int]:
    """Load an audio file and return a fixed-length waveform.

    Args:
        source:   File path, raw bytes, or a file-like object.
        sr:       Target sample rate.
        duration: Clip length in seconds.  Shorter clips are zero-padded;
                  longer clips are truncated.

    Returns:
        audio: 1-D float32 numpy array of length ``int(sr * duration)``.
        sr:    Sample rate (same as the *sr* argument).
    """
    audio, sr = librosa.load(source, sr=sr, duration=duration)

    target_length = int(sr * duration)
    if len(audio) < target_length:
        audio = np.pad(audio, (0, target_length - len(audio)))
    else:
        audio = audio[:target_length]

    return audio, sr


def load_audio_from_bytes(audio_bytes: bytes, sr: int = TARGET_SR, duration: float = DURATION) -> tuple[np.ndarray, int]:
    """Convenience wrapper: load audio from raw bytes.

    Args:
        audio_bytes: Raw audio file bytes (WAV, MP3, etc.).
        sr:          Target sample rate.
        duration:    Clip length in seconds.

    Returns:
        audio: 1-D float32 numpy array of length ``int(sr * duration)``.
        sr:    Sample rate (same as the *sr* argument).
    """
    return load_audio(io.BytesIO(audio_bytes), sr=sr, duration=duration)


# ─── Feature Extraction ───────────────────────────────────────────────────────

def compute_features(
    audio: np.ndarray,
    sr: int = TARGET_SR,
    n_mels: int = N_MELS,
    n_fft: int = N_FFT,
    hop_length: int = HOP_LENGTH,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute a mel spectrogram in dB and a normalised version for the model.

    Args:
        audio:      1-D float32 waveform array.
        sr:         Sample rate.
        n_mels:     Number of mel filterbank bins.
        n_fft:      FFT window size.
        hop_length: STFT hop length.

    Returns:
        mel_db:   2-D array ``(n_mels, time_frames)`` in dB — used for display.
        mel_norm: 2-D array ``(n_mels, time_frames)`` normalised to ``[0, 1]``
                  — used as model input.
    """
    mel = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)

    lo, hi = mel_db.min(), mel_db.max()
    mel_norm = (mel_db - lo) / (hi - lo + 1e-8)

    return mel_db, mel_norm


# ─── Visualisation ────────────────────────────────────────────────────────────

def plot_spectrogram(
    mel_db: np.ndarray,
    sr: int = TARGET_SR,
    hop_length: int = HOP_LENGTH,
) -> plt.Figure:
    """Return a matplotlib figure of the mel spectrogram.

    Args:
        mel_db:     2-D dB-scale mel spectrogram ``(n_mels, time_frames)``.
        sr:         Sample rate.
        hop_length: STFT hop length (needed for correct time axis).

    Returns:
        Matplotlib figure — caller is responsible for closing it.
    """
    fig, ax = plt.subplots(figsize=(10, 3))
    img = librosa.display.specshow(
        mel_db,
        sr=sr,
        hop_length=hop_length,
        x_axis="time",
        y_axis="mel",
        cmap="magma",
        ax=ax,
    )
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    ax.set_title("Mel Spectrogram")
    fig.tight_layout()
    return fig


def plot_probability_bar(
    probabilities: np.ndarray,
    class_names: list[str],
    predicted_idx: int,
) -> plt.Figure:
    """Return a bar chart of per-class probabilities.

    The predicted class bar is highlighted in a distinct colour.

    Args:
        probabilities:  1-D float array, one value per class.
        class_names:    List of class label strings.
        predicted_idx:  Index of the predicted (highest-probability) class.

    Returns:
        Matplotlib figure — caller is responsible for closing it.
    """
    labels = [c.replace("_", " ").title() for c in class_names]
    colors = [
        "#FF5722" if i == predicted_idx else "#2196F3"
        for i in range(len(class_names))
    ]

    fig, ax = plt.subplots(figsize=(9, 4))
    bars = ax.bar(labels, probabilities, color=colors, edgecolor="white", linewidth=0.5)

    # Annotate each bar with its percentage
    for bar, prob in zip(bars, probabilities):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{prob:.1%}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Probability")
    ax.set_title("Class Probability Distribution")
    ax.tick_params(axis="x", labelrotation=15)

    legend_elements = [
        Patch(facecolor="#FF5722", label="Predicted class"),
        Patch(facecolor="#2196F3", label="Other classes"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=8)

    fig.tight_layout()
    return fig
