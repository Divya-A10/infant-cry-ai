"""
model.py — CNN+CRNN model definition and prediction logic.

This module contains the core model architecture and inference utilities.
Do NOT modify this file when making UI/usability changes; edit app.py instead.
"""

import numpy as np
import torch
import torch.nn as nn

# ─── Class Labels ─────────────────────────────────────────────────────────────

CRY_CLASSES = ["hunger", "pain", "discomfort", "tiredness", "belly_pain", "burping"]

# Confidence threshold below which the prediction is reported as "Uncertain".
UNCERTAINTY_THRESHOLD = 0.50


# ─── Model Architecture ───────────────────────────────────────────────────────

class CNN_CRNN(nn.Module):
    """CNN + GRU-based model for infant cry classification.

    The CNN extracts local spectro-temporal features from a mel spectrogram,
    and the GRU captures long-range temporal dependencies before the final
    classification layer.
    """

    def __init__(self, num_classes: int = len(CRY_CLASSES)):
        super().__init__()

        # ── CNN feature extractor ──────────────────────────────────────────
        self.cnn = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # Adaptive pooling produces a fixed spatial output regardless of input size.
            nn.AdaptiveAvgPool2d((4, 16)),  # → (batch, 128, 4, 16)
        )

        # ── Recurrent layer ───────────────────────────────────────────────
        # After CNN: (batch, 128, 4, 16) → rearrange to (batch, 16, 512)
        rnn_input_size = 128 * 4  # channels × freq_bins = 512
        self.rnn = nn.GRU(
            input_size=rnn_input_size,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            dropout=0.3,
        )

        # ── Classification head ───────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Mel spectrogram tensor of shape (batch, 1, n_mels, time_frames).

        Returns:
            Raw logits of shape (batch, num_classes).
        """
        # CNN: (batch, 1, freq, time) → (batch, 128, 4, 16)
        features = self.cnn(x)

        # Reshape for RNN: (batch, 128, 4, 16) → (batch, 16, 512)
        batch, channels, freq, time = features.shape
        features = features.permute(0, 3, 1, 2).reshape(batch, time, channels * freq)

        # RNN: (batch, 16, 512) → (batch, 16, 256)
        rnn_out, _ = self.rnn(features)

        # Use the last time step's output for classification
        logits = self.classifier(rnn_out[:, -1, :])
        return logits


# ─── Model Loading ────────────────────────────────────────────────────────────

def load_model(model_path: str | None = None, device: str = "cpu") -> CNN_CRNN:
    """Instantiate and optionally load pre-trained weights.

    Args:
        model_path: Path to a saved ``state_dict`` (``.pt`` / ``.pth``).
                    When ``None`` the model is returned with random weights
                    (useful for UI development without trained weights).
        device:     Torch device string, e.g. ``"cpu"`` or ``"cuda"``.

    Returns:
        Model in evaluation mode.
    """
    model = CNN_CRNN(num_classes=len(CRY_CLASSES))
    if model_path:
        state = torch.load(model_path, map_location=device)
        model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


# ─── Inference ────────────────────────────────────────────────────────────────

def predict(
    model: CNN_CRNN,
    mel_spectrogram: np.ndarray,
    device: str = "cpu",
) -> tuple[str, float, np.ndarray]:
    """Run inference on a normalised mel spectrogram.

    Args:
        model:           Loaded ``CNN_CRNN`` model in eval mode.
        mel_spectrogram: 2-D float array of shape ``(n_mels, time_frames)``
                         with values normalised to ``[0, 1]``.
        device:          Torch device string.

    Returns:
        predicted_class: Human-readable class name, or ``"Uncertain"`` when
                         the top-1 confidence is below ``UNCERTAINTY_THRESHOLD``.
        confidence:      Probability of the top-1 class (0–1).
        probabilities:   1-D numpy array of per-class probabilities.
    """
    tensor = (
        torch.from_numpy(mel_spectrogram)
        .float()
        .unsqueeze(0)   # batch dim
        .unsqueeze(0)   # channel dim  → (1, 1, n_mels, time_frames)
        .to(device)
    )

    with torch.no_grad():
        logits = model(tensor)
        probabilities: np.ndarray = (
            torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
        )

    predicted_idx = int(probabilities.argmax())
    confidence = float(probabilities[predicted_idx])

    if confidence < UNCERTAINTY_THRESHOLD:
        predicted_class = "Uncertain"
    else:
        predicted_class = CRY_CLASSES[predicted_idx]

    return predicted_class, confidence, probabilities
