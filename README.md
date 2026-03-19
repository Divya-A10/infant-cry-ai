# Infant Cry Classification System

Ever wished you could tell *why* a baby is crying? That's exactly what this project tries to do.

This is an AI-powered system that listens to infant cry audio and predicts its likely cause — hunger, pain, discomfort, tiredness, or something else entirely. But unlike most classifiers that just throw out an answer no matter what, this one knows when it doesn't know. If the audio is noisy, ambiguous, or just doesn't match anything in its training, it says so — and tells you why.

---

## What it does

- Takes a `.wav` audio file as input
- Extracts mel spectrogram features from the audio
- Runs it through a hybrid CNN + CRNN model that picks up both spectral and temporal patterns
- Returns a prediction with a confidence score
- If the model isn't sure, it flags the result as **uncertain** and explains the likely reason

The whole thing runs in an interactive Streamlit dashboard where you can upload a file, see the spectrogram, and get an instant prediction.

---

## Why the uncertainty layer matters

Most ML models will confidently predict something even when they shouldn't. A baby monitor that tells you "hunger" when the model is actually 51% sure isn't useful — it's misleading.

This system takes a different approach. If confidence is too low, if the probabilities are spread too evenly across classes, or if the model predicts "other" (meaning it doesn't recognize the pattern), the output explicitly says **uncertain** along with a plain-English explanation. That way, a parent or caregiver always knows how much to trust the result.

---

## How it works

```
Audio file (.wav)
    → Voice Activity Detection
    → Mel Spectrogram
    → CNN (spatial features) + CRNN (temporal features)
    → Softmax probabilities
    → Uncertainty check
    → Prediction + explanation
```

### Cry categories
- `pain`
- `hunger`
- `discomfort`
- `tired`
- `other` (treated as uncertain if confidence is high — means the model is confident it doesn't fit)

---

## Project structure

```
infant-cry-ai/
│
├── app/
│   └── dashboard.py          # Streamlit interface
│
├── src/
│   ├── models/
│   │   └── cnn_crnn.py       # Model architecture
│   ├── features/
│   │   └── spectrogram.py    # Mel spectrogram extraction
│   ├── preprocessing/
│   │   └── vad.py            # Voice activity detection
│   ├── explainability/
│   │   └── gradcam.py        # Grad-CAM visualisation
│   └── severity/
│       └── severity_index.py # Cry severity scoring
│
├── training/
│   └── train_model.py        # Training script
│
├── data/
│   ├── raw/
│   └── processed/
│
├── cry_model.pth             # Trained model weights
├── requirements.txt
└── README.md
```

---

## Getting started

```bash
# Clone the repo
git clone https://github.com/Divya-A10/infant-cry-ai.git
cd infant-cry-ai

# Set up a virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the dashboard
streamlit run app/dashboard.py
```

Then open `http://localhost:8501` in your browser, upload a `.wav` file, and you're good to go.

---

## Known limitations

- Background noise can throw off predictions — cleaner recordings work better
- The `other` class can dominate if the training data is imbalanced
- The model has only been trained on a limited set of environments, so it may not generalise perfectly to every recording setup
- Real-time audio input isn't supported yet (file upload only)

---

## What's next

- Real-time microphone input
- Better class balance handling during training
- Improved uncertainty calibration
- Expanded dataset with more diverse acoustic environments
- Potential deployment on edge devices for real-world use

---

## License

Built for academic and research purposes.
