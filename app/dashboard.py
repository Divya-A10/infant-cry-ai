import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import torch
import librosa
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import time

from src.models.cnn_crnn import CryCNNCRNN

# ─── Config ────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Infant Cry Classifier",
    page_icon="👶",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap');

/* Global */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #F7F5F2;
    color: #1a1a1a;
}

.stApp {
    background-color: #F7F5F2;
}

/* Hide streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 3rem 3rem 3rem; max-width: 1200px; }

/* ── Hero Header ── */
.hero {
    display: flex;
    align-items: flex-end;
    justify-content: space-between;
    padding: 2.5rem 0 1.5rem 0;
    border-bottom: 2px solid #1a1a1a;
    margin-bottom: 2.5rem;
}
.hero-title {
    font-family: 'DM Serif Display', serif;
    font-size: 3rem;
    font-weight: 400;
    line-height: 1.1;
    letter-spacing: -0.02em;
    color: #1a1a1a;
    margin: 0;
}
.hero-title em {
    font-style: italic;
    color: #8B6F47;
}
.hero-badge {
    background: #1a1a1a;
    color: #F7F5F2;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    padding: 0.45rem 1rem;
    border-radius: 2rem;
    margin-bottom: 0.25rem;
}

/* ── Upload Card ── */
.upload-label {
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #6b6b6b;
    margin-bottom: 0.5rem;
}

/* ── Section Headers ── */
.section-header {
    font-family: 'DM Serif Display', serif;
    font-size: 1.6rem;
    font-weight: 400;
    color: #1a1a1a;
    letter-spacing: -0.01em;
    margin: 0 0 1.2rem 0;
}

/* ── Result Card ── */
.result-card {
    background: #ffffff;
    border: 1.5px solid #e8e4df;
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 2px 20px rgba(0,0,0,0.04);
}

/* ── Cry Type Display ── */
.cry-type-display {
    display: flex;
    align-items: center;
    gap: 1rem;
    margin-bottom: 1.5rem;
}
.cry-type-label {
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #6b6b6b;
}
.cry-type-value {
    font-family: 'DM Serif Display', serif;
    font-size: 2.5rem;
    font-weight: 400;
    color: #1a1a1a;
    text-transform: capitalize;
    line-height: 1;
}
.cry-type-dot {
    width: 14px;
    height: 14px;
    border-radius: 50%;
    background: #8B6F47;
    flex-shrink: 0;
}

/* ── Confidence Bar ── */
.confidence-row {
    margin-bottom: 1.5rem;
}
.confidence-label-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.4rem;
}
.confidence-label {
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #6b6b6b;
}
.confidence-value {
    font-family: 'DM Serif Display', serif;
    font-size: 1.5rem;
    color: #2D7A4F;
}
.confidence-track {
    height: 6px;
    background: #eee;
    border-radius: 99px;
    overflow: hidden;
}
.confidence-fill {
    height: 100%;
    background: linear-gradient(90deg, #2D7A4F, #5BAD7F);
    border-radius: 99px;
    transition: width 0.8s cubic-bezier(.16,1,.3,1);
}

/* ── Metric Pills ── */
.metric-grid {
    display: flex;
    gap: 1rem;
    flex-wrap: wrap;
    margin-bottom: 1.5rem;
}
.metric-pill {
    background: #F7F5F2;
    border: 1.5px solid #e8e4df;
    border-radius: 12px;
    padding: 0.9rem 1.4rem;
    flex: 1;
    min-width: 120px;
    text-align: center;
}
.metric-pill-label {
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #999;
    margin-bottom: 0.3rem;
}
.metric-pill-value {
    font-family: 'DM Serif Display', serif;
    font-size: 1.6rem;
    color: #1a1a1a;
}
.metric-pill-value.accent {
    color: #C17B2F;
}

/* ── Confusion matrix colours ── */
.cm-high   { background: #2563EB; color: #fff; }
.cm-mid    { background: #93C5FD; color: #1a1a1a; }
.cm-low    { background: #DBEAFE; color: #1a1a1a; }
.cm-zero   { background: #F8FAFC; color: #94a3b8; }
.cm-accent { background: #F59E42; color: #fff; }

/* ── File info chip ── */
.file-chip {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    background: #fff;
    border: 1.5px solid #e8e4df;
    border-radius: 8px;
    padding: 0.4rem 0.9rem;
    font-size: 0.82rem;
    color: #444;
    margin-bottom: 1.5rem;
}
.file-chip-dot {
    width: 8px; height: 8px;
    background: #2D7A4F;
    border-radius: 50%;
    flex-shrink: 0;
}

/* ── Divider ── */
.thin-divider {
    border: none;
    border-top: 1.5px solid #e8e4df;
    margin: 1.5rem 0;
}

/* Override streamlit uploader styling */
[data-testid="stFileUploader"] {
    background: #fff;
    border: 2px dashed #d0cbc4;
    border-radius: 14px;
    padding: 1rem;
    transition: border-color 0.2s;
}
[data-testid="stFileUploader"]:hover {
    border-color: #8B6F47;
}

/* ── Uncertainty Banner ── */
.uncertainty-banner {
    background: #FEF9EC;
    border: 1.5px solid #F5C842;
    border-radius: 12px;
    padding: 1rem 1.3rem;
    margin-top: 1rem;
    display: flex;
    gap: 0.8rem;
    align-items: flex-start;
}
.uncertainty-icon {
    font-size: 1.2rem;
    flex-shrink: 0;
    margin-top: 1px;
}
.uncertainty-reason {
    font-size: 0.85rem;
    font-weight: 500;
    color: #7A5C00;
    margin-bottom: 0.25rem;
}
.uncertainty-tip {
    font-size: 0.78rem;
    color: #A07820;
}

/* Spinner */
.stSpinner > div {
    border-top-color: #8B6F47 !important;
}
</style>
""", unsafe_allow_html=True)

# ─── Model Setup ───────────────────────────────────────────────────────────────
classes = ["pain", "hunger", "discomfort", "tired", "other"]
idx_to_label = {i: c for i, c in enumerate(classes)}

COLOR_MAP = {
    "pain":       "#E05252",
    "hunger":     "#2563EB",
    "discomfort": "#10B981",
    "tired":      "#8B5CF6",
    "other":      "#F59E0B",
}

device = "cpu"

@st.cache_resource(show_spinner=False)
def load_model():
    m = CryCNNCRNN(num_classes=len(classes))
    ckpt = torch.load("cry_model.pth", map_location=device)
    m.load_state_dict(ckpt, strict=False)
    m.to(device)
    m.eval()
    return m

model = load_model()

# ─── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div>
    <h1 class="hero-title">AI Infant Cry<br><em>Classifier</em></h1>
  </div>
  <div class="hero-badge">👶 &nbsp;Acoustic Analysis</div>
</div>
""", unsafe_allow_html=True)

# ─── Upload ────────────────────────────────────────────────────────────────────
st.markdown('<div class="upload-label">Upload Baby Cry Audio</div>', unsafe_allow_html=True)
uploaded_file = st.file_uploader("Upload audio", type=["wav"], label_visibility="collapsed")

# ─── Main Logic ────────────────────────────────────────────────────────────────
if uploaded_file is not None:

    with st.spinner("Analysing audio…"):
        time.sleep(0.4)  # small visual pause for polish

        y, sr = librosa.load(uploaded_file, sr=16000)
        duration = len(y) / sr

        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64)
        mel_db = librosa.power_to_db(mel)
        spec = torch.tensor(mel_db).unsqueeze(0).unsqueeze(0).float().to(device)

        with torch.no_grad():
            output = model(spec)
            probs = torch.softmax(output, dim=1)[0].cpu().numpy()
            pred = int(probs.argmax())
            confidence = float(probs[pred])

    raw_label = idx_to_label[pred]
    conf_pct  = round(confidence * 100, 1)

    # ── Uncertainty Explanation Layer ──────────────────────────────────────────
    LOW_CONF_THRESHOLD  = 0.55   # below this → model is genuinely unsure
    HIGH_ENTROPY_THRESHOLD = 1.8 # high spread across classes → uncertain

    entropy = float(-np.sum(probs * np.log(probs + 1e-9)))
    top2_gap = float(np.sort(probs)[-1] - np.sort(probs)[-2])

    is_other       = raw_label == "other"
    is_low_conf    = confidence < LOW_CONF_THRESHOLD
    is_high_entropy = entropy > HIGH_ENTROPY_THRESHOLD
    is_close_call  = top2_gap < 0.15   # top two classes nearly tied

    if is_other and confidence >= LOW_CONF_THRESHOLD:
        cry_label    = "uncertain"
        uncertainty_reason = "Acoustic features do not match any trained cry category"
        uncertainty_tip    = "Try a cleaner recording — background noise may be interfering"
        is_uncertain = True
    elif is_low_conf:
        cry_label    = "uncertain"
        uncertainty_reason = f"Model confidence is too low to make a reliable prediction ({conf_pct}%)"
        uncertainty_tip    = "The audio may be too short, too quiet, or contain mixed sounds"
        is_uncertain = True
    elif is_high_entropy:
        cry_label    = "uncertain"
        uncertainty_reason = "Probability is spread across multiple categories — no clear dominant pattern"
        uncertainty_tip    = "Try isolating a single continuous cry segment for better accuracy"
        is_uncertain = True
    elif is_close_call:
        cry_label    = raw_label
        uncertainty_reason = f"Close call — also resembles '{idx_to_label[int(np.argsort(probs)[-2])]}'"
        uncertainty_tip    = "Consider re-recording with less ambient noise"
        is_uncertain = True
    else:
        cry_label         = raw_label
        uncertainty_reason = None
        uncertainty_tip    = None
        is_uncertain       = False

    accent_color = "#9B8EA0" if is_uncertain else COLOR_MAP.get(cry_label, "#8B6F47")

    # ── File info chip ──
    dur_str = f"{int(duration//60):02d}:{int(duration%60):02d}"
    st.markdown(f"""
    <div class="file-chip">
      <div class="file-chip-dot"></div>
      <span><strong>{uploaded_file.name}</strong>&nbsp;&nbsp;({dur_str}) — Infant crying detected</span>
    </div>
    """, unsafe_allow_html=True)

    # ── Two-column layout ──
    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        # ── Result card ──
        st.markdown('<div class="result-card">', unsafe_allow_html=True)

        # Cry type row
        st.markdown(f"""
        <div class="cry-type-display">
          <div class="cry-type-dot" style="background:{accent_color}"></div>
          <div style="flex:1">
            <div class="cry-type-label">Detected Cry Type</div>
            <div class="cry-type-value">{cry_label}</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Reason row — separate block to avoid f-string quote conflicts
        if uncertainty_reason:
            st.markdown(
                '<div class="cry-type-label" style="margin-bottom:0.2rem;">Reason</div>'
                f'<div style="font-size:0.88rem;color:#555;font-style:italic;margin-bottom:1rem;">{uncertainty_reason}</div>',
                unsafe_allow_html=True,
            )

        # Confidence bar
        conf_color = "#C17B2F" if is_uncertain else "#2D7A4F"
        bar_gradient = "linear-gradient(90deg,#C17B2F,#E5A86A)" if is_uncertain else "linear-gradient(90deg,#2D7A4F,#5BAD7F)"
        st.markdown(f"""
        <div class="confidence-row">
          <div class="confidence-label-row">
            <span class="confidence-label">Confidence</span>
            <span class="confidence-value" style="color:{conf_color}">{conf_pct}%</span>
          </div>
          <div class="confidence-track">
            <div class="confidence-fill" style="width:{conf_pct}%;background:{bar_gradient}"></div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Tip banner
        if uncertainty_tip:
            st.markdown(
                '<div class="uncertainty-banner">'
                '<div class="uncertainty-icon">💡</div>'
                '<div>'
                '<div class="uncertainty-reason">What this means</div>'
                f'<div class="uncertainty-tip">{uncertainty_tip}</div>'
                '</div></div>',
                unsafe_allow_html=True,
            )

        st.markdown('<hr class="thin-divider">', unsafe_allow_html=True)

        # All class probabilities
        st.markdown('<div class="upload-label">All Class Probabilities</div>', unsafe_allow_html=True)
        for i, cls in enumerate(classes):
            p = float(probs[i]) * 100
            col = COLOR_MAP.get(cls, "#999")
            st.markdown(f"""
            <div style="margin-bottom:0.55rem;">
              <div style="display:flex;justify-content:space-between;margin-bottom:3px;">
                <span style="font-size:0.8rem;font-weight:500;text-transform:capitalize;color:#444">{cls}</span>
                <span style="font-size:0.8rem;font-weight:600;color:{col}">{p:.1f}%</span>
              </div>
              <div class="confidence-track">
                <div style="height:100%;width:{p}%;background:{col};border-radius:99px;"></div>
              </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    with col_right:
        # ── Spectrogram ──
        st.markdown('<div class="result-card" style="height:100%">', unsafe_allow_html=True)
        st.markdown('<div class="section-header">Spectrogram</div>', unsafe_allow_html=True)

        viridis = plt.cm.viridis
        fig, ax = plt.subplots(figsize=(6, 4))
        fig.patch.set_facecolor('#ffffff')
        ax.set_facecolor('#ffffff')

        img = ax.imshow(
            mel_db, aspect="auto", origin="lower",
            cmap="viridis", interpolation="bilinear"
        )
        fig.colorbar(img, ax=ax, format="%+2.0f dB", shrink=0.8,
                     label="Power (dB)", pad=0.02)
        ax.set_xlabel("Time Frames", fontsize=9, color="#666")
        ax.set_ylabel("Mel Frequency Bins", fontsize=9, color="#666")
        ax.tick_params(colors="#999", labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor("#e8e4df")
        plt.tight_layout(pad=1.2)
        st.pyplot(fig, width='stretch')
        plt.close(fig)

        st.markdown('</div>', unsafe_allow_html=True)

    # ─── Waveform ───────────────────────────────────────────────────────────────
    st.markdown('<div class="result-card" style="margin-top:1.5rem">', unsafe_allow_html=True)
    st.markdown('<div class="section-header">Waveform</div>', unsafe_allow_html=True)

    times = np.linspace(0, duration, len(y))
    fig2, ax2 = plt.subplots(figsize=(10, 2.2))
    fig2.patch.set_facecolor("#ffffff")
    ax2.set_facecolor("#ffffff")
    ax2.fill_between(times, y, alpha=0.25, color=accent_color)
    ax2.plot(times, y, color=accent_color, linewidth=0.6, alpha=0.9)
    ax2.axhline(0, color="#e8e4df", linewidth=0.8)
    ax2.set_xlabel("Time (s)", fontsize=9, color="#666")
    ax2.set_ylabel("Amplitude", fontsize=9, color="#666")
    ax2.tick_params(colors="#999", labelsize=8)
    for spine in ax2.spines.values():
        spine.set_edgecolor("#e8e4df")
    plt.tight_layout(pad=0.8)
    st.pyplot(fig2, width='stretch')
    plt.close(fig2)

    st.markdown('</div>', unsafe_allow_html=True)

else:
    # ─── Empty state ────────────────────────────────────────────────────────────
    st.markdown("""
    <div style="text-align:center;padding:4rem 0 2rem 0;color:#bbb;">
      <div style="font-size:3.5rem;margin-bottom:1rem;">🎙️</div>
      <div style="font-family:'DM Serif Display',serif;font-size:1.4rem;color:#ccc;margin-bottom:0.5rem;">
        Awaiting audio…
      </div>
      <div style="font-size:0.85rem;color:#d0cbc4;">Upload a .wav file above to begin analysis</div>
    </div>
    """, unsafe_allow_html=True)