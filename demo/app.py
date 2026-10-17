"""
================================================================================
EE136IU: Machine Learning for Wireless Communications
Capstone Mini-Project - Topic 02: Spectral & Temporal Feature Engineering
                                 vs. End-to-End Raw I/Q Deep Learning (AMC)
================================================================================
Application: demo/app.py
Description:
  Interactive 10-Minute Live Defense Web Application (Streamlit).
  Features:
    1. Real-time controls: SNR slider (-10 to +18 dB), Class selector, Random test burst.
    2. Dynamic channel impairment injection: CFO, Phase Jitter, Noise.
    3. Live Signal Visualizer:
       - Constellation Scatter Plot (I[n] vs Q[n])
       - Computed FFT Power Spectrum with Centroid (mu_f) and Spread (sigma_f)
       - Time-Domain I[n] & Q[n] Waveforms (n = 0...127)
    4. Real-time Extracted Feature Metric Cards (|C40|, C42, Phase Std, Centroid, Spread, Entropy).
    5. Side-by-side Model Predictions (XGBoost vs 1D-CNN) with probabilities and latency.
    6. Visual Highlighting (Green/Red) and Low-SNR Failure Diagnostic Panel.
================================================================================
"""

import os
import sys
import time
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd

# Add project root to sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dataset.data_generator import (
    MOD_CLASSES,
    MOD_TO_IDX,
    IDX_TO_MOD,
    CALIBRATED_SNR_GRID,
    generate_burst_signal,
    apply_awgn_and_rms_norm
)
from utils.feature_extractor import (
    extract_burst_features_1d,
    get_feature_dict,
    FEATURE_NAMES
)
from models.ml_classifier import FeatureMLClassifier
from models.cnn1d import CNN1DPipeline


# Page configuration
st.set_page_config(
    page_title="EE136 AMC Topic 02 | Live Defense Demo",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Clean White Academic Theme Styling
st.markdown("""
<style>
    /* Clean Academic Theme */
    .main {
        background-color: #FFFFFF;
        color: #0F172A;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 14px 18px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .metric-title {
        color: #64748B;
        font-size: 0.82rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-value {
        color: #0F172A;
        font-size: 1.45rem;
        font-weight: 700;
        margin-top: 4px;
    }
    .badge-correct {
        background-color: #DCFCE7;
        color: #166534;
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.85rem;
        display: inline-block;
    }
    .badge-wrong {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.85rem;
        display: inline-block;
    }
    .header-box {
        background: linear-gradient(135deg, #0284C7 0%, #1E3A8A 100%);
        color: white;
        padding: 20px 24px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_all_models():
    """Loads pre-trained XGBoost, RBF-SVM, and 1D-CNN models with caching."""
    saved_dir = os.path.join(PROJECT_ROOT, 'saved_models')
    xgb_path = os.path.join(saved_dir, 'xgboost_amc.joblib')
    svm_path = os.path.join(saved_dir, 'svm_amc.joblib')
    cnn_path = os.path.join(saved_dir, 'best_cnn1d_model.pt')

    models = {}
    if os.path.exists(xgb_path):
        models['xgboost'] = FeatureMLClassifier.load(xgb_path)
    else:
        models['xgboost'] = None

    if os.path.exists(svm_path):
        models['svm'] = FeatureMLClassifier.load(svm_path)
    else:
        models['svm'] = None

    if os.path.exists(cnn_path):
        cnn_pipe = CNN1DPipeline(num_classes=len(MOD_CLASSES))
        cnn_pipe.load_checkpoint(cnn_path)
        models['cnn1d'] = cnn_pipe
    else:
        models['cnn1d'] = None

    return models


models_dict = load_all_models()

# Header Banner
st.markdown("""
<div class="header-box">
    <div style="font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.1em; opacity: 0.9;">
        EE136IU: Machine Learning for Wireless Communications • Capstone Mini-Project
    </div>
    <div style="font-size: 1.7rem; font-weight: 800; margin-top: 4px;">
        Topic 02: Spectral & Temporal Feature Engineering vs. End-to-End Raw I/Q Deep Learning
    </div>
    <div style="font-size: 0.95rem; opacity: 0.95; margin-top: 6px;">
        Interactive Live Demonstration: Real-Time Baseband Synthesis, Cumulants, Differential Phase, FFT Spectrum, and Dual-Model Inference
    </div>
</div>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# SIDEBAR CONTROLS
# -----------------------------------------------------------------------------
st.sidebar.markdown("### 🎛️ RF Baseband Signal Controls")

# Session state for random burst seeding
if 'burst_seed' not in st.session_state:
    st.session_state.burst_seed = 42

if st.sidebar.button("🎲 Sample Random Unseen Test Burst", use_container_width=True):
    st.session_state.burst_seed = int(np.random.randint(1000, 999999))

snr_val = st.sidebar.select_slider(
    "Calibrated SNR Level (dB)",
    options=list(CALIBRATED_SNR_GRID),
    value=4,
    help="Select calibrated operational SNR from -10 dB to +18 dB (step 2 dB)."
)

ground_truth_mod = st.sidebar.selectbox(
    "Ground Truth Modulation Class",
    options=MOD_CLASSES,
    index=1,
    help="Select one of the 8 standardized digital modulation schemes."
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🌊 Channel Impairment Stress-Testing")

enable_impairments = st.sidebar.checkbox("Inject Realistic Impairments", value=False)
if enable_impairments:
    cfo_ratio = st.sidebar.slider("Carrier Frequency Offset (Δf / fs)", 0.0, 0.05, 0.005, step=0.001,
                                  help="Simulates receiver oscillator mismatch and Doppler drift (0 to 5% fs)")
    phase_jitter_std = st.sidebar.slider("Phase Jitter (rad)", 0.0, 0.20, 0.02, step=0.01,
                                         help="Simulates phase noise and local oscillator jitter")
else:
    cfo_ratio = 0.0
    phase_jitter_std = 0.0

st.sidebar.markdown("---")
st.sidebar.markdown("### ℹ️ Topic Specifications")
st.sidebar.info(
    "**L = 128 samples per burst**\n\n"
    "**Tensor shape: [2, 128]**\n\n"
    "**Per-burst RMS Normalization: Active**\n\n"
    "**Models Loaded:** "
    f"{'✓ XGBoost ' if models_dict['xgboost'] else '✗ XGBoost '}"
    f"{'✓ 1D-CNN ' if models_dict['cnn1d'] else '✗ 1D-CNN'}"
)


# -----------------------------------------------------------------------------
# SIGNAL SYNTHESIS & IMPAIRMENTS
# -----------------------------------------------------------------------------
rng = np.random.RandomState(st.session_state.burst_seed)
clean_burst = generate_burst_signal(ground_truth_mod, burst_length=128, rng=rng)

# Inject CFO / Phase Jitter if enabled
if enable_impairments:
    n_idx = np.arange(128)
    cfo_rotation = np.exp(1j * 2.0 * np.pi * cfo_ratio * n_idx)
    jitter = rng.normal(0.0, phase_jitter_std, 128)
    clean_burst = clean_burst * cfo_rotation * np.exp(1j * jitter)

# Add Calibrated AWGN and Apply Mandatory RMS Normalization
norm_burst, iq_tensor = apply_awgn_and_rms_norm(clean_burst, snr_val, rng=rng)

# Feature Extraction
feats = extract_burst_features_1d(norm_burst)
feat_dict = get_feature_dict(norm_burst)


# -----------------------------------------------------------------------------
# LIVE SIGNAL VISUALIZATION (3 PLOTS)
# -----------------------------------------------------------------------------
st.markdown("### 📊 Live Signal Visualizer (L = 128 Baseband Samples)")

vis_col1, vis_col2, vis_col3 = st.columns(3)

# Plot 1: Constellation Scatter Plot
with vis_col1:
    fig1, ax1 = plt.subplots(figsize=(4.5, 4.2), dpi=200)
    ax1.scatter(iq_tensor[0], iq_tensor[1], color='#0284C7', alpha=0.85, s=28, edgecolors='black', linewidth=0.5)
    circle = plt.Circle((0, 0), 1.0, color='gray', fill=False, linestyle='--', linewidth=0.8, alpha=0.6)
    ax1.add_patch(circle)
    ax1.axhline(0, color='#94A3B8', linestyle=':', linewidth=0.8)
    ax1.axvline(0, color='#94A3B8', linestyle=':', linewidth=0.8)
    ax1.set_xlim([-2.4, 2.4])
    ax1.set_ylim([-2.4, 2.4])
    ax1.set_title("Constellation Scatter (I[n] vs. Q[n])", fontsize=10, fontweight='bold')
    ax1.set_xlabel("In-Phase I[n]", fontsize=8, fontweight='bold')
    ax1.set_ylabel("Quadrature Q[n]", fontsize=8, fontweight='bold')
    ax1.grid(True, linestyle='--', alpha=0.4)
    st.pyplot(fig1, use_container_width=True)
    plt.close(fig1)

# Plot 2: Computed FFT Power Spectrum
with vis_col2:
    fig2, ax2 = plt.subplots(figsize=(4.5, 4.2), dpi=200)
    Z_fft = np.fft.fftshift(np.fft.fft(norm_burst))
    P_fft = np.abs(Z_fft)**2
    P_norm = P_fft / (np.sum(P_fft) + 1e-12)
    f_axis = np.fft.fftshift(np.fft.fftfreq(128, d=1.0))

    ax2.plot(f_axis, P_norm, color='#EA580C', linewidth=1.8, label='Normalized FFT')
    # Vertical line for Spectral Centroid
    mu_f = feat_dict['spectral_centroid']
    sigma_f = feat_dict['spectral_spread']
    gamma_max = feat_dict.get('gamma_max', float(np.max(P_norm)))
    H_f = feat_dict.get('spectral_entropy', 0.0)

    ax2.axvline(mu_f, color='#DC2626', linestyle='--', linewidth=1.5, label=f'Centroid ({mu_f:+.3f})')
    # Shaded band for Spectral Spread
    ax2.axvspan(mu_f - sigma_f, mu_f + sigma_f, color='#38BDF8', alpha=0.25, label=f'Spread (±{sigma_f:.3f})')
    # Horizontal line for Peak gamma_max
    ax2.axhline(gamma_max, color='#8B5CF6', linestyle=':', linewidth=1.3, label=f'Peak γ_max ({gamma_max:.3f})')

    ax2.set_xlim([-0.5, 0.5])
    ax2.set_ylim([0, max(np.max(P_norm) * 1.18, 0.05)])
    ax2.set_title("Computed FFT Power Spectrum", fontsize=10, fontweight='bold')
    ax2.set_xlabel("Normalized Frequency (f / fs)", fontsize=8, fontweight='bold')
    ax2.set_ylabel("Normalized Power P[k]", fontsize=8, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=6.8, frameon=True)
    ax2.grid(True, linestyle='--', alpha=0.4)
    st.pyplot(fig2, use_container_width=True)
    plt.close(fig2)

# Plot 3: Time-Domain Waveforms
with vis_col3:
    fig3, ax3 = plt.subplots(figsize=(4.5, 4.2), dpi=200)
    n_samples = np.arange(128)
    ax3.plot(n_samples, iq_tensor[0], color='#2563EB', linewidth=1.3, label='I[n] (In-Phase)')
    ax3.plot(n_samples, iq_tensor[1], color='#10B981', linewidth=1.3, linestyle='--', label='Q[n] (Quadrature)')
    ax3.set_xlim([0, 127])
    ax3.set_ylim([-2.6, 2.6])
    ax3.set_title("Time-Domain Waveforms (n = 0...127)", fontsize=10, fontweight='bold')
    ax3.set_xlabel("Sample Index n", fontsize=8, fontweight='bold')
    ax3.set_ylabel("Amplitude", fontsize=8, fontweight='bold')
    ax3.legend(loc='upper right', fontsize=7, frameon=True)
    ax3.grid(True, linestyle='--', alpha=0.4)
    st.pyplot(fig3, use_container_width=True)
    plt.close(fig3)


# -----------------------------------------------------------------------------
# EXTRACTED FEATURE VALUES PANEL
# -----------------------------------------------------------------------------
st.markdown("### 🔬 Extracted Physical Feature Values Panel")

# Row 1: Cumulants & Differential Phase
r1_c1, r1_c2, r1_c3, r1_c4 = st.columns(4)
with r1_c1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">|C40| (Cumulant)</div>
        <div class="metric-value">{feat_dict['C40_abs']:.3f}</div>
    </div>
    """, unsafe_allow_html=True)
with r1_c2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">C42 (Cumulant)</div>
        <div class="metric-value">{feat_dict['C42']:.3f}</div>
    </div>
    """, unsafe_allow_html=True)
with r1_c3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Phase Diff Std (σ_Δϕ)</div>
        <div class="metric-value">{feat_dict['phase_diff_std']:.3f}</div>
    </div>
    """, unsafe_allow_html=True)
with r1_c4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Max FFT Phase Diff</div>
        <div class="metric-value">{feat_dict.get('phase_diff_fft_max', 0.0):.3f}</div>
    </div>
    """, unsafe_allow_html=True)

# Row 2: Frequency-Domain (FFT) Spectral Statistics
r2_c1, r2_c2, r2_c3, r2_c4 = st.columns(4)
with r2_c1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">FFT Peak (γ_max)</div>
        <div class="metric-value">{feat_dict.get('gamma_max', float(np.max(P_norm))):.3f}</div>
    </div>
    """, unsafe_allow_html=True)
with r2_c2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Spectral Centroid (μ_f)</div>
        <div class="metric-value">{feat_dict['spectral_centroid']:+.3f}</div>
    </div>
    """, unsafe_allow_html=True)
with r2_c3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Spectral Spread (σ_f)</div>
        <div class="metric-value">{feat_dict['spectral_spread']:.3f}</div>
    </div>
    """, unsafe_allow_html=True)
with r2_c4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Spectral Entropy (H_f)</div>
        <div class="metric-value">{feat_dict['spectral_entropy']:.3f}</div>
    </div>
    """, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# SIDE-BY-SIDE MODEL PREDICTIONS
# -----------------------------------------------------------------------------
st.markdown("---")
st.markdown(f"### ⚔️ Side-by-Side Model Predictions (Ground Truth: **{ground_truth_mod}** @ SNR = **{snr_val:+d} dB**)")

col_xgb, col_cnn = st.columns(2)

# Column 1: Feature Engineering ML (XGBoost)
with col_xgb:
    st.markdown("#### 🌲 Branch 1: Feature-Based XGBoost")
    if models_dict['xgboost'] is not None:
        xgb_pred_idx, xgb_probs, xgb_lat = models_dict['xgboost'].predict_single_burst(feats)
        xgb_pred_mod = IDX_TO_MOD[xgb_pred_idx]
        is_xgb_correct = (xgb_pred_mod == ground_truth_mod)

        badge_html = f'<span class="badge-correct">✓ CORRECT</span>' if is_xgb_correct else f'<span class="badge-wrong">✗ MISCLASSIFIED</span>'

        st.markdown(f"""
        <div style="background-color: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 8px; padding: 16px; margin-bottom: 12px;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span style="font-size: 1.25rem; font-weight: 800; color: #0F172A;">Prediction: {xgb_pred_mod}</span>
                {badge_html}
            </div>
            <div style="font-size: 0.88rem; color: #64748B; margin-top: 6px;">
                Top Confidence: <b>{xgb_probs[xgb_pred_idx]*100:.1f}%</b> | Inference Latency: <b>{xgb_lat:.2f} ms</b>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Probability Bar Chart
        df_xgb = pd.DataFrame({'Modulation': MOD_CLASSES, 'Confidence': xgb_probs * 100})
        st.bar_chart(df_xgb.set_index('Modulation'), color='#EA580C')
    else:
        st.warning("XGBoost model not found. Run `python main.py` to train and save models.")

# Column 2: End-to-End Raw I/Q 1D-CNN
with col_cnn:
    st.markdown("#### 🧠 Branch 2: End-to-End Raw I/Q 1D-CNN")
    if models_dict['cnn1d'] is not None:
        cnn_pred_idx, cnn_probs, cnn_lat = models_dict['cnn1d'].predict_single_burst(iq_tensor)
        cnn_pred_mod = IDX_TO_MOD[cnn_pred_idx]
        is_cnn_correct = (cnn_pred_mod == ground_truth_mod)

        badge_html = f'<span class="badge-correct">✓ CORRECT</span>' if is_cnn_correct else f'<span class="badge-wrong">✗ MISCLASSIFIED</span>'

        st.markdown(f"""
        <div style="background-color: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 8px; padding: 16px; margin-bottom: 12px;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span style="font-size: 1.25rem; font-weight: 800; color: #0F172A;">Prediction: {cnn_pred_mod}</span>
                {badge_html}
            </div>
            <div style="font-size: 0.88rem; color: #64748B; margin-top: 6px;">
                Top Confidence: <b>{cnn_probs[cnn_pred_idx]*100:.1f}%</b> | Inference Latency: <b>{cnn_lat:.2f} ms</b>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Probability Bar Chart
        df_cnn = pd.DataFrame({'Modulation': MOD_CLASSES, 'Confidence': cnn_probs * 100})
        st.bar_chart(df_cnn.set_index('Modulation'), color='#0284C7')
    else:
        st.warning("1D-CNN model checkpoint not found. Run `python main.py` to train and save checkpoint.")


# -----------------------------------------------------------------------------
# LOW-SNR FAILURE DIAGNOSTIC ACCORDION (DEFENSE AID)
# -----------------------------------------------------------------------------
st.markdown("---")
with st.expander("🔍 Faculty Q&A Aid: Why Cumulants Fail at Low SNR vs. 1D-CNN Spatio-Temporal Filtering", expanded=(snr_val <= -4)):
    st.markdown("""
    **Theoretical Breakdown for Committee Defense:**
    
    1. **Cumulant Variance Explosion ($Var(\\hat{C}_{42}) \\propto \\sigma_w^4 / L$):**
       - Fourth-order cumulant $C_{42} = M_{42} - |M_{20}|^2 - 2 M_{21}^2$ relies on sample average of $|r[n]|^4$.
       - Under AWGN $r[n] = s[n] + w[n]$, expanding the 4th moment yields noise cross-terms ($2|s|^2|w|^2, 4\\text{Re}(sw^*)^2, |w|^4$).
       - In finite sample windows ($L = 128$), the variance of the estimator blows up at negative SNRs ($\text{SNR} \\le -2\\text{ dB}$), drowning the true signal cumulants in estimation noise.
    
    2. **Coherent Convolutional Bandpass Filtering:**
       - The 1D-CNN architecture applies learned convolutional kernels (length 7, 5, 3) across the temporal trajectory.
       - These kernels function as adaptive matched FIR filters, averaging out non-coherent white Gaussian noise across samples before non-linear pooling and dense classification.
       - Consequently, 1D-CNN retains robust classification down to $-4\\text{ dB}$, outperforming handcrafted statistical cumulants by $15\\text{--}30\\%$ in accuracy.
       
    3. **16QAM vs. 64QAM Lattice Ambiguity:**
       - Theoretical $C_{42}(\\text{16QAM}) = -0.6800$ and $C_{42}(\\text{64QAM}) = -0.6190$ differ by only **$0.061$**.
       - Handcrafted cumulant features struggle to resolve this slight margin under noise, whereas deep networks learn subtle constellation density transitions and multi-amplitude dynamics.
    """)
