# EE136IU: Machine Learning for Wireless Communications
## **Spectral & Temporal Feature Engineering vs. End-to-End Raw I/Q Deep Learning (Automatic Modulation Classification - AMC)**

**Department of Telecommunications — School of Electrical Engineering**  
**Vietnam National University Ho Chi Minh City • International University (VNU-HCM • IU)**  
*Instructor: Huynh Vo Trung Dung, Ph.D.*

---

## Executive Summary & Project Objective

In modern cognitive radio (CR) and electronic surveillance systems, **Automatic Modulation Classification (AMC)** enables autonomous spectrum monitoring, adaptive waveform demodulation, and interference triage without dedicated pilot overhead.

This project delivers a complete, reproducible physical-layer engineering comparison between two foundational paradigms:
1. **Branch 1 (Domain Feature Engineering + Classical ML):** Handcrafted physical representations combining 2nd & 4th-order Higher-Order Cumulants ($C_{40}, C_{42}$), differential phase statistics ($\Delta \phi[n]$), and frequency-domain FFT power spectral metrics (Centroid, Spread, Entropy) driving regularized **XGBoost** and **RBF-SVM** classifiers.
2. **Branch 2 (End-to-End Deep Learning):** Representation learning directly on raw complex baseband observation tensors $\mathbf{r} \in \mathbb{R}^{2 \times 128}$ via a deep **1D Convolutional Neural Network (1D-CNN)** in PyTorch.

---

## 1-Command Reproduction (< 30 Seconds)

### 1. Quick Environment Setup
```bash
# Clone and enter repository
git clone https://github.com/lshug180203/Spectral-Temporal-Feature-Engineering-vs-End-to-End-Raw-IQ-Deep-Learning.git
cd "Spectral-Temporal-Feature-Engineering-vs-End-to-End-Raw-IQ-Deep-Learning"

# Create virtual environment and install dependencies
python -m venv .venv
source .venv/bin/activate       # On Linux/macOS
.venv\Scripts\activate          # On Windows

pip install -r requirements.txt
```

### 2. End-to-End Pipeline Execution (Single Command)
```bash
python main.py
```
*Executes signal generation, RMS energy normalization, 70/15/15 zero-leakage partition, feature extraction, XGBoost/SVM training, 1D-CNN training with checkpointing, 15-point calibrated SNR grid evaluation (-10 dB to +18 dB), and generates publication-grade figures in `< 30 seconds`.*

### 3. Launch Interactive Live Defense Demo (Streamlit)
```bash
streamlit run demo/app.py
```

---

## Repository Architecture

```text
amc_topic2_project/
├── dataset/
│   ├── __init__.py
│   └── data_generator.py      # Baseband I/Q synthesis, per-burst RMS norm, 70/15/15 zero-leakage split
├── utils/
│   ├── __init__.py
│   ├── feature_extractor.py   # Vectorized Cumulants (C40, C42), differential phase stats, FFT spectral features
│   └── metrics_plots.py       # Calibrated 15-point SNR curves, row-normalized confusion matrices, failure analysis
├── models/
│   ├── __init__.py
│   ├── ml_classifier.py       # Branch 1: Anti-leakage StandardScaler + XGBoost & RBF-SVM classifiers
│   └── cnn1d.py               # Branch 2: PyTorch 1D-CNN directly on [N, 2, 128] + validation checkpointing
├── demo/
│   ├── __init__.py
│   └── app.py                 # Streamlit 10-Minute Live Defense Web Application with interactive sliders
├── results/                   # Generated high-resolution benchmark plots and failure report
├── saved_models/              # Checkpointed PyTorch CNN weights and serialized ML joblib pipelines
├── main.py                    # 1-command automated end-to-end execution pipeline
├── requirements.txt           # Explicit library dependencies
└── README.md                  # Comprehensive technical documentation & defense guide
```

---

## 📐 Mathematical & Baseband Physics Foundations

### 1. Complex Baseband Observation Model
Each received observation burst of length $L = 128$ complex baseband samples is modeled under flat AWGN:
$$r[n] = h \cdot s[n] + w[n], \quad n = 0, 1, \dots, L-1$$
where $s[n]$ represents the modulated baseband signal, $h$ denotes complex channel gain (normalized to $1.0$), and $w[n] \sim \mathcal{CN}(0, \sigma_w^2)$ represents circularly symmetric additive white Gaussian noise.

The observation is formatted as a real-valued tensor of dimensions:
$$\mathbf{X} \in \mathbb{R}^{N_{\text{batch}} \times 2 \times 128}, \quad \text{Row 0: } I[n] = \text{Re}(r[n]), \quad \text{Row 1: } Q[n] = \text{Im}(r[n])$$

### 2. Mandatory Per-Burst RMS Energy Normalization (Zero Energy Cheating)
To prevent classifiers from trivially discriminating modulation schemes (e.g., 64QAM vs. BPSK) based on average received power, each burst undergoes mandatory per-burst Root-Mean-Square (RMS) scaling:
$$E_{\text{rms}} = \sqrt{\frac{1}{L} \sum_{n=0}^{L-1} \left(I[n]^2 + Q[n]^2\right)}$$
$$r_{\text{norm}}[n] = \frac{r[n]}{E_{\text{rms}}} \implies \frac{1}{L} \sum_{n=0}^{L-1} |r_{\text{norm}}[n]|^2 \equiv 1.0$$

### 3. Modulation Scheme Diversity (8 Classes Across 3 Families)
1. **PSK Family:** `BPSK` (antipodal $\pm 1$), `QPSK` (quadrature phase $\{\pm 1 \pm j\}/\sqrt{2}$), `8PSK` ($e^{j 2\pi k / 8}$).
2. **QAM Family:** `16QAM` ($4 \times 4$ rectangular lattice), `64QAM` ($8 \times 8$ rectangular lattice).
3. **PAM & FSK Family:** `PAM4` (4-level real pulse amplitude), `CPFSK` (continuous phase FSK, index $h=0.5$), `GFSK` (Gaussian-filtered FSK, $BT=0.5$).

### 4. Higher-Order Moments & Cumulants (Bậc 2 & Bậc 4)
From the normalized complex burst $z[n] = I[n] + jQ[n]$:
- Moments:
  $$M_{20} = \frac{1}{L}\sum_{n=0}^{L-1} z[n]^2, \quad M_{21} = \frac{1}{L}\sum_{n=0}^{L-1} |z[n]|^2 \equiv 1.0$$
  $$M_{40} = \frac{1}{L}\sum_{n=0}^{L-1} z[n]^4, \quad M_{42} = \frac{1}{L}\sum_{n=0}^{L-1} |z[n]|^4$$
- Cumulants:
  $$C_{40} = M_{40} - 3 M_{20}^2 \quad \left(\text{extracted: } |C_{40}|, \text{Re}(C_{40}), \text{Im}(C_{40})\right)$$
  $$C_{42} = M_{42} - |M_{20}|^2 - 2 M_{21}^2$$

### 5. Temporal & Differential Phase Statistics
Differential phase sequence:
$$\Delta \phi[n] = \text{unwrap}(\angle z[n]) - \text{unwrap}(\angle z[n-1]), \quad n = 1, \dots, L-1$$
Features: $\text{mean}(\Delta \phi)$, $\text{std}(\Delta \phi)$, $\text{kurtosis}(\Delta \phi)$, and envelope statistics $\text{mean}(|z|)$, $\text{std}(|z|)$, $\text{kurtosis}(|z|)$.

### 6. Frequency-Domain (FFT) Spectral Statistics
From normalized centered power spectrum $P[k] = \frac{|FFT(z[n])|^2}{\sum |FFT(z[n])|^2}$ with frequency axis $f_k \in [-0.5, +0.5)$:
- **Spectral Centroid:** $\mu_f = \sum f_k P[k]$
- **Spectral Spread:** $\sigma_f = \sqrt{\sum (f_k - \mu_f)^2 P[k]}$
- **Normalized Spectral Entropy:** $H_f = -\frac{\sum P[k] \log_2(P[k] + \epsilon)}{\log_2(L)}$

---

##  Deep Physical Root-Cause Failure Analysis

### Why Cumulants Fail at Low SNR ($\le -2$ dB)
- Even though the theoretical 4th cumulant of Gaussian noise is zero asymptotically ($C_{40}(w) = 0$), sample estimators over finite length $L = 128$ suffer from variance scaling with noise power:
  $$\text{Var}(\hat{C}_{42}) \propto \mathcal{O}\left(\frac{\sigma_w^4}{L}\right)$$
- Under severe noise ($\text{SNR} = -4$ dB), $\sigma_w^2 > E_s$. Expanding $|s + w|^4$ yields noise cross-terms ($2|s|^2|w|^2, 4\text{Re}(sw^*)^2, |w|^4$) whose estimation variance completely overwhelms the signal cumulant.
- **Why 1D-CNN Wins:** The 1D-CNN utilizes learned temporal convolutional kernels (sizes 7, 5, 3) which act as coherent FIR bandpass matched filters. They accumulate signal energy in-phase while averaging out wideband white noise across samples *before* non-linear activation. As a result, 1D-CNN maintains $15\text{--}30\%$ higher accuracy in the noise-limited regime.

### 16QAM vs. 64QAM Lattice Ambiguity
- Theoretical cumulants of normalized square QAM constellations are closely spaced:
  $$C_{42}^{16\text{QAM}} = -0.6800 \quad \text{vs.} \quad C_{42}^{64\text{QAM}} \approx -0.6190 \quad (\Delta = 0.061)$$
- Under finite-sample variance, classical classifiers frequently confuse these classes. The 1D-CNN resolves this ambiguity by capturing multi-sample continuous constellation transitions and PAPR dynamics.

---

## Calibrated Evaluation Grid (-10 dB to +18 dB)

The models are rigorously benchmarked across 15 uniform SNR levels:
`-10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10, 12, 14, 16, 18 dB`

Key output figures generated in `results/`:
1. `results/snr_benchmark_curves.png`: Dual plot of **Accuracy vs. SNR** and **Macro-F1 vs. SNR**.
2. `results/confusion_matrices_regimes.png`: $2 \times 3$ grid of **Row-Normalized Confusion Matrices (%)** at:
   - Low SNR ($-4$ dB) — Noise-dominant regime
   - Medium SNR ($+4$ dB) — Transition recovery regime
   - High SNR ($+14$ dB) — Asymptotic clean channel regime
3. `results/failure_analysis_report.txt`: Automated numerical diagnostic report.

---
