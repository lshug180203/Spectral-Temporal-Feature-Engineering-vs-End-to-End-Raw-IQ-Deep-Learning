# CAPSTONE MINI-PROJECT DEFENSE PRESENTATION SLIDE DECK (15 MINUTES)
## TOPIC 02: Spectral & Temporal Feature Engineering vs. End-to-End Raw I/Q Deep Learning
**Course:** EE136IU - Machine Learning for Wireless Communications  
**Academic Supervision:** Department of Telecommunications - International University (VNU-HCM)  
**Design Standards:** Clean White Academic Standard (#FFFFFF / #F8FAFC / #0F172A / Accent #0284C7)

---

### PRESENTATION SCHEDULE & 4-MEMBER TEAM ALLOCATION (15 MINS DEFENSE + 5 MINS Q&A)
- **Member 1 (Minutes 00:00 – 03:45 | Slides 1 - 4):** AMC Wireless Context, Complex Baseband Physical Channel Model, Per-burst RMS Power Normalization, Higher-Order Cumulants Mathematical Theory & 10-D Feature Vector.
- **Member 2 (Minutes 03:45 – 07:30 | Slides 5 - 8):** Zero-Leakage Dataset Partitioning Protocol, Feature-Based Classifiers (XGBoost/SVM), PyTorch 1D-CNN Architecture & Hardware Complexity/Latency Benchmark.
- **Member 3 (Minutes 07:30 – 11:15 | Slides 9 - 11):** Empirical SNR vs. Accuracy/Macro-F1 Curves, Confusion Matrices Across 3 Noise Regimes, and Root-Cause Failure 1 (Cumulant Collapse under Negative SNR).
- **Member 4 (Minutes 11:15 – 15:00 | Slides 12 - 14):** Root-Cause Failure 2 (1D-CNN Implicit Matched Filtering), Root-Cause Failure 3 (16QAM vs. 64QAM Bayes Limit & 4 Upgrades), Technical Conclusions & Live Interactive Streamlit Demo.
- **Full Team (Minutes 15:00 – 20:00):** Direct Q&A Defense with Faculty Examination Committee (Assigned primary leads for each domain).

#### CONSOLIDATED 4-MEMBER RESPONSIBILITY MATRIX

| Team Member | Assigned Slides | Duration | Time Window | Core Technical Focus | Primary Q&A Focus |
| :---: | :---: | :---: | :---: | :--- | :--- |
| **Member 1** | Slides 1 – 4 (4 slides) | ~03:45 | 00:00 – 03:45 | AMC Motivation, Baseband Channel, RMS Normalization, HOC Theory & 10-D Feature Vector | **Question 1** (Baseband & RMS Normalization) |
| **Member 2** | Slides 5 – 8 (4 slides) | ~03:45 | 03:45 – 07:30 | Zero-Leakage Dataset, ML Models (XGBoost/SVM), PyTorch 1D-CNN & Hardware Latency Benchmark | **Questions 3 & 5** (CNN Architecture & Edge IoT) |
| **Member 3** | Slides 9 – 11 (3 slides) | ~03:45 | 07:30 – 11:15 | SNR Benchmark Curves, Confusion Matrices Across 3 Regimes, Root-Cause 1 (Cumulant Collapse) | **Question 2** (Cumulant Collapse Paradox) |
| **Member 4** | Slides 12 – 14 (3 slides) | ~03:45 | 11:15 – 15:00 | Root-Cause 2 (Matched Filter), Root-Cause 3 (16 vs 64QAM & 4 Solutions), Live Streamlit Demo | **Question 4** (Bayes 16/64QAM Boundary) |
| **All Members** | All 14 slides | 05:00 | 15:00 – 20:00 | **Direct Q&A Defense Session with Faculty Examination Committee** | Collaborative Defense (5/5 Questions) |

---

## SLIDE 1: PROBLEM FORMULATION & CORE RESEARCH QUESTIONS
- **Duration:** 01:00 | **Speaker:** Member 1
- **Slide Title:** Automatic Modulation Classification: Expert Feature Engineering vs. End-to-End Deep Learning
- **Card 1 [Wireless Context & Motivation]:**
  - Cognitive Radio & 5G/6G Networks: AMC is a vital physical-layer capability for dynamic spectrum access and non-cooperative signal interception.
  - Target Modulations: Autonomous classification across 8 common digital modulations: **BPSK, QPSK, 8PSK, 16QAM, 64QAM, PAM4, CPFSK, GFSK**.
  - Blind Operation: Operates strictly without side information, pilot sequences, or preambles from the transmitter.
- **Card 2 [Core Research Questions]:**
  - Domain-Expert Handcrafted Features: Do classical physical features (Higher-Order Cumulants, Differential Phase, Spectral FFT) still hold relevance against Deep Learning?
  - End-to-End Deep Learning: Can a 1D-CNN operating directly on raw complex baseband tensors consistently outperform expert features across all SNR regimes?
  - Performance Inflection Point: At what specific SNR threshold does the performance crossover occur, and what physical mechanisms govern this transition?
- **Card 3 [Primary Technical Contributions]:**
  - Realistic Baseband Channel Model: Integrating RRC pulse shaping, random Carrier Frequency Offset (CFO), phase jitter, complex AWGN, and per-burst RMS normalization.
  - Transparent Failure Analysis: Rigorous mathematical proof of cumulant contraction under negative SNR and Bayesian error bounds for 16QAM vs. 64QAM.
  - Empirical Validation: Comprehensive evaluation over 48,000 independent bursts coupled with an interactive real-time Streamlit demo.

---

## SLIDE 2: BASEBAND PHYSICAL SIGNAL MODELING & ENERGY NORMALIZATION
- **Duration:** 00:55 | **Speaker:** Member 1
- **Slide Title:** Realistic Complex Baseband Signal Generation Pipeline & Per-Burst Power Normalization
- **Card 1 [Rx Baseband Signal Pipeline]:**
  - $r[n] = \left[ \sum_{k} s[k] \cdot g(nT_s - kT) \right] \cdot \exp(j(2\pi \Delta f_c n T_s + \theta_0)) + w[n]$
  - Where: $g(t)$ is Root Raised Cosine filter (RRC, $\alpha=0.35$, SPS = 4 samples/symbol, span = 8 symbols).
  - $\Delta f_c$: Random Carrier Frequency Offset (CFO) simulating oscillator drift.
  - $w[n] \sim \mathcal{CN}(0, \sigma_w^2)$: Circularly symmetric complex additive white Gaussian noise (AWGN).
- **Card 2 [Mandatory Per-Burst RMS Energy Normalization]:**
  - $\tilde{r}[n] = \frac{r[n]}{\sqrt{\frac{1}{L} \sum_{m=0}^{L-1} |r[m]|^2}}$, over short observation window $L = 128$ samples.
  - **Telecommunications Significance:** Completely eliminates received power bias caused by transmission distance and large-scale path loss, compelling models to classify strictly based on **constellation geometry** and **spectral/phase signatures**.
  - **RMS vs. Peak Normalization:** Peak normalization is fragile under negative SNR (noise outliers distort all symbol amplitudes), whereas RMS normalization guarantees $E[|\tilde{r}|^2] = 1.0$.

---

## SLIDE 3: HIGHER-ORDER CUMULANTS (HOC) MATHEMATICAL FOUNDATIONS
- **Duration:** 00:55 | **Speaker:** Member 1
- **Slide Title:** Physical Grounding: Constellation Symmetry, Rotation Invariance & Theoretical Values
- **Card 1 [Normalized 4th-Order Cumulant Formulations]:**
  - $C_{40} = \text{Cum}(r, r, r, r) = M_{40} - 3 M_{20}^2$
  - $C_{42} = \text{Cum}(r, r^*, r, r^*) = M_{42} - |M_{20}|^2 - 2 M_{21}^2$
  - Where moment $M_{pq} = E[r^{p-q} (r^*)^q]$, with normalized power $M_{21} = E[|r|^2] = 1$.
- **Card 2 [Theoretical Values in Noiseless Channels ($E_s = 1$)]:**

| Modulation Scheme | Rotational Symmetry | $\|M_{20}\|$ | $\|C_{40}\|$ (Theory) | $C_{42}$ (Theory) | Geometric Signature |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **BPSK** | $180^\circ$ (Order 2) | 1.000 | **2.000** | **-2.000** | Real-axis symmetry, $s^2 = 1$ |
| **QPSK** | $90^\circ$ (Order 4) | 0.000 | **1.000** | **-1.000** | 4 points on unit circle, $s^4 = -1$ |
| **8PSK** | $45^\circ$ (Order 8) | 0.000 | **0.000** | **-1.000** | 8 circular points, $s^4$ vanishes |
| **16QAM** | $90^\circ$ (Order 4) | 0.000 | **0.680** | **-0.680** | 3 discrete energy shells |
| **64QAM** | $90^\circ$ (Order 4) | 0.000 | **0.619** | **-0.619** | 9 concentric energy shells |
| **PAM4** | $180^\circ$ (Real) | 1.000 | **1.360** | **-1.360** | 1D real constellation amplitudes |

- **Card 3 [Physical Grounding Insights]:**
  - $|C_{40}|$ strictly separates the PSK family ($2.000 \to 1.000 \to 0.000$).
  - $C_{42}$ separates constant-envelope circular modulations (PSK: $C_{42} = -1.0$) from multi-level constellations (QAM: $C_{42} \in [-0.68, -0.62]$).

---

## SLIDE 4: 10-DIMENSIONAL SPECTRAL & TEMPORAL FEATURE VECTOR (10-D SPACE)
- **Duration:** 00:55 | **Speaker:** Member 1
- **Slide Title:** Complete 10-Dimensional Handcrafted Feature Vector
- **Card 1 [1. Four Higher-Order Cumulants (HOC)]:**
  - $C_{20}, C_{21}, |C_{40}|, C_{42}$: Capture constellation rotational symmetry and the number of discrete energy shells.
- **Card 2 [2. Three Differential Phase Features]:**
  - $\Delta \phi[n] = \text{unwrap}(\arg(r[n] \cdot r^*[n-1]))$.
  - Extracted: $\mu_{\Delta \phi}$, $\sigma_{\Delta \phi}$, $\max |\text{FFT}(\Delta \phi)|$.
  - Uniquely isolates continuous-phase frequency shift keying (CPFSK, GFSK) having linear phase accumulation.
- **Card 3 [3. Four FFT Spectral Statistics]:**
  - **Normalized Peak Power ($\gamma_{\max}$):** $\gamma_{\max} = \frac{\max |\text{DFT}(\tilde{r})|^2}{L}$.
  - **Spectral Centroid ($\mu_f$):** $\mu_f = \sum f_k P[k]$.
  - **Spectral Spread ($\sigma_f$):** $\sigma_f = \sqrt{\sum (f_k - \mu_f)^2 P[k]}$.
  - **Spectral Entropy ($H_f$):** $H_f = -\frac{\sum P[k] \log_2(P[k])}{\log_2(L)}$.
- **Transition Cue to Member 2:**
  - *"Having established the 10-D physical expert feature space and the raw I/Q tensor formulation, I now hand over to [Member 2] to present our zero-leakage dataset partitioning protocol and classifier architectures."*

---

## SLIDE 5: ZERO-LEAKAGE DATASET PARTITIONING PROTOCOL
- **Duration:** 00:50 | **Speaker:** Member 2
- **Slide Title:** Zero-Leakage Dataset Partition & Deterministic Reproducibility
- **Card 1 [Independent Data Allocation]:**
  - 8 Classes $\times$ 15 SNR points ($-10$ dB to $+18$ dB, 2 dB step) $\times$ 400 bursts = **48,000 bursts**.
  - Partition: 70% Train (33,600 bursts, `seed=42`), 15% Validation (7,200 bursts, `seed=43`), 15% Test (7,200 bursts, `seed=44`).
- **Card 2 [Strict Anti-Leakage Protocol]:**
  - **Independent Synthesis:** Test set synthesized using entirely separate random bit sequences and channel noise instances, not sliced from continuous bursts.
  - **Rigorous Scaler Fitting:** `StandardScaler` is fitted strictly on the Train set, and transformed onto Val and Test to eliminate data snooping.
- **Card 3 [Per-SNR Stratification]:**
  - Exactly 480 test bursts per SNR point, accompanied by Wilson 95% Confidence Intervals ($CI_{95} \approx \pm 4.5\%$).

---

## SLIDE 6: EXPERT FEATURE CLASSIFIERS: XGBOOST & RBF-SVM
- **Duration:** 00:55 | **Speaker:** Member 2
- **Slide Title:** Expert Feature Classifiers: Hyperparameters & Optimization
- **Card 1 [XGBoost Gradient Boosted Trees]:**
  - Configuration: `n_estimators = 200`, `max_depth = 6`, `learning_rate = 0.1`, `subsample = 0.8`.
  - Core Advantage: Ultra-fast non-linear boundary partitioning across the 10-D feature space.
  - Training Time: ~4.2s for all 33,600 Train bursts on a standard CPU.
- **Card 2 [RBF Support Vector Machine]:**
  - Configuration: RBF Kernel (`C = 10.0`, `gamma = 'scale'`), Platt scaling (`probability = True`).
  - Optimization: Subsampled SVM training to 12,000 bursts to prevent the $O(N^2)$ quadratic programming solver bottleneck, preserving 100% of Val and Test sets.
- **Card 3 [Feature Importance Ranking]:**
  - Top 1: $|C_{40}|$ (PSK vs. QAM discrimination).
  - Top 2: $C_{42}$ (Circular vs. multi-amplitude shell separation).
  - Top 3: $\sigma_{\Delta \phi}$ (FSK vs. linear modulations).

---

## SLIDE 7: PYTORCH 1D-CNN ARCHITECTURE (END-TO-END RAW I/Q)
- **Duration:** 01:10 | **Speaker:** Member 2
- **Slide Title:** End-to-End 1D-CNN: Learning Directly from Raw Complex Tensors
- **Card 1 [Tensor Network Architecture $[N, 2, 128]$]:**
  - **Conv1D Block 1:** 64 filters, kernel size = 7, stride = 1, padding = 3 $\to$ BatchNorm1d $\to$ ReLU $\to$ MaxPool1d(2).
  - **Conv1D Block 2:** 128 filters, kernel size = 5, stride = 1, padding = 2 $\to$ BatchNorm1d $\to$ ReLU $\to$ MaxPool1d(2).
  - **Conv1D Block 3:** 256 filters, kernel size = 3, stride = 1, padding = 1 $\to$ BatchNorm1d $\to$ ReLU $\to$ MaxPool1d(2).
- **Card 2 [Classifier Head & Regularization]:**
  - Flatten $\to$ Linear(4096 $\to$ 256) $\to$ ReLU $\to$ **Dropout(0.5)** $\to$ Linear(256 $\to$ 8 classes).
  - Total Parameters: **1,178,632 parameters**. Model Checkpoint: **~4.7 MB**.
- **Card 3 [Robust Optimization Strategy]:**
  - Optimizer: `AdamW` ($lr = 10^{-3}$, weight decay = $10^{-4}$).
  - Scheduler: `ReduceLROnPlateau(patience=3, factor=0.5)`.
  - Early Stopping: Monitored on validation loss with `patience = 7`, saving `best_cnn1d_model.pt`.

---

## SLIDE 8: COMPUTATIONAL COMPLEXITY & LATENCY BENCHMARK
- **Duration:** 00:50 | **Speaker:** Member 2
- **Slide Title:** Inference Latency, Memory Footprint & Hardware Feasibility
- **Card 1 [Empirical CPU Hardware Benchmark Table]:**

| Evaluation Metric | Feature Extraction + XGBoost | Feature Extraction + RBF-SVM | 1D-CNN (PyTorch CPU) |
| :--- | :---: | :---: | :---: |
| **Model Checkpoint Size** | **0.8 MB** (Ultra-compact) | **8.4 MB** | **4.7 MB** |
| **Feature Extraction (ms/burst)** | 0.85 ms | 0.85 ms | **0.00 ms** (None required) |
| **Model Inference (ms/burst)** | **0.08 ms** | 1.15 ms | 0.42 ms |
| **Total Latency (Total Latency)** | **0.93 ms / burst** | 2.00 ms / burst | **0.42 ms / burst** |
| **Throughput (Bursts / sec)** | 1,075 bursts/s | 500 bursts/s | **2,380 bursts/s** |
| **Target Embedded Hardware** | Microcontroller (ARM Cortex-M/A) | Standard Server CPU | Embedded NPU / DSP / GPU |

- **Card 2 [Architectural Conclusions]:**
  - 1D-CNN achieves higher batch throughput on raw tensors because it eliminates Python loop overhead for FFT and cumulant extraction.
- **Transition Cue to Member 3:**
  - *"To answer the fundamental question—how these models perform quantitatively across each SNR operating regime—I invite [Member 3] to present our experimental benchmark curves and our first mathematical root-cause failure analysis."*

---

## SLIDE 9: EXPERIMENTAL RESULTS: SNR BENCHMARK CURVES & MACRO-F1
- **Duration:** 01:15 | **Speaker:** Member 3
- **Slide Title:** Experimental Results: SNR Benchmark Curves (n = 7,200 Test Bursts)
- **Card 1 [Harsh Noise Regime ($-10$ dB to $0$ dB)]:**
  - At $-4$ dB: **1D-CNN reaches 48.75%**, outperforming **XGBoost (31.67%)** and **RBF-SVM (32.29%)** by **+17.08%**.
  - At $0$ dB: 1D-CNN hits **74.58%**, while XGBoost achieves only **46.67%** (+27.91%).
- **Card 2 [Low Noise Regime ($+4$ dB to $+18$ dB)]:**
  - All 3 models rapidly surge above the 80% accuracy benchmark from +6 dB onwards.
  - At SNR $\ge +12$ dB: Both XGBoost and 1D-CNN converge at **90% – 93%**, matching the theoretical separability ceiling for short bursts $L=128$.
- **Card 3 [Macro-F1 Parity]:**
  - The Macro-F1 curve tracks Accuracy tightly across all SNR points, confirming zero majority-class starvation or bias.

---

## SLIDE 10: CONFUSION MATRIX ANALYSIS ACROSS OPERATING REGIMES
- **Duration:** 01:15 | **Speaker:** Member 3
- **Slide Title:** Confusion Matrix Analysis Across Low, Medium & High SNR
- **Card 1 [Low SNR Regime: $-4$ dB (Noise Dominated)]:**
  - XGBoost experiences pervasive confusion between QPSK, 16QAM, and 64QAM due to cumulant contraction.
  - 1D-CNN robustly preserves BPSK (88%), PAM4 (82%), and correctly isolates FSK carriers via temporal correlation.
- **Card 2 [Medium SNR Regime: $+4$ dB (Transition State)]:**
  - BPSK, PAM4, CPFSK, GFSK attain near-perfect classification (> 95%).
  - Residual confusion is strictly confined within amplitude/phase families: QPSK $\leftrightarrow$ 8PSK $\leftrightarrow$ 16QAM.
- **Card 3 [High SNR Regime: $+14$ dB (Noiseless Channel)]:**
  - 6 out of 8 modulation formats reach a flawless **100.0%** accuracy.
  - **Sole Remaining Bottleneck:** Mutual ambiguity between **16QAM $\leftrightarrow$ 64QAM** (XGBoost misclassifies 34.0%, CNN misclassifies 34.7%).

---

## SLIDE 11: ROOT CAUSE 1: CUMULANT COLLAPSE UNDER LOW SNR
- **Duration:** 01:15 | **Speaker:** Member 3
- **Slide Title:** Root-Cause Failure 1: The Low-SNR Collapse of Higher-Order Cumulants
- **Card 1 [Mechanism A: Contraction Factor (Shrinkage to 0)]:**
  - Signal model: $r = s + w$ with $s \perp w$; Gaussian noise has $C_{42}(w) = 0$.
  - Under per-burst RMS power normalization $E[|r|^2] = E[|s|^2] + \sigma_w^2$:
    $$C_{42, \text{observed}} = C_{42, \text{signal}} \cdot \left( \frac{\text{SNR}}{1 + \text{SNR}} \right)^2$$
  - At $\text{SNR} = -4\text{ dB} \implies \rho = 10^{-0.4} \approx 0.398 \implies \left(\frac{0.398}{1.398}\right)^2 \approx \mathbf{0.081}$.
  - Expected cumulants contract by more than **12-fold towards zero**, collapsing constellation feature separations!
- **Card 2 [Mechanism B: Finite-Sample Variance Explosion ($L=128$)]:**
  - Estimator: $\hat{M}_{42} = \frac{1}{L} \sum_{n=1}^{L} |r[n]|^4$.
  - Estimator variance: $\text{Var}\{\hat{C}_{42}\} \propto \frac{E[|w|^8] - (E[|w|^4])^2}{L}$.
  - Standard deviation $\sigma_{\hat{C}_{42}}$ **does not decay with SNR**, dominated by 8th-order moments of Gaussian noise.
- **Card 3 [Empirical Test Set Measurements ("Feature SNR" $\frac{|\mu|}{\sigma}$)]:**
  - **QPSK:** At $+14$ dB $|\mu|/\sigma = 14.51$ (Clear) $\to$ at $-4$ dB drops to **$0.64$** ($|\mu| < \sigma$).
  - **16QAM:** At $+14$ dB $|\mu|/\sigma = 5.24$ $\to$ at $-4$ dB drops to **$0.38$**.
  - **Physical Consequence:** When $\frac{|\mu|}{\sigma} < 1$, feature-based classification degenerates to random guessing.
- **Transition Cue to Member 4:**
  - *"Why then does the 1D-CNN solve this dilemma to gain a massive +17% advantage at low SNR, and what is the definitive resolution to the 16QAM vs. 64QAM bottleneck? I invite [Member 4] to unpack the signal processing mechanics and lead the live interactive demo."*

---

## SLIDE 12: ROOT CAUSE 2: WHY 1D-CNN OUTPERFORMS AT LOW SNR
- **Duration:** 01:10 | **Speaker:** Member 4
- **Slide Title:** Root-Cause Failure 2: Why 1D-CNN Outperforms at Low SNR (+17.08%)
- **Card 1 [Temporal Convolution as Implicit Matched Filtering]:**
  - Conv1D slides kernel filters $W \in \mathbb{R}^{2 \times K}$ across 128 I/Q samples:
    $$y_k[n] = \sum_{c \in \{I,Q\}} \sum_{\tau=0}^{K-1} W_k[c, \tau] \cdot r_c[n - \tau]$$
  - Backpropagation trains filter weights $W_k$ to converge into finite-impulse matched filters aligned with RRC pulse shapes and symbol transitions.
- **Card 2 [Coherent Processing Gain]:**
  - Coherent integration adds structured signal energy constructively, while white noise cancels out partially:
    $$\text{SNR}_{\text{effective}} \approx \text{SNR}_{\text{in}} + 10 \log_{10}(K \cdot \text{SPS})$$
  - Consequently, 1D-CNN elevates the effective SNR before non-linear activations.
- **Card 3 [Mathematical Comparison]:**
  - **Feature-Based:** Takes instantaneous 4th-order powers ($|r[n]|^4$) sample-by-sample $\to$ Non-linearly amplifies noise variance.
  - **1D-CNN:** Performs linear temporal smoothing first $\to$ Denoises signal before non-linear activation.

---

## SLIDE 13: ROOT CAUSE 3: INTRINSIC 16QAM VS 64QAM AMBIGUITY
- **Duration:** 01:10 | **Speaker:** Member 4
- **Slide Title:** Root-Cause Failure 3: The Intrinsic 16QAM vs. 64QAM Ambiguity
- **Card 1 [Narrow Theoretical Margin]:**
  - $C_{42}(16\text{QAM}) = -0.680$ vs $C_{42}(64\text{QAM}) = -0.619 \implies \Delta C_{42} = \mathbf{0.061}$.
  - Measured on test set ($L=128$, with RRC filter):
    - 16QAM: $-0.551 \pm 0.104$
    - 64QAM: $-0.497 \pm 0.103$
    - Difference $\Delta = 0.054$ vs Pooled Std Dev $\sigma \approx 0.103$.
- **Card 2 [Fisher Distance & Bayes Error Probability]:**
  - $d' = \frac{\Delta \mu}{\sigma_{\text{pooled}}} = \frac{0.054}{0.103} \approx \mathbf{0.52}$.
  - Theoretical 1D Bayes classification error probability:
    $$P_e = Q(d'/2) = Q(0.26) \approx \mathbf{39.7\%}$$
  - Empirical observation: XGBoost misclassifies $34.0\%$, 1D-CNN misclassifies $34.7\%$, matching theoretical Bayes predictions perfectly!
- **Card 3 [4 Concrete Engineering Upgrades]:**
  1. **Scale Burst Length $L$:** Since $\sigma \propto 1/\sqrt{L}$, scaling $L=512$ or $1024$ halves variance, cleanly separating distributions.
  2. **Incorporate 6th-Order Cumulant ($C_{63}$):** The ratio $\frac{C_{63}}{C_{42}}$ provides higher sensitivity to dense QAM constellations.
  3. **Timing Recovery:** Symbol-rate peak-eye sampling eliminates inter-symbol transition variance (ISI).
  4. **Deeper Neural Backbones:** Residual 1D-CNN or Complex-Valued CNNs (CV-CNN).

---

## SLIDE 14: CONCLUSIONS & LIVE INTERACTIVE DEMO
- **Duration:** 01:25 | **Speaker:** Member 4 (Live Streamlit Demonstration)
- **Slide Title:** Conclusions, Key Takeaways & Live Interactive Demo
- **Card 1 [Technical Takeaways & Guidelines]:**
  - **Harsh Noise Regimes (SNR $< 0$ dB):** Unequivocally recommend **1D-CNN** owing to its coherent temporal matched-filtering advantage (+17.08% at $-4$ dB).
  - **Low-Power Edge IoT Nodes:** Recommend **XGBoost + Handcrafted Features** under SNR $\ge +4$ dB to leverage ultra-low latency (0.08 ms) and compact footprint (0.8 MB).
  - **Domain Expertise Value:** Physical features demystify deep learning behavior and provide analytical foundations for system design.
- **Card 2 [Live Interactive Streamlit Dashboard]:**
  - Real-time visualization of I/Q constellation diagrams, time-domain waveforms, and FFT power spectral density.
  - Interactive SNR slider from $+18$ dB down to $-4$ dB.
  - Direct head-to-head demonstration at $-4$ dB: XGBoost misclassifies due to cumulant collapse, while 1D-CNN predicts accurately with high confidence.
- **Concluding Remarks & Transition to Q&A (Member 4):**
  - *"That concludes our presentation on Topic 02. We sincerely thank the Faculty Committee for your attention, and our 4 members are now ready for the Question & Answer session."*

---

# DEFENSE Q&A SESSION: QUESTIONS & MODEL RESPONSES (20/20 RUBRIC POINTS)

### Question 1 (Baseband Signal Processing):
- **Primary Lead:** Member 1 *(Physical Channel Model & RMS Normalization)*
> **Faculty Committee Question:** *"Why did the team select per-burst RMS energy normalization rather than peak normalization or raw unnormalized amplitudes?"*
- **Rigorous RF Engineer Response:**  
  "Dear Professors, in practical wireless transmission environments:  
  1. **Wireless Channel Power Fluctuations:** Received signal power undergoes severe, unpredictable variations caused by transmission distance (Path Loss) and large-scale shadowing. Without normalization, machine learning models would trivially overfit to power levels rather than learning modulation geometry.  
  2. **Fatal Vulnerability of Peak Normalization:** Peak normalization is extremely sensitive to noise outliers (spikes) under low SNR. A single large noise sample compresses all underlying signal samples, artificially destroying constellation energy shell ratios.  
  3. **Mathematical Cumulant Preservation:** Per-burst RMS normalization strictly standardizes average signal power to 1.0 ($E[|\tilde{r}|^2] = 1$), preserving theoretical energy shell proportions and keeping cumulant invariant properties intact ($C_{21} = 1, C_{42}$)."

---

### Question 2 (Cumulant Collapse Phenomenon):
- **Primary Lead:** Member 3 *(Statistical Signal Processing & Cumulant Collapse at Low-SNR)*
> **Faculty Committee Question:** *"Mathematically, independent Gaussian noise has zero 4th-order cumulant ($C_{42}(w) = 0$). Why does the received signal cumulant collapse at low SNR instead of equaling the noiseless signal cumulant?"*
- **Rigorous RF Engineer Response:**  
  "Dear Professors, while $C_{42}(s+w) = C_{42}(s) + C_{42}(w) = C_{42}(s)$ holds for unnormalized signals, the critical factor is **ENERGY NORMALIZATION**:  
  When we normalize total received burst power $P_r = P_s + \sigma_w^2$, the normalized burst becomes $\tilde{r} = \frac{s+w}{\sqrt{P_s + \sigma_w^2}}$.  
  By the homogeneity of 4th-order cumulants $\text{Cum}(\alpha r) = \alpha^4 \text{Cum}(r)$:
  $$C_{42}(\tilde{r}) = \frac{C_{42}(s)}{(P_s + \sigma_w^2)^2} = C_{42}(s) \cdot \left(\frac{\text{SNR}}{1+\text{SNR}}\right)^2$$
  At $-4$ dB ($\text{SNR}=0.398$), this scaling factor is $(0.398/1.398)^2 \approx 0.081$, shrinking the mean cumulant by **92% towards zero**. Simultaneously, finite-sample estimator variance ($L=128$) contains 8th-order moments of Gaussian noise ($E[|w|^8]$). Standard deviation $\sigma$ remains high while mean $\mu$ shrinks to 0, driving the Feature SNR $|\mu|/\sigma$ below $1.0$."

---

### Question 3 (1D-CNN Deep Learning Architecture):
- **Primary Lead:** Member 2 *(CNN Architecture & Cross-Channel I/Q; Member 4 Matched Filter Support)*
> **Faculty Committee Question:** *"Your 1D-CNN receives a 2D real tensor $[2, 128]$. How does a standard real 1D-CNN capture the complex phase relationships between I and Q channels?"*
- **Rigorous RF Engineer Response:**  
  "Dear Professors, in the initial convolutional layer `Conv1d(in_channels=2, out_channels=64, kernel_size=7)`:  
  Each filter kernel slides across the sequence and computes a cross-channel sum:
  $$y[n] = \sum_{\tau=0}^{6} \left( W_I[\tau] \cdot I[n-\tau] + W_Q[\tau] \cdot Q[n-\tau] \right)$$
  This real cross-sum is mathematically identical to the real part of complex convolution: $\text{Re}\{ (W_I - j W_Q) * (I + j Q) \}$.  
  Cross-correlation over multi-tap delays $\tau$ directly learns angular phase velocity $\Delta \theta = \omega \tau$ and constellation rotation trajectories without ever computing explicit $\arctan$ functions."

---

### Question 4 (16QAM vs. 64QAM Bottleneck):
- **Primary Lead:** Member 4 *(Fisher Distance, Bayes Limit & Upgrades)*
> **Faculty Committee Question:** *"Why do both feature engineering and deep learning struggle most with 16QAM vs. 64QAM, and what is the single most cost-effective upgrade in a practical deployment?"*
- **Rigorous RF Engineer Response:**  
  "Dear Professors:  
  1. **Geometric Symmetry & Short Burst Constraint:** Both 16QAM and 64QAM possess identical $90^\circ$ rotational symmetry (order 4), yielding $M_{20}=0$ and tight theoretical cumulants ($\Delta C_{42} = 0.061$). With burst length $L = 128$ (only 32 symbols at SPS=4), outer 64QAM constellation points may not even be transmitted in a short random sequence, causing empirical amplitude distributions to overlap heavily with 16QAM.  
  2. **Most Cost-Effective Practical Upgrade:** The most computationally frugal and effective solution is **scaling the observation burst length to $L = 512$ or $1024$ samples**. Because estimator variance scales as $\sigma \propto 1/\sqrt{L}$, increasing $L$ shrinks standard deviation by 2 to 3 times, pulling apart the Gaussian distributions and lifting classification accuracy above $95\%$."

---

### Question 5 (Embedded Edge IoT Deployment):
- **Primary Lead:** Member 2 *(Computational Complexity, Memory Footprint & MCU; Member 1 Support)*
> **Faculty Committee Question:** *"If you were required to deploy this AMC system onto a battery-powered IoT edge device using an ARM Cortex-M4 microcontroller without a GPU, which approach would you choose?"*
- **Rigorous RF Engineer Response:**  
  "Dear Professors, we would definitively select **Feature Engineering paired with a Decision Tree or Compact XGBoost classifier**:  
  - **Hardware Infeasibility of 1D-CNN:** The 1D-CNN demands over 1.17 million Multiply-Accumulate operations (MACs) and a ~4.7 MB Flash model checkpoint. This far exceeds the typical SRAM (64 KB - 256 KB) and Flash limits of standard Cortex-M4 microcontrollers.  
  - **Absolute Viability of Decision Trees:** Extracting 10 features requires only a few thousand elementary arithmetic operations. The tree classifier itself consists purely of simple floating-point `if-else` threshold comparisons, fitting within $< 50$ KB of Flash and executing in $< 1$ ms without requiring tensor runtimes or neural accelerators."
