"""
================================================================================
EE136IU: Machine Learning for Wireless Communications
Capstone Mini-Project - Topic 02: Spectral & Temporal Feature Engineering
                                 vs. End-to-End Raw I/Q Deep Learning (AMC)
================================================================================
Module: utils/feature_extractor.py
Description:
  Vectorized feature extraction module for complex baseband bursts z[n] = I[n] + jQ[n].
  Computes:
    1. Higher-Order Moments & Cumulants (2nd & 4th order: M20, M21, M40, M42, C40, C42).
    2. Temporal & Differential Phase Statistics (delta_phi mean, std, kurtosis, amp stats).
    3. Spectral Statistics from FFT (Power spectrum, Spectral Centroid, Spread, Entropy).
================================================================================
"""

import numpy as np
from scipy import stats
from typing import Union, Tuple, List, Dict

FEATURE_NAMES: List[str] = [
    'C40_abs',              # |C40|
    'C40_real',             # Re(C40)
    'C40_imag',             # Im(C40)
    'C42',                  # C42
    'M20_abs',              # |M20|
    'M20_real',             # Re(M20)
    'M20_imag',             # Im(M20)
    'M40_abs',              # |M40|
    'M42',                  # M42
    'phase_diff_mean',      # Mean of differential phase
    'phase_diff_std',       # Std of differential phase
    'phase_diff_kurtosis',  # Kurtosis of differential phase
    'amp_mean',             # Mean instantaneous envelope
    'amp_std',              # Std instantaneous envelope
    'amp_kurtosis',         # Kurtosis of envelope
    'spectral_centroid',    # Spectral Centroid (mu_f)
    'spectral_spread',      # Spectral Spread (sigma_f)
    'spectral_entropy',     # Normalized Spectral Entropy (H_f)
    'spectral_skewness',    # Spectral Skewness
    'spectral_kurtosis'     # Spectral Kurtosis
]


def extract_burst_features_1d(z: np.ndarray) -> np.ndarray:
    """
    Extracts the 20 physical domain features for a single 128-sample complex burst z[n].
    
    Args:
        z: 1D complex numpy array of length L (e.g. L=128).
        
    Returns:
        1D float32 array of shape [20] containing the extracted features.
    """
    L = len(z)
    eps = 1e-12

    # -------------------------------------------------------------------------
    # 1. Higher-Order Moments and Cumulants (2nd & 4th Order)
    # -------------------------------------------------------------------------
    # Moments
    M20 = np.mean(z**2)
    M21 = np.mean(np.abs(z)**2)
    M40 = np.mean(z**4)
    M42 = np.mean(np.abs(z)**4)

    # Cumulants
    # C40 = M40 - 3 * M20^2
    C40 = M40 - 3.0 * (M20**2)
    # C42 = M42 - |M20|^2 - 2 * M21^2
    C42 = np.real(M42 - (np.abs(M20)**2) - 2.0 * (M21**2))

    C40_abs = np.abs(C40)
    C40_real = np.real(C40)
    C40_imag = np.imag(C40)

    M20_abs = np.abs(M20)
    M20_real = np.real(M20)
    M20_imag = np.imag(M20)
    M40_abs = np.abs(M40)
    M42_val = np.real(M42)

    # -------------------------------------------------------------------------
    # 2. Temporal & Differential Phase Statistics
    # -------------------------------------------------------------------------
    # Instantaneous phase unwrapped
    inst_phase = np.unwrap(np.angle(z))
    # Differential phase: delta_phi[n] = unwrap(angle(z[n])) - unwrap(angle(z[n-1]))
    delta_phi = np.diff(inst_phase)

    dphi_mean = float(np.mean(delta_phi))
    dphi_std = float(np.std(delta_phi))
    dphi_kurt = float(stats.kurtosis(delta_phi)) if len(delta_phi) > 3 else 0.0

    # Instantaneous envelope A[n] = |z[n]|
    amp = np.abs(z)
    amp_mean = float(np.mean(amp))
    amp_std = float(np.std(amp))
    amp_kurt = float(stats.kurtosis(amp)) if len(amp) > 3 else 0.0

    # -------------------------------------------------------------------------
    # 3. Frequency-Domain (FFT) Spectral Statistics
    # -------------------------------------------------------------------------
    # Centered FFT
    Z_fft = np.fft.fftshift(np.fft.fft(z))
    power_spectrum = np.abs(Z_fft)**2
    total_power = np.sum(power_spectrum)
    if total_power < eps:
        total_power = eps

    P_k = power_spectrum / total_power  # Normalized power distribution sum(P_k) = 1

    # Centered normalized frequency axis f_k in [-0.5, 0.5)
    f_k = np.fft.fftshift(np.fft.fftfreq(L, d=1.0))

    # Spectral Centroid: mu_f = sum(f_k * P[k])
    mu_f = float(np.sum(f_k * P_k))

    # Spectral Spread: sigma_f = sqrt(sum((f_k - mu_f)^2 * P[k]))
    variance_f = float(np.sum(((f_k - mu_f)**2) * P_k))
    sigma_f = np.sqrt(max(variance_f, 0.0))

    # Spectral Entropy: H_f = - sum(P[k] * log2(P[k] + eps)) / log2(L)
    H_f = float(-np.sum(P_k * np.log2(P_k + eps)) / np.log2(L))

    # Spectral Skewness and Kurtosis
    if sigma_f > 1e-6:
        spec_skew = float(np.sum(((f_k - mu_f) / sigma_f)**3 * P_k))
        spec_kurt = float(np.sum(((f_k - mu_f) / sigma_f)**4 * P_k))
    else:
        spec_skew = 0.0
        spec_kurt = 0.0

    features = np.array([
        C40_abs, C40_real, C40_imag, C42,
        M20_abs, M20_real, M20_imag, M40_abs, M42_val,
        dphi_mean, dphi_std, dphi_kurt,
        amp_mean, amp_std, amp_kurt,
        mu_f, sigma_f, H_f,
        spec_skew, spec_kurt
    ], dtype=np.float32)

    # Sanitize any NaNs or Infs
    features = np.nan_to_num(features, nan=0.0, posinf=10.0, neginf=-10.0)
    return features


def extract_features_batch(iq_tensors: np.ndarray) -> np.ndarray:
    """
    Extracts features for a batch of bursts.
    
    Args:
        iq_tensors: numpy array of shape [N, 2, L] where Row 0 is I, Row 1 is Q.
                    Or complex array of shape [N, L].
                    
    Returns:
        numpy array of shape [N, 20] containing feature vectors.
    """
    if np.iscomplexobj(iq_tensors):
        complex_bursts = iq_tensors
    else:
        if iq_tensors.ndim == 2 and iq_tensors.shape[0] == 2:
            # Single burst [2, L]
            complex_bursts = (iq_tensors[0] + 1j * iq_tensors[1])[np.newaxis, :]
        elif iq_tensors.ndim == 3 and iq_tensors.shape[1] == 2:
            # Batch [N, 2, L]
            complex_bursts = iq_tensors[:, 0, :] + 1j * iq_tensors[:, 1, :]
        else:
            raise ValueError(f"Unexpected shape for iq_tensors: {iq_tensors.shape}")

    N = complex_bursts.shape[0]
    num_features = len(FEATURE_NAMES)
    features_batch = np.zeros((N, num_features), dtype=np.float32)

    for i in range(N):
        features_batch[i] = extract_burst_features_1d(complex_bursts[i])

    return features_batch


def get_feature_dict(z_or_iq: np.ndarray) -> Dict[str, float]:
    """Returns a readable dictionary of extracted features for a single burst."""
    if not np.iscomplexobj(z_or_iq) and z_or_iq.ndim == 2 and z_or_iq.shape[0] == 2:
        z = z_or_iq[0] + 1j * z_or_iq[1]
    else:
        z = z_or_iq.squeeze()

    feats = extract_burst_features_1d(z)
    return {name: float(val) for name, val in zip(FEATURE_NAMES, feats)}
