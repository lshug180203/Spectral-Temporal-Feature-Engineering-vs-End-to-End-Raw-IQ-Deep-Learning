"""
================================================================================
EE136IU: Machine Learning for Wireless Communications
Capstone Mini-Project - Topic 02: Spectral & Temporal Feature Engineering
                                 vs. End-to-End Raw I/Q Deep Learning (AMC)
================================================================================
Module: dataset/data_generator.py
Description:
  Synthesizes physically grounded complex baseband I/Q observation bursts (L=128),
  converts them to real-valued tensors of shape [N, 2, 128], performs mandatory
  per-burst RMS energy normalization (r_norm = r / E_rms) to prevent energy cheating,
  and executes strict anti-leakage train/validation/test partitioning (70/15/15).
================================================================================
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Dict, Optional, List

# Standard 8 modulation classes across 3 distinct families
MOD_CLASSES: List[str] = [
    'BPSK',    # PSK Family
    'QPSK',    # PSK Family
    '8PSK',    # PSK Family
    '16QAM',   # QAM Family
    '64QAM',   # QAM Family
    'PAM4',    # PAM Family
    'CPFSK',   # FSK Family
    'GFSK'     # FSK Family
]

MOD_TO_IDX: Dict[str, int] = {mod: i for i, mod in enumerate(MOD_CLASSES)}
IDX_TO_MOD: Dict[int, str] = {i: mod for i, mod in enumerate(MOD_CLASSES)}

# Standard 15-point calibrated SNR grid (-10 dB to +18 dB, step 2 dB)
CALIBRATED_SNR_GRID: np.ndarray = np.arange(-10, 20, 2)  # -10, -8, ..., +18


def set_seed(seed: int = 42) -> None:
    """Sets deterministic random seeds across all libraries."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def generate_baseband_symbols(mod_type: str, num_symbols: int, rng: np.random.RandomState) -> np.ndarray:
    """
    Generates complex baseband constellation symbols for a given digital modulation.
    All constellations are normalized to average symbol energy E_s = 1.0.
    """
    if mod_type == 'BPSK':
        # Symbols in {-1, +1}
        bits = rng.randint(0, 2, num_symbols)
        symbols = 2.0 * bits - 1.0
        return symbols.astype(np.complex64)

    elif mod_type == 'QPSK':
        # Gray-coded QPSK symbols in { (+-1 +- j) / sqrt(2) }
        bits_i = rng.randint(0, 2, num_symbols)
        bits_q = rng.randint(0, 2, num_symbols)
        symbols = ((2.0 * bits_i - 1.0) + 1j * (2.0 * bits_q - 1.0)) / np.sqrt(2.0)
        return symbols.astype(np.complex64)

    elif mod_type == '8PSK':
        # 8-ary PSK on unit circle: exp(j * 2pi * k / 8)
        phases = rng.randint(0, 8, num_symbols) * (2.0 * np.pi / 8.0)
        symbols = np.exp(1j * phases)
        return symbols.astype(np.complex64)

    elif mod_type == '16QAM':
        # 16-QAM square lattice: {-3, -1, +1, +3} + j{-3, -1, +1, +3}
        # Average power before normalization = (2/16) * 4 * (1+9+9+9...) = 10.0
        sym_i = rng.choice([-3, -1, 1, 3], size=num_symbols)
        sym_q = rng.choice([-3, -1, 1, 3], size=num_symbols)
        symbols = (sym_i + 1j * sym_q) / np.sqrt(10.0)
        return symbols.astype(np.complex64)

    elif mod_type == '64QAM':
        # 64-QAM square lattice: {-7, -5, -3, -1, 1, 3, 5, 7}
        # Average power before normalization = (1+9+25+49)/4 = 84/2 = 42.0
        levels = np.array([-7, -5, -3, -1, 1, 3, 5, 7], dtype=np.float32)
        sym_i = rng.choice(levels, size=num_symbols)
        sym_q = rng.choice(levels, size=num_symbols)
        symbols = (sym_i + 1j * sym_q) / np.sqrt(42.0)
        return symbols.astype(np.complex64)

    elif mod_type == 'PAM4':
        # 4-level Pulse Amplitude Modulation: {-3, -1, 1, 3} on In-phase axis
        # Average power before normalization = (1+9)/2 = 5.0
        sym_i = rng.choice([-3, -1, 1, 3], size=num_symbols)
        symbols = (sym_i + 0.0j) / np.sqrt(5.0)
        return symbols.astype(np.complex64)

    else:
        raise ValueError(f"Unknown discrete symbol modulation: {mod_type}")


def generate_burst_signal(mod_type: str, burst_length: int = 128, rng: Optional[np.random.RandomState] = None) -> np.ndarray:
    """
    Synthesizes a single complex baseband burst of length L=128 maintaining
    continuous phase trajectory and realistic pulse shaping.
    """
    if rng is None:
        rng = np.random.RandomState(42)

    samples_per_symbol = 4
    num_symbols = burst_length // samples_per_symbol + 8  # Extra symbols to avoid transient edge effects

    if mod_type in ['BPSK', 'QPSK', '8PSK', '16QAM', '64QAM', 'PAM4']:
        symbols = generate_baseband_symbols(mod_type, num_symbols, rng)
        # Upsample with rectangular / half-sine shaping for continuous baseband trajectory
        upsampled = np.zeros(num_symbols * samples_per_symbol, dtype=np.complex64)
        upsampled[::samples_per_symbol] = symbols

        # Root-Raised Cosine (RRC) / Gaussian pulse shaping impulse response
        # To maintain clean continuous trajectory without external heavy filter overhead:
        t = np.arange(-2 * samples_per_symbol, 2 * samples_per_symbol + 1)
        pulse = np.sinc(t / samples_per_symbol) * np.hamming(len(t))
        pulse = pulse / np.sqrt(np.sum(pulse**2))

        shaped = np.convolve(upsampled, pulse, mode='same')
        # Slice exactly burst_length samples from steady-state center
        start_idx = 4 * samples_per_symbol
        burst = shaped[start_idx : start_idx + burst_length]

    elif mod_type == 'CPFSK':
        # Continuous Phase Frequency Shift Keying (modulation index h = 0.5, MSK style)
        bits = rng.randint(0, 2, burst_length // samples_per_symbol + 4)
        nrz = 2.0 * bits - 1.0  # {-1, +1}
        # Frequency deviation pulse
        freq_dev = np.repeat(nrz, samples_per_symbol)[:burst_length + 16]
        h = 0.5
        phase = np.cumsum(freq_dev * (np.pi * h / samples_per_symbol))
        # Initial random phase
        init_phase = rng.uniform(0, 2 * np.pi)
        burst = np.exp(1j * (phase[:burst_length] + init_phase))

    elif mod_type == 'GFSK':
        # Gaussian Frequency Shift Keying (BT = 0.5)
        bits = rng.randint(0, 2, burst_length // samples_per_symbol + 8)
        nrz = 2.0 * bits - 1.0
        upsampled_nrz = np.repeat(nrz, samples_per_symbol)

        # Gaussian filter
        bt = 0.5
        t_span = 3 * samples_per_symbol
        t = np.arange(-t_span, t_span + 1)
        alpha = np.sqrt(2 * np.pi / np.log(2)) * bt / samples_per_symbol
        gaussian_filter = np.exp(-0.5 * (alpha * t)**2)
        gaussian_filter /= np.sum(gaussian_filter)

        freq_shaped = np.convolve(upsampled_nrz, gaussian_filter, mode='same')
        start_idx = 3 * samples_per_symbol
        freq_slice = freq_shaped[start_idx : start_idx + burst_length]

        h = 0.5
        phase = np.cumsum(freq_slice * (np.pi * h / samples_per_symbol))
        init_phase = rng.uniform(0, 2 * np.pi)
        burst = np.exp(1j * (phase + init_phase))

    else:
        raise ValueError(f"Unsupported modulation type: {mod_type}")

    # Ensure length is exactly burst_length
    if len(burst) < burst_length:
        burst = np.pad(burst, (0, burst_length - len(burst)), mode='edge')
    else:
        burst = burst[:burst_length]

    return burst.astype(np.complex64)


def apply_awgn_and_rms_norm(
    clean_burst: np.ndarray,
    snr_db: float,
    rng: Optional[np.random.RandomState] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Applies calibrated AWGN to the clean complex burst and enforces MANDATORY
    per-burst RMS energy normalization:
        r_norm[n] = r[n] / E_rms
        E_rms = sqrt( (1/L) * sum( |r[n]|^2 ) )
    This prevents the classifier from exploiting average signal power differences.

    Returns:
        r_norm: complex baseband burst of shape [L] with unit RMS energy.
        iq_tensor: real-valued 2D array of shape [2, L] (Row 0: I[n], Row 1: Q[n]).
    """
    if rng is None:
        rng = np.random.RandomState(42)

    L = len(clean_burst)
    signal_power = np.mean(np.abs(clean_burst)**2)
    if signal_power == 0:
        signal_power = 1.0

    snr_linear = 10.0**(snr_db / 10.0)
    noise_power = signal_power / snr_linear

    # Complex circularly symmetric Gaussian noise: CN(0, noise_power)
    noise_i = rng.normal(0.0, np.sqrt(noise_power / 2.0), L)
    noise_q = rng.normal(0.0, np.sqrt(noise_power / 2.0), L)
    noise = noise_i + 1j * noise_q

    received = clean_burst + noise

    # Per-burst RMS Energy Normalization (Zero Energy Cheating)
    e_rms = np.sqrt(np.mean(np.abs(received)**2))
    if e_rms < 1e-12:
        e_rms = 1e-12

    r_norm = received / e_rms

    # Format into real tensor [2, L]
    iq_tensor = np.stack([np.real(r_norm), np.imag(r_norm)], axis=0).astype(np.float32)

    return r_norm.astype(np.complex64), iq_tensor


def generate_amc_dataset(
    bursts_per_class_snr: int = 100,
    burst_length: int = 128,
    snr_grid: Optional[np.ndarray] = None,
    seed: int = 42,
    cache_path: Optional[str] = None
) -> Dict[str, np.ndarray]:
    """
    Generates or loads cached standardized AMC dataset across 8 classes and 15 SNR points.
    Strictly partitions into 70% Train, 15% Validation, 15% Test with zero data leakage.

    Dataset dictionary keys:
      - 'X_train_iq': [N_train, 2, 128]
      - 'y_train': [N_train]
      - 'snr_train': [N_train]
      - 'X_val_iq': [N_val, 2, 128]
      - 'y_val': [N_val]
      - 'snr_val': [N_val]
      - 'X_test_iq': [N_test, 2, 128]
      - 'y_test': [N_test]
      - 'snr_test': [N_test]
      - 'classes': MOD_CLASSES
      - 'snr_grid': CALIBRATED_SNR_GRID
    """
    if cache_path is not None and os.path.exists(cache_path):
        print(f"[*] Loading cached AMC dataset from: {cache_path}")
        data = np.load(cache_path, allow_pickle=True)
        return {key: data[key] for key in data.files}

    if snr_grid is None:
        snr_grid = CALIBRATED_SNR_GRID

    set_seed(seed)
    rng = np.random.RandomState(seed)

    print(f"[*] Generating AMC Dataset: {len(MOD_CLASSES)} classes, {len(snr_grid)} SNRs, "
          f"{bursts_per_class_snr} bursts/class/SNR (Total = {len(MOD_CLASSES) * len(snr_grid) * bursts_per_class_snr} bursts)...")

    all_iq = []
    all_labels = []
    all_snrs = []

    for mod_idx, mod_type in enumerate(MOD_CLASSES):
        for snr in snr_grid:
            for _ in range(bursts_per_class_snr):
                clean = generate_burst_signal(mod_type, burst_length, rng=rng)
                _, iq_tensor = apply_awgn_and_rms_norm(clean, snr, rng=rng)
                all_iq.append(iq_tensor)
                all_labels.append(mod_idx)
                all_snrs.append(snr)

    all_iq = np.array(all_iq, dtype=np.float32)      # [N, 2, 128]
    all_labels = np.array(all_labels, dtype=np.int64) # [N]
    all_snrs = np.array(all_snrs, dtype=np.float32)   # [N]

    # Strict Anti-Leakage Partitioning: 70% Train, 15% Validation, 15% Test
    # Stratified per (mod_type, snr) cell
    total_samples = len(all_labels)
    train_indices = []
    val_indices = []
    test_indices = []

    num_cells = len(MOD_CLASSES) * len(snr_grid)
    for c in range(num_cells):
        cell_start = c * bursts_per_class_snr
        cell_end = (c + 1) * bursts_per_class_snr
        indices = np.arange(cell_start, cell_end)
        rng.shuffle(indices)

        n_train = int(0.70 * bursts_per_class_snr)
        n_val = int(0.15 * bursts_per_class_snr)

        train_indices.extend(indices[:n_train])
        val_indices.extend(indices[n_train : n_train + n_val])
        test_indices.extend(indices[n_train + n_val :])

    train_indices = np.array(train_indices)
    val_indices = np.array(val_indices)
    test_indices = np.array(test_indices)

    # Shuffle each partition independently
    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    rng.shuffle(test_indices)

    dataset = {
        'X_train_iq': all_iq[train_indices],
        'y_train': all_labels[train_indices],
        'snr_train': all_snrs[train_indices],
        'X_val_iq': all_iq[val_indices],
        'y_val': all_labels[val_indices],
        'snr_val': all_snrs[val_indices],
        'X_test_iq': all_iq[test_indices],
        'y_test': all_labels[test_indices],
        'snr_test': all_snrs[test_indices],
        'classes': np.array(MOD_CLASSES),
        'snr_grid': np.array(snr_grid)
    }

    if cache_path is not None:
        os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
        np.savez_compressed(cache_path, **dataset)
        print(f"[OK] Saved standardized AMC dataset to: {cache_path}")

    print(f"[OK] Dataset partitioned: Train = {len(dataset['y_train'])}, "
          f"Val = {len(dataset['y_val'])}, Test = {len(dataset['y_test'])} | Zero Leakage Guaranteed.")

    return dataset


class AMCTensorDataset(Dataset):
    """PyTorch Dataset wrapper for raw I/Q baseband tensors [2, 128]."""
    def __init__(self, iq_tensors: np.ndarray, labels: np.ndarray, snrs: Optional[np.ndarray] = None):
        self.X = torch.from_numpy(iq_tensors).float()
        self.y = torch.from_numpy(labels).long()
        self.snrs = torch.from_numpy(snrs).float() if snrs is not None else None

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]
