"""
================================================================================
EE136IU: Machine Learning for Wireless Communications
Capstone Mini-Project - Topic 02: Spectral & Temporal Feature Engineering
                                 vs. End-to-End Raw I/Q Deep Learning (AMC)
================================================================================
Module: utils/metrics_plots.py
Description:
  Comprehensive evaluation & plotting suite:
    1. Calibrated 15-point SNR Grid evaluation (Accuracy & Macro-F1).
    2. Publication-quality SNR benchmark curves (Accuracy vs SNR, Macro-F1 vs SNR).
    3. Row-Normalized Confusion Matrices in % at -4 dB (Low), +4 dB (Med), +14 dB (High).
    4. Baseband physics failure mode analysis (Cumulant breakdown vs CNN spatial filtering).
================================================================================
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from typing import Dict, List, Tuple, Optional, Any

from dataset.data_generator import MOD_CLASSES, CALIBRATED_SNR_GRID


def evaluate_models_per_snr(
    test_iq: np.ndarray,
    test_feats: np.ndarray,
    test_labels: np.ndarray,
    test_snrs: np.ndarray,
    xgb_clf: Any,
    svm_clf: Optional[Any],
    cnn_pipeline: Any,
    snr_grid: Optional[np.ndarray] = None
) -> Dict[str, Dict[str, List[float]]]:
    """
    Evaluates Feature-based ML (XGBoost, SVM) and Raw I/Q 1D-CNN across all 15 SNR points.
    
    Returns:
        Dictionary mapping model names to 'accuracy' and 'macro_f1' arrays across the SNR grid.
    """
    if snr_grid is None:
        snr_grid = CALIBRATED_SNR_GRID

    results = {
        'snr': list(snr_grid),
        'XGBoost': {'accuracy': [], 'macro_f1': []},
        '1D-CNN': {'accuracy': [], 'macro_f1': []}
    }
    if svm_clf is not None:
        results['RBF-SVM'] = {'accuracy': [], 'macro_f1': []}

    print("\n" + "=" * 80)
    print(f"{'CALIBRATED 15-POINT SNR EVALUATION (-10 dB to +18 dB)':^80}")
    print("=" * 80)
    print(f"{'SNR (dB)':^10} | {'XGBoost Acc':^12} {'XGBoost F1':^12} | {'1D-CNN Acc':^12} {'1D-CNN F1':^12} |")
    print("-" * 80)

    for snr in snr_grid:
        mask = (test_snrs == snr)
        if np.sum(mask) == 0:
            continue

        y_true = test_labels[mask]
        feats_snr = test_feats[mask]
        iq_snr = test_iq[mask]

        # 1. XGBoost predictions
        xgb_preds = xgb_clf.predict(feats_snr)
        xgb_acc = accuracy_score(y_true, xgb_preds)
        xgb_f1 = f1_score(y_true, xgb_preds, average='macro', zero_division=0)
        results['XGBoost']['accuracy'].append(xgb_acc)
        results['XGBoost']['macro_f1'].append(xgb_f1)

        # 2. RBF-SVM predictions (if available)
        if svm_clf is not None:
            svm_preds = svm_clf.predict(feats_snr)
            svm_acc = accuracy_score(y_true, svm_preds)
            svm_f1 = f1_score(y_true, svm_preds, average='macro', zero_division=0)
            results['RBF-SVM']['accuracy'].append(svm_acc)
            results['RBF-SVM']['macro_f1'].append(svm_f1)

        # 3. 1D-CNN predictions
        cnn_preds = []
        for i in range(len(iq_snr)):
            pred_c, _, _ = cnn_pipeline.predict_single_burst(iq_snr[i])
            cnn_preds.append(pred_c)
        cnn_preds = np.array(cnn_preds)
        cnn_acc = accuracy_score(y_true, cnn_preds)
        cnn_f1 = f1_score(y_true, cnn_preds, average='macro', zero_division=0)
        results['1D-CNN']['accuracy'].append(cnn_acc)
        results['1D-CNN']['macro_f1'].append(cnn_f1)

        print(f"{snr:^10.1f} | {xgb_acc*100:^11.2f}% {xgb_f1*100:^11.2f}% | {cnn_acc*100:^11.2f}% {cnn_f1*100:^11.2f}% |")

    print("=" * 80)
    return results


def plot_snr_curves(
    results: Dict[str, Any],
    save_path: str = 'results/snr_benchmark_curves.png'
) -> None:
    """
    Plots high-contrast, publication-quality Accuracy vs. SNR and Macro-F1 vs. SNR curves.
    """
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    snrs = results['snr']

    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), dpi=300)

    # Palette
    color_cnn = '#0284C7'    # Tech Blue
    color_xgb = '#EA580C'    # Deep Orange
    color_svm = '#16A34A'    # Forest Green

    # Subplot 1: Accuracy vs SNR
    ax1 = axes[0]
    ax1.plot(snrs, [a * 100 for a in results['1D-CNN']['accuracy']],
             marker='o', markersize=7, linewidth=2.5, color=color_cnn, label='1D-CNN (End-to-End Raw I/Q)')
    ax1.plot(snrs, [a * 100 for a in results['XGBoost']['accuracy']],
             marker='s', markersize=7, linewidth=2.5, color=color_xgb, label='XGBoost (Cumulants + FFT Features)')

    if 'RBF-SVM' in results and len(results['RBF-SVM']['accuracy']) > 0:
        ax1.plot(snrs, [a * 100 for a in results['RBF-SVM']['accuracy']],
                 marker='^', markersize=7, linewidth=2.0, linestyle='--', color=color_svm, label='RBF-SVM (Feature Baseline)')

    # Operational Regimes Shading
    ax1.axvspan(-10, -2, alpha=0.10, color='red', label='Noise-Limited Regime (Cumulant Breakdown)')
    ax1.axvspan(-2, 6, alpha=0.08, color='gold', label='Transition Recovery Regime')
    ax1.axvspan(6, 18, alpha=0.10, color='green', label='Asymptotic High-SNR Regime')

    ax1.set_title('Classification Accuracy vs. SNR (-10 dB to +18 dB)', fontsize=13, fontweight='bold', pad=12)
    ax1.set_xlabel('Signal-to-Noise Ratio (SNR in dB)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Accuracy (%)', fontsize=11, fontweight='bold')
    ax1.set_xlim([-10.5, 18.5])
    ax1.set_ylim([10, 102])
    ax1.set_xticks(snrs)
    ax1.legend(loc='lower right', frameon=True, framealpha=0.9, fontsize=9)
    ax1.grid(True, linestyle='--', alpha=0.6)

    # Subplot 2: Macro-F1 vs SNR
    ax2 = axes[1]
    ax2.plot(snrs, [f * 100 for f in results['1D-CNN']['macro_f1']],
             marker='o', markersize=7, linewidth=2.5, color=color_cnn, label='1D-CNN (End-to-End Raw I/Q)')
    ax2.plot(snrs, [f * 100 for f in results['XGBoost']['macro_f1']],
             marker='s', markersize=7, linewidth=2.5, color=color_xgb, label='XGBoost (Cumulants + FFT Features)')

    if 'RBF-SVM' in results and len(results['RBF-SVM']['macro_f1']) > 0:
        ax2.plot(snrs, [f * 100 for f in results['RBF-SVM']['macro_f1']],
                 marker='^', markersize=7, linewidth=2.0, linestyle='--', color=color_svm, label='RBF-SVM (Feature Baseline)')

    ax2.set_title('Macro-F1 Score vs. SNR (Balanced Performance)', fontsize=13, fontweight='bold', pad=12)
    ax2.set_xlabel('Signal-to-Noise Ratio (SNR in dB)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Macro-F1 (%)', fontsize=11, fontweight='bold')
    ax2.set_xlim([-10.5, 18.5])
    ax2.set_ylim([10, 102])
    ax2.set_xticks(snrs)
    ax2.legend(loc='lower right', frameon=True, framealpha=0.9, fontsize=9)
    ax2.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved SNR benchmark curves to: {save_path}")


def plot_row_normalized_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    classes: List[str],
    ax: plt.Axes,
    title: str,
    cmap: str = 'Blues'
) -> None:
    """Plots a single row-normalized confusion matrix with percentage annotations."""
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(classes)))
    cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-12) * 100.0

    sns.heatmap(
        cm_norm,
        annot=True,
        fmt='.1f',
        cmap=cmap,
        xticklabels=classes,
        yticklabels=classes,
        cbar=False,
        ax=ax,
        vmin=0.0,
        vmax=100.0,
        annot_kws={'size': 8, 'weight': 'bold'}
    )
    ax.set_title(title, fontsize=11, fontweight='bold', pad=8)
    ax.set_ylabel('Ground Truth Class', fontsize=9, fontweight='bold')
    ax.set_xlabel('Predicted Class', fontsize=9, fontweight='bold')
    ax.tick_params(axis='x', rotation=45)


def plot_confusion_matrices_at_regimes(
    test_iq: np.ndarray,
    test_feats: np.ndarray,
    test_labels: np.ndarray,
    test_snrs: np.ndarray,
    xgb_clf: Any,
    cnn_pipeline: Any,
    target_snrs: Tuple[int, int, int] = (-4, 4, 14),
    save_path: str = 'results/confusion_matrices_regimes.png'
) -> None:
    """
    Plots row-normalized confusion matrices comparing XGBoost (Features) vs 1D-CNN (Raw I/Q)
    at 3 key operational SNR points:
      - Low SNR (-4 dB)
      - Medium SNR (+4 dB)
      - High SNR (+14 dB)
    """
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(18, 11), dpi=300)

    regime_names = {
        target_snrs[0]: f"Low SNR ({target_snrs[0]} dB) [Noise-Dominant]",
        target_snrs[1]: f"Medium SNR ({target_snrs[1]} dB) [Transition]",
        target_snrs[2]: f"High SNR ({target_snrs[2]} dB) [Clean Channel]"
    }

    for col_idx, snr_val in enumerate(target_snrs):
        mask = (test_snrs == snr_val)
        y_true = test_labels[mask]
        feats = test_feats[mask]
        iq = test_iq[mask]

        # XGBoost Predictions
        xgb_preds = xgb_clf.predict(feats)
        # 1D-CNN Predictions
        cnn_preds = []
        for i in range(len(iq)):
            p, _, _ = cnn_pipeline.predict_single_burst(iq[i])
            cnn_preds.append(p)
        cnn_preds = np.array(cnn_preds)

        # Plot Row 0: XGBoost
        plot_row_normalized_confusion_matrix(
            y_true, xgb_preds, MOD_CLASSES,
            ax=axes[0, col_idx],
            title=f"XGBoost (Features) @ {regime_names[snr_val]}",
            cmap='Oranges'
        )

        # Plot Row 1: 1D-CNN
        plot_row_normalized_confusion_matrix(
            y_true, cnn_preds, MOD_CLASSES,
            ax=axes[1, col_idx],
            title=f"1D-CNN (Raw I/Q) @ {regime_names[snr_val]}",
            cmap='Blues'
        )

    plt.suptitle("Row-Normalized Confusion Matrices (% per class) across 3 SNR Regimes",
                 fontsize=15, fontweight='bold', y=0.99)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved 3-regime row-normalized confusion matrices to: {save_path}")


def print_failure_analysis_report(
    test_iq: np.ndarray,
    test_feats: np.ndarray,
    test_labels: np.ndarray,
    test_snrs: np.ndarray,
    xgb_clf: Any,
    cnn_pipeline: Any
) -> str:
    """
    Produces a detailed RF physical failure analysis report explaining why models fail,
    especially focusing on Cumulant degradation vs CNN temporal filtering and QAM lattice ambiguity.
    """
    report = []
    report.append("=" * 80)
    report.append("EE136 CAPSTONE MINI-PROJECT - TOPIC 02: ROOT-CAUSE FAILURE ANALYSIS REPORT")
    report.append("=" * 80)

    # 1. Low SNR Failure (-4 dB)
    mask_low = (test_snrs == -4)
    y_low = test_labels[mask_low]
    xgb_pred_low = xgb_clf.predict(test_feats[mask_low])

    cnn_pred_low = []
    for i in range(np.sum(mask_low)):
        p, _, _ = cnn_pipeline.predict_single_burst(test_iq[mask_low][i])
        cnn_pred_low.append(p)
    cnn_pred_low = np.array(cnn_pred_low)

    acc_xgb_low = accuracy_score(y_low, xgb_pred_low) * 100
    acc_cnn_low = accuracy_score(y_low, cnn_pred_low) * 100

    report.append("\n[1] LOW-SNR NOISE COLLAPSE REGIME (SNR = -4 dB):")
    report.append(f"  • Feature-Based XGBoost Accuracy: {acc_xgb_low:.2f}%")
    report.append(f"  • End-to-End 1D-CNN Accuracy:    {acc_cnn_low:.2f}% (Delta = +{acc_cnn_low - acc_xgb_low:.2f}%)")
    report.append("  • Physical Root Cause:")
    report.append("    Higher-Order Cumulants (C40, C42) depend on 4th moments E[|z|^4] and E[z^4].")
    report.append("    Under AWGN r[n] = s[n] + w[n], expanding |r[n]|^4 yields severe noise cross-terms:")
    report.append("      |r[n]|^4 = (|s[n]|^2 + |w[n]|^2 + 2 Re{s[n] w*[n]})^2.")
    report.append("    Over finite burst length L = 128, the sample variance of the cumulant estimator explodes:")
    report.append("      Var(C42_est) ~ O(sigma_w^4 / L) or O(sigma_w^8 / L).")
    report.append("    At SNR = -4 dB, noise power sigma_w^2 > signal power E_s, completely corrupting")
    report.append("    the cumulant feature space. In contrast, 1D-CNN applies learned spatio-temporal")
    report.append("    convolutional filters (kernels of size 7, 5, 3) acting as coherent bandpass match-filters")
    report.append("    that integrate and average out wideband out-of-band Gaussian noise before non-linear activation.")

    # 2. QAM Lattice Ambiguity (16QAM vs 64QAM)
    mask_high = (test_snrs >= 10)
    idx_16qam = MOD_CLASSES.index('16QAM')
    idx_64qam = MOD_CLASSES.index('64QAM')

    qam_mask = mask_high & ((test_labels == idx_16qam) | (test_labels == idx_64qam))
    y_qam = test_labels[qam_mask]
    xgb_pred_qam = xgb_clf.predict(test_feats[qam_mask])

    cnn_pred_qam = []
    for i in range(np.sum(qam_mask)):
        p, _, _ = cnn_pipeline.predict_single_burst(test_iq[qam_mask][i])
        cnn_pred_qam.append(p)
    cnn_pred_qam = np.array(cnn_pred_qam)

    acc_xgb_qam = accuracy_score(y_qam, xgb_pred_qam) * 100
    acc_cnn_qam = accuracy_score(y_qam, cnn_pred_qam) * 100

    report.append("\n[2] DENSE CONSTELLATION LATTICE AMBIGUITY (16QAM vs. 64QAM @ High SNR >= 10 dB):")
    report.append(f"  • XGBoost QAM Discrimination Accuracy: {acc_xgb_qam:.2f}%")
    report.append(f"  • 1D-CNN QAM Discrimination Accuracy:  {acc_cnn_qam:.2f}%")
    report.append("  • Physical Root Cause:")
    report.append("    Theoretical 4th-order cumulants of normalized square QAM constellations are mathematically close:")
    report.append("      C42(16QAM) = -0.6800")
    report.append("      C42(64QAM) = -0.6190  (Difference delta = 0.061)")
    report.append("    Under finite-sample estimation (L=128), statistical variance often overlaps this 0.061 gap,")
    report.append("    causing classical feature classifiers to confuse 16QAM and 64QAM.")
    report.append("    1D-CNN overcomes this by learning complex transition trajectories and multilevel peak-to-average")
    report.append("    ratio (PAPR) characteristics across multiple consecutive samples.")

    # 3. Phase Continuity & FSK Separation
    report.append("\n[3] CONTINUOUS PHASE TRAJECTORY (CPFSK vs. GFSK):")
    report.append("  • CPFSK and GFSK both exhibit constant envelopes (|z[n]| = 1.0) and identical C42 = -1.0.")
    report.append("  • However, their differential phase statistics (Delta_phi) and FFT Spectral Spread differ sharply:")
    report.append("    GFSK passes binary data through a Gaussian filter (BT=0.5), resulting in a significantly")
    report.append("    narrower Spectral Spread and smaller differential phase std than CPFSK.")
    report.append("=" * 80)

    full_report_str = "\n".join(report)
    print(full_report_str)
    return full_report_str
