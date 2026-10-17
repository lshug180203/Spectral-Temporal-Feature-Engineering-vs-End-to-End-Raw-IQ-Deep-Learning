"""
================================================================================
EE136IU: Machine Learning for Wireless Communications
Capstone Mini-Project - Topic 02: Spectral & Temporal Feature Engineering
                                 vs. End-to-End Raw I/Q Deep Learning (AMC)
================================================================================
Module: utils/metrics_plots.py
Description:
  Comprehensive evaluation & plotting suite:
    1. Calibrated 15-point SNR Grid evaluation (Accuracy & Macro-F1) with 95% CIs.
    2. Publication-quality SNR benchmark curves (Accuracy vs SNR, Macro-F1 vs SNR).
    3. Row-Normalized Confusion Matrices in % at -4 dB (Low), +4 dB (Med), +14 dB (High).
    4. CNN training curves (loss / validation accuracy / validation Macro-F1).
    5. Data-driven root-cause failure analysis (every number is computed from the
       test set; no hard-coded performance claims).
================================================================================
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from math import erf, sqrt
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from typing import Dict, List, Tuple, Optional, Any

from dataset.data_generator import MOD_CLASSES, CALIBRATED_SNR_GRID
from utils.feature_extractor import FEATURE_NAMES


def _norm_cdf(x: float) -> float:
    """Standard normal CDF."""
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def _ci95(acc: float, n: int) -> float:
    """Half-width of the 95% normal-approximation binomial confidence interval."""
    if n <= 0:
        return 0.0
    return 1.96 * sqrt(max(acc * (1.0 - acc), 1e-12) / n)


def evaluate_models_per_snr(
    test_iq: np.ndarray,
    test_feats: np.ndarray,
    test_labels: np.ndarray,
    test_snrs: np.ndarray,
    xgb_clf: Any,
    svm_clf: Optional[Any],
    cnn_pipeline: Any,
    snr_grid: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """
    Evaluates Feature-based ML (XGBoost, SVM) and Raw I/Q 1D-CNN across all 15 SNR points.

    Returns:
        Dictionary with 'snr', 'n_per_snr' and, per model, 'accuracy', 'macro_f1',
        and 'acc_ci95' (half-width of the 95% confidence interval on accuracy).
    """
    if snr_grid is None:
        snr_grid = CALIBRATED_SNR_GRID

    results: Dict[str, Any] = {
        'snr': [int(s) for s in snr_grid],
        'n_per_snr': [],
        'XGBoost': {'accuracy': [], 'macro_f1': [], 'acc_ci95': []},
        '1D-CNN': {'accuracy': [], 'macro_f1': [], 'acc_ci95': []}
    }
    if svm_clf is not None:
        results['RBF-SVM'] = {'accuracy': [], 'macro_f1': [], 'acc_ci95': []}

    print("\n" + "=" * 92)
    print(f"{'CALIBRATED 15-POINT SNR EVALUATION (-10 dB to +18 dB)':^92}")
    print("=" * 92)
    print(f"{'SNR (dB)':^9} | {'N':^5} | {'XGB Acc (+-95%CI)':^20} {'XGB F1':^8} | "
          f"{'CNN Acc (+-95%CI)':^20} {'CNN F1':^8} |")
    print("-" * 92)

    for snr in snr_grid:
        mask = (test_snrs == snr)
        n = int(np.sum(mask))
        if n == 0:
            continue

        y_true = test_labels[mask]
        results['n_per_snr'].append(n)

        # 1. XGBoost
        xgb_preds = xgb_clf.predict(test_feats[mask])
        xgb_acc = accuracy_score(y_true, xgb_preds)
        xgb_f1 = f1_score(y_true, xgb_preds, average='macro', zero_division=0)
        results['XGBoost']['accuracy'].append(xgb_acc)
        results['XGBoost']['macro_f1'].append(xgb_f1)
        results['XGBoost']['acc_ci95'].append(_ci95(xgb_acc, n))

        # 2. RBF-SVM
        if svm_clf is not None:
            svm_preds = svm_clf.predict(test_feats[mask])
            svm_acc = accuracy_score(y_true, svm_preds)
            results['RBF-SVM']['accuracy'].append(svm_acc)
            results['RBF-SVM']['macro_f1'].append(f1_score(y_true, svm_preds, average='macro', zero_division=0))
            results['RBF-SVM']['acc_ci95'].append(_ci95(svm_acc, n))

        # 3. 1D-CNN (batched)
        cnn_preds, _ = cnn_pipeline.predict_batch(test_iq[mask])
        cnn_acc = accuracy_score(y_true, cnn_preds)
        cnn_f1 = f1_score(y_true, cnn_preds, average='macro', zero_division=0)
        results['1D-CNN']['accuracy'].append(cnn_acc)
        results['1D-CNN']['macro_f1'].append(cnn_f1)
        results['1D-CNN']['acc_ci95'].append(_ci95(cnn_acc, n))

        print(f"{snr:^9.0f} | {n:^5d} | "
              f"{xgb_acc*100:6.2f}% +-{_ci95(xgb_acc, n)*100:4.1f} {xgb_f1*100:7.2f}% | "
              f"{cnn_acc*100:6.2f}% +-{_ci95(cnn_acc, n)*100:4.1f} {cnn_f1*100:7.2f}% |")

    print("=" * 92)
    return results


def plot_snr_curves(
    results: Dict[str, Any],
    save_path: str = 'results/snr_benchmark_curves.png'
) -> None:
    """
    Plots Accuracy vs. SNR and Macro-F1 vs. SNR curves. Accuracy panel includes 95% CI bands.
    """
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    snrs = np.array(results['snr'])

    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), dpi=300)

    color_cnn = '#0284C7'
    color_xgb = '#EA580C'
    color_svm = '#16A34A'

    def pct(values):
        return np.array(values) * 100.0

    # Subplot 1: Accuracy vs SNR (with 95% CI bands)
    ax1 = axes[0]
    for name, color, marker, ls, label in [
        ('1D-CNN', color_cnn, 'o', '-', '1D-CNN (End-to-End Raw I/Q)'),
        ('XGBoost', color_xgb, 's', '-', 'XGBoost (Cumulants + FFT Features)'),
        ('RBF-SVM', color_svm, '^', '--', 'RBF-SVM (Feature Baseline)')
    ]:
        if name not in results or len(results[name]['accuracy']) == 0:
            continue
        acc = pct(results[name]['accuracy'])
        ci = pct(results[name]['acc_ci95'])
        ax1.plot(snrs, acc, marker=marker, markersize=7, linewidth=2.5 if ls == '-' else 2.0,
                 linestyle=ls, color=color, label=label)
        if name != 'RBF-SVM':
            ax1.fill_between(snrs, acc - ci, acc + ci, color=color, alpha=0.12, linewidth=0)

    ax1.axvspan(-10, -2, alpha=0.10, color='red', label='Noise-Limited Regime (Cumulant Breakdown)')
    ax1.axvspan(-2, 6, alpha=0.08, color='gold', label='Transition Recovery Regime')
    ax1.axvspan(6, 18, alpha=0.10, color='green', label='Asymptotic High-SNR Regime')

    n_min = min(results['n_per_snr']) if results.get('n_per_snr') else 0
    ax1.set_title(f'Classification Accuracy vs. SNR (shaded: 95% CI, n >= {n_min}/SNR)',
                  fontsize=12, fontweight='bold', pad=12)
    ax1.set_xlabel('Signal-to-Noise Ratio (SNR in dB)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Accuracy (%)', fontsize=11, fontweight='bold')
    ax1.set_xlim([-10.5, 18.5])
    ax1.set_ylim([0, 102])
    ax1.set_xticks(snrs)
    ax1.axhline(100.0 / len(MOD_CLASSES), color='gray', linestyle=':', linewidth=1.0)
    ax1.text(-10.2, 100.0 / len(MOD_CLASSES) + 1.2, 'chance (12.5%)', fontsize=8, color='gray')
    ax1.legend(loc='lower right', frameon=True, framealpha=0.9, fontsize=9)
    ax1.grid(True, linestyle='--', alpha=0.6)

    # Subplot 2: Macro-F1 vs SNR
    ax2 = axes[1]
    for name, color, marker, ls, label in [
        ('1D-CNN', color_cnn, 'o', '-', '1D-CNN (End-to-End Raw I/Q)'),
        ('XGBoost', color_xgb, 's', '-', 'XGBoost (Cumulants + FFT Features)'),
        ('RBF-SVM', color_svm, '^', '--', 'RBF-SVM (Feature Baseline)')
    ]:
        if name not in results or len(results[name]['macro_f1']) == 0:
            continue
        ax2.plot(snrs, pct(results[name]['macro_f1']), marker=marker, markersize=7,
                 linewidth=2.5 if ls == '-' else 2.0, linestyle=ls, color=color, label=label)

    ax2.set_title('Macro-F1 Score vs. SNR (Balanced Performance)', fontsize=12, fontweight='bold', pad=12)
    ax2.set_xlabel('Signal-to-Noise Ratio (SNR in dB)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Macro-F1 (%)', fontsize=11, fontweight='bold')
    ax2.set_xlim([-10.5, 18.5])
    ax2.set_ylim([0, 102])
    ax2.set_xticks(snrs)
    ax2.legend(loc='lower right', frameon=True, framealpha=0.9, fontsize=9)
    ax2.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved SNR benchmark curves to: {save_path}")


def plot_training_curves(
    history: Dict[str, List[float]],
    save_path: str = 'results/cnn_training_curves.png'
) -> None:
    """
    Plots 1D-CNN training diagnostics: train/validation loss, and validation Accuracy / Macro-F1.
    The best-checkpoint epoch (max validation Macro-F1) is marked.
    """
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    epochs = np.arange(1, len(history['train_loss']) + 1)
    best_epoch = int(np.argmax(history['val_f1'])) + 1

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), dpi=300)

    ax = axes[0]
    ax.plot(epochs, history['train_loss'], marker='o', linewidth=2.2, color='#0284C7', label='Train loss')
    ax.plot(epochs, history['val_loss'], marker='s', linewidth=2.2, color='#EA580C', label='Validation loss')
    ax.axvline(best_epoch, color='gray', linestyle='--', linewidth=1.2, label=f'Best checkpoint (epoch {best_epoch})')
    ax.set_title('1D-CNN Loss (CrossEntropy)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Epoch', fontweight='bold')
    ax.set_ylabel('Loss', fontweight='bold')
    ax.legend(frameon=True, fontsize=9)
    ax.grid(True, linestyle='--', alpha=0.5)

    ax = axes[1]
    ax.plot(epochs, np.array(history['val_acc']) * 100, marker='o', linewidth=2.2, color='#16A34A', label='Val accuracy')
    ax.plot(epochs, np.array(history['val_f1']) * 100, marker='^', linewidth=2.2, color='#2563EB', label='Val Macro-F1')
    ax.axvline(best_epoch, color='gray', linestyle='--', linewidth=1.2)
    ax.set_title('Validation Accuracy & Macro-F1', fontsize=12, fontweight='bold')
    ax.set_xlabel('Epoch', fontweight='bold')
    ax.set_ylabel('%', fontweight='bold')
    ax.legend(frameon=True, fontsize=9)
    ax.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved CNN training curves to: {save_path}")


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
    at 3 key operational SNR points: Low (-4 dB), Medium (+4 dB), High (+14 dB).
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
        xgb_preds = xgb_clf.predict(test_feats[mask])
        cnn_preds, _ = cnn_pipeline.predict_batch(test_iq[mask])

        plot_row_normalized_confusion_matrix(
            y_true, xgb_preds, MOD_CLASSES, ax=axes[0, col_idx],
            title=f"XGBoost (Features) @ {regime_names[snr_val]}", cmap='Oranges'
        )
        plot_row_normalized_confusion_matrix(
            y_true, cnn_preds, MOD_CLASSES, ax=axes[1, col_idx],
            title=f"1D-CNN (Raw I/Q) @ {regime_names[snr_val]}", cmap='Blues'
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
    Root-cause failure analysis. Every performance number below is COMPUTED from the
    test set; the theory (cumulant contraction, estimator variance) is stated as
    mechanism, and its predictions are compared with the measurements.
    """
    c42_idx = FEATURE_NAMES.index('C42')
    idx = {m: MOD_CLASSES.index(m) for m in MOD_CLASSES}

    xgb_all = xgb_clf.predict(test_feats)
    cnn_all, _ = cnn_pipeline.predict_batch(test_iq)

    def acc_at(mask, preds):
        return accuracy_score(test_labels[mask], preds[mask]) * 100.0

    lines: List[str] = []
    lines.append("=" * 80)
    lines.append("EE136 CAPSTONE MINI-PROJECT - TOPIC 02: ROOT-CAUSE FAILURE ANALYSIS REPORT")
    lines.append("(all numbers computed from the held-out test set)")
    lines.append("=" * 80)

    # ---------------------------------------------------------------- [1] Low SNR
    low = (test_snrs == -4)
    n_low = int(np.sum(low))
    a_x, a_c = acc_at(low, xgb_all), acc_at(low, cnn_all)
    rho = 10 ** (-4 / 10.0)
    shrink = (rho / (1.0 + rho)) ** 2

    lines.append("\n[1] LOW-SNR COLLAPSE OF CUMULANT FEATURES (SNR = -4 dB, n = %d bursts)" % n_low)
    lines.append(f"  - XGBoost accuracy: {a_x:.2f}% | 1D-CNN accuracy: {a_c:.2f}% (CNN - XGB = {a_c - a_x:+.2f} pts)")
    lines.append("  - Mechanism A (contraction / bias): 4th-order cumulants of independent signals add,")
    lines.append("    and the 4th-order cumulant of Gaussian noise is 0. After RMS normalization:")
    lines.append("        C42_obs = C42_signal * (rho / (1 + rho))^2")
    lines.append(f"    At -4 dB, rho = {rho:.3f} -> contraction factor = {shrink:.3f}.")
    lines.append("  - Mechanism B (estimator variance, finite L = 128): the sample estimate of M42 averages |r|^4;")
    lines.append("    its variance contains 8th-order noise moments, so the standard deviation of C42_hat does NOT")
    lines.append("    shrink with SNR while its mean contracts toward 0.")
    lines.append("  - Measured on the test set (feature C42), |mean| / std = 'feature SNR':")
    for m in ['QPSK', '16QAM', '64QAM']:
        row = []
        for snr in (-4, 4, 14):
            sel = (test_snrs == snr) & (test_labels == idx[m])
            v = test_feats[sel, c42_idx]
            row.append(f"{snr:+d} dB: mean={v.mean():+.3f}, std={v.std():.3f}, |mu|/sigma={abs(v.mean()) / max(v.std(), 1e-9):.2f}")
        lines.append(f"      {m:6s} " + " | ".join(row))
    lines.append("    When |mu|/sigma < 1 the class cumulant is below its own estimation noise.")

    # ---------------------------------------------------------------- [2] QAM
    hi = (test_snrs >= 10)
    q16, q64 = idx['16QAM'], idx['64QAM']
    v16 = test_feats[hi & (test_labels == q16), c42_idx]
    v64 = test_feats[hi & (test_labels == q64), c42_idx]
    gap = abs(v16.mean() - v64.mean())
    pooled = np.sqrt(0.5 * (v16.var() + v64.var()))
    d_prime = gap / max(pooled, 1e-9)
    bayes_err = _norm_cdf(-d_prime / 2.0) * 100.0

    def cross_rate(preds, true_cls, other_cls):
        sel = hi & (test_labels == true_cls)
        return float(np.mean(preds[sel] == other_cls)) * 100.0

    lines.append("\n[2] 16QAM vs 64QAM AMBIGUITY (SNR >= 10 dB)")
    lines.append("  - Theoretical noise-free cumulants (1 sample/symbol): C42(16QAM) = -0.680, C42(64QAM) = -0.619 (gap 0.061).")
    lines.append(f"  - Measured C42 on test set (4 samples/symbol, pulse shaped, L=128, SNR>=10 dB):")
    lines.append(f"      16QAM: {v16.mean():+.3f} +- {v16.std():.3f} | 64QAM: {v64.mean():+.3f} +- {v64.std():.3f} | gap = {gap:.3f}")
    lines.append(f"  - Discriminability d' = gap / pooled_std = {d_prime:.2f} -> Bayes error for this single feature ~ {bayes_err:.1f}%")
    lines.append(f"  - Observed 16QAM->64QAM / 64QAM->16QAM confusion:")
    lines.append(f"      XGBoost: {cross_rate(xgb_all, q16, q64):.1f}% / {cross_rate(xgb_all, q64, q16):.1f}%")
    lines.append(f"      1D-CNN : {cross_rate(cnn_all, q16, q64):.1f}% / {cross_rate(cnn_all, q64, q16):.1f}%")
    lines.append("  - Proposed remedies (NOT yet evaluated by this project): longer bursts (sigma ~ 1/sqrt(L)),")
    lines.append("    6th-order cumulants, timing recovery to 1 sample/symbol, deeper residual 1D-CNN, more training data.")

    # ---------------------------------------------------------------- [3] FSK
    f_cp, f_gf = idx['CPFSK'], idx['GFSK']
    lines.append("\n[3] CPFSK vs GFSK (SNR >= 10 dB): constant envelope, similar C42, differ in spectrum / phase")
    for feat_name in ['C42', 'spectral_spread', 'phase_diff_std']:
        fi = FEATURE_NAMES.index(feat_name)
        a = test_feats[hi & (test_labels == f_cp), fi]
        b = test_feats[hi & (test_labels == f_gf), fi]
        lines.append(f"  - {feat_name:16s} CPFSK = {a.mean():+.4f} +- {a.std():.4f} | GFSK = {b.mean():+.4f} +- {b.std():.4f}")
    lines.append(f"  - XGBoost CPFSK->GFSK confusion: {cross_rate(xgb_all, f_cp, f_gf):.1f}% | CNN: {cross_rate(cnn_all, f_cp, f_gf):.1f}%")

    # ---------------------------------------------------------------- [4] Honesty
    lines.append("\n[4] STATISTICAL CAVEAT")
    n_per = int(np.sum(test_snrs == -4))
    lines.append(f"  - Each SNR point has n = {n_per} test bursts; the 95% CI on an accuracy near 50% is +-{_ci95(0.5, n_per) * 100:.1f} pts.")
    lines.append("    Differences smaller than the CI (typically at high SNR) are not statistically significant.")
    lines.append("=" * 80)

    report = "\n".join(lines)
    print(report)
    return report
