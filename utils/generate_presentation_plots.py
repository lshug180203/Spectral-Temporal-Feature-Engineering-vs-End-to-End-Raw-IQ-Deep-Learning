"""
================================================================================
EE136IU: Machine Learning for Wireless Communications
Capstone Mini-Project - Topic 02: Spectral & Temporal Feature Engineering
                                 vs. End-to-End Raw I/Q Deep Learning (AMC)
================================================================================
Script: utils/generate_presentation_plots.py
Description:
  Generates 300-DPI high-resolution academic presentation plots for:
    1. assets/slide9_snr_curves.png:
       - Classification Accuracy vs. SNR and Macro-F1 vs. SNR (15 points: -10 to +18 dB, step 2 dB).
       - Compares 1D-CNN (Raw I/Q), XGBoost (Features), and RBF-SVM (Features).
       - Annotated arrows showing 1D-CNN superiority: +17.08% at -4 dB and +27.91% at 0 dB.
       - Clean White Academic styling (#FFFFFF, #E2E8F0 grid, #0284C7, #EA580C, #16A34A).
    2. assets/slide10_confusion_matrices.png:
       - Row-Normalized Confusion Matrices (%) across 3 operational regimes:
         Low SNR (-4 dB), Medium SNR (+4 dB), High SNR (+14 dB) for all 8 classes.
       - Highlights symmetric 16QAM <-> 64QAM ambiguity (~34%) at +14 dB.
================================================================================
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sklearn.metrics import confusion_matrix
import seaborn as sns

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dataset.data_generator import MOD_CLASSES
from models.cnn1d import CNN1DPipeline
from models.ml_classifier import FeatureMLClassifier


def generate_slide9_snr_curves(metrics_json_path: str, save_path: str):
    """Generates dual-panel SNR benchmark plot for Slide 9."""
    with open(metrics_json_path, 'r', encoding='utf-8') as f:
        metrics = json.load(f)

    snrs = np.array(metrics['snr'])
    xgb_acc = np.array(metrics['XGBoost']['accuracy']) * 100.0
    xgb_f1 = np.array(metrics['XGBoost']['macro_f1']) * 100.0
    cnn_acc = np.array(metrics['1D-CNN']['accuracy']) * 100.0
    cnn_f1 = np.array(metrics['1D-CNN']['macro_f1']) * 100.0
    cnn_ci = np.array(metrics['1D-CNN']['acc_ci95']) * 100.0
    svm_acc = np.array(metrics['RBF-SVM']['accuracy']) * 100.0
    svm_f1 = np.array(metrics['RBF-SVM']['macro_f1']) * 100.0

    # Clean White Academic Style
    plt.rcParams['font.family'] = 'DejaVu Sans'
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 6.2), dpi=300)
    fig.patch.set_facecolor('#FFFFFF')

    # Color palette
    color_cnn = '#0284C7'   # Sky/Blue Accent
    color_xgb = '#EA580C'   # Vibrant Orange
    color_svm = '#16A34A'   # Emerald Green

    # -------------------------------------------------------------------------
    # Panel (a): Classification Accuracy vs. SNR
    # -------------------------------------------------------------------------
    ax1 = axes[0]
    ax1.set_facecolor('#FFFFFF')

    # Regime Background Shading
    ax1.axvspan(-10, -2, alpha=0.12, color='#EF4444', label='Noise-Limited Regime (Cumulant Collapse)')
    ax1.axvspan(-2, 6, alpha=0.10, color='#F59E0B', label='Transition Recovery Regime')
    ax1.axvspan(6, 18, alpha=0.10, color='#10B981', label='Asymptotic High-SNR Regime')

    # CI Bands
    ax1.fill_between(snrs, cnn_acc - cnn_ci, cnn_acc + cnn_ci, color=color_cnn, alpha=0.15, linewidth=0)

    # Lines
    ax1.plot(snrs, cnn_acc, marker='o', markersize=7.5, linewidth=2.8, color=color_cnn,
             label='1D-CNN (End-to-End Raw I/Q)')
    ax1.plot(snrs, xgb_acc, marker='s', markersize=7.0, linewidth=2.4, color=color_xgb,
             label='XGBoost (Handcrafted 10-D Features)')
    ax1.plot(snrs, svm_acc, marker='^', markersize=6.5, linewidth=2.0, linestyle='--', color=color_svm,
             label='RBF-SVM (Handcrafted 10-D Features)')

    # Annotations at -4 dB and 0 dB
    # -4 dB: CNN = 48.75%, XGB = 31.67% -> +17.08%
    ax1.annotate(
        "1D-CNN: +17.08% vs. XGBoost\n(Implicit Matched Filtering)",
        xy=(-4, 48.75), xytext=(-9.5, 66),
        arrowprops=dict(facecolor='#0284C7', edgecolor='#0284C7', width=1.5, headwidth=7, shrink=0.08),
        fontsize=9, fontweight='bold', color='#0F172A',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#E0F2FE', edgecolor='#0284C7', alpha=0.95)
    )

    # 0 dB: CNN = 74.58%, XGB = 46.67% -> +27.91%
    ax1.annotate(
        "1D-CNN: +27.91% vs. XGBoost\n(Coherent Temporal Gain)",
        xy=(0, 74.58), xytext=(-5.5, 87),
        arrowprops=dict(facecolor='#0284C7', edgecolor='#0284C7', width=1.5, headwidth=7, shrink=0.08),
        fontsize=9, fontweight='bold', color='#0F172A',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#E0F2FE', edgecolor='#0284C7', alpha=0.95)
    )

    ax1.set_title("(a) Modulation Classification Accuracy vs. SNR", fontsize=12.5, fontweight='bold', pad=12)
    ax1.set_xlabel("Operational Signal-to-Noise Ratio (SNR in dB)", fontsize=10, fontweight='bold')
    ax1.set_ylabel("Classification Accuracy (%)", fontsize=10, fontweight='bold')
    ax1.set_xlim([-10.5, 18.5])
    ax1.set_ylim([10, 102])
    ax1.set_xticks(snrs)
    ax1.grid(True, linestyle='--', color='#E2E8F0', alpha=0.8, linewidth=0.8)
    ax1.legend(loc='lower right', frameon=True, framealpha=0.95, edgecolor='#CBD5E1', fontsize=8.2)

    # -------------------------------------------------------------------------
    # Panel (b): Macro-F1 Score vs. SNR
    # -------------------------------------------------------------------------
    ax2 = axes[1]
    ax2.set_facecolor('#FFFFFF')

    ax2.axvspan(-10, -2, alpha=0.12, color='#EF4444')
    ax2.axvspan(-2, 6, alpha=0.10, color='#F59E0B')
    ax2.axvspan(6, 18, alpha=0.10, color='#10B981')

    ax2.plot(snrs, cnn_f1, marker='o', markersize=7.5, linewidth=2.8, color=color_cnn,
             label='1D-CNN (End-to-End Raw I/Q)')
    ax2.plot(snrs, xgb_f1, marker='s', markersize=7.0, linewidth=2.4, color=color_xgb,
             label='XGBoost (Handcrafted Features)')
    ax2.plot(snrs, svm_f1, marker='^', markersize=6.5, linewidth=2.0, linestyle='--', color=color_svm,
             label='RBF-SVM (Feature Baseline)')

    ax2.set_title("(b) Macro-F1 Score vs. SNR (Balanced Performance)", fontsize=12.5, fontweight='bold', pad=12)
    ax2.set_xlabel("Operational Signal-to-Noise Ratio (SNR in dB)", fontsize=10, fontweight='bold')
    ax2.set_ylabel("Macro-F1 Score (%)", fontsize=10, fontweight='bold')
    ax2.set_xlim([-10.5, 18.5])
    ax2.set_ylim([10, 102])
    ax2.set_xticks(snrs)
    ax2.grid(True, linestyle='--', color='#E2E8F0', alpha=0.8, linewidth=0.8)
    ax2.legend(loc='lower right', frameon=True, framealpha=0.95, edgecolor='#CBD5E1', fontsize=8.5)

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"[OK] Generated Slide 9 SNR Curves at 300 DPI: {save_path}")


def generate_slide10_confusion_matrices(dataset_path: str, cnn_path: str, save_path: str):
    """Generates 3-regime row-normalized confusion matrices for Slide 10."""
    ds = np.load(dataset_path)
    X_test_iq = ds['X_test_iq']
    y_test = ds['y_test']
    snr_test = ds['snr_test']

    cnn = CNN1DPipeline(num_classes=len(MOD_CLASSES))
    cnn.load_checkpoint(cnn_path)

    target_snrs = [-4, 4, 14]
    regime_labels = [
        "Low SNR (-4 dB)\n[Noise-Dominant Regime]",
        "Medium SNR (+4 dB)\n[Transition Recovery]",
        "High SNR (+14 dB)\n[Clean Channel Regime]"
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18.5, 6.2), dpi=300)
    fig.patch.set_facecolor('#FFFFFF')

    for col_idx, snr_val in enumerate(target_snrs):
        mask = (snr_test == snr_val)
        sub_iq = X_test_iq[mask]
        sub_y = y_test[mask]

        preds, _ = cnn.predict_batch(sub_iq)
        cm = confusion_matrix(sub_y, preds, labels=np.arange(len(MOD_CLASSES)))
        cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-12) * 100.0

        ax = axes[col_idx]
        ax.set_facecolor('#FFFFFF')

        # Custom colormap
        sns.heatmap(
            cm_norm,
            annot=True,
            fmt='.1f',
            cmap='Blues',
            xticklabels=MOD_CLASSES,
            yticklabels=MOD_CLASSES,
            cbar=(col_idx == 2),
            cbar_kws={'label': 'Recall (%)'} if col_idx == 2 else None,
            ax=ax,
            vmin=0.0,
            vmax=100.0,
            annot_kws={'size': 9, 'weight': 'bold'},
            linewidths=0.5,
            linecolor='#CBD5E1'
        )

        ax.set_title(f"1D-CNN @ {regime_labels[col_idx]}", fontsize=11.5, fontweight='bold', pad=10)
        ax.set_xlabel("Predicted Modulation Class", fontsize=9.5, fontweight='bold')
        if col_idx == 0:
            ax.set_ylabel("Ground Truth Class", fontsize=9.5, fontweight='bold')
        else:
            ax.set_ylabel("")
        ax.tick_params(axis='x', rotation=40, labelsize=9)
        ax.tick_params(axis='y', rotation=0, labelsize=9)

        # Highlight 16QAM <-> 64QAM ambiguity at High SNR (+14 dB)
        if snr_val == 14:
            idx_16 = MOD_CLASSES.index('16QAM')
            idx_64 = MOD_CLASSES.index('64QAM')

            # Box on (16QAM true -> 64QAM pred)
            rect1 = patches.Rectangle(
                (idx_64, idx_16), 1, 1,
                linewidth=2.5, edgecolor='#DC2626', facecolor='none', linestyle='--'
            )
            # Box on (64QAM true -> 16QAM pred)
            rect2 = patches.Rectangle(
                (idx_16, idx_64), 1, 1,
                linewidth=2.5, edgecolor='#DC2626', facecolor='none', linestyle='--'
            )
            ax.add_patch(rect1)
            ax.add_patch(rect2)

            ax.text(
                0.5, -0.22,
                "★ Red Dashed Boxes: 16QAM ↔ 64QAM Ambiguity (~34.7% Confusion)\nCaused by constellation lattice overlap under short bursts (L = 128)",
                transform=ax.transAxes,
                ha='center', va='top', fontsize=8.5, fontweight='bold', color='#DC2626',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='#FEF2F2', edgecolor='#DC2626', alpha=0.95)
            )

    plt.suptitle("Row-Normalized Confusion Matrices (% Recall per Class) Across Key Noise Regimes",
                 fontsize=14.5, fontweight='bold', y=0.99)
    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"[OK] Generated Slide 10 Confusion Matrices at 300 DPI: {save_path}")


if __name__ == '__main__':
    metrics_path = os.path.join(PROJECT_ROOT, 'results', 'benchmark_metrics.json')
    ds_path = os.path.join(PROJECT_ROOT, 'data', 'amc_dataset_seed42_n400.npz')
    cnn_pt = os.path.join(PROJECT_ROOT, 'saved_models', 'best_cnn1d_model.pt')

    out_slide9 = os.path.join(PROJECT_ROOT, 'assets', 'slide9_snr_curves.png')
    out_slide10 = os.path.join(PROJECT_ROOT, 'assets', 'slide10_confusion_matrices.png')

    generate_slide9_snr_curves(metrics_path, out_slide9)
    generate_slide10_confusion_matrices(ds_path, cnn_pt, out_slide10)
    print("\n[SUCCESS] Generated all presentation assets at 300 DPI!")
