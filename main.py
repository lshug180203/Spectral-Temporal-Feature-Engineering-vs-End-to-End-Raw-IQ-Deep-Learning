"""
================================================================================
EE136IU: Machine Learning for Wireless Communications
Capstone Mini-Project - Topic 02: Spectral & Temporal Feature Engineering
                                 vs. End-to-End Raw I/Q Deep Learning (AMC)
================================================================================
Script: main.py
Description:
  End-to-End Execution Pipeline (first run: a few minutes on CPU; later runs reuse cached data):
    1. Deterministic seed initialization (seed=42).
    2. Baseband I/Q burst generation & per-burst RMS normalization (L=128, 8 classes, 15 SNRs).
    3. Anti-leakage partition 70/15/15 with INDEPENDENT seeds per split (train=42, val=43, test=44).
    4. Higher-order cumulant, differential phase, & FFT spectral feature extraction.
    5. Training Branch 1 (StandardScaler + XGBoost & RBF-SVM).
    6. Training Branch 2 (End-to-End PyTorch 1D-CNN with validation checkpointing).
    7. Calibrated 15-point SNR Grid benchmark evaluation.
    8. Publication-quality figure generation (Accuracy/F1 vs SNR & Confusion Matrices at -4dB, +4dB, +14dB).
    9. Root-cause physical failure diagnosis synthesis.
================================================================================
"""

import os
import sys
import time
import json
import numpy as np

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dataset.data_generator import (
    generate_amc_dataset,
    set_seed,
    MOD_CLASSES,
    CALIBRATED_SNR_GRID
)
from utils.feature_extractor import extract_features_batch
from utils.metrics_plots import (
    evaluate_models_per_snr,
    plot_snr_curves,
    plot_training_curves,
    plot_confusion_matrices_at_regimes,
    print_failure_analysis_report
)
from models.ml_classifier import FeatureMLClassifier, train_ml_models
from models.cnn1d import CNN1DPipeline


def run_pipeline(
    bursts_per_class_snr: int = 400,
    cnn_epochs: int = 12,
    seed: int = 42,
    cache_dataset: bool = True
) -> None:
    """Executes the complete AMC Topic 02 engineering pipeline."""
    total_start_time = time.time()
    set_seed(seed)

    print("=" * 80)
    print(" EE136IU CAPSTONE MINI-PROJECT: AUTOMATIC MODULATION CLASSIFICATION (TOPIC 02)")
    print(" Spectral & Temporal Feature Engineering vs. End-to-End Raw I/Q Deep Learning")
    print("=" * 80)

    data_dir = os.path.join(PROJECT_ROOT, 'data')
    saved_models_dir = os.path.join(PROJECT_ROOT, 'saved_models')
    results_dir = os.path.join(PROJECT_ROOT, 'results')
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(saved_models_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    tag = f'seed{seed}_n{bursts_per_class_snr}'
    cache_path = os.path.join(data_dir, f'amc_dataset_{tag}.npz') if cache_dataset else None

    # Step 1: Baseband Signal Generation & Anti-Leakage Partition
    print("\n--- STEP 1: PHYSICAL BASEBAND SIGNAL GENERATION & RMS NORMALIZATION ---")
    dataset = generate_amc_dataset(
        bursts_per_class_snr=bursts_per_class_snr,
        burst_length=128,
        snr_grid=CALIBRATED_SNR_GRID,
        seed=seed,
        cache_path=cache_path
    )

    X_train_iq = dataset['X_train_iq']
    y_train = dataset['y_train']
    snr_train = dataset['snr_train']

    X_val_iq = dataset['X_val_iq']
    y_val = dataset['y_val']
    snr_val = dataset['snr_val']

    X_test_iq = dataset['X_test_iq']
    y_test = dataset['y_test']
    snr_test = dataset['snr_test']

    # Step 2: Vectorized Physical Feature Extraction
    print("\n--- STEP 2: PHYSICAL DOMAIN FEATURE EXTRACTION (CUMULANTS + FFT) ---")
    feat_cache = os.path.join(data_dir, f'features_{tag}.npz')
    if os.path.exists(feat_cache):
        print(f"[*] Loading cached features from: {feat_cache}")
        f_data = np.load(feat_cache)
        X_train_feat = f_data['X_train_feat']
        X_val_feat = f_data['X_val_feat']
        X_test_feat = f_data['X_test_feat']
    else:
        t_feat0 = time.time()
        print(f"[*] Extracting features for Train ({len(X_train_iq)}), Val ({len(X_val_iq)}), Test ({len(X_test_iq)})...")
        X_train_feat = extract_features_batch(X_train_iq)
        X_val_feat = extract_features_batch(X_val_iq)
        X_test_feat = extract_features_batch(X_test_iq)
        np.savez_compressed(
            feat_cache,
            X_train_feat=X_train_feat,
            X_val_feat=X_val_feat,
            X_test_feat=X_test_feat
        )
        print(f"[OK] Feature extraction complete in {time.time() - t_feat0:.2f}s (Cached to disk)")

    # Step 3: Branch 1 - Feature-Based ML Training (XGBoost & RBF-SVM)
    print("\n--- STEP 3: BRANCH 1 - FEATURE-BASED ML (XGBOOST & RBF-SVM) ---")
    xgb_clf, svm_clf = train_ml_models(
        X_train_feat, y_train,
        X_val_feat, y_val,
        save_dir=saved_models_dir
    )

    # Step 4: Branch 2 - End-to-End Raw I/Q 1D-CNN Training
    print("\n--- STEP 4: BRANCH 2 - END-TO-END RAW I/Q 1D-CNN (PYTORCH) ---")
    cnn_pipeline = CNN1DPipeline(num_classes=len(MOD_CLASSES))
    cnn_ckpt_path = os.path.join(saved_models_dir, 'best_cnn1d_model.pt')
    history = cnn_pipeline.train_pipeline(
        X_train_iq, y_train,
        X_val_iq, y_val,
        epochs=cnn_epochs,
        batch_size=64,
        lr=1e-3,
        save_path=cnn_ckpt_path
    )
    plot_training_curves(history, save_path=os.path.join(results_dir, 'cnn_training_curves.png'))

    # Step 5: Calibrated 15-Point SNR Grid Benchmark Evaluation
    print("\n--- STEP 5: CALIBRATED 15-POINT SNR EVALUATION & BENCHMARKING ---")
    snr_results = evaluate_models_per_snr(
        test_iq=X_test_iq,
        test_feats=X_test_feat,
        test_labels=y_test,
        test_snrs=snr_test,
        xgb_clf=xgb_clf,
        svm_clf=svm_clf,
        cnn_pipeline=cnn_pipeline,
        snr_grid=CALIBRATED_SNR_GRID
    )

    # Step 6: Visualization - SNR Benchmark Curves
    print("\n--- STEP 6: PUBLICATION-QUALITY VISUALIZATION GENERATION ---")
    curve_plot_path = os.path.join(results_dir, 'snr_benchmark_curves.png')
    plot_snr_curves(snr_results, save_path=curve_plot_path)

    # Step 7: Visualization - Row-Normalized Confusion Matrices at 3 Regimes (-4 dB, +4 dB, +14 dB)
    cm_plot_path = os.path.join(results_dir, 'confusion_matrices_regimes.png')
    plot_confusion_matrices_at_regimes(
        test_iq=X_test_iq,
        test_feats=X_test_feat,
        test_labels=y_test,
        test_snrs=snr_test,
        xgb_clf=xgb_clf,
        cnn_pipeline=cnn_pipeline,
        target_snrs=(-4, 4, 14),
        save_path=cm_plot_path
    )

    # Step 8: Physical Root-Cause Failure Diagnosis
    print("\n--- STEP 8: BASEBAND PHYSICAL ROOT-CAUSE FAILURE ANALYSIS ---")
    report_text = print_failure_analysis_report(
        test_iq=X_test_iq,
        test_feats=X_test_feat,
        test_labels=y_test,
        test_snrs=snr_test,
        xgb_clf=xgb_clf,
        cnn_pipeline=cnn_pipeline
    )
    with open(os.path.join(results_dir, 'failure_analysis_report.txt'), 'w', encoding='utf-8') as f:
        f.write(report_text)

    # Save numeric benchmark results
    with open(os.path.join(results_dir, 'benchmark_metrics.json'), 'w', encoding='utf-8') as f:
        json.dump(snr_results, f, indent=2)

    total_duration = time.time() - total_start_time
    print("\n" + "=" * 80)
    print(f"[OK] END-TO-END PIPELINE SUCCESSFULLY COMPLETED IN {total_duration:.2f} SECONDS!")
    print(f"    - All Trained Models Saved In: {saved_models_dir}/")
    print(f"    - High-Res Figures Saved In:   {results_dir}/")
    print(f"    - Ready for Live Defense Demo: streamlit run demo/app.py")
    print("=" * 80)


if __name__ == '__main__':
    run_pipeline(bursts_per_class_snr=400, cnn_epochs=12, seed=42)
