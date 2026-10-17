"""
================================================================================
EE136IU: Machine Learning for Wireless Communications
Capstone Mini-Project - Topic 02: Spectral & Temporal Feature Engineering
                                 vs. End-to-End Raw I/Q Deep Learning (AMC)
================================================================================
Module: models/ml_classifier.py
Description:
  Branch 1: Feature-Based Machine Learning Models (XGBoost & RBF-SVM).
  Implements feature scaling via StandardScaler (strictly fitted on Train set),
  hyperparameter tuning, model training, persistence, and latency profiling.
================================================================================
"""

import os
import time
import joblib
import numpy as np
from typing import Dict, Tuple, Optional, Any
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score


class FeatureMLClassifier:
    """
    Wrapper for Feature-Based ML classification (XGBoost & RBF-SVM)
    incorporating anti-leakage StandardScaler.
    """
    def __init__(self, model_type: str = 'xgboost', random_state: int = 42):
        self.model_type = model_type.lower()
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.is_fitted = False

        if self.model_type == 'xgboost':
            self.model = XGBClassifier(
                n_estimators=120,
                max_depth=5,
                learning_rate=0.08,
                subsample=0.85,
                colsample_bytree=0.85,
                objective='multi:softprob',
                eval_metric='mlogloss',
                random_state=random_state,
                n_jobs=-1,
                tree_method='hist'
            )
        elif self.model_type in ['svm', 'rbf_svm']:
            self.model = SVC(
                C=10.0,
                kernel='rbf',
                gamma='scale',
                probability=True,
                random_state=random_state
            )
        else:
            raise ValueError(f"Unsupported model_type: {model_type}. Choose 'xgboost' or 'svm'.")

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None
    ) -> 'FeatureMLClassifier':
        """
        Fits the scaler strictly on X_train, then fits the classifier.
        """
        print(f"[*] Fitting StandardScaler and training {self.model_type.upper()} on {X_train.shape[0]} samples...")
        start_time = time.time()

        # Strict anti-leakage scaling: fit ONLY on training set
        X_train_scaled = self.scaler.fit_transform(X_train)

        if self.model_type == 'xgboost' and X_val is not None and y_val is not None:
            X_val_scaled = self.scaler.transform(X_val)
            self.model.fit(
                X_train_scaled,
                y_train,
                eval_set=[(X_val_scaled, y_val)],
                verbose=False
            )
        else:
            self.model.fit(X_train_scaled, y_train)

        train_time = time.time() - start_time
        self.is_fitted = True

        train_preds = self.model.predict(X_train_scaled)
        train_acc = accuracy_score(y_train, train_preds)
        print(f"[OK] {self.model_type.upper()} training complete in {train_time:.2f}s | Train Acc: {train_acc * 100:.2f}%")

        if X_val is not None and y_val is not None:
            X_val_scaled = self.scaler.transform(X_val)
            val_preds = self.model.predict(X_val_scaled)
            val_acc = accuracy_score(y_val, val_preds)
            val_f1 = f1_score(y_val, val_preds, average='macro')
            print(f"    Validation Acc: {val_acc * 100:.2f}% | Val Macro-F1: {val_f1 * 100:.2f}%")

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predicts class labels for unscaled feature inputs."""
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet.")
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predicts class probability distributions for unscaled feature inputs."""
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet.")
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)

    def predict_single_burst(self, feature_vector: np.ndarray) -> Tuple[int, np.ndarray, float]:
        """
        Runs single-burst inference and measures latency in milliseconds.
        
        Args:
            feature_vector: 1D array of shape [20].
            
        Returns:
            predicted_class: integer class index.
            probabilities: 1D array of shape [8].
            latency_ms: inference time in milliseconds.
        """
        if feature_vector.ndim == 1:
            feature_vector = feature_vector.reshape(1, -1)

        t0 = time.perf_counter()
        probs = self.predict_proba(feature_vector)[0]
        latency_ms = (time.perf_counter() - t0) * 1000.0
        pred_class = int(np.argmax(probs))

        return pred_class, probs, latency_ms

    def save(self, filepath: str) -> None:
        """Saves model and scaler to disk."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        joblib.dump({'model': self.model, 'scaler': self.scaler, 'model_type': self.model_type}, filepath)
        print(f"[OK] Saved {self.model_type.upper()} model to: {filepath}")

    @classmethod
    def load(cls, filepath: str) -> 'FeatureMLClassifier':
        """Loads model and scaler from disk."""
        data = joblib.load(filepath)
        instance = cls(model_type=data['model_type'])
        instance.model = data['model']
        instance.scaler = data['scaler']
        instance.is_fitted = True
        return instance


def train_ml_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    save_dir: str = 'saved_models',
    svm_max_train: int = 12000
) -> Tuple[FeatureMLClassifier, FeatureMLClassifier]:
    """
    Trains both XGBoost and RBF-SVM classifiers and saves them to disk.

    XGBoost uses the full training set. The RBF-SVM (cost ~ O(n^2) with probability
    calibration) is trained on a seeded random subset of at most `svm_max_train`
    training bursts; validation and test sets are untouched.
    """
    os.makedirs(save_dir, exist_ok=True)

    # 1. XGBoost Classifier
    xgb_clf = FeatureMLClassifier(model_type='xgboost', random_state=42)
    xgb_clf.fit(X_train, y_train, X_val, y_val)
    xgb_clf.save(os.path.join(save_dir, 'xgboost_amc.joblib'))

    # 2. RBF-SVM Classifier (seeded subsample for tractable training time)
    X_svm, y_svm = X_train, y_train
    if len(X_train) > svm_max_train:
        sub_idx = np.random.RandomState(42).choice(len(X_train), svm_max_train, replace=False)
        X_svm, y_svm = X_train[sub_idx], y_train[sub_idx]
        print(f"[*] RBF-SVM trained on a seeded subset of {svm_max_train} / {len(X_train)} training bursts.")
    svm_clf = FeatureMLClassifier(model_type='svm', random_state=42)
    svm_clf.fit(X_svm, y_svm, X_val, y_val)
    svm_clf.save(os.path.join(save_dir, 'svm_amc.joblib'))

    return xgb_clf, svm_clf
