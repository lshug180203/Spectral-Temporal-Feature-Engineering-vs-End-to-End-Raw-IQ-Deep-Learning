"""
================================================================================
EE136IU: Machine Learning for Wireless Communications
Capstone Mini-Project - Topic 02: Spectral & Temporal Feature Engineering
                                 vs. End-to-End Raw I/Q Deep Learning (AMC)
================================================================================
Module: models/cnn1d.py
Description:
  Branch 2: End-to-End 1D Convolutional Neural Network (PyTorch) operating directly
  on raw complex baseband I/Q observation tensors [N_batch, 2, 128].
  Incorporates 4 Conv1D blocks with BatchNorm, ReLU, and pooling, followed by
  regularized dense projection, validation checkpointing, and latency profiling.
================================================================================
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Tuple, Dict, Optional, List
from sklearn.metrics import accuracy_score, f1_score

from dataset.data_generator import AMCTensorDataset


class AMC_1DCNN(nn.Module):
    """
    End-to-End 1D-CNN Receiver for Automatic Modulation Classification.
    Accepts raw baseband I/Q tensors of shape [N, 2, 128].
    """
    def __init__(self, num_classes: int = 8, in_channels: int = 2):
        super(AMC_1DCNN, self).__init__()

        # Conv Block 1: Large receptive field to capture long symbol pulse transitions
        self.block1 = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=7, stride=1, padding=3, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2)  # 128 -> 64
        )

        # Conv Block 2: Intermediate features & carrier phase tracking
        self.block2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2)  # 64 -> 32
        )

        # Conv Block 3: Deep constellation pattern representation
        self.block3 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2)  # 32 -> 16
        )

        # Conv Block 4: High-level spatial/temporal feature aggregation
        self.block4 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1)  # 16 -> 1
        )

        # Regularized Dense Classification Head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.3),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x: Real tensor of shape [batch_size, 2, 128]
        Returns:
            logits: Tensor of shape [batch_size, num_classes]
        """
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        logits = self.classifier(x)
        return logits


class CNN1DPipeline:
    """Wrapper for training, evaluating, and checkpointing the 1D-CNN model."""
    def __init__(self, num_classes: int = 8, device: Optional[str] = None):
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self.model = AMC_1DCNN(num_classes=num_classes).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.num_classes = num_classes

    def train_pipeline(
        self,
        X_train_iq: np.ndarray,
        y_train: np.ndarray,
        X_val_iq: np.ndarray,
        y_val: np.ndarray,
        epochs: int = 15,
        batch_size: int = 64,
        lr: float = 1e-3,
        save_path: str = 'saved_models/best_cnn1d_model.pt'
    ) -> Dict[str, List[float]]:
        """
        Trains the 1D-CNN with early checkpointing on validation Macro-F1.
        """
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)

        train_dataset = AMCTensorDataset(X_train_iq, y_train)
        val_dataset = AMCTensorDataset(X_val_iq, y_val)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)

        history = {'train_loss': [], 'val_loss': [], 'val_acc': [], 'val_f1': []}
        best_val_f1 = -1.0

        print(f"[*] Training AMC 1D-CNN on device: {self.device} ({epochs} epochs, batch_size={batch_size})...")
        start_time = time.time()

        for epoch in range(1, epochs + 1):
            self.model.train()
            running_loss = 0.0

            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * batch_x.size(0)

            epoch_train_loss = running_loss / len(train_dataset)

            # Validation Evaluation
            val_loss, val_acc, val_f1, _, _ = self.evaluate(val_loader)
            scheduler.step(val_f1)

            history['train_loss'].append(epoch_train_loss)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            history['val_f1'].append(val_f1)

            # Checkpoint best model based on validation Macro-F1
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                torch.save(self.model.state_dict(), save_path)
                saved_flag = " [BEST CHECKPOINT SAVED]"
            else:
                saved_flag = ""

            print(f"  Epoch [{epoch:02d}/{epochs:02d}] Train Loss: {epoch_train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}% | Val Macro-F1: {val_f1*100:.2f}%{saved_flag}")

        total_time = time.time() - start_time
        print(f"[OK] 1D-CNN Training finished in {total_time:.2f}s | Best Val Macro-F1: {best_val_f1*100:.2f}%")

        # Load best weights
        if os.path.exists(save_path):
            self.model.load_state_dict(torch.load(save_path, map_location=self.device))

        return history

    def evaluate(self, data_loader: DataLoader) -> Tuple[float, float, float, np.ndarray, np.ndarray]:
        """Evaluates model over DataLoader, returning loss, accuracy, Macro-F1, preds, and probs."""
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_probs = []
        all_targets = []

        with torch.no_grad():
            for batch_x, batch_y in data_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                outputs = self.model(batch_x)
                loss = self.criterion(outputs, batch_y)
                total_loss += loss.item() * batch_x.size(0)

                probs = torch.softmax(outputs, dim=1).cpu().numpy()
                preds = np.argmax(probs, axis=1)

                all_probs.append(probs)
                all_preds.append(preds)
                all_targets.append(batch_y.cpu().numpy())

        total_samples = sum(len(p) for p in all_preds)
        avg_loss = total_loss / max(total_samples, 1)

        all_preds = np.concatenate(all_preds, axis=0)
        all_probs = np.concatenate(all_probs, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)

        acc = accuracy_score(all_targets, all_preds)
        f1 = f1_score(all_targets, all_preds, average='macro')

        return avg_loss, acc, f1, all_preds, all_probs

    def predict_batch(self, iq_batch: np.ndarray, batch_size: int = 1024) -> Tuple[np.ndarray, np.ndarray]:
        """
        Batched inference on raw I/Q tensors [N, 2, 128].

        Returns:
            preds: integer class indices, shape [N]
            probs: softmax probabilities, shape [N, num_classes]
        """
        self.model.eval()
        all_probs = []
        with torch.no_grad():
            for start in range(0, len(iq_batch), batch_size):
                chunk = torch.from_numpy(iq_batch[start:start + batch_size]).float().to(self.device)
                all_probs.append(torch.softmax(self.model(chunk), dim=1).cpu().numpy())
        probs = np.concatenate(all_probs, axis=0)
        return np.argmax(probs, axis=1), probs

    def predict_single_burst(self, iq_burst: np.ndarray) -> Tuple[int, np.ndarray, float]:
        """
        Runs single-burst inference on raw I/Q array [2, 128] and measures latency in ms.
        
        Returns:
            predicted_class: integer index
            probabilities: shape [8]
            latency_ms: time in milliseconds
        """
        self.model.eval()
        if iq_burst.ndim == 2:
            burst_tensor = torch.from_numpy(iq_burst).unsqueeze(0).float().to(self.device)
        else:
            burst_tensor = torch.from_numpy(iq_burst).float().to(self.device)

        t0 = time.perf_counter()
        with torch.no_grad():
            logits = self.model(burst_tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        latency_ms = (time.perf_counter() - t0) * 1000.0

        pred_class = int(np.argmax(probs))
        return pred_class, probs, latency_ms

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Loads state dictionary checkpoint from path."""
        self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        self.model.eval()
        print(f"[OK] Loaded 1D-CNN checkpoint from: {checkpoint_path}")
