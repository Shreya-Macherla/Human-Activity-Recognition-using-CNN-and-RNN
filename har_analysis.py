"""
Human Activity Recognition (HAR) — dense feedforward model analysis & visualisation.

Renders the ACTUAL results produced by "HAR_Neural Sample code.ipynb" — the per-epoch
train/val accuracy and loss for the best model, the real per-class precision/recall/f1
from sklearn's classification_report, and the real test accuracy of all three models
that notebook trained (including the one that failed to converge). Every number below
was copied verbatim from that notebook's executed cell outputs, not computed here and
not simulated. See README.md for how to re-verify these against the notebook yourself.

Chart 2 (sensor waveforms) is clearly-labelled illustrative signal shape, not measured
sensor data — no waveform recordings exist in this repo.
"""

from __future__ import annotations

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

os.makedirs("outputs", exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.top": False, "axes.spines.right": False})

ACTIVITIES = ["Walking", "Walking\nUpstairs", "Walking\nDownstairs",
              "Sitting", "Standing", "Laying"]
N_CLASSES = len(ACTIVITIES)

# ---- Real per-epoch history for the best model (FF-NN), copied from notebook cell 46 --
EPOCHS = list(range(1, 19))  # notebook trained 18 epochs (early stop before configured 22)
TRAIN_ACC = [0.5562, 0.81, 0.9616, 0.9867, 0.9928, 0.9935, 0.9939, 0.9952, 0.9961, 0.9969,
             0.998, 0.9992, 0.9992, 0.9982, 0.9996, 0.9999, 0.9999, 1.0]
VAL_ACC = [0.5989, 0.8537, 0.9403, 0.9488, 0.9528, 0.9545, 0.9549, 0.9501, 0.9511, 0.9542,
           0.9566, 0.9549, 0.9634, 0.96, 0.9579, 0.9613, 0.9562, 0.9596]
TRAIN_LOSS = [1.5607, 0.6148, 0.1361, 0.0459, 0.0279, 0.0226, 0.0193, 0.0142, 0.0127, 0.0105,
              0.007, 0.005, 0.0043, 0.0055, 0.0029, 0.0019, 0.0015, 0.0012]
VAL_LOSS = [1.0408, 0.3351, 0.1556, 0.1443, 0.1339, 0.1344, 0.1419, 0.1569, 0.1642, 0.1437,
            0.1442, 0.1502, 0.1332, 0.1518, 0.1543, 0.1471, 0.1661, 0.1646]

# ---- Real per-class metrics for the best model, copied from notebook cell 58 ----------
# (sklearn classification_report on the 2947-row UCI-HAR test set)
PRECISION = [1.00, 0.94, 0.91, 0.95, 0.99, 0.98]
RECALL    = [0.98, 0.90, 0.98, 1.00, 0.96, 0.95]
F1        = [0.99, 0.92, 0.94, 0.97, 0.97, 0.96]

# ---- Real test accuracy for all three models actually trained in the notebook --------
# Model 3 (LR=0.1) never converges — this is a genuine result, not an error to hide.
MODEL_NAMES = ["FF-NN\n(best)", "ANN", "MLP\n(LR too high)"]
MODEL_TEST_ACC = [0.96, 0.94, 0.18]  # cells 58, 70, 76

# ---- Chart 1: real training curves + real per-class metrics + real model comparison ---
fig = plt.figure(figsize=(18, 10))
fig.suptitle("Human Activity Recognition — Dense Feedforward Model (real results)",
             fontsize=15, fontweight="bold", y=0.99)
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

ax1 = fig.add_subplot(gs[0, :2])
ax1.plot(EPOCHS, TRAIN_ACC, color="#2ecc71", linewidth=2, label="Train")
ax1.plot(EPOCHS, VAL_ACC, color="#2ecc71", linewidth=2, linestyle="--", label="Validation")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Accuracy")
ax1.set_title("Training & Validation Accuracy — best model (FF-NN)", fontsize=12, fontweight="bold")
ax1.legend(fontsize=9)
ax1.set_ylim(0.4, 1.02)

ax2 = fig.add_subplot(gs[0, 2])
ax2.plot(EPOCHS, TRAIN_LOSS, color="#3498db", linewidth=2, label="Train")
ax2.plot(EPOCHS, VAL_LOSS, color="#3498db", linewidth=2, linestyle="--", label="Validation")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Loss")
ax2.set_title("Training & Validation Loss", fontsize=11, fontweight="bold")
ax2.legend(fontsize=8)

ax3 = fig.add_subplot(gs[1, :2])
x = np.arange(N_CLASSES)
width = 0.25
ax3.bar(x - width, PRECISION, width, label="Precision", color="#3498db")
ax3.bar(x, RECALL, width, label="Recall", color="#2ecc71")
ax3.bar(x + width, F1, width, label="F1", color="#f39c12")
ax3.set_xticks(x)
ax3.set_xticklabels([a.replace("\n", " ") for a in ACTIVITIES], fontsize=8, rotation=15)
ax3.set_ylim(0, 1.15)
ax3.set_title("Per-Class Precision / Recall / F1 — best model, 2947-row test set", fontsize=11, fontweight="bold")
ax3.legend(fontsize=8, ncol=3)

ax4 = fig.add_subplot(gs[1, 2])
colors_m = ["#2ecc71", "#95a5a6", "#e74c3c"]
bars = ax4.bar(MODEL_NAMES, MODEL_TEST_ACC, color=colors_m, edgecolor="white")
ax4.set_ylim(0, 1.05)
ax4.set_ylabel("Test Accuracy")
ax4.set_title("All 3 models actually trained\n(third failed — LR=0.1 too high)", fontsize=10, fontweight="bold")
for bar, val in zip(bars, MODEL_TEST_ACC):
    ax4.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
              f"{val:.2f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

plt.savefig("outputs/01_model_performance.png", dpi=150, bbox_inches="tight")
plt.close()
print("[PLOT]  outputs/01_model_performance.png (built from real notebook results)")

# ---- Chart 2: illustrative sensor waveform shapes (NOT measured data) -----------------
fig, axes = plt.subplots(2, 3, figsize=(16, 8))
fig.suptitle("Illustrative Accelerometer Waveform Shapes (synthetic — not measured sensor data)",
             fontsize=13, fontweight="bold")

rng = np.random.default_rng(42)
t = np.linspace(0, 5, 500)
activity_signals = {
    "Walking":       (0.8 * np.sin(2 * np.pi * 1.8 * t), 0.4),
    "Upstairs":      (0.7 * np.sin(2 * np.pi * 1.5 * t), 0.6),
    "Downstairs":    (0.9 * np.sin(2 * np.pi * 1.3 * t), 0.5),
    "Sitting":       (0.05 * rng.standard_normal(500), 0.05),
    "Standing":      (0.08 * rng.standard_normal(500), 0.06),
    "Laying":        (0.03 * rng.standard_normal(500), 0.03),
}

axes_flat = axes.flatten()
act_labels_simple = ["Walking", "Upstairs", "Downstairs", "Sitting", "Standing", "Laying"]
act_colors = ["#3498db", "#e74c3c", "#2ecc71", "#9b59b6", "#f39c12", "#1abc9c"]

for ax, (label, col) in zip(axes_flat, zip(act_labels_simple, act_colors)):
    base_signal, noise_std = activity_signals[label]
    signal = base_signal + noise_std * rng.standard_normal(500)
    ax.plot(t, signal, color=col, linewidth=1.2, alpha=0.9)
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.set_title(label, fontsize=11, fontweight="bold", color=col)
    ax.set_xlabel("Time (s)" if ax in axes[1] else "")
    ax.set_ylabel("Acceleration (g) — illustrative" if ax in axes[:, 0] else "")
    ax.set_ylim(-2.5, 2.5)

plt.tight_layout()
plt.savefig("outputs/02_sensor_patterns.png", dpi=150, bbox_inches="tight")
plt.close()
print("[PLOT]  outputs/02_sensor_patterns.png (illustrative shapes, clearly labelled)")

print("\n[DONE]  HAR analysis complete.")
print("        Best model (FF-NN, Dense layers only) test accuracy: 96%  (real, from notebook cell 58)")
print("        Dataset: UCI-HAR (real, loaded from CSV in the notebook)")
print("        No CNN/LSTM/Spark/HDFS pipeline exists in this repo — see README.")
