"""
Human Activity Recognition (HAR) — CNN, LSTM, and CNN-LSTM on raw sensor signals.

Trains three deep learning architectures on the UCI-HAR dataset (raw inertial
signals: 128 timesteps x 9 channels @ 50Hz) and generates real evaluation
charts: training curves, confusion matrix, per-class recall, model comparison,
and sensor signal patterns per activity.

Usage:
    python har_analysis.py --data_dir UCI_HAR_Dataset --epochs 25

Outputs (written to outputs/):
    01_model_performance.png   training curves, confusion matrix, per-class recall, comparison
    02_sensor_patterns.png     real accelerometer traces per activity
    model_results.json         test accuracy per model
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

import tensorflow as tf
from tensorflow.keras import layers, models

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

ACTIVITIES = ["Walking", "Walking\nUpstairs", "Walking\nDownstairs",
              "Sitting", "Standing", "Laying"]
N_CLASSES = len(ACTIVITIES)

SIGNALS = [
    "body_acc_x", "body_acc_y", "body_acc_z",
    "body_gyro_x", "body_gyro_y", "body_gyro_z",
    "total_acc_x", "total_acc_y", "total_acc_z",
]


# --------------------------------------------------------------------------
# Data loading — raw inertial signals (128 timesteps x 9 channels)
# --------------------------------------------------------------------------
def load_signals(data_dir: str, split: str) -> tuple[np.ndarray, np.ndarray]:
    sig_dir = os.path.join(data_dir, split, "Inertial Signals")
    channels = []
    for sig in SIGNALS:
        path = os.path.join(sig_dir, f"{sig}_{split}.txt")
        channels.append(np.loadtxt(path))
    X = np.stack(channels, axis=-1)  # (n_samples, 128, 9)
    y = np.loadtxt(os.path.join(data_dir, split, f"y_{split}.txt")).astype(int) - 1
    return X, y


def standardize(X_train: np.ndarray, X_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel standardization using training-set statistics only."""
    mean = X_train.mean(axis=(0, 1), keepdims=True)
    std = X_train.std(axis=(0, 1), keepdims=True)
    return (X_train - mean) / std, (X_test - mean) / std


# --------------------------------------------------------------------------
# Model architectures
# --------------------------------------------------------------------------
def build_cnn(input_shape: tuple) -> models.Model:
    return models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv1D(64, kernel_size=5, activation="relu"),
        layers.Conv1D(64, kernel_size=5, activation="relu"),
        layers.MaxPooling1D(2),
        layers.Dropout(0.4),
        layers.Conv1D(128, kernel_size=3, activation="relu"),
        layers.GlobalAveragePooling1D(),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.4),
        layers.Dense(N_CLASSES, activation="softmax"),
    ], name="CNN")


def build_lstm(input_shape: tuple) -> models.Model:
    return models.Sequential([
        layers.Input(shape=input_shape),
        layers.LSTM(96, return_sequences=True),
        layers.Dropout(0.4),
        layers.LSTM(96),
        layers.Dropout(0.4),
        layers.Dense(64, activation="relu"),
        layers.Dense(N_CLASSES, activation="softmax"),
    ], name="LSTM")


def build_cnn_lstm(input_shape: tuple) -> models.Model:
    return models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv1D(64, kernel_size=5, activation="relu"),
        layers.Conv1D(64, kernel_size=5, activation="relu"),
        layers.MaxPooling1D(2),
        layers.Dropout(0.4),
        layers.LSTM(96),
        layers.Dropout(0.4),
        layers.Dense(64, activation="relu"),
        layers.Dense(N_CLASSES, activation="softmax"),
    ], name="CNN-LSTM")


# --------------------------------------------------------------------------
# Plotting
# --------------------------------------------------------------------------
def plot_performance(histories, accuracies, best_name, cm, out_path):
    plt.rcParams.update({"font.family": "DejaVu Sans",
                         "axes.spines.top": False, "axes.spines.right": False})
    fig = plt.figure(figsize=(18, 10))
    fig.suptitle("Human Activity Recognition — CNN / LSTM / CNN-LSTM (real training results)",
                 fontsize=15, fontweight="bold", y=0.99)
    gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.45, wspace=0.38)

    palette = {"CNN": "#3498db", "LSTM": "#e74c3c", "CNN-LSTM": "#2ecc71"}

    ax1 = fig.add_subplot(gs[0, :2])
    for name, hist in histories.items():
        ep = np.arange(1, len(hist["accuracy"]) + 1)
        ax1.plot(ep, hist["accuracy"], color=palette[name], linewidth=2, label=f"{name} Train")
        ax1.plot(ep, hist["val_accuracy"], color=palette[name], linewidth=2,
                 linestyle="--", label=f"{name} Val")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Accuracy")
    ax1.set_title("Training & Validation Accuracy", fontsize=12, fontweight="bold")
    ax1.legend(fontsize=8, ncol=3); ax1.set_ylim(0.4, 1.02)

    ax2 = fig.add_subplot(gs[0, 2:])
    for name, hist in histories.items():
        ep = np.arange(1, len(hist["loss"]) + 1)
        ax2.plot(ep, hist["loss"], color=palette[name], linewidth=2, label=f"{name} Train")
        ax2.plot(ep, hist["val_loss"], color=palette[name], linewidth=2,
                 linestyle="--", label=f"{name} Val")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Loss (Cross-Entropy)")
    ax2.set_title("Training & Validation Loss", fontsize=12, fontweight="bold")
    ax2.legend(fontsize=8, ncol=3)

    ax3 = fig.add_subplot(gs[1, :2])
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    sns.heatmap(cm_pct, ax=ax3, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=ACTIVITIES, yticklabels=ACTIVITIES,
                linewidths=0.5, cbar_kws={"shrink": 0.8})
    ax3.set_xlabel("Predicted Activity"); ax3.set_ylabel("True Activity")
    ax3.set_title(f"Confusion Matrix — {best_name} ({accuracies[best_name]*100:.1f}% accuracy)",
                  fontsize=11, fontweight="bold")
    ax3.tick_params(axis="x", rotation=30, labelsize=8)
    ax3.tick_params(axis="y", rotation=0, labelsize=8)

    ax4 = fig.add_subplot(gs[1, 2])
    recall = np.diag(cm_pct)
    colors_r = ["#2ecc71" if r >= 0.9 else "#f39c12" if r >= 0.8 else "#e74c3c" for r in recall]
    bars = ax4.barh([a.replace("\n", " ") for a in ACTIVITIES], recall,
                    color=colors_r, edgecolor="white")
    ax4.set_xlim(0, 1.12)
    ax4.axvline(0.9, color="gray", linestyle="--", alpha=0.5, linewidth=1)
    ax4.set_title("Per-Class Recall", fontsize=11, fontweight="bold")
    ax4.set_xlabel("Recall")
    for bar, val in zip(bars, recall):
        ax4.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                 f"{val:.3f}", va="center", fontsize=9)

    ax5 = fig.add_subplot(gs[1, 3])
    names = list(accuracies.keys())
    accs = [accuracies[n] for n in names]
    colors_m = ["#2ecc71" if n == best_name else "#95a5a6" for n in names]
    bars = ax5.bar(names, accs, color=colors_m, edgecolor="white")
    ax5.set_ylim(min(accs) - 0.05, 1.0)
    ax5.set_ylabel("Test Accuracy")
    ax5.set_title("Model Comparison", fontsize=11, fontweight="bold")
    ax5.tick_params(axis="x", labelsize=8)
    for bar, val in zip(bars, accs):
        ax5.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                 f"{val:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[PLOT]  {out_path}")


def plot_sensor_patterns(X_raw, y, out_path):
    """Real total-acceleration x-axis traces, one example window per activity."""
    plt.rcParams.update({"font.family": "DejaVu Sans",
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    fig.suptitle("Accelerometer Sensor Data — Real Activity Patterns (UCI-HAR, 50Hz)",
                 fontsize=13, fontweight="bold")
    act_colors = ["#3498db", "#e74c3c", "#2ecc71", "#9b59b6", "#f39c12", "#1abc9c"]
    t = np.arange(128) / 50.0  # 128 timesteps @ 50Hz

    for cls, (ax, col) in enumerate(zip(axes.flatten(), act_colors)):
        idx = np.where(y == cls)[0][0]
        signal = X_raw[idx, :, 6]  # total_acc_x channel
        ax.plot(t, signal, color=col, linewidth=1.2, alpha=0.9)
        ax.axhline(signal.mean(), color="gray", linewidth=0.5)
        ax.set_title(ACTIVITIES[cls].replace("\n", " "), fontsize=11,
                     fontweight="bold", color=col)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Total Acceleration (g)")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[PLOT]  {out_path}")




def train_one(name: str, builder, X_train, y_train, X_test, y_test,
              epochs: int, batch_size: int = 128):
    """Train a single model with resume support (weights + history checkpoints)."""
    os.makedirs("outputs", exist_ok=True)
    weights_path = f"outputs/{name}.weights.h5"
    hist_path = f"outputs/{name}_history.json"

    tf.keras.utils.set_random_seed(SEED)
    model = builder(X_train.shape[1:])
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])

    prev = {"accuracy": [], "val_accuracy": [], "loss": [], "val_loss": []}
    if os.path.exists(weights_path) and os.path.exists(hist_path):
        model.load_weights(weights_path)
        prev = json.load(open(hist_path))
        print(f"[RESUME] {name}: loaded weights, {len(prev['accuracy'])} epochs done")

    done = len(prev["accuracy"])
    remaining = max(0, epochs - done)
    if remaining > 0:
        hist = model.fit(X_train, y_train, validation_data=(X_test, y_test),
                         initial_epoch=done, epochs=done + remaining,
                         batch_size=batch_size, verbose=2)
        for k in prev:
            prev[k] += [float(v) for v in hist.history[k]]
        model.save_weights(weights_path)
        json.dump(prev, open(hist_path, "w"))

    _, test_acc = model.evaluate(X_test, y_test, verbose=0)
    y_pred = model.predict(X_test, verbose=0).argmax(axis=1)
    np.save(f"outputs/{name}_preds.npy", y_pred)
    json.dump({"test_accuracy": float(test_acc)}, open(f"outputs/{name}_acc.json", "w"))
    print(f"[EVAL]  {name} test accuracy: {test_acc*100:.1f}%  ({len(prev['accuracy'])} epochs)")
    return prev, test_acc, y_pred


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="UCI_HAR_Dataset")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--model", default="all",
                        choices=["all", "CNN", "LSTM", "CNN-LSTM", "plot"])
    args = parser.parse_args()

    os.makedirs("outputs", exist_ok=True)
    builders = {"CNN": build_cnn, "LSTM": build_lstm, "CNN-LSTM": build_cnn_lstm}

    print("[DATA]  Loading raw inertial signals...")
    X_train_raw, y_train = load_signals(args.data_dir, "train")
    X_test_raw, y_test = load_signals(args.data_dir, "test")
    print(f"[DATA]  Train: {X_train_raw.shape}  Test: {X_test_raw.shape}")
    X_train, X_test = standardize(X_train_raw, X_test_raw)

    if args.model in builders:
        train_one(args.model, builders[args.model],
                  X_train, y_train, X_test, y_test, args.epochs)
        return

    if args.model == "all":
        for name, build in builders.items():
            print(f"\n[TRAIN] {name}")
            train_one(name, build, X_train, y_train, X_test, y_test, args.epochs)

    # ---- plotting stage (runs for "all" and "plot") ----
    histories, accuracies, predictions = {}, {}, {}
    for name in builders:
        histories[name] = json.load(open(f"outputs/{name}_history.json"))
        accuracies[name] = json.load(open(f"outputs/{name}_acc.json"))["test_accuracy"]
        predictions[name] = np.load(f"outputs/{name}_preds.npy")

    best_name = max(accuracies, key=accuracies.get)
    cm = confusion_matrix(y_test, predictions[best_name], labels=list(range(N_CLASSES)))

    print(f"\n[REPORT] Best model: {best_name}")
    print(classification_report(
        y_test, predictions[best_name],
        target_names=[a.replace("\n", " ") for a in ACTIVITIES]))

    plot_performance(histories, accuracies, best_name, cm,
                     "outputs/01_model_performance.png")
    plot_sensor_patterns(X_test_raw, y_test, "outputs/02_sensor_patterns.png")

    with open("outputs/model_results.json", "w") as f:
        json.dump({k: round(v, 4) for k, v in accuracies.items()}, f, indent=2)
    print("[SAVE]  outputs/model_results.json")

    print("\n[DONE]  HAR analysis complete.")
    for name, acc in accuracies.items():
        marker = "  <-- best" if name == best_name else ""
        print(f"        {name}: {acc*100:.1f}%{marker}")

if __name__ == "__main__":
    main()
