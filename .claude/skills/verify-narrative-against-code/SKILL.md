---
name: verify-narrative-against-code
description: Load before editing README.md, har_analysis.py, the notebook, or any claim about model architecture/accuracy in this repo. Trigger on keywords — CNN, RNN, LSTM, accuracy, subject-independent, Apache Spark, HDFS, simulated.
---

# Every number in this repo must trace to a real training run

## History (resolved — do not re-fix)

This repo previously had two layers of fabrication, both repaired:
1. `har_analysis.py` was a chart generator that simulated all results with
   `np.random` (95.6% "CNN-LSTM accuracy" that was never trained). It is now a
   real training script (CNN / LSTM / CNN-LSTM on raw UCI-HAR inertial signals,
   seeded, with checkpoint/resume).
2. The README claimed CNN/LSTM architectures and an Apache Spark/HDFS pipeline
   that didn't exist. The CNN/LSTM/CNN-LSTM models now genuinely exist in
   `har_analysis.py`; the Spark/HDFS pipeline claims stay removed — never
   re-add them.

## Current verified numbers (independent re-run, TF 2.21, seed 42, 18 epochs)

| Model | Test accuracy | Source |
|---|---|---|
| CNN (raw signals) | 90.7% | `outputs/model_results.json` + `CNN_history.json` |
| CNN-LSTM (raw signals) | 90.7% | same |
| LSTM (raw signals) | 90.6% | same |
| Dense NN baseline (561 features) | 96.0% | `HAR_Neural Sample code.ipynb`, cell 58 |

Evaluation is the standard subject-independent UCI-HAR split. Rules of thumb:
- Numbers drift ~±1pp across TensorFlow versions even with the seed fixed. If a
  re-run changes them, update README, `outputs/model_results.json`, the history
  files, and this table in the same commit.
- The three deep models are statistically tied — never claim one architecture
  "wins" at a sub-1pp margin.
- The 96.0% belongs to the feature-engineered dense baseline (notebook), NOT to
  the CNN/LSTM raw-signal models. Resume/profile wording must not attach 96% to
  the deep models.
- Sitting↔Standing confusion (recalls ~0.85/0.78) is the known weakness — it's a
  documented sensor limitation, keep it visible rather than hiding it.

## Rules when editing

1. New accuracy claim → run `har_analysis.py` (or the notebook) first, copy the
   number from its output, and commit the regenerated
   `model_results.json`/history artifacts in the same commit.
2. `outputs/*.weights.h5`, `*_preds.npy`, `*_acc.json`, and the dataset dir are
   gitignored — don't force-add them.
3. Chart 2 (`02_sensor_patterns.png`) now plots REAL test-set windows — if you
   touch `plot_sensor_patterns`, keep it reading from the loaded dataset, never
   synthetic waveforms.

## When NOT to use this skill

Non-metric edits (typos, structure lists, .gitignore) don't need the full trace
check — but any edit that adds, changes, or rephrases a performance or
architecture claim does.
