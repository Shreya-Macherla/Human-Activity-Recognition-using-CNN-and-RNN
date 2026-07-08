---
name: verify-narrative-against-code
description: Load before editing README.md, har_analysis.py, or the training notebook, or any claim about model architecture/accuracy in this repo. Trigger on keywords — CNN, RNN, LSTM, Conv1D, accuracy, Apache Spark, HDFS.
---

# har_analysis.py fabricates every result; the real notebook doesn't match the repo's own name

Verified directly (not from an unverified claim): `har_analysis.py`'s docstring says "Simulates
training." It never loads data or trains a model — `numpy.random` generates the training curves,
confusion matrix, and per-model accuracy numbers (95.6%, 93.4%, etc.) that the README embeds as
`outputs/01_model_performance.png`.

Separately, the real notebook (`HAR_Neural Sample code.ipynb`) does load the actual UCI-HAR
dataset and does train real models — but:

```bash
grep -o "Conv1D\|Conv2D\|LSTM\|GRU\|Dense(" "HAR_Neural Sample code.ipynb" | sort | uniq -c
#   16 Dense(
#    1 GRU        (one incidental mention, not an actual layer in the trained models)
```

All three trained models are plain `Dense` feedforward networks. There is no `Conv1D` and no
`LSTM` anywhere, despite the repo name, README architecture diagram, and Tools badges all
claiming a CNN+LSTM pipeline with an Apache Spark/HDFS/Sqoop/Flume/Hive/Cassandra stack that
appears nowhere in any file in this repo.

The real notebook's genuine result — 96.0% test accuracy from a well-executed comparison of three
dense architectures, with real EDA, PCA, and per-class evaluation — is solid and defensible on
its own. It doesn't need the simulated chart or the CNN/LSTM claim to be a good result.

## Before touching this repo again

1. Replace `outputs/01_model_performance.png` and `02_sensor_patterns.png` with real outputs
   saved from the notebook (it already produces training curves, a confusion matrix heatmap, and
   classification reports).
2. Either add real `Conv1D`/`LSTM` models to the notebook so the repo matches its own name, or
   rename the repo/README/architecture diagram to describe what's actually there (a dense-network
   comparison).
3. Reconcile the accuracy number everywhere it appears — resume, README (95.6% simulated), and
   notebook (96.0% real) currently disagree; use the real, traceable 96.0% figure everywhere.
4. Drop the Apache Spark/HDFS/Sqoop/Flume/Hive/Zeppelin/Cassandra "Pipeline Stack" badges — none
   of that appears in any code file here.

## When NOT to use this skill

This repo's real notebook is legitimate work worth preserving as-is technically — this skill is
about the gap between the marketing layer (README, badges, `har_analysis.py`) and what the
notebook actually does, not about the modeling approach itself.
