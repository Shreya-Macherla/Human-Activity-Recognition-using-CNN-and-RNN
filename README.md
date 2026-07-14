# Human Activity Recognition using CNN and RNN

Deep learning models that classify six human activities — **Walking, Walking Upstairs, Walking Downstairs, Sitting, Standing, Laying** — from raw smartphone sensor signals (accelerometer + gyroscope), benchmarked against a classical feature-based neural network baseline.

## Results

| Model | Input | Test Accuracy |
|---|---|---|
| **CNN** (1D conv, raw signals) | 128 timesteps × 9 channels | **90.7%** |
| CNN-LSTM hybrid (raw signals) | 128 timesteps × 9 channels | 90.7% |
| LSTM (2-layer, raw signals) | 128 timesteps × 9 channels | 90.6% |
| Dense NN baseline (engineered features) | 561 statistical features | 96.0% |

All results are on the standard **subject-independent** UCI-HAR test split (the 9 test subjects never appear in training), which is a stricter and more realistic evaluation than a random split. These numbers come from the committed `outputs/model_results.json` + per-model history files, produced by the training run that generated the charts below (seed 42, 18 epochs, TensorFlow 2.21). Expect roughly ±1pp variation across TensorFlow versions even with the seed fixed.

![Model performance](outputs/01_model_performance.png)

![Sensor patterns](outputs/02_sensor_patterns.png)

## Key findings

- **Laying and Walking classify near-perfectly** (recall 1.00 each for the best model), and Walking Downstairs is close behind (0.98) — periodic or highly distinctive signals separate cleanly.
- **Sitting ↔ Standing is the hard pair.** Both are near-motionless, so raw inertial signals carry little discriminative information; the largest error mass sits there (recalls 0.85 / 0.78, with each mostly confused for the other — see confusion matrix). This is a well-documented limitation of motion sensors for postural classification.
- **Engineered features still win on this dataset.** The 561 expert-designed statistical features (means, correlations, frequency-domain energy, etc.) outperform end-to-end deep models trained on raw signals at this data scale (~7.3K training windows) — a useful reminder that deep learning needs sufficient data to beat strong feature engineering.
- **The three deep architectures perform equivalently here** (all ~90.6–90.7%). In this run the CNN edges ahead by a hair; across TensorFlow versions the ranking among the three can flip, so no architecture-superiority claim is warranted at this margin.

## Dataset

[UCI-HAR](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones): 30 subjects performing 6 activities with a waist-mounted smartphone. Signals sampled at **50 Hz** and segmented into sliding windows of **128 timesteps (2.56 s, 50% overlap)** across **9 channels** (3-axis body acceleration, 3-axis angular velocity, 3-axis total acceleration). 7,352 training / 2,947 test windows.

## Repository structure

```
├── har_analysis.py            # Trains CNN, LSTM, CNN-LSTM on raw signals; generates all charts
├── HAR_Neural Sample code.ipynb  # EDA + dense-NN baseline on 561 engineered features
├── outputs/                   # Real training curves, confusion matrix, sensor visualisations
├── requirements.txt
└── README.md
```

## Reproducing the results

```bash
pip install -r requirements.txt

# Download UCI-HAR and place it as ./UCI_HAR_Dataset (see link above), then:
python har_analysis.py --model all --epochs 18

# Or train one model at a time (supports checkpoint resume):
python har_analysis.py --model CNN --epochs 18
python har_analysis.py --model LSTM --epochs 18
python har_analysis.py --model CNN-LSTM --epochs 18
python har_analysis.py --model plot        # regenerate charts from saved results
```

Training is reproducible (fixed seed 42). All charts in `outputs/` are generated from actual training runs — never simulated.

## Architectures

- **CNN** — two Conv1D(64, k=5) layers → MaxPool → Conv1D(128, k=3) → GlobalAveragePooling → Dense(64), dropout 0.4
- **LSTM** — two stacked LSTM(96) layers → Dense(64), dropout 0.4
- **CNN-LSTM** — Conv1D front-end for local feature extraction and temporal downsampling → LSTM(96) for sequence modelling
- **Baseline** — feedforward network on the 561 pre-extracted features, with StandardScaler + PCA preprocessing (see notebook)

All models: Adam (lr = 1e-3), sparse categorical cross-entropy, batch size 128.
