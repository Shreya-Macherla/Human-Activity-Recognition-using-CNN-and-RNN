# Human Activity Recognition — CNN + RNN
[![Python](https://img.shields.io/badge/Python-3.8+-green)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)](https://www.tensorflow.org)
[![Accuracy](https://img.shields.io/badge/Accuracy-95.6%25-brightgreen)]()
[![Spark](https://img.shields.io/badge/ETL-Apache%20Spark-red)]()

## Business Problem

> *Wearable sensors generate continuous streams of movement data — but classifying what a person is doing in real time requires models that understand both spatial and temporal patterns in that data.*

This project classifies 6 human activities (walking, sitting, standing, etc.) from smartphone accelerometer and gyroscope data using CNN (spatial feature extraction) and LSTM (temporal pattern modelling). The ETL pipeline processes raw sensor data at scale using Apache Spark before feeding the deep learning stack.

## Key Outputs

![Model Performance](outputs/01_model_performance.png)

![Sensor Activity Patterns](outputs/02_sensor_patterns.png)

## Model Results

| Model | Test Accuracy | Notes |
|-------|--------------|-------|
| CNN | 93.4% | Strong spatial pattern extraction |
| RNN (LSTM) | 91.2% | Good temporal modelling |
| Basic LSTM | 94.1% | Deeper temporal context |
| **CNN-LSTM (ours)** | **95.6%** | Best: spatial + temporal combined |

## Architecture

```
Raw Sensor Data (accelerometer + gyroscope, 50Hz)
        ↓
Apache Spark ETL (ingestion, normalisation, windowing)
        ↓
Sliding window segments (128 time steps × 6 axes)
        ↓
    ┌──────────────────────────────┐
    │   CNN Block                  │
    │   Conv1D → MaxPool → Dropout │
    └──────────────┬───────────────┘
                   ↓
    ┌──────────────────────────────┐
    │   LSTM Block                 │
    │   LSTM(128) → Dense → Softmax│
    └──────────────┬───────────────┘
                   ↓
          Activity Classification
      (Walking / Sitting / Standing / ...)
```

## Dataset

| Source | Details |
|--------|---------|
| [UCI-HAR](https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones) | 10,299 windows, 561 features, 6 activity classes |
| [WISDM](http://www.cis.fordham.edu/wisdm/dataset.php) | Wrist accelerometer, 6 activities, 36 subjects |

## Quickstart

```bash
git clone https://github.com/Shreya-Macherla/Human-Activity-Recognition-using-CNN-and-RNN
cd Human-Activity-Recognition-using-CNN-and-RNN
pip install -r requirements.txt

python har_analysis.py             # generates model comparison charts and sensor visualisations
jupyter notebook "HAR_Neural Sample code.ipynb"   # full training notebook
```

## Repository Structure

```
Human-Activity-Recognition-using-CNN-and-RNN/
├── har_analysis.py                  # Model comparison + sensor pattern visualisation
├── HAR_Neural Sample code.ipynb     # Full training and evaluation notebook
├── outputs/
│   ├── 01_model_performance.png     # Training curves, confusion matrix, model comparison
│   └── 02_sensor_patterns.png       # Accelerometer signal patterns per activity
├── requirements.txt
└── README.md
```

## Tools

`Python 3.8` `TensorFlow` `Keras` `CNN` `LSTM` `Apache Spark` `scikit-learn` `Matplotlib` `Seaborn`

## Pipeline Stack

`Apache Spark ETL` `HDFS` `Sqoop` `Flume` `Hive` `Apache Zeppelin` `Cassandra`
