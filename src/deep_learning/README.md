# 🧠 Deep Learning (LSTM-based Sales Forecasting)

## 📌 Overview

TASK 3 extends the sales forecasting system by introducing a **Deep Learning approach using LSTM (Long Short-Term Memory networks)** to model temporal dependencies in historical sales data.

While TASK 2 focused on **feature-based machine learning models (Linear, Random Forest, XGBoost)**, this task focuses on **sequence learning**, where the model learns directly from past values without explicit calendar features.

> ⚠️ Important Note
> This task is **not about outperforming ML at all costs**.
> It is about:
>
> - Understanding sequence models
> - Learning DL pipeline design
> - Comparing ML vs DL objectively

---

## 🎯 Objectives

1. Convert historical daily sales into a time-series format
2. Build an LSTM model using sliding windows
3. Train and evaluate the LSTM model
4. Generate a **365-day future sales forecast**
5. Compare Deep Learning results with ML models from TASK 2

---

## 🧩 Why LSTM?

Traditional ML models require **manual feature engineering** (day, month, weekend, etc.).

LSTM:

- Learns **temporal dependencies implicitly**
- Captures trends, cycles, and short-term memory
- Is suitable for **sequential forecasting problems**

However:

- Requires more data
- Is slower to train
- Often **underperforms tree-based models on tabular data**

This project intentionally demonstrates this trade-off.

---

## 🗂️ Folder Structure (TASK 3)

```text
src/deep_learning/
├── README.md               # This document
├── data_prep.py            # Scaling + train/test preparation
├── sequence_builder.py     # Sliding window sequence creation
├── model.py                # LSTM architecture
├── train.py                # Training orchestration
├── evaluate.py             # RMSE calculation
├── predict.py              # (optional) inference helpers
└── forecast.py             # 1-year autoregressive forecast
```

Execution entry points:

```text
scripts/
├── run_train_dl.py
└── run_forecast_dl.py
```

---

## ⚙️ Environment Requirements

### Python

- Python **3.10+**

### Dependencies

Ensure `requirements.txt` contains:

```txt
tensorflow>=2.15
numpy>=1.24
pandas>=2.0
scikit-learn>=1.3
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Verify TensorFlow:

```bash
python -c "import tensorflow as tf; print(tf.__version__)"
```

> CPU-only TensorFlow is sufficient. GPU is **not required**.

---

## 🔹 Step 1 – Data Preparation for LSTM

### Purpose

Prepare data in a format suitable for sequence models.

### Key Design Decisions

- Aggregate **global daily sales**
- Use **last 30 days → predict next day**
- Apply **MinMax scaling**
- Strict **time-based split** (no shuffling)

### Core Logic

- Load `master_sales.csv`
- Aggregate by date
- Split:

  - ~85% train
  - ~15% test (last ~6 months)

- Build sliding windows

### Output Shapes

```text
X_train: (samples, 30, 1)
y_train: (samples,)
X_test : (samples, 30, 1)
y_test : (samples,)
```

📄 Implemented in:

- `data_prep.py`
- `sequence_builder.py`

> ⚠️ This module is **not run directly**.
> It is invoked internally during training and forecasting.

---

## 🔹 Step 2 – LSTM Model Architecture

### Model Design

```text
Input (30 timesteps)
   ↓
LSTM (64 units, tanh)
   ↓
Dense (1 output)
```

### Design Rationale

- Single LSTM layer (avoids overfitting)
- No dropout initially (baseline clarity)
- MSE loss (regression task)
- Adam optimizer

📄 Implemented in:
`model.py`

---

## 🔹 Step 3 – Model Training & Evaluation

### Runner Script

```bash
py -m scripts.run_train_dl
```

### What Happens Internally

1. Load & scale daily sales
2. Create sliding windows
3. Train LSTM for 20 epochs
4. Validate on last 6 months
5. Print RMSE (scaled)

### Example Output

```text
Epoch 20/20
loss: 0.0076 - val_loss: 0.0076
LSTM RMSE (scaled): 0.0871
```

### Interpretation

- Stable loss and validation loss
- No divergence → no overfitting
- RMSE is on **scaled data**
- Used for **model sanity**, not direct ML comparison

📄 Implemented in:
`train.py`, `evaluate.py`

---

## 🔹 Step 4 – 1-Year Sales Forecast (LSTM)

### Forecast Strategy

- Retrain LSTM on **full historical data**
- Use last 30 days as seed
- Predict **one day at a time**
- Feed prediction back as input (autoregressive)
- Repeat for 365 days
- Inverse-scale predictions

### Runner Script

```bash
py -m scripts.run_forecast_dl
```

### Output File

```text
outputs/sales_forecast_lstm_next_year.csv
```

Example:

```csv
predicted_sales
6254.70
3332.51
1628.17
...
```

📄 Implemented in:
`forecast.py`

---

## ✅ What TASK 3 Achieved

✔ Built a full Deep Learning pipeline
✔ Implemented sequence modeling correctly
✔ Generated a 365-day forecast
✔ Preserved clean modular architecture
✔ Enabled ML vs DL comparison

---

# 📊 ML vs DL – Comparison & Learnings

## Model Performance Summary

| Approach | Model             | RMSE                                      |
| -------- | ----------------- | ----------------------------------------- |
| ML       | Linear Regression | 2258                                      |
| ML       | Random Forest     | 841                                       |
| ML       | **XGBoost**       | **666 (Best)**                            |
| DL       | LSTM              | Higher than XGBoost (scaled RMSE ≈ 0.087) |

---

## Key Observations

### ✅ Why XGBoost Performed Better

- Explicit calendar features
- Handles non-linear interactions well
- Strong bias-variance balance
- Ideal for structured/tabular data

### ⚠️ Why LSTM Didn’t Win

- No explicit weekday / seasonality features
- Limited data length
- More parameters to learn
- Autoregressive error accumulation

---

## Engineering Conclusion (Important)

> **For this problem, feature-based ML is superior to DL.**

This is a **correct and professional conclusion**, not a failure.

---

## When LSTM Would Win

- Much larger datasets
- Multiple correlated time series
- External signals (holidays, promotions)
- Long-term dependency dominance

---

## 🧠 Final Takeaway

| Aspect                  | Machine Learning | Deep Learning |
| ----------------------- | ---------------- | ------------- |
| Feature engineering     | Required         | Not required  |
| Training speed          | Fast             | Slower        |
| Interpretability        | High             | Low           |
| Performance (this task) | ✅ Best          | ⚠️ Acceptable |
| Learning value          | Medium           | **High**      |
