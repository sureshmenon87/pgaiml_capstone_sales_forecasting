## Machine Learning Forecasting

### 1. Objective

The objective of TASK 2 is to **build and evaluate machine learning models that can predict future daily sales using historical data and calendar-based features**.

Specifically, this task aims to:

- Transform historical sales data into a supervised learning problem
- Learn the relationship between calendar variables and sales demand
- Compare multiple machine learning algorithms objectively
- Select the model with the best generalization performance
- Use the selected model to forecast sales for the next one year

---

### 2. Problem Formulation

This task is formulated as a **regression problem**:

- **Input (X):** Time-based features derived from date
- **Output (y):** Daily total sales amount

The goal is to minimize **prediction error** on unseen future data.

---

### 3. Dataset Used

Input dataset:

```
data/processed/master_sales.csv
```

This dataset is:

- Cleaned and validated
- Aggregated to daily level
- Free of structural inconsistencies

---

### 4. Feature Engineering

Based on insights from TASK 1 (EDA), the following features are engineered:

| Feature      | Description                       |
| ------------ | --------------------------------- |
| year         | Calendar year                     |
| month        | Month of the year                 |
| day          | Day of the month                  |
| day_of_week  | Monday–Sunday encoded numerically |
| week_of_year | ISO week number                   |
| quarter      | Business quarter (Q1–Q4)          |
| is_weekend   | Weekend indicator                 |

These features encode **seasonality, weekly patterns, and calendar effects** observed during EDA.

---

### 5. Train–Test Strategy

To avoid data leakage and simulate real forecasting:

- The dataset is sorted by date
- The **last 6 months** are used as the test set
- All earlier data is used for training
- No random shuffling is applied

This ensures the model is evaluated on **future data only**.

---

### 6. Models Evaluated

Three supervised regression models are trained and compared:

| Model                   | Purpose                         |
| ----------------------- | ------------------------------- |
| Linear Regression       | Baseline and interpretability   |
| Random Forest Regressor | Capture non-linear patterns     |
| XGBoost Regressor       | High-performance boosting model |

---

### 7. Evaluation Metric

Model performance is evaluated using:

**Root Mean Squared Error (RMSE)**

RMSE penalizes large prediction errors and is well-suited for continuous sales forecasting.

---

### 8. Model Performance Comparison

| Model             | RMSE       |
| ----------------- | ---------- |
| Linear Regression | 2258.03    |
| Random Forest     | 841.94     |
| **XGBoost**       | **666.90** |

---

### 9. Model Selection

The **XGBoost Regressor** is selected as the final machine learning model because:

- It achieves the lowest RMSE
- It generalizes well on unseen data
- It captures non-linear interactions effectively
- It maintains reasonable computational cost on a laptop

---

### 10. Forecasting Outcome

The selected model is retrained on the **full historical dataset** and used to generate a **1-year daily sales forecast**.

Output file:

```
outputs/sales_forecast_next_year.csv
```

This forecast can be used for:

- Demand planning
- Resource allocation
- Strategic business decisions

---

### 11. Key Takeaways

- Calendar features strongly influence sales
- Non-linear models significantly outperform linear baselines
- Boosting methods are highly effective for tabular time-derived data
- Machine learning provides **quantifiable, data-driven forecasts**
