from sklearn.ensemble import RandomForestRegressor
import pandas as pd

from src.common.utils import log
from src.machine_learning.data_prep import prepare_ml_dataset


def train_rf_model():
    """
    Train a Random Forest Regressor using time-aware split.
    """
    df = prepare_ml_dataset()
    df = df.sort_values("date")

    split_date = df["date"].max() - pd.DateOffset(months=6)
    log(f"Train/Test split date: {split_date.date()}")

    train_df = df[df["date"] <= split_date]
    test_df = df[df["date"] > split_date]

    feature_cols = [
        "year", "month", "day",
        "day_of_week", "week_of_year",
        "quarter", "is_weekend"
    ]

    X_train = train_df[feature_cols]
    y_train = train_df["sales_amount"]

    X_test = test_df[feature_cols]
    y_test = test_df["sales_amount"]

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    return model, X_test, y_test
