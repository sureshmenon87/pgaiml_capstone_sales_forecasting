import pandas as pd
from xgboost import XGBRegressor

from src.common.utils import log
from src.common.config import OUTPUT_DIR
from src.machine_learning.feature_engineering import add_time_features
from src.machine_learning.data_prep import prepare_ml_dataset


def train_final_model() -> XGBRegressor:
    """
    Train XGBoost on full historical data.
    """
    df = prepare_ml_dataset()

    feature_cols = [
        "year", "month", "day",
        "day_of_week", "week_of_year",
        "quarter", "is_weekend"
    ]

    X = df[feature_cols]
    y = df["sales_amount"]

    model = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=4
    )

    model.fit(X, y)
    log("Final XGBoost model trained on full data")

    return model


def generate_future_dates(last_date: pd.Timestamp, days: int = 365) -> pd.DataFrame:
    """
    Generate future dates for forecasting.
    """
    future_dates = pd.date_range(
        start=last_date + pd.Timedelta(days=1),
        periods=days,
        freq="D"
    )

    return pd.DataFrame({"date": future_dates})


def forecast_next_year() -> pd.DataFrame:
    """
    Forecast sales for the next year.
    """
    df = prepare_ml_dataset()
    last_date = df["date"].max()

    model = train_final_model()

    future_df = generate_future_dates(last_date)
    future_df = add_time_features(future_df)

    feature_cols = [
        "year", "month", "day",
        "day_of_week", "week_of_year",
        "quarter", "is_weekend"
    ]

    future_df["predicted_sales"] = model.predict(
        future_df[feature_cols]
    )

    return future_df


def save_forecast(df: pd.DataFrame) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "sales_forecast_next_year.csv"

    df.to_csv(output_path, index=False)
    log(f"Forecast saved to {output_path}")
