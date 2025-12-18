import numpy as np
import pandas as pd

from src.deep_learning.data_prep import load_daily_series
from src.common.config import OUTPUT_DIR
from src.common.utils import log


def forecast_next_year_lstm(model, scaler, window_size=30):
    series = load_daily_series().values.reshape(-1, 1)
    scaled = scaler.transform(series)

    last_window = scaled[-window_size:].reshape(1, window_size, 1)
    predictions = []

    for _ in range(365):
        next_val = model.predict(last_window)[0][0]
        predictions.append(next_val)

        last_window = np.append(
            last_window[:, 1:, :],
            [[[next_val]]],
            axis=1
        )

    preds = scaler.inverse_transform(
        np.array(predictions).reshape(-1, 1)
    )

    dates = pd.date_range(
        start=pd.Timestamp(series.shape[0]),
        periods=365
    )

    return pd.DataFrame({
        "predicted_sales": preds.flatten()
    })


def save_forecast(df):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / "sales_forecast_lstm_next_year.csv"
    df.to_csv(path, index=False)
    log(f"LSTM forecast saved to {path}")
