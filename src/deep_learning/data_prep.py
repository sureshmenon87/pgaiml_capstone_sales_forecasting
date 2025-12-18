import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from src.common.config import PROCESSED_DATA_DIR
from src.common.utils import log


def load_daily_series() -> pd.Series:
    path = PROCESSED_DATA_DIR / "master_sales.csv"
    log(f"Loading master dataset from {path}")

    df = pd.read_csv(path, parse_dates=["date"])
    daily = df.groupby("date")["sales_amount"].sum().sort_index()
    return daily


def create_sequences(data: np.ndarray, window_size: int = 30):
    X, y = [], []
    for i in range(window_size, len(data)):
        X.append(data[i - window_size:i])
        y.append(data[i])
    return np.array(X), np.array(y)


def prepare_lstm_data(window_size: int = 30):
    series = load_daily_series().values.reshape(-1, 1)

    split_idx = int(len(series) * 0.85)  # approx last 6 months
    train_data = series[:split_idx]
    test_data = series[split_idx - window_size:]

    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_data)
    test_scaled = scaler.transform(test_data)

    X_train, y_train = create_sequences(train_scaled, window_size)
    X_test, y_test = create_sequences(test_scaled, window_size)

    # LSTM expects 3D input
    X_train = X_train.reshape(X_train.shape[0], window_size, 1)
    X_test = X_test.reshape(X_test.shape[0], window_size, 1)

    return X_train, y_train, X_test, y_test, scaler
