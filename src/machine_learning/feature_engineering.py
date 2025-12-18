import pandas as pd
from src.common.utils import log


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate time-based features from date column.
    """
    log("Generating time-based features")

    df = df.copy()
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df["day_of_week"] = df["date"].dt.dayofweek
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
    df["quarter"] = df["date"].dt.quarter
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

    return df
