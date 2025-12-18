import pandas as pd
from src.common.config import PROCESSED_DATA_DIR
from src.common.utils import log
from src.machine_learning.feature_engineering import add_time_features


def prepare_ml_dataset() -> pd.DataFrame:
    """
    Prepare aggregated daily dataset for ML models.
    """
    path = PROCESSED_DATA_DIR / "master_sales.csv"
    log(f"Loading master dataset from {path}")

    df = pd.read_csv(path, parse_dates=["date"])

    # Aggregate to daily level (global sales)
    daily_df = (
        df.groupby("date", as_index=False)["sales_amount"]
        .sum()
    )

    daily_df = add_time_features(daily_df)

    return daily_df
