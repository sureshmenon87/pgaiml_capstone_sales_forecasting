import pandas as pd
from src.common.config import (
    RESTAURANTS_FILE,
    ITEMS_FILE,
    SALES_FILE,
)
from src.common.utils import log, read_csv_safe


def load_restaurants() -> pd.DataFrame:
    log("Loading restaurants data")
    df = read_csv_safe(
        RESTAURANTS_FILE,
        dtype={
            "id": "int32",
            "name": "string"
        }
    )
    return df


def load_items() -> pd.DataFrame:
    log("Loading items data")
    df = read_csv_safe(
        ITEMS_FILE,
        dtype={
            "id": "int32",
            "store_id": "int32",
            "name": "string",
            "kcal": "float32",
            "cost": "float32",
        }
    )
    return df


def load_sales() -> pd.DataFrame:
    log("Loading sales data")
    df = read_csv_safe(
        SALES_FILE,
        parse_dates=["date"],
        dtype={
            "item_id": "string",
            "price": "float32",
            "item_count": "int32",
        }
    )
    return df
