import pandas as pd
from src.common.utils import log


def validate_restaurants(df: pd.DataFrame) -> None:
    log("Validating restaurants data")

    required_cols = {"id", "name"}
    if not required_cols.issubset(df.columns):
        raise ValueError("Restaurants dataset missing required columns")

    if df["id"].duplicated().any():
        raise ValueError("Duplicate restaurant IDs found")


def validate_items(df: pd.DataFrame, restaurants_df: pd.DataFrame) -> None:
    log("Validating items data")

    required_cols = {"id", "store_id", "name", "kcal", "cost"}
    if not required_cols.issubset(df.columns):
        raise ValueError("Items dataset missing required columns")

    if (df["cost"] < 0).any():
        raise ValueError("Negative item cost detected")

    if not df["store_id"].isin(restaurants_df["id"]).all():
        raise ValueError("Items contain invalid store_id references")


def validate_sales(df: pd.DataFrame) -> None:
    log("Validating sales data")
    # TEMP: schema inspection during development
    log(f"Sales columns found: {list(df.columns)}")

    
    required_cols = {"date", "item_id", "price", "item_count"}

    if not required_cols.issubset(df.columns):
        raise ValueError("Sales dataset missing required columns")

   

    if (df["price"] < 0).any():
        raise ValueError("Negative price detected")
