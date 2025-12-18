import pandas as pd

from src.common.utils import log
from src.common.config import PROCESSED_DATA_DIR
from src.datascience.ingestion import (
    load_sales,
    load_items,
    load_restaurants,
)


def preprocess_and_merge() -> pd.DataFrame:
    """
    Preprocess raw datasets and create a master sales dataset.

    Returns
    -------
    pd.DataFrame
        Cleaned and merged master dataset.
    """

    log("Starting preprocessing and merge step")

    # Load raw data
    sales = load_sales()
    items = load_items()
    restaurants = load_restaurants()

    # -------------------------
    # Business rule 1:
    # Drop non-positive sales
    # -------------------------
    initial_rows = len(sales)
    sales = sales[sales["item_count"] > 0]
    dropped_rows = initial_rows - len(sales)

    log(f"Dropped {dropped_rows} rows with non-positive item_count")

    # -------------------------
    # Enforce join key dtypes
    # -------------------------
    sales["item_id"] = sales["item_id"].astype("int32")
    items["id"] = items["id"].astype("int32")
    log(f"sales.item_id dtype: {sales['item_id'].dtype}")
    log(f"items.id dtype: {items['id'].dtype}")



    # -------------------------
    # Merge sales -> items
    # -------------------------
    merged = sales.merge(
        items,
        left_on="item_id",
        right_on="id",
        how="inner",
        suffixes=("", "_item"),
    )

    log(f"After merging sales and items: {merged.shape}")

    # -------------------------
    # Merge with restaurants
    # -------------------------
    merged = merged.merge(
        restaurants,
        left_on="store_id",
        right_on="id",
        how="inner",
        suffixes=("", "_restaurant"),
    )

    log(f"After merging with restaurants: {merged.shape}")

    # -------------------------
    # Feature engineering (minimal, justified)
    # -------------------------
    merged["sales_amount"] = merged["price"] * merged["item_count"]

    # -------------------------
    # Column cleanup
    # -------------------------
    final_columns = {
        "date": "date",
        "item_id": "item_id",
        "name": "item_name",
        "store_id": "store_id",
        "name_restaurant": "store_name",
        "price": "price",
        "item_count": "item_count",
        "kcal": "kcal",
        "cost": "cost",
        "sales_amount": "sales_amount",
    }

    master_df = merged[list(final_columns.keys())].rename(
        columns=final_columns
    )

    log(f"Final master dataset shape: {master_df.shape}")

    return master_df


def save_master_dataset(df: pd.DataFrame) -> None:
    """
    Persist the processed master dataset.
    """
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_path = PROCESSED_DATA_DIR / "master_sales.csv"

    df.to_csv(output_path, index=False)
    log(f"Master dataset saved to {output_path}")
