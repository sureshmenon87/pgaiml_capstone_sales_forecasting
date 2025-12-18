import pandas as pd
import matplotlib.pyplot as plt

from src.common.config import PROCESSED_DATA_DIR, FIGURES_DIR
from src.common.utils import log


def load_master_dataset() -> pd.DataFrame:
    path = PROCESSED_DATA_DIR / "master_sales.csv"
    log(f"Loading master dataset from {path}")
    return pd.read_csv(path, parse_dates=["date"])


def plot_daily_sales(df: pd.DataFrame) -> None:
    log("Plotting overall daily sales trend")

    daily_sales = (
        df.groupby("date")["sales_amount"]
        .sum()
        .reset_index()
    )

    plt.figure(figsize=(12, 5))
    plt.plot(daily_sales["date"], daily_sales["sales_amount"])
    plt.title("Overall Daily Sales Trend")
    plt.xlabel("Date")
    plt.ylabel("Sales Amount")
    plt.tight_layout()

    output = FIGURES_DIR / "daily_sales_trend.png"
    plt.savefig(output)
    plt.close()

    log(f"Saved {output}")


def plot_weekday_sales(df: pd.DataFrame) -> None:
    log("Analyzing weekday sales pattern")

    df["weekday"] = df["date"].dt.day_name()

    weekday_sales = (
        df.groupby("weekday")["sales_amount"]
        .mean()
        .reindex([
            "Monday", "Tuesday", "Wednesday",
            "Thursday", "Friday", "Saturday", "Sunday"
        ])
    )

    plt.figure(figsize=(8, 5))
    weekday_sales.plot(kind="bar")
    plt.title("Average Sales by Day of Week")
    plt.xlabel("Day")
    plt.ylabel("Average Sales Amount")
    plt.tight_layout()

    output = FIGURES_DIR / "weekday_sales.png"
    plt.savefig(output)
    plt.close()

    log(f"Saved {output}")


def plot_monthly_trends(df: pd.DataFrame) -> None:
    log("Analyzing monthly sales trends")

    df["month"] = df["date"].dt.month

    monthly_sales = (
        df.groupby("month")["sales_amount"]
        .mean()
    )

    plt.figure(figsize=(8, 5))
    monthly_sales.plot(marker="o")
    plt.title("Average Monthly Sales Trend")
    plt.xlabel("Month")
    plt.ylabel("Average Sales Amount")
    plt.tight_layout()

    output = FIGURES_DIR / "monthly_sales_trend.png"
    plt.savefig(output)
    plt.close()

    log(f"Saved {output}")


def plot_quarterly_distribution(df: pd.DataFrame) -> None:
    log("Analyzing quarterly sales distribution")

    df["quarter"] = df["date"].dt.quarter

    quarterly_sales = (
        df.groupby("quarter")["sales_amount"]
        .mean()
    )

    plt.figure(figsize=(6, 5))
    quarterly_sales.plot(kind="bar")
    plt.title("Average Sales by Quarter")
    plt.xlabel("Quarter")
    plt.ylabel("Average Sales Amount")
    plt.tight_layout()

    output = FIGURES_DIR / "quarterly_sales.png"
    plt.savefig(output)
    plt.close()

    log(f"Saved {output}")


def analyze_restaurant_performance(df: pd.DataFrame) -> pd.DataFrame:
    log("Analyzing restaurant-wise performance")

    restaurant_sales = (
        df.groupby("store_name")["sales_amount"]
        .sum()
        .sort_values(ascending=False)
        .reset_index()
    )

    log("Top performing restaurants:")
    log(restaurant_sales.head(5).to_string(index=False))

    return restaurant_sales
