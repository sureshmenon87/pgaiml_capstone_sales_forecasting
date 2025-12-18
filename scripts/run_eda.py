from src.datascience.eda import (
    load_master_dataset,
    plot_daily_sales,
    plot_weekday_sales,
    plot_monthly_trends,
    plot_quarterly_distribution,
    analyze_restaurant_performance,
)
from src.common.config import FIGURES_DIR


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    df = load_master_dataset()

    plot_daily_sales(df)
    plot_weekday_sales(df)
    plot_monthly_trends(df)
    plot_quarterly_distribution(df)

    analyze_restaurant_performance(df)

    print("EDA completed successfully")


if __name__ == "__main__":
    main()
