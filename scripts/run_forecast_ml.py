from src.machine_learning.forecast import forecast_next_year, save_forecast


def main():
    forecast_df = forecast_next_year()
    save_forecast(forecast_df)

    print("1-year sales forecast generated successfully")


if __name__ == "__main__":
    main()
