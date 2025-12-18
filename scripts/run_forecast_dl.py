from src.deep_learning.train import train_lstm
from src.deep_learning.forecast import forecast_next_year_lstm, save_forecast
from src.common.utils import log


def main():
    log("Starting LSTM forecast for next 1 year")

    model, _, _, scaler = train_lstm()
    forecast_df = forecast_next_year_lstm(model, scaler)
    save_forecast(forecast_df)

    log(f"LSTM forecast generated with {len(forecast_df)} days")
    log("Output file: outputs/sales_forecast_lstm_next_year.csv")


if __name__ == "__main__":
    main()
