from src.deep_learning.train import train_lstm
from src.deep_learning.evaluate import rmse


def main():
    model, X_test, y_test, scaler = train_lstm()
    preds = model.predict(X_test)

    error = rmse(y_test, preds)
    print(f"LSTM RMSE (scaled): {error:.4f}")


if __name__ == "__main__":
    main()
