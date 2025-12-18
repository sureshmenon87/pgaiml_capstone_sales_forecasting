from src.machine_learning.train_linear import train_linear_model
from src.machine_learning.train_rf import train_rf_model
from src.machine_learning.train_xgb import train_xgb_model
from src.machine_learning.evaluate import rmse


def main():
    # Linear Regression
    lin_model, X_test, y_test = train_linear_model()
    lin_rmse = rmse(y_test, lin_model.predict(X_test))
    print(f"Linear Regression RMSE: {lin_rmse:.2f}")

    # Random Forest
    rf_model, X_test, y_test = train_rf_model()
    rf_rmse = rmse(y_test, rf_model.predict(X_test))
    print(f"Random Forest RMSE: {rf_rmse:.2f}")

    # XGBoost
    xgb_model, X_test, y_test = train_xgb_model()
    xgb_rmse = rmse(y_test, xgb_model.predict(X_test))
    print(f"XGBoost RMSE: {xgb_rmse:.2f}")


if __name__ == "__main__":
    main()
