from src.deep_learning.data_prep import prepare_lstm_data
from src.deep_learning.model import build_lstm_model
from src.common.utils import log


def train_lstm():
    X_train, y_train, X_test, y_test, scaler = prepare_lstm_data()

    model = build_lstm_model(
        input_shape=(X_train.shape[1], X_train.shape[2])
    )

    log("Training LSTM model")

    model.fit(
        X_train,
        y_train,
        epochs=20,
        batch_size=32,
        validation_data=(X_test, y_test),
        verbose=1
    )

    return model, X_test, y_test, scaler
