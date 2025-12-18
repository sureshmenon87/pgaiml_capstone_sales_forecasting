from sklearn.metrics import mean_squared_error
import numpy as np


def rmse(y_true, y_pred) -> float:
    """
    Compute Root Mean Squared Error.
    """
    return np.sqrt(mean_squared_error(y_true, y_pred))
