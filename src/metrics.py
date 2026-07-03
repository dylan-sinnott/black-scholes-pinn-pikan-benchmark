import numpy as np


def _to_arrays(y_pred, y_true):
    return np.asarray(y_pred), np.asarray(y_true)


def mse(y_pred, y_true):
    y_pred, y_true = _to_arrays(y_pred, y_true)
    return np.mean((y_pred - y_true) ** 2)


def rmse(y_pred, y_true):
    return np.sqrt(mse(y_pred, y_true))


def mae(y_pred, y_true):
    y_pred, y_true = _to_arrays(y_pred, y_true)
    return np.mean(np.abs(y_pred - y_true))


def max_abs_error(y_pred, y_true):
    y_pred, y_true = _to_arrays(y_pred, y_true)
    return np.max(np.abs(y_pred - y_true))


def relative_l2_error(y_pred, y_true):
    y_pred, y_true = _to_arrays(y_pred, y_true)
    denom = np.linalg.norm(y_true)

    if denom == 0:
        return np.nan

    return np.linalg.norm(y_pred - y_true) / denom


def error_summary(y_pred, y_true):
    return {
        "MSE": mse(y_pred, y_true),
        "RMSE": rmse(y_pred, y_true),
        "MAE": mae(y_pred, y_true),
        "Max Abs Error": max_abs_error(y_pred, y_true),
        "Relative L2 Error": relative_l2_error(y_pred, y_true),
    }