import pandas as pd

from sklearn.metrics import mean_squared_error, mean_absolute_error, \
    mean_absolute_percentage_error


def get_reg_metrics(
    pred: pd.Series,
    ground_truth: pd.Series
):
    """Compute and print MSE, MAE, and MAPE for predictions againt ground truth

    Args:
        pred (pd.Series): pandas Series corresponding to model predictions
        ground_truth (pd.Series): Series with ground truth values

    Returns:
        List[float]: [MSE, MAE, MAPE]
    """

    mse = mean_squared_error(pred, ground_truth)
    mae = mean_absolute_error(pred, ground_truth)
    mape = mean_absolute_percentage_error(pred, ground_truth)

    return [mse, mae, mape]
