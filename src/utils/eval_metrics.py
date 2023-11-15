import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
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


def plot_model_coeffs(
    model: LinearRegression,
    X_train: pd.DataFrame,
    save_dir: str,
    top_k: int = 15
):
    """Save top_k significant linear regression model coefficients

    Args:
        model (LinearRegression): linear regression model
        X_train (pd.DataFrame): train data
        save_dir (str): output directory
        top_k (int, optional): top-k significant coefficient. Defaults to 15.
    """

    coefs = pd.DataFrame(
        model.coef_, columns=["Coefficients"], index=X_train.columns
    )

    coefs = coefs.iloc[: top_k, :]

    coefs.plot(kind="barh", figsize=(9, 7))
    plt.title("Linear Regression")
    plt.axvline(x=0, color=".5")
    plt.subplots_adjust(left=0.2)

    plt.savefig(f"{save_dir}/model_coeffs.png")
