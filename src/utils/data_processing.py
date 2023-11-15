import pandas as pd
import numpy as np

from typing import List
from sklearn.preprocessing import MinMaxScaler


def remove_na_rows(
    df: pd.DataFrame,
    min_thresh: int = 1
):
    """Returns dataframe after removing rows with at least min no. of nas

    Args:
        df (pd.DataFrame): dataframe to remove na rows from
        min_thresh (int, optional): minimum no. of nas for row removal. Defaults to 1.

    Returns:
        pd.DataFrame: dataframe with rows meeting the min threshold no of na.
    """

    na_ct = df.isna().apply(lambda x: sum(x), axis=1)
    mask = ~na_ct.apply(lambda x: True if x >= min_thresh else False)

    return df[mask]


def remove_outliers(
    df: pd.DataFrame,
    column: str,
    no_std: int = 3,
):
    """Return dataframe after removing rows whose values in col are outliers

    Args:
        df (pd.DataFrame): dataframe having column labeled column
        column (str): column label
        no_std (int, optional): no. of standard deviations. Defaults to 3.

    Returns:
        pd.DataFrame: dataframe with rows corresponding to outliers in column
                      removed
    """

    label_median, label_std = df[column].median(), df[column].std()

    return df[df[column] < label_median + no_std * label_std]


def convert_to_datetime(
    df: pd.DataFrame,
    date_column: str = "Date",
    
):
    """Converts datetime column values to datetime objects

    Args:
        df (pd.DataFrame): dataframe with date column (str)
        date_column (str, optional): date column label. Defaults to "Date".
    """

    date_entry = df[date_column].iloc[0]
    # convert if needed
    if not isinstance(date_entry, pd._libs.tslibs.timestamps.Timestamp):
        df.loc[:, date_column] = pd.to_datetime(df[date_column])


def add_province(
    df: pd.DataFrame,
    city_col: str = 'Region'
):
    """Adds columns to dataframe via a region to province dict map

    Args:
        df (pd.DataFrame): dataframe with column region
        city_col (str, optional): city column label. Defaults to 'Region'.
    """

    city_province_dict = {
        'Seoul': 'Gyeonggi', 'Incheon': 'Gyeonggi', 'deagu': 'North Gyeongsang',
        'Anyang': 'Gyeonggi', 'Ulsan': 'South Gyeongsang',
        'Busan': 'South Gyeongsang', 'Daejon': 'South Chungcheong',
        'Jeju': "Jeju", 'Gwangju': 'Gyeonggi', 'Gangeung': 'Gangwon',
        'Pyeongchang': 'Gangwon', 'Andong': 'North Gyeongsang', 
        'Asan': 'South Chungcheong', 'Boryeong': 'South Chungcheong',
        'Chungju': 'North Chungcheong', 'Geoje': 'South Gyeongsang',
        'Gimpo': 'Gyeonggi', 'Gongju': 'South Chungcheong',
        'Gunpo': 'Gyeonggi', 'Hanam': 'Gyeonggi', 'Jeonju': 'North Jeolla',
        'Suwon': 'Gyeonggi', 'Taebaek': 'Gangwon', 'Pyeongtaek': 'Gyeonggi',
        'Yeoju': 'Gyeonggi', 'Wonju': 'Gangwon', 'Sangju': 'North Gyeongsang',
        'Miryang': 'South Gyeongsang', 'Suncheon': 'South Jeolla',
        'Iksan': 'North Jeolla', 'Namwon': 'North Jeolla',
        'Siheung': 'Gyeonggi', 'Tongyeong': 'South Gyeongsang',
        'Yangju': 'Gyeonggi', 'Yangsan': 'South Gyeongsang',
        'Donghae': 'Gangwon', 'Gyeongju': 'North Gyeongsang',
        'Gyeryong': 'South Chungcheong', 'Gimje': 'North Jeolla',
        'Gwangmyeong': 'Gyeonggi', 'Icheon': 'Gyeonggi'
    }

    df.loc[:, "Province"] = df[city_col].apply(lambda x: city_province_dict[x])


def split_by_horizon(
    df: pd.DataFrame,
    features: list,
    label: str,
    horizon: int = 3, 
    date_col: str = "Date",
    has_categorical: bool = True,
):
    """Return trainX, trainy, testX, testy

    Args:
        df (pd.DataFrame): dataframe with date column
        features (list): features to include to train model
        label (str): target label to train model
        horizon (int, optional): month to split data. Defaults to 3.
        date_col (str, optional): date column label. Defaults to "Date".
        has_categorical (bool, optional): whether there are categorical
                                          columns. Defaults to True.

    Returns:
        List[pd.DataFrame]: [trainX, trainy, testX, testy]
    """

    # we assume the date column is type datetime
    split_mask = df[date_col].apply(lambda x: x.month) >= horizon
    # split
    df_train, df_test = df[~split_mask], df[split_mask]

    # features
    trainX, testX = df_train[features], df_test[features]
    # one-hot encoding
    if has_categorical:
        trainX, testX = pd.get_dummies(trainX), pd.get_dummies(testX)

    # labels
    trainy, testy = df_train[label], df_test[label]

    return trainX, trainy, testX, testy


def normalize_columns(
    df: pd.DataFrame,
    columns: list,
    scaler: MinMaxScaler,
):
    """Normalize numeric columns using min-max scaler

    Args:
        df (pd.DataFrame): dataframe to scale columns from
        columns (list): list of columns to scale
        scaler (MinMaxScaler): scaler used to normalize col values
    """

    for column in columns:
        df[column] = scaler.fit_transform(
            np.array(df[column]).reshape(-1, 1)
        )
