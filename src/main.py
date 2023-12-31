import os
import yaml
import argparse

import pandas as pd

from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import MinMaxScaler

from utils.eval_metrics import get_reg_metrics, plot_model_coeffs

from utils.data_processing import remove_na_rows, remove_outliers, \
    convert_to_datetime, add_province, split_by_horizon, normalize_columns, \
    generate_random_data

parser = argparse.ArgumentParser()
parser.add_argument('-r', '--random_data', action='store_true')
parser.add_argument('-s', '--save', action='store_false')


if __name__ == "__main__":

    pargs = parser.parse_args()

    with open('src/config.yaml') as file:
        config = yaml.safe_load(file)

    file_path = os.path.join(
        config['data_directory'],
        config['file_name']
    )

    features = config['features']  # feature columns
    label = config['target_name']  # target label
    save_dir = config['save_dir']  # dir to save output
    save = pargs.save              # whether to save output

    if pargs.random_data:  # for github workflow
        df = generate_random_data()
    else:
        df = pd.read_csv(file_path)

######################################PROCESSING###############################

    # remove rows that have at least 1 na
    df_nona = remove_na_rows(df)

    # remove outliers 3 std deviations away
    df_nona_noout = remove_outliers(df, label)

    # remove once more
    df_nona_noout = remove_outliers(df_nona_noout, label)

    # we output the some stats such as min, max, median, and mean
    print(df_nona_noout[label].describe())  # now we have a more normal
                                            # distribution w/o skew

    # rename dataframe variable since we don't have a skew
    df_normal = df_nona_noout

    # convert date column to datetime
    convert_to_datetime(df_normal)

    # add province column given city
    add_province(df_normal)

###############################################################################


######################################TRAIN####################################

    # first we split by month
    # since there are 3 months, we choose the last month as the test set

    # first normalize numeric features
    numeric_features = [
        'Total Volume', 'Total Boxes', 'Small Boxes',
        'Large Boxes', 'XLarge Boxes'
    ]

    normalize_columns(df_normal, numeric_features, MinMaxScaler())

    trainX, trainy, testX, testy = split_by_horizon(
        df_normal,
        features,
        label,
    )

    # parametric models
    lr = LinearRegression()
    lasso = Lasso()
    ridge = Ridge()

    # non-paramteric models
    dt = DecisionTreeRegressor()
    knn = KNeighborsRegressor(n_neighbors=2)

    # let's tabulate some metrics results into a dataframe
    model_names = ['lr', 'lasso', 'ridge', 'dt', 'knn']
    eval_dfs = {}
    for name in model_names:
        eval_dict = {}

        # train
        model = eval(name)
        model.fit(trainX, trainy)

        # train metrics
        train_pred = model.predict(trainX)
        eval_dict['metric'] = ['mse', 'mae', 'mape']
        eval_dict['train'] = get_reg_metrics(
            train_pred, trainy
        )

        # predict
        pred = model.predict(testX)
        eval_dict['test'] = get_reg_metrics(
            pred,
            testy
        )
        eval_dfs[name] = pd.DataFrame(eval_dict).set_index('metric')

    evals = pd.concat([eval_dfs[name] for name in model_names], axis=1)
    evals.columns = pd.MultiIndex.from_product([
        model_names,
        ['train', 'test']
    ])

    # save
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if save:
        evals.to_csv(os.path.join(save_dir, "model_evals.csv"))

    # print
    print(evals)

    # save model coeffs
    plot_model_coeffs(
        lr,
        trainX,
        save_dir,
        save
    )

    # linear regression performance by region and province
    provinces = [col for col in testX.columns if col.startswith('Province')]
    lr_eval_metrics_by_province = {}
    # set metric column
    lr_eval_metrics_by_province["metric"] = [
        "MSE",
        "MAE",
        "MAPE",
    ]
    for province in provinces:
        mask = testX[province] == True
        testX_province = testX[mask]
        testy_province = testy[mask]

        pred = lr.predict(testX_province)
        gt = testy_province

        province_name = province.split('_')[1]
        lr_eval_metrics_by_province[province_name] = get_reg_metrics(pred, gt)

    lr_metrics_df = pd.DataFrame(lr_eval_metrics_by_province).\
        set_index('metric')

    # save
    if save:
        lr_metrics_df.to_csv(os.path.join(save_dir, "lr_metrics.csv"))
    # stdout
    print(lr_metrics_df)
    # sort provinces by ascending MSE metric
    print(lr_metrics_df.loc['MSE'].sort_values())

###############################################################################
