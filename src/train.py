import os
import yaml


import pandas as pd

from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor

from utils.eval_metrics import get_reg_metrics

from utils.data_processing import remove_na_rows, remove_outliers, \
    convert_to_datetime, add_province, split_by_horizon


if __name__ == "__main__":

    with open('src/config.yaml') as file:
        config = yaml.safe_load(file)

    file_path = os.path.join(
        config['data_directory'],
        config['file_name']
    )

    features = config['features']  # feature columns
    label = config['target_name']  # target label
    save_dir = config['save_dir']  # dir to save output

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


    # # go through 3 more iterations
    # # stop early if there's nothing to remove
    # run_iterations = 0
    # for _ in range(3):
    #     prev_len = df_nona_noout.shape[0]

    #     df_nona_noout = remove_outliers(
    #         df_nona_noout, label
    #     )

    #     if prev_len == df_nona_noout.shape[0]:
    #         break
    #     run_iterations += 1

    # print(f"Run iterations: {run_iterations}")


######################################TRAIN####################################

    # first we split by month
    # since there are 3 months, we choose the last month as the test set

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
    evals.to_csv(os.path.join(save_dir, "model_evals.csv"))

    # print
    print(evals)

###############################################################################