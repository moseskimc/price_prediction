# price_prediction



## Introduction

We perform time-series forecasting. More specifically, we predict future product prices based on previously observed values and/or features. 


## Setup

First, clone this repo. Make sure you have downloaded the dataset and copied it to `data/` and have defined necessary config parameters in `src/config.yaml` (e.g. file name, feature column labels, target label, etc.).

To run the notebooks and script locally, make sure to create your environment and install the necessary dependencies running:

    pip install -r requirements.txt


## Scripts


### `train.py`

From the root directory, run

    PYTHONPATH=./ python3 src/train.py

There will be some stdout as well as files saved inside `output/` once the script is run.

If you wish to run via docker, first build the image

    docker compose build

And, second, run the script

    docker compose up



## Analysis

The presence of a few outliers skewed the distribution of price values. Binning by intervals of 500, we get:

|bin | frequency |
|---------------|------|
|(0.999, 501.0] | 643 |
|(501.0, 1001.0] | 0 |
|(1001.0, 1501.0]| 0 |
|(1501.0, 2001.0]| 0 |
|(2001.0, 2501.0]| 0 |
|(2501.0, 3001.0]| 0 |
|(3001.0, 3501.0]| 1 |
 

Once the outlier is removed, we have the following bins.

|bin | frequency |
|---------------|------|
|(0.999, 33.0]  |   642
|(33.0, 65.0]   |     0
|(65.0, 97.0]   |     0
|(97.0, 129.0]  |     0
|(129.0, 161.0] |     0
|(161.0, 193.0] |     0
|(193.0, 225.0]  |    1

Once we remove it, we we have following distribution of price values that resemble a normal distribution.

![](resources/normal.png)


There wasn't much correlation between numeric features (i.e. Total Volume, Total Boxes, etc.) and Price. For instance, the correlation between total volume and price looked as follows:

![](resources/pricetotalboxcorr.png)

But we do observe fluctuations usually occur for lower total volume values (other numeric values such as total boxes show a similar behavior)


Moreover, we examine any difference in means from region to region and province to province. One way to visualize the significance of t-tests is to plot boxplots as below.

![](resources/boxplotbycity.png)


![](resources/boxplotbyprovince.png)


## Results


For model evaluation we chose the first two months to be our train set while the third month, our test set. Upon careful observation, one can see the prices come on a weekly basis.

We trained three parametric models (linear regression, lasso, and ridge) and two non-parametric models (knearest and decision tree). As expected, the non-paramtric models overfit the data while the linear parametric models generalized much better with the linear regression model performing the best.


                lr                  lasso               ridge               dt              knn          
                train     test      train     test      train     test      train test      train     test
        metric                                                                                                
        mse     0.023307  0.023664  0.049532  0.047414  0.023659  0.023559  0.0   0.031771  0.004156  0.016890
        mae     0.105841  0.118169  0.178126  0.174618  0.107968  0.118392  0.0   0.129393  0.045280  0.095070
        mape    0.065787  0.074955  0.112790  0.110568  0.067164  0.075171  0.0   0.085764  0.028411  0.060617


The top-15 most significant coefficients in our linear model are shown in the plot below.


![](resources/model_coeffs.png)