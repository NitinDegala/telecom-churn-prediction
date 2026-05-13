import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings # for warnings
warnings.filterwarnings("ignore")
import os
import logging
from logging_code import setup_logging
logger = setup_logging("var_out")
from scipy.stats import yeojohnson
from sklearn.feature_selection import VarianceThreshold

def vt_outliers(X_train_num, X_test_num):
    try:
        logger.info(f"Before train column names : {X_train_num.columns}")
        logger.info(f"Before test column names : {X_test_num.columns}")

        for i in X_train_num.columns:
            X_train_num[i+'_yeo'],lam_value = yeojohnson(X_train_num[i])
            X_test_num[i+'_yeo'],lam_value = yeojohnson(X_test_num[i])
            X_train_num = X_train_num.drop([i], axis=1)
            X_test_num = X_test_num.drop([i], axis=1)
            #trimming
            iqr = X_train_num[i+'_yeo'].quantile(0.75) - X_train_num[i+'_yeo'].quantile(0.25)
            upper_limit = X_train_num[i+'_yeo'].quantile(0.75) + (1.5 * iqr)
            lower_limit = X_train_num[i+'_yeo'].quantile(0.25) - (1.5 * iqr)
            X_train_num[i+'_trim'] = np.where(X_train_num[i+'_yeo'] > upper_limit , upper_limit,
                    np.where(X_train_num[i+'_yeo'] < lower_limit, lower_limit, X_train_num[i+'_yeo']))

            X_test_num[i+'_trim'] = np.where(X_test_num[i+'_yeo'] > upper_limit, upper_limit,
                                                np.where(X_test_num[i+'_yeo'] < lower_limit, lower_limit,
                                                         X_test_num[i+'_yeo']))

            X_train_num = X_train_num.drop([i+'_yeo'], axis=1)
            X_test_num = X_test_num.drop([i+'_yeo'], axis=1)

        logger.info(f"After train column names : {X_train_num.columns}")
        logger.info(f"After test column names : {X_test_num.columns}")

        return X_train_num, X_test_num



    except Exception as e:

        logger.exception(e)
