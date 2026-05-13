import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings # for warnings
warnings.filterwarnings("ignore")
import os
import sys
import logging
from logging_code import setup_logging
logger = setup_logging("filter_methods")
import sys
from sklearn.feature_selection import VarianceThreshold
from scipy.stats import pearsonr

def fm(X_train_num, X_test_num, y_train, y_test):
    try:
        logger.info(f"Before Train columns  : {X_train_num.shape} \n : {X_train_num.columns}")
        logger.info(f"Before Test columns  : {X_test_num.shape} \n : {X_test_num.columns}")
        reg = VarianceThreshold(threshold=0)
        reg.fit(X_train_num)
        logger.info(f"Number of good columns : {sum(reg.get_support())} : {X_train_num.columns[reg.get_support()]}")
        logger.info(f"Number of bad columns : {sum(~reg.get_support())} : {X_test_num.columns[~reg.get_support()]}")
        X_train_num = X_train_num.drop(["SeniorCitizen_trim"], axis = 1)
        X_test_num = X_test_num.drop("SeniorCitizen_trim", axis = 1)
        logger.info(f"After Train columns  : {X_train_num.shape} \n : {X_train_num.columns}")
        logger.info(f"After Train columns  : {X_test_num.shape} \n : {X_test_num.columns}")
        logger.info(f"======================Hypothesis Testing==================")
        c = []
        for i in X_train_num.columns:
            results = pearsonr(X_train_num[i], y_train)
            c.append(results)
        t = np.array(c)
        p_value =  pd.series(t[: , 1] , index = X_train_num.columns)




    except Exception as e:
        er_type, er_msg, er_line = sys.exc_info()
        logger.info(f"Error in Line no {er_line.tb_lineno} due to : {er_msg}")