import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings # for warnings
warnings.filterwarnings("ignore")
import os
import logging
from logging_code import setup_logging
logger = setup_logging("feature_scaling")
import sys
from sklearn.preprocessing import StandardScaler #z_score
from all_models import common
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
import pickle


def fs(X_train , y_train, X_test , y_test ):
    try:
        logger.info(f"Training data independent size : {X_train.shape}")
        logger.info(f"Training data dependent size : {y_train.shape}")
        logger.info(f"Testing data independent size : {X_test.shape}")
        logger.info(f"Testing data dependent size : {y_test.shape}")
        logger.info(f"before : {X_train.head(1)}")
        X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)

        sc = StandardScaler()
        sc.fit(X_train)
        X_train_sc = sc.transform(X_train)
        X_test_sc = sc.transform(X_test)

        with open('standar_scaler.pkl' , 'wb') as f:
            pickle.dump(sc, f)

        logger.info(f"{X_train_sc}")
        #common(X_train_sc, y_train, X_test_sc, y_test)
        xgb = XGBClassifier()
        xgb.fit(X_train_sc, y_train)
        logger.info(f"test accuracy : {accuracy_score(y_test, xgb.predict(X_test_sc))}")
        logger.info(f"test confusion matrix : {confusion_matrix(y_test, xgb.predict(X_test_sc))}")
        logger.info(f"classification report : {classification_report    (y_test, xgb.predict(X_test_sc))}")
            
        with open('Model.pkl', 'wb') as t:
            pickle.dump(xgb, t)

    except Exception as e:
        er_type, er_msg, er_line = sys.exc_info()
        logger.info(f"Error in Line no {er_line.tb_lineno} due to : {er_msg}")