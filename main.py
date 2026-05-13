'''
In this file we are going to cal all related functions for data cleaning and development
'''
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
logger = setup_logging("main")
import sys
from sklearn.model_selection import train_test_split
from var_out import vt_outliers
from filter_methods import fm
from categorical_to_num import c_t_n
from imblearn.over_sampling import SMOTE
from feature_scaling import fs


class CHURNPREDICTION:
    def __init__(self,path):
        try:
            self.path = path
            self.df = pd.read_csv(self.path)
            self.df["TotalCharges"] = pd.to_numeric(self.df["TotalCharges"], errors="coerce")
            self.df["TotalCharges"].fillna(self.df["TotalCharges"].mean(), inplace=True)
            logger.info(f"Total data size is : {self.df.shape}")
            logger.info(f"Null values  : \n : {self.df.isnull().sum()}")

            self.X = self.df.drop("Churn", axis=1) #independent
            self.y = self.df["Churn"] #dependent
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X,self.y,test_size=0.2, random_state=42)
            self.y_train = self.y_train.map({'Yes': 1, 'No': 0}).astype(int)
            self.y_test = self.y_test.map({'Yes': 1, 'No': 0}).astype(int)
            logger.info(f"Train data size : {len(self.X_train)} : {len(self.y_train)} \n: Total train data: {self.X_train.shape}")
            logger.info(f"Test data size : {len(self.X_test)} : {len(self.y_test)} \n: Total test data: {self.X_test.shape}")

        except Exception as e:
            er_type, er_msg, er_line = sys.exc_info()
            logger.info(f"Error in Line no {er_line.tb_lineno} due to : {er_msg}")

    def data_seperation(self):
        try:
            self.X_train_num_cols = self.X_train.select_dtypes(exclude='object')
            self.X_test_num_cols = self.X_test.select_dtypes(exclude='object')

            self.X_train_cat_cols = self.X_train.select_dtypes(include='object')
            self.X_test_cat_cols = self.X_test.select_dtypes(include='object')

            logger.info(f"{self.X_train_num_cols.columns} : {self.X_train_num_cols.shape}")
            logger.info(f"{self.X_test_num_cols.columns} : {self.X_test_num_cols.shape}")
            logger.info(f"{self.X_train_cat_cols.columns} : {self.X_train_cat_cols.shape}")
            logger.info(f"{self.X_test_cat_cols.columns} : {self.X_test_cat_cols.shape}")
        except Exception as e:
            er_type, er_msg, er_line = sys.exc_info()
            logger.info(f"Error in Line no {er_line.tb_lineno} due to : {er_msg}")


    def variable_transformation(self):
        try:
            logger.info(f"Before train column names : {self.X_train_num_cols.columns}")
            logger.info(f"Before test column names : {self.X_test_num_cols.columns}")
            self.X_train_num_cols, self.X_test_num_cols = vt_outliers(self.X_train_num_cols, self.X_test_num_cols)
            logger.info(f"After train column names : {self.X_train_num_cols.columns}")
            logger.info(f"After test column names : {self.X_test_num_cols.columns}")
        except Exception as e:
            er_type, er_msg, er_line = sys.exc_info()
            logger.info(f"Error in Line no {er_line.tb_lineno} due to : {er_msg}")


    def feature_selection(self):
        try:
            fm(self.X_train_num_cols, self.X_test_num_cols , self.y_train, self.y_test)
        except Exception as e:
            er_type, er_msg, er_line = sys.exc_info()
            logger.info(f"Error in Line no {er_line.tb_lineno} due to : {er_msg}")

    def cat_to_num(self):
        try:
            self.X_train_cat_cols, self.X_test_cat_cols = c_t_n(self.X_train_cat_cols, self.X_test_cat_cols)
            # combine the data
            self.X_train_num_cols.reset_index(drop=True, inplace=True)
            self.X_train_cat_cols.reset_index(drop=True, inplace=True)
            self.X_test_num_cols.reset_index(drop=True, inplace=True)
            self.X_test_cat_cols.reset_index(drop=True, inplace=True)

            self.training_data = pd.concat([self.X_train_num_cols, self.X_train_cat_cols], axis=1)
            self.testing_data = pd.concat([self.X_test_num_cols, self.X_test_cat_cols], axis=1)

            logger.info(f"============================================================================")
            logger.info(f"Final Training data : {self.training_data.shape}")
            logger.info(f"{self.training_data.columns}")
            logger.info(f"{self.training_data.isnull().sum()}")

            self.training_data = self.training_data.drop(["customerID"], axis=1)
            self.testing_data = self.testing_data.drop(["customerID"], axis=1)

            logger.info(f"Final Testing data : {self.testing_data.shape}")
            logger.info(f"{self.testing_data.columns}")
            logger.info(f"{self.testing_data.isnull().sum()}")

        except Exception as e:
            er_type, er_msg, er_line = sys.exc_info()
            logger.info(f"Error in Line no {er_line.tb_lineno} due to : {er_msg}")


    def data_balancing(self):
         try:
             logger.info(f"Number of Rows for Good customer {1} : {sum(self.y_train==1)}")
             logger.info(f"Number of Rows for Bad customer {0} : {sum(self.y_train==0)}")
             logger.info(f"Training data size : {self.training_data.shape}")
             self.training_data.fillna(self.training_data.mean(), inplace=True)
             self.testing_data.fillna(self.testing_data.mean(), inplace=True)

             sm = SMOTE(random_state=42)

             self.training_data_bal, self.y_train_bal = sm.fit_resample(self.training_data, self.y_train)

             logger.info(f"Number of Rows for Good customer {1} : {sum(self.y_train_bal == 1)}")
             logger.info(f"Number of Rows for Bad customer {0} : {sum(self.y_train_bal == 0)}")
             logger.info(f"Training data size : {self.training_data_bal.shape}")

             fs(self.training_data_bal, self.y_train_bal,self.testing_data,  self.y_test)


         except Exception as e:
             er_type, er_msg, er_line = sys.exc_info()
             logger.info(f"Error in Line no {er_line.tb_lineno} due to : {er_msg}")






if __name__ == "__main__":
    try:
        obj = CHURNPREDICTION('churn_data.csv')
        obj.data_seperation()
        obj.variable_transformation()
        obj.feature_selection()
        obj.cat_to_num()
        obj.data_balancing()

    except Exception as e:
        er_type, er_msg, er_line = sys.exc_info()
        logger.info(f"Error in Line no {er_line.tb_lineno} due to : {er_msg}")
