print ("Prepare Test Data - Start")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import keras
import keras.backend as K
import pandas
import matplotlib.pyplot as plt

from keras.models import Sequential, load_model
from keras.layers.core import Activation
from keras.layers import Dense, Dropout, LSTM, Flatten
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing
from Prediction import Predict

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# fix random seed for reproducibility
np.random.seed(7)

# pick a large window size of 50 cycles
sequence_length = 50

# pick the feature columns 
sequence_cols = ['LtLt', 'LtRt', 'RtLt', 'RtRt']

class PrepareAndPredict:

    def PrepareForTesting(self):
        train_df = pandas.read_csv('../KerasFeasibility/Data/DataToTeach.csv', engine='python')
        train_df.columns = ['id', 'cycle', 'LtLt', 'LtRt', 'RtLt', 'RtRt']
        #train_df = array(train_df.values)
        train_df = train_df.sort_values(['id','cycle'])
        
        #######
        # TRAIN
        #######
        # Data Labeling - generate column RUL(Remaining Usefull Life or Time to Failure)
        rul = pd.DataFrame(train_df.groupby('id')['cycle'].max()).reset_index()
        rul.columns = ['id', 'max']
        train_df = train_df.merge(rul, on=['id'], how='left')
        train_df['RUL'] = train_df['max'] - train_df['cycle']
        train_df.drop('max', axis=1, inplace=True)
        
        # MinMax normalization (from 0 to 1)
        train_df['cycle_norm'] = train_df['cycle']
        self.cols_normalize = train_df.columns.difference(['id','cycle','RUL'])
        self.min_max_scaler = preprocessing.MinMaxScaler()
        norm_train_df = pd.DataFrame(self.min_max_scaler.fit_transform(train_df[self.cols_normalize]), 
                                     columns=self.cols_normalize, 
                                     index=train_df.index)
        join_df = train_df[train_df.columns.difference(self.cols_normalize)].join(norm_train_df)
        train_df = join_df.reindex(columns = train_df.columns)

# # read test data - It is the sealer temperature data without failure events recorded.
# test_df = pd.read_csv('test_data.csv', engine='python')
# test_df.columns = ['id', 'cycle', 'LtLt', 'LtRt', 'RtLt', 'RtRt']

# # Data Labeling - generate column RUL(Remaining Usefull Life or Time to Failure)
# rul = pd.DataFrame(test_df.groupby('id')['cycle'].max()).reset_index()
# rul.columns = ['id', 'max']
# test_df = test_df.merge(rul, on=['id'], how='left')
# test_df['RUL'] = test_df['max'] - test_df['cycle']
# test_df.drop('max', axis=1, inplace=True)
# test_df.to_csv("test_data_with_rul.csv")
# test_df.drop('RUL', axis=1, inplace=True)

    def NormalizeData(self, test_df):
        # MinMax normalization (from 0 to 1)
        test_df['cycle_norm'] = test_df['cycle']
        norm_test_df = pd.DataFrame(self.min_max_scaler.transform(test_df[self.cols_normalize]), 
                                    columns=self.cols_normalize, 
                                    index=test_df.index)
        test_join_df = test_df[test_df.columns.difference(self.cols_normalize)].join(norm_test_df)
        test_df = test_join_df.reindex(columns = test_df.columns)
        test_df = test_df.reset_index(drop=True)
        print(test_df.head())
        return test_df

    def PredictRUL(self, seq_array_test):
        y_pred_test = Predict(seq_array_test)      
        return y_pred_test