print ("Prediction - Start")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import keras
import keras.backend as K
import pandas
import matplotlib.pyplot as plt

from keras.models import Sequential, load_model

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
   
# define path to save model
model_path = 'sealTemp_model.h5'

# fix random seed for reproducibility
np.random.seed(7)

def r2_keras(y_true, y_pred):
    """Coefficient of Determination 
    """
    SS_res =  K.sum(K.square( y_true - y_pred ))
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) )
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )

##################################
# EVALUATE ON TEST DATA
##################################

# # We pick the last sequence for each id in the test data
# seq_array_test_last = [test_df[test_df['id']==id][sequence_cols].values[-sequence_length:] 
#                        for id in test_df['id'].unique() if len(test_df[test_df['id']==id]) >= sequence_length]
# 
# seq_array_test_last = np.asarray(seq_array_test_last).astype(np.float32)
# print("seq_array_test_last")
# #print(seq_array_test_last)
# print(seq_array_test_last.shape)

def Predict(seq_array_test):
    # if best iteration's model was saved then load and use it
    if os.path.isfile(model_path):
        estimator = load_model(model_path,custom_objects={'r2_keras': r2_keras})
        y_pred_test = estimator.predict(seq_array_test)
        test_set = pd.DataFrame(y_pred_test)
        test_set.to_csv('submit_test.csv', index = None)   
        
    return y_pred_test    
print ("Prediction - End")