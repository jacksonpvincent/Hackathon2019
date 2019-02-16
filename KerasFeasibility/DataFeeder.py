print ("Data Feeder - Start")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import keras
import keras.backend as K
import pandas
import matplotlib.pyplot as plt

from PrepareAndPredict import PrepareAndPredict

predictionClass = PrepareAndPredict()

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# fix random seed for reproducibility
np.random.seed(7)

# pick a large window size of 50 cycles
sequence_length = 50

# pick the feature columns 
sequence_cols = ['LtLt', 'LtRt', 'RtLt', 'RtRt']

# # read test data - It is the sealer temperature data without failure events recorded.
# test_df = pd.read_csv('test_data.csv', engine='python')
# test_df.columns = ['id', 'cycle', 'LtLt', 'LtRt', 'RtLt', 'RtRt']

# function to reshape features into (samples, time steps, features) 
def gen_sequence(id_df, seq_length, seq_cols):
    """ Only sequences that meet the window-length are considered, no padding is used. This means for testing
    we need to drop those which are below the window-length. An alternative would be to pad sequences so that
    we can use shorter ones """
    # for one id I put all the rows in a single matrix
    data_matrix = id_df[seq_cols].values
    num_elements = data_matrix.shape[0]
    # Iterate over two lists in parallel.
    # For example id1 have 192 rows and sequence_length is equal to 50
    # so zip iterate over two following list of numbers (0,112),(50,192)
    # 0 50 -> from row 0 to row 50
    # 1 51 -> from row 1 to row 51
    # 2 52 -> from row 2 to row 52
    # ...
    # 111 191 -> from row 111 to 191
    for start, stop in zip(range(0, num_elements-seq_length), range(seq_length, num_elements)):
        yield data_matrix[start:stop, :]

class DataFeeder:
    
    def __init__(self, dataFrame):
        self.df = dataFrame
        self.counter = 0
    
    def SetDataRecord(self, dataFrame):  
        if self.counter == 0:
            self.df = dataFrame
            self.df.columns = ['id', 'cycle', 'LtLt', 'LtRt', 'RtLt', 'RtRt']
            self.counter = 1     
        else:
            self.df = self.df.append(dataFrame)
           
        if len(self.df) >= 50:
            # generator for the sequences
            # transform each id of the train dataset in a sequence
            predictionClass.PrepareForTesting() 
            normalizedData = predictionClass.NormalizeData(self.df)
#             seq_gen = (list(gen_sequence(normalizedData[normalizedData['id']==id], sequence_length, sequence_cols)) 
#                        for id in normalizedData['id'].unique())  
            seq_gen = [normalizedData[normalizedData['id']==id][sequence_cols].values[-sequence_length:] 
                        for id in normalizedData['id'].unique() if len(normalizedData[normalizedData['id']==id]) >= sequence_length]
            seq_array = np.concatenate(list(seq_gen)).astype(np.float32)
            seq_array_test = np.asarray(seq_array).astype(np.float32)
            y_pred_test = predictionClass.PredictRUL(seq_array_test)
            
            timepoints=[];taperTempRul={}
            taperTempRul["name"] = "RUL"
            taperTempRul["value"] = y_pred_test
            taperTempRul["timestamp"] = int(time.time()*1000)
            timepoints.append(taperTempRul)
            r = requests.post("http://localhost:8083/api/v1/datapoints", data=json.dumps(timepoints))
            print ("RUL sent")


# # # generate sequences and convert to numpy array
# # seq_array = np.concatenate(list(seq_gen)).astype(np.float32)
# # seq_array_test = np.asarray(seq_array).astype(np.float32)
# 
# # Plot in blue color the predicted data and in green color the
# # actual data to verify visually the accuracy of the model.
# fig_verify = plt.figure(figsize=(50, 40))
# plt.plot(y_pred_test, color="blue")
# plt.title('prediction')
# plt.ylabel('value')
# plt.xlabel('row')
# plt.legend(['predicted', 'actual data'], loc='upper left')
# plt.show()
# fig_verify.savefig("model_regression_verify1.png")

print ("Data Feeder - End")