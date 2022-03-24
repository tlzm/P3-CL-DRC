import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import math
from deepRC import SimpleDeepReservoirLayer
from tensorflow.python.keras.layers.core import Flatten,Dense,Dropout
from tensorflow.python.keras.models import Sequential,load_model
import tensorflow.python.keras.backend as K

output_path = 'model/regression_model_v7'
sequence_length = 30
test_data1 = pd.read_csv("input/Bearing1_1_Set.csv")
test_data2 = pd.read_csv("input/Bearing1_2_Set.csv")
test_data3 = pd.read_csv("input/Bearing1_3_Set.csv")
test_data4 = pd.read_csv("input/Bearing1_4_Set.csv")
test_data5 = pd.read_csv("input/Bearing1_5_Set.csv")
test_data6 = pd.read_csv("input/Bearing1_6_Set.csv")
test_data7 = pd.read_csv("input/Bearing1_7_Set.csv")
#test_data = pd.read_csv("input/TestingSet2.csv")
#test_data = pd.read_csv("input/TestingSet3.csv")

feature_cols = ['haccel_var','vaccel_var','haccel_skew','vaccel_skew','haccel_rms','vaccel_rms','haccel_freq_vars','vaccel_freq_vars','haccel_freq_means','vaccel_freq_means']

def reshapeFeatures(id_df, seq_length, Feature):
    """
    Only sequences that meet the window-length are considered, no padding is used. This means for testing
    we need to drop those which are below the window-length.
    """
    data_matrix = id_df[Feature].values
    num_elements = data_matrix.shape[0] # 输出行数
    for start, stop in zip(range(0, num_elements-seq_length+1), range(seq_length, num_elements+1)):
        yield data_matrix[start:stop,:]

# function to generate label
def reshapeLabel(id_df, seq_length=sequence_length, label=['RUL']):
    """ Only sequences that meet the window-length are considered, no padding is used. This means for testing
    we need to drop those which are below the window-length."""
    data_matrix = id_df[label].values
    num_elements = data_matrix.shape[0]
    return data_matrix[seq_length-1: num_elements, :]


fea_gen1 = [list(reshapeFeatures(test_data1, sequence_length, feature_cols))] 
fea_gen2 = [list(reshapeFeatures(test_data2, sequence_length, feature_cols))] 
fea_gen3 = [list(reshapeFeatures(test_data3, sequence_length, feature_cols))] 
fea_gen4 = [list(reshapeFeatures(test_data4, sequence_length, feature_cols))] 
fea_gen5 = [list(reshapeFeatures(test_data5, sequence_length, feature_cols))] 
fea_gen6 = [list(reshapeFeatures(test_data6, sequence_length, feature_cols))] 
fea_gen7 = [list(reshapeFeatures(test_data7, sequence_length, feature_cols))] 
fea_array1 = np.concatenate(list(fea_gen1)).astype(np.float32)
fea_array2 = np.concatenate(list(fea_gen2)).astype(np.float32)
fea_array3 = np.concatenate(list(fea_gen3)).astype(np.float32)
fea_array4 = np.concatenate(list(fea_gen4)).astype(np.float32)
fea_array5 = np.concatenate(list(fea_gen5)).astype(np.float32)
fea_array6 = np.concatenate(list(fea_gen6)).astype(np.float32)
fea_array7 = np.concatenate(list(fea_gen7)).astype(np.float32)



# generate labels
label_gen1 = [reshapeLabel(test_data1)]
label_gen2 = [reshapeLabel(test_data2)]
label_gen3 = [reshapeLabel(test_data3)]
label_gen4 = [reshapeLabel(test_data4)]
label_gen5 = [reshapeLabel(test_data5)]
label_gen6 = [reshapeLabel(test_data6)]
label_gen7 = [reshapeLabel(test_data7)]
label_array1 = np.concatenate(label_gen1).astype(np.float32)
label_array2 = np.concatenate(label_gen2).astype(np.float32)
label_array3 = np.concatenate(label_gen3).astype(np.float32)
label_array4 = np.concatenate(label_gen4).astype(np.float32)
label_array5 = np.concatenate(label_gen5).astype(np.float32)
label_array6 = np.concatenate(label_gen6).astype(np.float32)
label_array7 = np.concatenate(label_gen7).astype(np.float32)



def root_mean_squared_error(y_true, y_pred): 
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))

def exps(y_true, y_pred):
        return K.mean(K.square(y_pred**2 - y_true**2), axis=-1) /10000




estimator = tf.saved_model.load(output_path)


y_pred_test = estimator(fea_array1)
y_pred_test = np.transpose(y_pred_test)
y_true_test = label_array1
y_true_test = np.transpose(y_true_test)

print(root_mean_squared_error(y_true_test,y_pred_test))


y_pred_test = estimator(fea_array2)
y_pred_test = np.transpose(y_pred_test)
y_true_test = label_array2
y_true_test = np.transpose(y_true_test)

print(root_mean_squared_error(y_true_test,y_pred_test))


y_pred_test = estimator(fea_array3)
y_pred_test = np.transpose(y_pred_test)
y_true_test = label_array3
y_true_test = np.transpose(y_true_test)

print(root_mean_squared_error(y_true_test,y_pred_test))


y_pred_test = estimator(fea_array4)
y_pred_test = np.transpose(y_pred_test)
y_true_test = label_array4
y_true_test = np.transpose(y_true_test)

print(root_mean_squared_error(y_true_test,y_pred_test))


y_pred_test = estimator(fea_array5)
y_pred_test = np.transpose(y_pred_test)
y_true_test = label_array5
y_true_test = np.transpose(y_true_test)

print(root_mean_squared_error(y_true_test,y_pred_test))


y_pred_test = estimator(fea_array6)
y_pred_test = np.transpose(y_pred_test)
y_true_test = label_array6
y_true_test = np.transpose(y_true_test)

print(root_mean_squared_error(y_true_test,y_pred_test))


y_pred_test = estimator(fea_array7)
y_pred_test = np.transpose(y_pred_test)
y_true_test = label_array7
y_true_test = np.transpose(y_true_test)

print(root_mean_squared_error(y_true_test,y_pred_test))

