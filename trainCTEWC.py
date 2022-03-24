from unittest.main import MODULE_EXAMPLES
import tensorflow as tf
from tensorflow import keras
#import keras.backend as K
#from keras.layers.core import Activation
#from keras.models import Sequential
#from keras.layers import Dense, Dropout, LSTM,Conv1D,Flatten 
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from deepRC import SimpleDeepReservoirLayer

#from keras.layers.core import Flatten,Dense,Dropout
#from keras.models import Sequential,load_model
#import keras.backend as K
from tensorflow.python.keras.regularizers import l2
from tensorflow.python.keras.layers.core import Flatten,Dense,Dropout
from tensorflow.python.keras.models import Sequential,load_model
import tensorflow.python.keras.backend as K
from copy import deepcopy


output_path1 = 'model/regression_model_v1'
output_path2 = 'model/regression_model_v2'
output_path3 = 'model/regression_model_v3'
output_path4 = 'model/regression_model_v4'
output_path5 = 'model/regression_model_v5'
output_path6 = 'model/regression_model_v6'
output_path7 = 'model/regression_model_v7'

train_data1 = pd.read_csv('input/Bearing1_1_set.csv')
train_data2 = pd.read_csv('input/Bearing1_2_set.csv')
train_data3 = pd.read_csv('input/Bearing1_3_set.csv')
train_data4 = pd.read_csv('input/Bearing1_4_set.csv')
train_data5 = pd.read_csv('input/Bearing1_5_set.csv')
train_data6 = pd.read_csv('input/Bearing1_6_set.csv')
train_data7 = pd.read_csv('input/Bearing1_7_set.csv')


# time windows
sequence_length = 30

def reshapeFeatures(id_df, seq_length, Feature):
    """
    Only sequences that meet the window-length are considered, no padding is used. This means for testing
    we need to drop those which are below the window-length.
    """
    data_matrix = id_df[Feature].values
    num_elements = data_matrix.shape[0] # 输出行数
    for start, stop in zip(range(0, num_elements-seq_length+1), range(seq_length, num_elements+1)):
        yield data_matrix[start:stop, :]

# pick the feature columns 
feature_col = ['haccel_var','vaccel_var','haccel_skew','vaccel_skew','haccel_rms','vaccel_rms','haccel_freq_vars','vaccel_freq_vars','haccel_freq_means','vaccel_freq_means']
#print(feature_col)

# generator for the sequences
fea_gen1 = [list(reshapeFeatures(train_data1, sequence_length, feature_col))]
fea_gen2 = [list(reshapeFeatures(train_data2, sequence_length, feature_col))]
fea_gen3 = [list(reshapeFeatures(train_data3, sequence_length, feature_col))]
fea_gen4 = [list(reshapeFeatures(train_data4, sequence_length, feature_col))]
fea_gen5 = [list(reshapeFeatures(train_data5, sequence_length, feature_col))]
fea_gen6 = [list(reshapeFeatures(train_data6, sequence_length, feature_col))]
fea_gen7 = [list(reshapeFeatures(train_data7, sequence_length, feature_col))]

# generate sequences and convert to numpy array
fea_array1 = np.concatenate(list(fea_gen1)).astype(np.float32)
fea_array2 = np.concatenate(list(fea_gen2)).astype(np.float32)
fea_array3 = np.concatenate(list(fea_gen3)).astype(np.float32)
fea_array4 = np.concatenate(list(fea_gen4)).astype(np.float32)
fea_array5 = np.concatenate(list(fea_gen5)).astype(np.float32)
fea_array6 = np.concatenate(list(fea_gen6)).astype(np.float32)
fea_array7 = np.concatenate(list(fea_gen7)).astype(np.float32)

print(fea_array1.shape)
#print("The data set has now shape: {} entries, {} time windows and {} features.".format(fea_array.shape[0],fea_array.shape[1],fea_array.shape[2]))

# function to generate label
def reshapeLabel(id_df, seq_length=sequence_length, label=['RUL']):
    """ Only sequences that meet the window-length are considered, no padding is used. This means for testing
    we need to drop those which are below the window-length."""
    data_matrix = id_df[label].values
    num_elements = data_matrix.shape[0]
    return data_matrix[seq_length-1: num_elements, :]

# generate labels
label_gen1 = [reshapeLabel(train_data1)]
label_gen2 = [reshapeLabel(train_data2)]
label_gen3 = [reshapeLabel(train_data3)]
label_gen4 = [reshapeLabel(train_data4)]
label_gen5 = [reshapeLabel(train_data5)]
label_gen6 = [reshapeLabel(train_data6)]
label_gen7 = [reshapeLabel(train_data7)]

label_array1 = np.concatenate(label_gen1).astype(np.float32)
label_array2 = np.concatenate(label_gen2).astype(np.float32)
label_array3 = np.concatenate(label_gen3).astype(np.float32)
label_array4 = np.concatenate(label_gen4).astype(np.float32)
label_array5 = np.concatenate(label_gen5).astype(np.float32)
label_array6 = np.concatenate(label_gen6).astype(np.float32)
label_array7 = np.concatenate(label_gen7).astype(np.float32)


print(label_array1.shape)



# EWC

def fisher_matrix(model, dataset, samples):
    inputs, labels = dataset
    weights = model.trainable_weights
    variance = [tf.zeros_like(tensor) for tensor in weights]

    for _ in range(samples):
        # Select a random element from the dataset.
        index = np.random.randint(len(inputs))
        data = inputs[index]

        # When extracting from the array we lost a dimension so put it back.
        data = tf.expand_dims(data, axis=0)

        # Collect gradients.
        with tf.GradientTape() as tape:
            output = model(data)
            log_likelihood = tf.math.log(output)

        gradients = tape.gradient(log_likelihood, weights)

        # If the model has converged, we can assume that the current weights
        # are the mean, and each gradient we see is a deviation. The variance is
        # the average of the square of this deviation.
        variance = [var + (grad ** 2) for var, grad in zip(variance, gradients)]

    fisher_diagonal = [tensor / samples for tensor in variance]
    return fisher_diagonal

def ewc_loss(lam, model, dataset, samples):

    optimal_weights = deepcopy(model.trainable_weights)
    fisher_diagonal = fisher_matrix(model, dataset, samples)

    def loss_fn(new_model):
        # We're computing:
        # sum [(lambda / 2) * F * (current weights - optimal weights)^2]
        loss = 0
        current = new_model.trainable_weights
        for f, c, o in zip(fisher_diagonal, current, optimal_weights):
            loss += tf.reduce_mean(f * ((c - o) ** 2))

        return loss * (lam / 2)

    return loss_fn

def root_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1)) 

def custom_loss(y_true, y_pred):
    loss = root_mean_squared_error(y_true, y_pred)/100
    extra_losses = regularisers
    for fn in extra_losses:
        loss += fn(model)

    return loss

def apply_mask(gradients, mask):

    if mask is not None:
        gradients = [grad * tf.cast(mask, tf.float32)
                     for grad, mask in zip(gradients, mask)]
    return gradients



# MODEL


nb_features = fea_array1.shape[2]
nb_out = label_array1.shape[1]

model = Sequential()
model.add(SimpleDeepReservoirLayer(layers=5))
#model.add(SimpleDeepReservoirLayer())
#model.add(Conv1D(filters=64,kernel_size=2,activation='swish',input_shape=(sequence_length, nb_features)))
#model.add(Conv1D(filters=16,kernel_size=2))
#model.add(Flatten())

model.add(Dense(units=100, name="dense_0"))
model.add(Dropout(0.2, name="dropout_2"))

model.add(Dense(units=16, name="dense_1"))
model.add(Dropout(0.2, name="dropout_3"))
model.add(Dense(units=nb_out, name="dense_2"))
#model.add(Activation("relu", name="activation_0"))
model.compile(loss='mse', optimizer='adam', metrics=['mae'])

fepoch = 1000
epochs = 100
batch_size = 100 


class SaveModel1(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.val_loss = []
        self.loss =[]
        self.D_loss =[]
    def on_epoch_end(self, epoch, logs=None):
        current_D_loss = logs.get("val_mae")/2 + logs.get("mae") 
        self.D_loss.append(logs.get("val_mae")/2+logs.get("mae"))
        if current_D_loss <= min(self.D_loss):
            print('Find better ones. Saving entire model.')
            model.save(output_path1)
            model.save_weights(output_path1)

class SaveModel2(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.val_loss = []
        self.loss =[]
        self.D_loss =[]
    def on_epoch_end(self, epoch, logs=None):
        current_D_loss = logs.get("val_loss")/2 + logs.get("loss") 
        self.D_loss.append(logs.get("val_loss")/2+logs.get("loss"))
        if current_D_loss <= min(self.D_loss):
            print('Find better ones. Saving entire model.')
            model.save(output_path2)
            model.save_weights(output_path2)
            
class SaveModel3(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.val_loss = []
        self.loss =[]
        self.D_loss =[]
    def on_epoch_end(self, epoch, logs=None):
        current_D_loss = logs.get("val_loss")/2 + logs.get("loss") 
        self.D_loss.append(logs.get("val_loss")/2+logs.get("loss"))
        if current_D_loss <= min(self.D_loss):
            print('Find better ones. Saving entire model.')
            model.save(output_path3)
            model.save_weights(output_path3)

class SaveModel4(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.val_loss = []
        self.loss =[]
        self.D_loss =[]
    def on_epoch_end(self, epoch, logs=None):
        current_D_loss = logs.get("val_loss")/2 + logs.get("loss") 
        self.D_loss.append(logs.get("val_loss")/2+logs.get("loss"))
        if current_D_loss <= min(self.D_loss):
            print('Find better ones. Saving entire model.')
            model.save(output_path4)
            model.save_weights(output_path4)
            
class SaveModel5(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.val_loss = []
        self.loss =[]
        self.D_loss =[]
    def on_epoch_end(self, epoch, logs=None):
        current_D_loss = logs.get("val_loss")/2 + logs.get("loss") 
        self.D_loss.append(logs.get("val_loss")/2+logs.get("loss"))
        if current_D_loss <= min(self.D_loss):
            print('Find better ones. Saving entire model.')
            model.save(output_path5)
            model.save_weights(output_path5)
            
class SaveModel6(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.val_loss = []
        self.loss =[]
        self.D_loss =[]
    def on_epoch_end(self, epoch, logs=None):
        current_D_loss = logs.get("val_loss")/2 + logs.get("loss") 
        self.D_loss.append(logs.get("val_loss")/2+logs.get("loss"))
        if current_D_loss <= min(self.D_loss):
            print('Find better ones. Saving entire model.')
            model.save(output_path6)
            model.save_weights(output_path6)
            
class SaveModel7(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.val_loss = []
        self.loss =[]
        self.D_loss =[]
    def on_epoch_end(self, epoch, logs=None):
        current_D_loss = logs.get("val_loss")/2 + logs.get("loss") 
        self.D_loss.append(logs.get("val_loss")/2+logs.get("loss"))
        if current_D_loss <= min(self.D_loss):
            print('Find better ones. Saving entire model.')
            model.save(output_path7)
            model.save_weights(output_path7)

save_model1 = SaveModel1()
save_model2 = SaveModel2()
save_model3 = SaveModel3()
save_model4 = SaveModel4()
save_model5 = SaveModel5()
save_model6 = SaveModel6()
save_model7 = SaveModel7()

# fit the network
history = model.fit(fea_array1, label_array1, epochs=fepoch, batch_size=batch_size,shuffle=True, validation_split=0.1, verbose=1,
          callbacks =save_model1
          )

print(history.history.keys())
print("Model saved as {}".format(output_path1))


regularisers = []
gradient_mask = []

loss_fn = ewc_loss(0.8, model, (fea_array2, label_array2),100)
regularisers.append(loss_fn)
model.compile(loss=custom_loss,optimizer="adam", metrics=[root_mean_squared_error])
training_data = (fea_array2, label_array2)
dataset = tf.data.Dataset.from_tensor_slices(training_data)
dataset = dataset.shuffle(len(training_data[0])).batch(100)

for inputs, labels in dataset:
    with tf.GradientTape() as tape:
        outputs = model(inputs)
        loss = model.compiled_loss(labels, outputs)
    gradients = tape.gradient(loss, model.trainable_weights)
    gradients = apply_mask(gradients, gradient_mask)
    model.optimizer.apply_gradients(zip(gradients, model.trainable_weights))


history = model.fit(fea_array2, label_array2, epochs=epochs, batch_size=batch_size,shuffle=True, validation_split=0.1, verbose=1,
          callbacks =save_model2
          )

print(history.history.keys())
print("Model saved as {}".format(output_path2))



loss_fn = ewc_loss(0.8, model, (fea_array3, label_array3),100)
regularisers.append(loss_fn)
model.compile(loss=custom_loss,optimizer="adam", metrics=[root_mean_squared_error])
training_data = (fea_array3, label_array3)
dataset = tf.data.Dataset.from_tensor_slices(training_data)
dataset = dataset.shuffle(len(training_data[0])).batch(100)

for inputs, labels in dataset:
    with tf.GradientTape() as tape:
        outputs = model(inputs)
        loss = model.compiled_loss(labels, outputs)
    gradients = tape.gradient(loss, model.trainable_weights)
    gradients = apply_mask(gradients, gradient_mask)
    model.optimizer.apply_gradients(zip(gradients, model.trainable_weights))

history = model.fit(fea_array3, label_array3, epochs=epochs, batch_size=batch_size,shuffle=True, validation_split=0.1, verbose=1,
          callbacks =save_model3
          )

print(history.history.keys())
print("Model saved as {}".format(output_path3))




loss_fn = ewc_loss(0.8, model, (fea_array4, label_array4),100)
regularisers.append(loss_fn)
model.compile(loss=custom_loss,optimizer="adam", metrics=[root_mean_squared_error])
training_data = (fea_array4, label_array4)
dataset = tf.data.Dataset.from_tensor_slices(training_data)
dataset = dataset.shuffle(len(training_data[0])).batch(100)

for inputs, labels in dataset:
    with tf.GradientTape() as tape:
        outputs = model(inputs)
        loss = model.compiled_loss(labels, outputs)
    gradients = tape.gradient(loss, model.trainable_weights)
    gradients = apply_mask(gradients, gradient_mask)
    model.optimizer.apply_gradients(zip(gradients, model.trainable_weights))

history = model.fit(fea_array4, label_array4, epochs=epochs, batch_size=batch_size,shuffle=True, validation_split=0.1, verbose=1,
          callbacks =save_model4
          )

print(history.history.keys())
print("Model saved as {}".format(output_path4))


loss_fn = ewc_loss(0.9, model, (fea_array5, label_array5),100)
regularisers.append(loss_fn)
model.compile(loss=custom_loss,optimizer="adam", metrics=[root_mean_squared_error])
training_data = (fea_array5, label_array5)
dataset = tf.data.Dataset.from_tensor_slices(training_data)
dataset = dataset.shuffle(len(training_data[0])).batch(100)

for inputs, labels in dataset:
    with tf.GradientTape() as tape:
        outputs = model(inputs)
        loss = model.compiled_loss(labels, outputs)
    gradients = tape.gradient(loss, model.trainable_weights)
    gradients = apply_mask(gradients, gradient_mask)
    model.optimizer.apply_gradients(zip(gradients, model.trainable_weights))

history = model.fit(fea_array5, label_array5, epochs=epochs, batch_size=batch_size,shuffle=True, validation_split=0.1, verbose=1,
          callbacks =save_model5
          )

print(history.history.keys())
print("Model saved as {}".format(output_path5))


loss_fn = ewc_loss(0.8, model, (fea_array6, label_array6),100)
regularisers.append(loss_fn)
model.compile(loss=custom_loss,optimizer="adam", metrics=[root_mean_squared_error])
training_data = (fea_array6, label_array6)
dataset = tf.data.Dataset.from_tensor_slices(training_data)
dataset = dataset.shuffle(len(training_data[0])).batch(100)

for inputs, labels in dataset:
    with tf.GradientTape() as tape:
        outputs = model(inputs)
        loss = model.compiled_loss(labels, outputs)
    gradients = tape.gradient(loss, model.trainable_weights)
    gradients = apply_mask(gradients, gradient_mask)
    model.optimizer.apply_gradients(zip(gradients, model.trainable_weights))

history = model.fit(fea_array6, label_array6, epochs=epochs, batch_size=batch_size,shuffle=True, validation_split=0.1, verbose=1,
          callbacks =save_model6
          )

print(history.history.keys())
print("Model saved as {}".format(output_path6))


loss_fn = ewc_loss(0.8, model, (fea_array7, label_array7),100)
regularisers.append(loss_fn)
model.compile(loss=custom_loss,optimizer="adam", metrics=[root_mean_squared_error])
training_data = (fea_array7, label_array7)
dataset = tf.data.Dataset.from_tensor_slices(training_data)
dataset = dataset.shuffle(len(training_data[0])).batch(100)

for inputs, labels in dataset:
    with tf.GradientTape() as tape:
        outputs = model(inputs)
        loss = model.compiled_loss(labels, outputs)
    gradients = tape.gradient(loss, model.trainable_weights)
    gradients = apply_mask(gradients, gradient_mask)
    model.optimizer.apply_gradients(zip(gradients, model.trainable_weights))

history = model.fit(fea_array7, label_array7, epochs=epochs, batch_size=batch_size,shuffle=True, validation_split=0.1, verbose=1,
          callbacks =save_model7
          )

print(history.history.keys())
print("Model saved as {}".format(output_path7))