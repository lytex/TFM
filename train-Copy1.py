# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.2
#   kernelspec:
#     display_name: TFM
#     language: python
#     name: tfm
# ---

# %% id="GfWBD_IJKfGJ"
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
from LCWavelet import *

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense, concatenate,Conv1D, Flatten,Dropout , BatchNormalization, MaxPooling1D

# %% id="xwin1Ts_RROr"
dataset_path= 'waveletsK/'
df_path = 'light_curves_K_stars_filter.csv'
train_split = .80

# %% colab={"base_uri": "https://localhost:8080/"} id="xl3jfjC22rxO" outputId="659e74e8-34c0-4a2e-c5ab-6817ead11bd7"
ds_p_8,ds_i_8,label_8 = generate_dataset_model_1(dataset_path,level=8, progress=False)
ds_p_8 = normalize_data(ds_p_8)
ds_i_8 = normalize_data(ds_i_8)
X_train_8, X_test_8, y_train_8, y_test_8 = split_dataset(ds_p_8, ds_i_8, label_8)

print(f"datos entrenamiento:{len(X_train_8[0])}, labels:{len(y_train_8)}")
print(f"datos validacion:{len(X_test_8[0])}, labels:{len(y_test_8)}")
print(f"input shape par:{np.shape(X_train_8[0])} , inpar:{np.shape(X_train_8[0])}")

# %%
from binning import bin_and_aggregate
np.squeeze(X_train_8).shape
bin_and_aggregate(X_train_8[0][:, :, 0].shape
# bin_and_aggregate

# %% [markdown] id="4QdUCI1qmkzS"
# # Construnccion del modelo 1
#     [Conv1D(16,5)]    [Conv1D(16,5)]
#     [Conv1D(16,5)]    [Conv1D(16,5)]
#     [MaxPool(3,2)]    [MaxPool(3,2)]
#     [Conv1D(32,5)]    [Conv1D(32,5)]
#     [Conv1D(32,5)]    [Conv1D(32,5)]
#     [MaxPool(3,2)]    [MaxPool(3,2)]
#       [Flatten()]      [Flatten()]
#                 [Concat]
#             [Dense(512relu)]
#             [Dense(512relu)]
#             [Dense(512relu)]
#             [Dense(512relu)]
#             [Dense(1sigmoid)]
#
#
#
#

# %% id="3QQJ5W2d4gfZ"
def gen_model_1_level(ds,activation = 'relu'): 
  input_shape = np.shape(ds)[2:]
  print(input_shape)
  model_p = tf.keras.Sequential()
  model_p.add(Conv1D(filters=16, kernel_size=5, activation='relu', input_shape=input_shape))
  model_p.add(Conv1D(filters=16, kernel_size=5, activation='relu')) 
  model_p.add(MaxPooling1D(pool_size=3, strides=1)) 
  model_p.add(Conv1D(32,5, activation='relu'))
  model_p.add(Conv1D(32,5, activation='relu'))
  model_p.add(MaxPooling1D(pool_size=3, strides=1))
  model_p.add(Flatten())

  model_i = tf.keras.Sequential()
  model_i.add(Conv1D(filters=16, kernel_size=5, activation='relu', input_shape=input_shape))
  model_i.add(Conv1D(filters=16, kernel_size=5, activation='relu')) 
  model_i.add(MaxPooling1D(pool_size=3, strides=1)) 
  model_i.add(Conv1D(32,5, activation='relu'))
  model_i.add(Conv1D(32,5, activation='relu'))
  model_i.add(MaxPooling1D(pool_size=3, strides=1))
  model_i.add(Flatten())

  model_f = concatenate([model_p.output,model_i.output], axis=-1)
  model_f = Dense(512,activation='relu')(model_f)
  model_f = Dense(512,activation='relu')(model_f)
  model_f = Dense(512,activation='relu')(model_f)
  model_f = Dense(512,activation='relu')(model_f)
  model_f = Dense(1,activation='sigmoid')(model_f)
  model_f = Model([model_p.input,model_i.input],model_f)
  return model_f


# %% colab={"base_uri": "https://localhost:8080/", "height": 563} id="s-vaJmZIe9dM" outputId="bf223032-ebbc-4286-b623-64cf808cb4cb"
model_1 =  gen_model_1_level(X_train_8 ,activation = tf.keras.layers.LeakyReLU())
model_1.compile(loss = 'binary_crossentropy', optimizer=tf.keras.optimizers.Adam(), metrics=['accuracy','binary_crossentropy'])
history_1 = model_1.fit(X_train_8, y_train_8, epochs=1000, batch_size=64, validation_data=(X_test_8, y_test_8))

# %% colab={"base_uri": "https://localhost:8080/", "height": 563} id="s-vaJmZIe9dM" outputId="bf223032-ebbc-4286-b623-64cf808cb4cb"
plot_results(history_1)
for key in history_1.history.keys():
    fig, ax = plt.subplots()
    ax.plot(history_1.history[key], 'b*-')
    ax.set_title(key)
