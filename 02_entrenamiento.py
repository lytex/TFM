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

# %%
import warnings
warnings.filterwarnings(action="ignore", category=DeprecationWarning)
warnings.filterwarnings(action="ignore", category=FutureWarning)
from LCWavelet import *
from tqdm import tqdm
from collections import defaultdict
from parallelbar import progress_map
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, concatenate,Conv1D, Flatten,Dropout , BatchNormalization, MaxPooling1D
from tensorflow.keras.models import Model, Sequential
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from functools import partial
import datetime

path = "all_data_2024-06-11/"
files = [file for file in os.listdir(path) if file.endswith(".pickle") and "wavelet" in file]
lightcurves = []

def load_files(file, path):
    global_local = LightCurveWaveletGlobalLocalCollection.from_pickle(path+file)
    # try:
    #     getattr(global_local, "levels")
    # except AttributeError:
    #     global_local.levels = [1, 2, 3, 4]
    return global_local

func = partial(load_files, path=path)

lightcurves = progress_map(func, files, n_cpu=64, total=len(files), executor='processes', error_behavior='raise')

# for file in tqdm(files):
#     lightcurves.append(func(file))

# %%

pliegue_par_global = defaultdict(list)
pliegue_impar_global = defaultdict(list)
pliegue_par_local = defaultdict(list)
pliegue_impar_local = defaultdict(list)

for lc in lightcurves:
    for level in range(1, lc.levels_global+1):
        pliegue_par_global[level].append(lc.pliegue_par_global.get_approximation_coefficent(level=level))
        pliegue_impar_global[level].append(lc.pliegue_impar_global.get_approximation_coefficent(level=level))
    for level in range(1, lc.levels_local+1):
        pliegue_par_local[level].append(lc.pliegue_par_local.get_approximation_coefficent(level=level))
        pliegue_impar_local[level].append(lc.pliegue_impar_local.get_approximation_coefficent(level=level))
        

pliegue_par_global = {k: np.array(v) for k, v in pliegue_par_global.items() if k in (1 ,2)}
pliegue_par_global = {k: v.reshape(list(v.shape)+[1]) for k, v in pliegue_par_global.items()}
pliegue_impar_global = {k: np.array(v) for k, v in pliegue_impar_global.items() if k in (1, 2)}
pliegue_impar_global = {k: v.reshape(list(np.shape(v))+[1]) for k, v in pliegue_impar_global.items()}


pliegue_par_local = {k: np.array(v) for k, v in pliegue_par_local.items() if k in (1, 2)}
pliegue_par_local = {k: v.reshape(list(np.shape(v))+[1]) for k, v in pliegue_par_local.items()}
pliegue_impar_local = {k: np.array(v) for k, v in pliegue_impar_local.items() if k in (1, 2)}
pliegue_impar_local = {k: v.reshape(list(np.shape(v))+[1]) for k, v in pliegue_impar_local.items()}


inputs = (pliegue_par_global, pliegue_impar_global), (pliegue_par_local, pliegue_impar_local)

# %%

# %%
# inputs
# %pdb off
from math import ceil
def gen_model_2_levels(inputs, classes, activation = 'relu',summary=False):
    
    (pliegue_par_global, pliegue_impar_global), (pliegue_par_local, pliegue_impar_local) = inputs    

    input_shape_global = [x.shape for x in pliegue_par_global.values()]
    assert input_shape_global == [x.shape for x in pliegue_impar_global.values()]
    
    input_shape_local = [x.shape for x in pliegue_par_local.values()]
    assert input_shape_local == [x.shape for x in pliegue_impar_local.values()]


    net = defaultdict(list)
 
    for n, data in pliegue_par_global.items():
        block = Sequential()
        layer_depth = ceil(data.shape[1]/2001)
        for i in range(layer_depth):
            if i == 0:
                block.add( Conv1D(16*2**i, 5, activation=activation, input_shape=data.shape[1:], ))
            else:
                block.add(Conv1D(16*2**i, 5, activation=activation, ))
            block.add(Conv1D(16*2**i, 5, activation=activation, ))
            block.add(MaxPooling1D(pool_size=3, strides=1, ))
        block.add(Flatten())
        net["global_par"].append(block)
        
    for n, data in pliegue_impar_global.items():
        block = Sequential()
        layer_depth = ceil(data.shape[1]/2001)
        for i in range(layer_depth):
            if i == 0:
                block.add( Conv1D(16*2**i, 5, activation=activation, input_shape=data.shape[1:], ))
            else:
                block.add(Conv1D(16*2**i, 5, activation=activation, ))
            block.add(Conv1D(16*2**i, 5, activation=activation, ))
            block.add(MaxPooling1D(pool_size=3, strides=1, ))
        block.add(Flatten())
        net["global_impar"].append(block)

    for n, data in pliegue_par_local.items():
        block = Sequential()
        layer_depth = ceil(data.shape[1]/201)
        for i in range(layer_depth):
            if i == 0:
                block.add( Conv1D(16*2**i, 5, activation=activation, input_shape=data.shape[1:], ))
            else:
                block.add(Conv1D(16*2**i, 5, activation=activation, ))
            block.add(Conv1D(16*2**i, 5, activation=activation, ))
            block.add(MaxPooling1D(pool_size=3, strides=1, ))
        block.add(Flatten())
        net["local_par"].append(block)
        
    for n, data in pliegue_impar_local.items():
        block = Sequential()
        layer_depth = ceil(data.shape[1]/201)
        for i in range(layer_depth):
            if i == 0:
                block.add( Conv1D(16*2**i, 5, activation=activation, input_shape=data.shape[1:], ))
            else:
                block.add(Conv1D(16*2**i, 5, activation=activation, ))
            block.add(Conv1D(16*2**i, 5, activation=activation, ))
            block.add(MaxPooling1D(pool_size=3, strides=1, ))
        block.add(Flatten())
        net["local_impar"].append(block)
                             
    model_f = concatenate([m.output for m in net["global_par"]] + [m.output for m in net["global_impar"]] + [m.output for m in net["local_par"]] + [m.output for m in net["local_impar"]], axis=-1)
    model_f = BatchNormalization(axis=-1)(model_f)
    model_f = Dense(512,activation=activation)(model_f)
    model_f = Dense(512,activation=activation)(model_f)
    model_f = Dense(512,activation=activation)(model_f)
    model_f = Dense(512,activation=activation)(model_f)
    model_f = Dense(2,activation='softmax')(model_f)
    
    model_f = Model([[m.input for m in net["global_par"]], [m.input for m in net["global_impar"]]  , [m.input for m in net["local_par"]], [m.input for m in net["local_impar"]]],model_f)
    # model_f = Model([[model_p_7.input,model_i_7.input],[model_p_8.input,model_i_8.input]],model_f)


    if summary:
      model_f.summary()
    return model_f
    
    # model_f = concatenate([model_p_7.output,model_i_7.output,
    #                        model_p_8.output,model_i_8.output], axis=-1)
    # model_f = BatchNormalization(axis=-1)(model_f)
    # model_f = Dense(512,activation=activation)(model_f)
    # model_f = Dense(512,activation=activation)(model_f)
    # model_f = Dense(512,activation=activation)(model_f)
    # model_f = Dense(512,activation=activation)(model_f)
    # model_f = Dense(1,activation='sigmoid')(model_f)
    # model_f = Model([[model_p_7.input,model_i_7.input],[model_p_8.input,model_i_8.input]],model_f)

def gen_astronet(inputs, classes, activation = 'relu',summary=False):
    
    (pliegue_par_global, pliegue_impar_global), (pliegue_par_local, pliegue_impar_local) = inputs    

    input_shape_global = [x.shape for x in pliegue_par_global.values()]
    assert input_shape_global == [x.shape for x in pliegue_impar_global.values()]
    
    input_shape_local = [x.shape for x in pliegue_par_local.values()]
    assert input_shape_local == [x.shape for x in pliegue_impar_local.values()]

    
    global_layers = tf.keras.Sequential([
        Conv1D(16, 5, activation=activation, input_shape=data.shape[1:],),
        Conv1D(16, 5, activation=activation),
        MaxPooling1D(pool_size=5, strides=2),
        Conv1D(32, 5, activation=activation),
        Conv1D(32, 5, activation=activation),
        MaxPooling1D(pool_size=5, strides=2),
        Conv1D(64, 5, activation=activation),
        Conv1D(64, 5, activation=activation),
        MaxPooling1D(pool_size=5, strides=2),
        Conv1D(128, 5, activation=activation),
        Conv1D(128, 5, activation=activation),
        MaxPooling1D(pool_size=5, strides=2),
        Conv1D(256, 5, activation=activation),
        Conv1D(256, 5, activation=activation),
        MaxPooling1D(pool_size=5, strides=2),
    ])
                                        
    local_layers = tf.keras.Sequential([
        Conv1D(16, 5, activation=activation, input_shape=data.shape[1:],),
        Conv1D(16, 5, activation=activation),
        MaxPooling1D(pool_size=7, strides=2),
        Conv1D(32, 5, activation=activation),
        Conv1D(32, 5, activation=activation),
        MaxPooling1D(pool_size=7, strides=2),
    ])
    
    model_f = concatenate([global_layers.output,
                           local_layers.output], axis=-1)
    
    model_f = Dense(512,activation=activation)(model_f)
    model_f = Dense(512,activation=activation)(model_f)
    model_f = Dense(512,activation=activation)(model_f)
    model_f = Dense(512,activation=activation)(model_f)
    model_f = Dense(1,activation='sigmoid')(model_f)
    model_f = Model([lobal_layers.input,local_layers.input],model_f)
    if summary:
      model_f.summary()
    return model_f



flatten = []
for (n, data) in sorted(pliegue_par_global.items(), key=lambda d: d[0]):
    flatten.append(data)
    
for (n, data) in sorted(pliegue_impar_global.items(), key=lambda d: d[0]):
    flatten.append(data)

for (n, data) in sorted(pliegue_impar_local.items(), key=lambda d: d[0]):
    flatten.append(data)
    
for (n, data) in sorted(pliegue_par_local.items(), key=lambda d: d[0]):
    flatten.append(data)

# %%
import pandas as pd
# Permitir cargar un archivo distinto de descargado
df_path = 'cumulative_2024.06.01_09.08.01.csv'
df = pd.read_csv(df_path ,skiprows=144)
df_data = pd.DataFrame([{"class": lc.headers['class'], "kepoi_name": lc.headers['Kepler_name'] } for lc in lightcurves])
df_merge = pd.merge(df_data, df, on='kepoi_name')
y = df_merge['koi_disposition'].to_numpy()
y_original = np.array([lc.headers['class'] for lc in lightcurves])
# Diferencia
df_merge.loc[df_merge["class"] != df_merge.koi_disposition][["kepid", "kepoi_name", "class", "koi_disposition"]]

# %%
if np.any(y == "CANDIDATE"):
    flatten = [array[y != "CANDIDATE"] for array in flatten]
    y = y[y != "CANDIDATE"]
output_classes = np.unique(y)
class2num = {label: n for n, label in enumerate(output_classes)}
num2class = {n: label for n, label in enumerate(output_classes)}
y = to_categorical([class2num[x] for x in y], num_classes=2)

# %%
res = train_test_split(*(flatten+[y]), test_size=0.3, shuffle=False)
*X_train, y_train = [r for n, r in enumerate(res) if n % 2 == 0 ]
*X_test, y_test = [r for n, r in enumerate(res) if n % 2 == 1 ]

model_1 = gen_model_2_levels(inputs, output_classes)
tf.keras.utils.plot_model(model_1, "model.png")
tf.keras.utils.model_to_dot(model_1).write("model.dot")
model_1.compile(loss = 'binary_crossentropy', optimizer=tf.keras.optimizers.Adam(), metrics=['accuracy', tf.keras.metrics.Recall(), tf.keras.metrics.Precision(), 'binary_crossentropy'])

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=log_dir,
                                                 save_weights_only=True,
                                                 verbose=1)

history_1 = model_1.fit(X_train, y_train, epochs=100, batch_size=64, validation_data=(X_test, y_test),
                        callbacks=[tensorboard_callback, cp_callback])

# %%
# summarize history_1 for accuracy
plt.plot(history_1.history['accuracy'])
plt.plot(history_1.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history_1 for loss
plt.plot(history_1.history['loss'])
plt.plot(history_1.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# %%
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

num2class_vec = np.vectorize(num2class.get)
y_predict = model_1.predict(X_test)
# Escoger la clase que tiene mayor probabilidad
y_predict_sampled = y_predict.argmax(axis=1)
y_test_sampled = y_test.argmax(axis=1)

cm = confusion_matrix(num2class_vec(y_test_sampled), num2class_vec(y_predict_sampled), labels=[str(v) for v in num2class.values()])
ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[str(v) for v in num2class.values()]).plot(xticks_rotation='vertical')
