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
from tensorflow.keras.layers import Input, Dense, concatenate,Conv1D, Flatten,Dropout , BatchNormalization, MaxPooling1D, AveragePooling1D, ActivityRegularization
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.regularizers import L1L2
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from functools import partial
import datetime
use_wavelet = True

path = "all_data_2024-07-04/"
if use_wavelet:
    files = [file for file in os.listdir(path) if file.endswith(".pickle") and "wavelet" in file]
else:
    files = [file for file in os.listdir(path) if file.endswith(".pickle") and "wavelet" not in file]
lightcurves = []

def load_files(file, path):
    try:
        global_local = LightCurveWaveletGlobalLocalCollection.from_pickle(path+file)
    except Exception as e:
        import traceback
        print(f"Error con archivo {path}/{file}")
        traceback.print_exc()
        return None
        # return e
        
    # try:
    #     getattr(global_local, "levels")
    # except AttributeError:
    #     global_local.levels = [1, 2, 3, 4]
    return global_local

func = partial(load_files, path=path)

lightcurves = progress_map(func, files, n_cpu=64, total=len(files), executor='processes', error_behavior='raise')

# for file in tqdm(files):
#     lightcurves.append(func(file))
lightcurves = [lc for lc in lightcurves if lc is not None]

# %%
if use_wavelet:
    lightcurves = sorted(lightcurves, key=lambda lc: lc.headers["id"])
    lightcurves = [lc for lc in lightcurves if lc.headers["class"] != "CANDIDATE"]
else:
    lightcurves = sorted(lightcurves, key=lambda lc: lc.headers["kepid"])
    lightcurves = [lc for lc in lightcurves if lc.headers["kepid"] != "CANDIDATE"]


lightcurves_train, lightcurves_test = train_test_split(lightcurves, test_size=0.3, shuffle=True)

def inputs_from_dataset(lightcurves):
    if use_wavelet:
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
                
        
        global_level_list = (1, 3, 5,)
        local_level_list = (1, 3,)
        
        pliegue_par_global = {k: np.array(v) for k, v in pliegue_par_global.items() if k in global_level_list}
        pliegue_par_global = {k: v.reshape(list(v.shape)+[1]) for k, v in pliegue_par_global.items()}
        pliegue_impar_global = {k: np.array(v) for k, v in pliegue_impar_global.items() if k in global_level_list}
        pliegue_impar_global = {k: v.reshape(list(np.shape(v))+[1]) for k, v in pliegue_impar_global.items()}
        
        
        pliegue_par_local = {k: np.array(v) for k, v in pliegue_par_local.items() if k in local_level_list}
        pliegue_par_local = {k: v.reshape(list(np.shape(v))+[1]) for k, v in pliegue_par_local.items()}
        pliegue_impar_local = {k: np.array(v) for k, v in pliegue_impar_local.items() if k in local_level_list}
        pliegue_impar_local = {k: v.reshape(list(np.shape(v))+[1]) for k, v in pliegue_impar_local.items()}
        
        
        inputs = (pliegue_par_global, pliegue_impar_global), (pliegue_par_local, pliegue_impar_local)
    else:
        inputs = np.array([lc.lc_global for lc in lightcurves]),  np.array([lc.lc_local for lc in lightcurves]),  
        
    return inputs

def flatten_from_inputs(inputs, use_wavelets=True):
    if use_wavelet:
        (pliegue_par_global, pliegue_impar_global), (pliegue_par_local, pliegue_impar_local) = inputs    
        
        flatten = []
        for (n, data) in sorted(pliegue_par_global.items(), key=lambda d: d[0]):
            flatten.append(data)
            
        for (n, data) in sorted(pliegue_impar_global.items(), key=lambda d: d[0]):
            flatten.append(data)
        
        for (n, data) in sorted(pliegue_impar_local.items(), key=lambda d: d[0]):
            flatten.append(data)
            
        for (n, data) in sorted(pliegue_par_local.items(), key=lambda d: d[0]):
            flatten.append(data)
    else:
        flatten = inputs

    return flatten


if use_wavelet:
    y = np.array([lc.headers['class'] for lc in lightcurves])
else:
    y = np.array([lc.headers['koi_disposition'] for lc in lightcurves])
output_classes = np.unique(y)
class2num = {label: n for n, label in enumerate(sorted(output_classes))}
num2class = {n: label for n, label in enumerate(sorted(output_classes))}

if use_wavelet:
    y = to_categorical([class2num[x] for x in y], num_classes=2)
else:
    y = np.array([class2num[x] for x in y])


if use_wavelet:
    y_train = np.array([lc.headers['class'] for lc in lightcurves_train])
    y_test = np.array([lc.headers['class'] for lc in lightcurves_test])
    kepid_test = np.array([lc.headers["id"] for lc in lightcurves_test])
    kepid_train = np.array([lc.headers["id"] for lc in lightcurves_train])
    y_train = to_categorical([class2num[x] for x in y_train], num_classes=2)
    y_test = to_categorical([class2num[x] for x in y_test], num_classes=2)
else:
    y_train = np.array([lc.headers['koi_disposition'] == "CONFIRMED" for lc in lightcurves_train]).astype(float)
    y_test = np.array([lc.headers['koi_disposition'] == "CONFIRMED" for lc in lightcurves_test]).astype(float)
    kepid_test = np.array([lc.headers["kepid"] for lc in lightcurves_test])
    kepid_train = np.array([lc.headers["kepid"] for lc in lightcurves_train])


inputs = inputs_from_dataset(lightcurves_train)
X_train = flatten_from_inputs(inputs_from_dataset(lightcurves_train))
X_test = flatten_from_inputs(inputs_from_dataset(lightcurves_test))

if not use_wavelet:
    X_train = list(X_train)
    X_train[0] = X_train[0].reshape(list(X_train[0].shape)+[1])
    X_train[1] = X_train[1].reshape(list(X_train[1].shape)+[1])
    X_test = list(X_test)
    X_test[0] = X_test[0].reshape(list(X_test[0].shape)+[1])
    X_test[1] = X_test[1].reshape(list(X_test[1].shape)+[1])


# *X_train, y_train, kepid_train = [r for n, r in enumerate(res) if n % 2 == 0 ]
# *X_test, y_test, kepid_test = [r for n, r in enumerate(res) if n % 2 == 1 ]

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
        layer_depth = ceil(data.shape[1]*5.0/2001 + 1)
        print(f"par global: {layer_depth}")
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
        layer_depth = ceil(data.shape[1]*5.0/2001 + 1)
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
        layer_depth = ceil(data.shape[1]*2.0/201 + 1)
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
        layer_depth = ceil(data.shape[1]*2.0/201 + 1)
        for i in range(layer_depth):
            if i == 0:
                block.add( Conv1D(16*2**i, 5, activation=activation, input_shape=data.shape[1:], ))
            else:
                block.add(Conv1D(16*2**i, 5, activation=activation, ))
            block.add(Conv1D(16*2**i, 5, activation=activation, ))
            block.add(MaxPooling1D(pool_size=3, strides=1, ))
        block.add(Flatten())
        net["local_impar"].append(block)


    l1 = 0.01
    l2 = 0.0
    
    model_f = concatenate([m.output for m in net["global_par"]] + [m.output for m in net["global_impar"]] + [m.output for m in net["local_par"]] + [m.output for m in net["local_impar"]], axis=-1)
    model_f = BatchNormalization(axis=-1)(model_f)
    model_f = Dropout(0.5)(model_f)
    model_f = Dense(256,activation=activation, kernel_regularizer=L1L2( l1=l1, l2=l2,))(model_f)
    model_f = Dropout(l1)(model_f)
    model_f = Dense(256,activation=activation, kernel_regularizer=L1L2( l1=l1, l2=l2,))(model_f)
    model_f = Dropout(l1)(model_f)
    model_f = Dense(256,activation=activation, kernel_regularizer=L1L2( l1=l1, l2=l2,))(model_f)
    model_f = Dropout(l1)(model_f)
    model_f = Dense(256,activation=activation, kernel_regularizer=L1L2( l1=l1, l2=l2,))(model_f)
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
    
    global_view, local_view = inputs    

    global_layers = tf.keras.Sequential([
        Conv1D(16, 5, activation=activation, input_shape=global_view[0].reshape(-1, 1).shape,),
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
        Flatten(),
    ])
                                        
    local_layers = tf.keras.Sequential([
        Conv1D(16, 5, activation=activation, input_shape=local_view[0].reshape(-1, 1).shape,),
        Conv1D(16, 5, activation=activation),
        MaxPooling1D(pool_size=7, strides=2),
        Conv1D(32, 5, activation=activation),
        Conv1D(32, 5, activation=activation),
        MaxPooling1D(pool_size=7, strides=2),
        Flatten(),
    ])
    
    model_f = concatenate([global_layers.output,
                           local_layers.output], axis=-1)
    
    model_f = Dense(512,activation=activation)(model_f)
    model_f = Dense(512,activation=activation)(model_f)
    model_f = Dense(512,activation=activation)(model_f)
    model_f = Dense(512,activation=activation)(model_f)
    model_f = Dense(1,activation='sigmoid')(model_f)
    model_f = Model([global_layers.input,local_layers.input],model_f)
    if summary:
      model_f.summary()
    return model_f

# %%

# %%
# import pandas as pd
# # TODO Permitir cargar un archivo distinto de descargado
# df_path = 'cumulative_2024.06.01_09.08.01.csv'
# df = pd.read_csv(df_path ,skiprows=144)
# df = df.sort_values(by="kepid")
# df_data = pd.DataFrame([{"class": lc.headers['class'], "kepoi_name": lc.headers['Kepler_name'] } for lc in lightcurves])
# df_merge = pd.merge(df_data, df, on='kepoi_name')
# df_merge = df_merge.sort_values(by="kepid")
# y = df_merge['koi_disposition'].to_numpy()
# y_original = np.array([lc.headers['class'] for lc in lightcurves])
# # Diferencia
# df_merge.loc[df_merge["class"] != df_merge.koi_disposition][["kepid", "kepoi_name", "class", "koi_disposition"]]

# %%
# y = np.array([lc.headers['class'] for lc in lightcurves])
# if np.any(y == "CANDIDATE"):
#     # Están bien ordenados los datos en flatten??
#     # Comprobar que hay el mismo número de datos
#     assert all([len(array) == len(y) for array in flatten])
#     flatten = [array[y != "CANDIDATE"] for array in flatten]
#     df_merge = df_merge.query("koi_disposition != 'CANDIDATE'")
#     y = y[y != "CANDIDATE"]
#     assert len(df_merge) == len(y)


# %%
# from sklearn.model_selection import KFold
# kf = KFold(n_splits=5)
# split = list(kf.split(flatten, y))
# split[0][0].shape
# y.shape, [x.shape for x in flatten], np.hstack(flatten).shape

# %%
# np.unique(y_train)

# %%
# res = train_test_split(*(flatten+[y]+[np.array([lc.headers["id"] for lc in lightcurves])]), test_size=0.3, shuffle=True)

# *X_train, y_train, kepid_train = [r for n, r in enumerate(res) if n % 2 == 0 ]
# *X_test, y_test, kepid_test = [r for n, r in enumerate(res) if n % 2 == 1 ]


# def get_real(kepid_test):
#     class2num_vec = np.vectorize(class2num.get)
#     df_path = 'cumulative_2024.06.01_09.08.01.csv'
#     df = pd.read_csv(df_path ,skiprows=144)
#     df_kepid = pd.DataFrame({"kepid": kepid_test})
#     df_merge = pd.merge(df_kepid, df, how="inner")
#     df_merge = df_merge.query("koi_disposition != 'CANDIDATE'")
#     print(df_merge.koi_disposition.unique())
#     return df_merge.koi_disposition.apply(class2num_vec).to_numpy().astype(np.int)

# print(y_train.shape, y_test.shape)
# y_train = get_real(kepid_train)
# y_test = get_real(kepid_test)

# y_train = to_categorical([x for x in y_train], num_classes=2)
# y_test = to_categorical([x for x in y_test], num_classes=2)
# print(y_train.shape, y_test.shape)

# https://github.com/tensorflow/tensorflow/issues/48545
import gc
if globals().get("model_1"):
    print("Erasing model_1")
    del model_1
    import gc
    gc.collect()
    tf.keras.backend.clear_session()
tf.keras.backend.clear_session()
# from numba import cuda 
# device = cuda.get_current_device()
# device.reset()
inputs = inputs_from_dataset(lightcurves_train)
if use_wavelet:
    model_1 = gen_model_2_levels(inputs, output_classes)
else:
    model_1 = gen_astronet(inputs, output_classes)
tf.keras.utils.plot_model(model_1, "model.png")
tf.keras.utils.model_to_dot(model_1).write("model.dot")
if use_wavelet:
    model_1.compile(loss = 'binary_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4,),
                    metrics=['accuracy', tf.keras.metrics.Recall(), tf.keras.metrics.Precision(), 'binary_crossentropy'])

else:
    # TODO NaN metrics ??
    model_1.compile(loss = 'binary_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4,),
                    metrics=[])

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
# tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=log_dir,
#                                                  save_weights_only=True,
#                                                  verbose=1)


cp_callback = tf.keras.callbacks.BackupAndRestore(log_dir)


history_1 = model_1.fit(X_train, y_train, epochs=100, batch_size=16, validation_data=(X_test, y_test),
                        callbacks=[cp_callback])

# %%
# summarize history_1 for accuracy
plt.plot(history_1.history['accuracy'])
plt.plot(history_1.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history_1 for precision
plt.plot(history_1.history['precision'])
plt.plot(history_1.history['val_precision'])
plt.title('model precision')
plt.ylabel('precision')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history_1 for recall
plt.plot(history_1.history['recall'])
plt.plot(history_1.history['val_recall'])
plt.title('model recall')
plt.ylabel('recall')
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
# summarize history_1 for precision
plt.plot(history_1.history['precision'])
plt.plot(history_1.history['val_precision'])
plt.title('model precision')
plt.ylabel('precision')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history_1 for recall
plt.plot(history_1.history['recall'])
plt.plot(history_1.history['val_recall'])
plt.title('model recall')
plt.ylabel('recall')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


plt.plot(np.array(history_1.history['recall'])-np.array(history_1.history['precision']))
plt.plot(np.array(history_1.history['val_recall'])-np.array(history_1.history['val_precision']))
plt.title('model recall - model precision')
plt.ylabel('recall - precision')
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

# %%
wrong = y_predict_sampled != y_test_sampled


download_dir="data3/"
import importlib  
descarga = importlib.import_module("01_descarga")
process_light_curve = descarga.process_light_curve

process_func =  partial(process_light_curve, levels_global=5, levels_local=3, wavelet_family="sym5", sigma=20, sigma_upper=5,
                        plot=True, plot_comparative=False, save=False, path=path, download_dir=download_dir, plot_folder=log_dir, use_download_cache=True)

def process_func_continue(row, title):
    try:
        print(title)
        return process_func(row, title)
    except Exception as e:
        print(f"Exception on {row.kepid}")
        import traceback
        traceback.print_exc()
        return e


df_path = 'cumulative_2024.06.01_09.08.01.csv'
df = pd.read_csv(df_path ,skiprows=144)
df_kepid = pd.DataFrame({"kepid": kepid_test[wrong], "predicted": y_predict_sampled[wrong], "true": y_test_sampled[wrong]})
df_wrong = pd.merge(df_kepid, df, how="inner")

results = []
for _, row in tqdm(df_wrong.iterrows(), total=len(df_wrong)):
    try:
        results.append(process_light_curve(row, title=f" Predicho: {num2class[row.predicted]} Real: {num2class[row.true]}",
                                           levels_global=5, levels_local=3, wavelet_family="sym5", sigma=20, sigma_upper=5,
                                           plot=True, plot_comparative=False, save=False, path=path, download_dir=download_dir, plot_folder=log_dir, use_download_cache=True))
    except Exception as e:
        print(f"Exception on {row.kepid}")
        import traceback
        traceback.print_exc()
        results.append(e)
# results = progress_map(process_func, [row for _, row in df.iterrows()], n_cpu=20, total=len(df), error_behavior='coerce')

