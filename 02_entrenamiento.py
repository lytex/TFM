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

path = "all_data_2024-07-17/"
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

# %%
binary_classification = False

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
                
        
        global_level_list = (1, 5,)
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

def flatten_from_inputs(inputs, use_wavelet=True):
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


def get_data_split(lightcurves, binary_classification=False, use_wavelet=True, k_fold=None, ind=None):

    if use_wavelet:
        lightcurves = sorted(lightcurves, key=lambda lc: lc.headers["id"])
        lightcurves = [lc for lc in lightcurves if lc.headers["class"] != "CANDIDATE"]
    else:
        lightcurves = sorted(lightcurves, key=lambda lc: lc.headers["kepid"])
        lightcurves = [lc for lc in lightcurves if lc.headers["kepid"] != "CANDIDATE"]



    if k_fold:
        if ind >= k_fold:
            raise ValueError("Index ind must be strictly smaller than k_fold")
        from sklearn.model_selection import KFold
        kf = KFold(n_splits=k_fold, shuffle=True)
        train_index, test_index =  [(train_index, test_index) for i, (train_index, test_index) in enumerate(kf.split(lightcurves)) if i == ind][0]
        lightcurves_train, lightcurves_test = np.array(lightcurves)[train_index], np.array(lightcurves)[test_index]
    else:
        lightcurves_train, lightcurves_test = train_test_split(lightcurves, test_size=0.3, shuffle=True)


    if use_wavelet:
        y = np.array([lc.headers['class'] for lc in lightcurves])
    else:
        y = np.array([lc.headers['koi_disposition'] for lc in lightcurves])

    output_classes = np.unique(y)
    class2num = {label: n for n, label in enumerate(sorted(output_classes))}
    num2class = {n: label for n, label in enumerate(sorted(output_classes))}

    if use_wavelet:
        if binary_classification:
            y = np.array([class2num[x] for x in y])
        else:
            y = to_categorical([class2num[x] for x in y], num_classes=2)
    else:
        y = np.array([class2num[x] for x in y])


    if use_wavelet:
        if binary_classification:
            y_train = np.array([lc.headers['class'] == "CONFIRMED" for lc in lightcurves_train]).astype(float)
            y_test = np.array([lc.headers['class'] == "CONFIRMED" for lc in lightcurves_test]).astype(float)
        else:
            y_train = np.array([lc.headers['class'] for lc in lightcurves_train])
            y_test = np.array([lc.headers['class'] for lc in lightcurves_test])
            y_train = to_categorical([class2num[x] for x in y_train], num_classes=2)
            y_test = to_categorical([class2num[x] for x in y_test], num_classes=2)
        
        kepid_test = np.array([lc.headers["id"] for lc in lightcurves_test])
        kepid_train = np.array([lc.headers["id"] for lc in lightcurves_train])
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

    # aa = np.bincount(y_train.astype(int))
    # print(aa, aa[0]/sum(aa), aa[1]/sum(aa))
    # aa = np.bincount(y_test.astype(int))
    # print(aa, aa[0]/sum(aa), aa[1]/sum(aa))

    return inputs, X_train, X_test, y_train, y_test, y, kepid_test, kepid_train, num2class, num2class, num2class, lightcurves_train, lightcurves_test, output_classes


# *X_train, y_train, kepid_train = [r for n, r in enumerate(res) if n % 2 == 0 ]
# *X_test, y_test, kepid_test = [r for n, r in enumerate(res) if n % 2 == 1 ]

# %%

# %%
# inputs
# %pdb off
from math import ceil
def gen_model_2_levels(inputs, classes, activation = 'relu',summary=False, binary_classification=False):
    
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


    l1 = 0.00
    l2 = 0.0
    dropout = 0.0
    
    model_f = concatenate([m.output for m in net["global_par"]] + [m.output for m in net["global_impar"]] + [m.output for m in net["local_par"]] + [m.output for m in net["local_impar"]], axis=-1)
    model_f = BatchNormalization(axis=-1)(model_f)
    model_f = Dropout(dropout)(model_f)
    model_f = Dense(256,activation=activation, kernel_regularizer=L1L2( l1=l1, l2=l2,))(model_f)
    model_f = Dropout(dropout)(model_f)
    model_f = Dense(256,activation=activation, kernel_regularizer=L1L2( l1=l1, l2=l2,))(model_f)
    model_f = Dropout(dropout)(model_f)
    model_f = Dense(256,activation=activation, kernel_regularizer=L1L2( l1=l1, l2=l2,))(model_f)
    model_f = Dropout(dropout)(model_f)
    model_f = Dense(256,activation=activation, kernel_regularizer=L1L2( l1=l1, l2=l2,))(model_f)
    if binary_classification:
        model_f = Dense(1,activation='sigmoid')(model_f)
    else:
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
class F1_Score(tf.keras.metrics.Metric):

    def __init__(self, name='f1_score', **kwargs):
        super().__init__(name=name, **kwargs)
        self.f1 = self.add_weight(name='f1', initializer='zeros')
        self.precision_fn = tf.keras.metrics.Precision(thresholds=0.5)
        self.recall_fn = tf.keras.metrics.Recall(thresholds=0.5)

    def update_state(self, y_true, y_pred, sample_weight=None):
        p = self.precision_fn(y_true, y_pred)
        r = self.recall_fn(y_true, y_pred)
        # since f1 is a variable, we use assign
        self.f1.assign(2 * ((p * r) / (p + r + 1e-6)))

    def result(self):
        return self.f1

    def reset_state(self):
        # we also need to reset the state of the precision and recall objects
        self.precision_fn.reset_states()
        self.recall_fn.reset_states()
        self.f1.assign(0)


# %%
# https://github.com/tensorflow/tensorflow/issues/48545
import gc
if globals().get("model_1"):
    print("Erasing model_1")
    del model_1
    import gc
    gc.collect()
    tf.keras.backend.clear_session()
# from numba import cuda 
# device = cuda.get_current_device()
# device.reset()
k_fold = 5


inputs, X_train, X_test, y_train, y_test, y, kepid_test, kepid_train, num2class, num2class, num2class, \
    lightcurves_train, lightcurves_test, output_classes = get_data_split(lightcurves, binary_classification=binary_classification, use_wavelet=use_wavelet, k_fold=k_fold, ind=0)

if use_wavelet:
    model_1 = gen_model_2_levels(inputs, output_classes, binary_classification=binary_classification)
else:
    model_1 = gen_astronet(inputs, output_classes)

if use_wavelet:
    if binary_classification:
        model_1.compile(loss = 'binary_crossentropy', optimizer=tf.keras.optimizers.Adam(),
                        metrics=['accuracy', tf.keras.metrics.Recall(), tf.keras.metrics.Precision(),])
    else:
        from weighted_loss import WeightedCategoricalCrossentropy
        
        # model_1.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(),
        #                 metrics=[F1_Score(),])
        model_1.compile(loss=WeightedCategoricalCrossentropy(weights=[1.0, 0.2]), optimizer=tf.keras.optimizers.Adam(),
                        metrics=[F1_Score(),])

else:
    # TODO NaN metrics ??
    model_1.compile(loss = 'binary_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=1e-7,),
                    metrics=[])

tf.keras.utils.plot_model(model_1, "model.png")
tf.keras.utils.model_to_dot(model_1).write("model.dot")
    

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


cp_callback = tf.keras.callbacks.BackupAndRestore(log_dir)

# %%
history_1 =  pd.DataFrame()

inputs, X_train, X_test, y_train, y_test, y, kepid_test, kepid_train, num2class, num2class, num2class, \
    lightcurves_train, lightcurves_test, output_classes = get_data_split(lightcurves, binary_classification=binary_classification, use_wavelet=use_wavelet)
temp = model_1.fit(X_train, y_train, epochs=100, batch_size=128, validation_data=(X_test, y_test),
                        callbacks=[cp_callback])
history_1 = history_1.append(pd.DataFrame(temp.history))

# for ind in tqdm(range(k_fold)):
#     inputs, X_train, X_test, y_train, y_test, y, kepid_test, kepid_train, num2class, num2class, num2class, \
#         lightcurves_train, lightcurves_test, output_classes = get_data_split(lightcurves, binary_classification=binary_classification, use_wavelet=use_wavelet, k_fold=k_fold, ind=ind)
#     temp = model_1.fit(X_train, y_train, epochs=100, batch_size=128, validation_data=(X_test, y_test),
#                             callbacks=[cp_callback])
#     history_1 = history_1.append(pd.DataFrame(temp.history))

history_1 = history_1.reset_index().rename(columns={"index": "epoch"})

# %%
if not binary_classification:
# summarize history_1 for loss
    plt.plot(history_1['loss'])
    plt.plot(history_1['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    
    
    plt.plot(history_1['f1_score'])
    plt.plot(history_1['val_f1_score'])
    plt.title('model f1_score')
    plt.ylabel('f1_score')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    
else:
    # summarize history_1 for accuracy
    plt.plot(history_1['accuracy'])
    plt.plot(history_1['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history_1 for precision
    plt.plot(history_1['precision'])
    plt.plot(history_1['val_precision'])
    plt.title('model precision')
    plt.ylabel('precision')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history_1 for recall
    plt.plot(history_1['recall'])
    plt.plot(history_1['val_recall'])
    plt.title('model recall')
    plt.ylabel('recall')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

# %%

# %%
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

num2class_vec = np.vectorize(num2class.get)
y_predict = model_1.predict(X_test)
# Escoger la clase que tiene mayor probabilidad
y_predict_sampled = y_predict.argmax(axis=1)
if binary_classification:
    y_test_sampled = y_test
else:
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

def process_func_title(row):
    title=f" Predicho: {num2class[row.predicted]} Real: {num2class[row.true]}"
    return process_func(row, title=title)

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
df_wrong = pd.merge(df_kepid, df, how="inner", on="kepid")

# results = []
# for _, row in tqdm(df_wrong.iterrows(), total=len(df_wrong)):
#     try:
#         results.append(process_light_curve(row, title=f" Predicho: {num2class[row.predicted]} Real: {num2class[row.true]}",
#                                            levels_global=5, levels_local=3, wavelet_family="sym5", sigma=20, sigma_upper=5,
#                                            plot=True, plot_comparative=False, save=False, path=path, download_dir=download_dir, plot_folder=log_dir, use_download_cache=True))
#     except Exception as e:
#         print(f"Exception on {row.kepid}")
#         import traceback
#         traceback.print_exc()
#         results.append(e)
results = progress_map(process_func_title, [row for _, row in df_wrong.iterrows()], n_cpu=20, total=len(df_wrong), error_behavior='coerce')

