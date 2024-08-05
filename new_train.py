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
from parallelbar import progress_imap
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, concatenate,Conv1D, Flatten,Dropout , BatchNormalization, MaxPooling1D, AveragePooling1D, ActivityRegularization
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.regularizers import L1L2
from keras.utils import to_categorical
from keras.utils.layer_utils import count_params
from sklearn.model_selection import train_test_split
from functools import partial
import datetime
import importlib  
import matplotlib as mpl
import gc
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
mpl.use("agg")

descarga = importlib.import_module("01_descarga")
entrenamiento = importlib.import_module("02_entrenamiento")

GetBest = entrenamiento.GetBest
get_data_split = entrenamiento.get_data_split
gen_model_2_levels = entrenamiento.gen_model_2_levels
gen_astronet = entrenamiento.gen_astronet
from weighted_loss import WeightedBinaryCrossentropy,  WeightedCategoricalCrossentropy
process_light_curve = descarga.process_light_curve
load_files = entrenamiento.load_files


def descarga_process_light_curve(
    df_path=None,
    sigma=None, sigma_upper=None,
    num_bins_global=None, bin_width_factor_global=None,
    num_bins_local=None, bin_width_factor_local=None,
    plot=False, plot_comparative=False, save=False, path=None, download_dir=None, plot_folder=None, use_download_cache=True,
    parallel=None
):
                        

    process_func =  partial(process_light_curve,
                            sigma=sigma, sigma_upper=sigma_upper,
                            num_bins_global=num_bins_global, bin_width_factor_global=bin_width_factor_global,
                            num_bins_local=num_bins_local, bin_width_factor_local=bin_width_factor_local,
                            levels_global=levels_global, levels_local=levels_local, wavelet_family=wavelet_family,
                            plot=False, plot_comparative=False, save=False, path=path, download_dir=download_dir, plot_folder=None, use_download_cache=use_download_cache, cache_dict=dict())

    def process_func_continue(row):
        try:
            return process_func(row)
        except Exception as e:
            print(f"Exception on {row.kepid}")
            import traceback
            traceback.print_exc()
            return e


    df = pd.read_csv(df_path ,skiprows=144)[:100]

    if not parallel:
        results = []
        for _, row in tqdm(df.iterrows(), total=len(df)):
            results.append(process_func_continue(row))
    else:
        results = progress_imap(process_func, [row for _, row in df.iterrows()], n_cpu=n_proc, total=len(df), error_behavior='coerce', chunk_size=len(df)//n_proc)
    
    return results

# %%
# %pdb on
# results = descarga_process_light_curve(
#     df_path=df_path,
#     sigma=sigma, sigma_upper=sigma_upper,
#     num_bins_global=num_bins_global, bin_width_factor_global=bin_width_factor_global,
#     num_bins_local=num_bins_local, bin_width_factor_local=bin_width_factor_local,
#     plot=False, plot_comparative=False, save=False, path=path, download_dir=download_dir, plot_folder=None, use_download_cache=use_download_cache,
#     parallel=True
# )
# lightcurves = [x for x in results if type(x) in (LightCurveWaveletGlobalLocalCollection, )]


# def load_files(file, path):
#     try:
#         global_local = LightCurveWaveletGlobalLocalCollection.from_pickle(path+file)
#     except Exception as e:
#         import traceback
#         print(f"Error con archivo {path}/{file}")
#         traceback.print_exc()
#         return None
        
#     return global_local

# if use_wavelet:
#     files = [file for file in os.listdir(path) if file.endswith(".pickle") and "wavelet" in file]
# else:
#     files = [file for file in os.listdir(path) if file.endswith(".pickle") and "wavelet" not in file]
    
# func = partial(load_files, path=path)

# lightcurves = progress_imap(func, files, n_cpu=64, total=len(files), executor='processes', error_behavior='raise', chunk_size=len(files)//64//10)

# %%
# lightcurves = [lc for lc in lightcurves if lc is not None]

# %%
def get_model_wrapper(lightcurves, use_wavelet=True, binary_classification=False, frac=0.5, model_name="model"):
    
    if use_wavelet:
        lightcurves_filtered = sorted(lightcurves, key=lambda lc: lc.headers["id"])
        lightcurves_filtered = [lc for lc in lightcurves if lc.headers["class"] != "CANDIDATE"]
    else:
        lightcurves_filtered = sorted(lightcurves, key=lambda lc: lc.headers["kepid"])
        lightcurves_filtered = [lc for lc in lightcurves if lc.headers["koi_disposition"] != "CANDIDATE"]
    
    inputs, _, X_entire, _, y_entire, y_class, _, kepid_train, num2class, \
        output_classes = get_data_split(lightcurves, binary_classification=binary_classification, use_wavelet=use_wavelet, test_size=len(lightcurves_filtered)-1,
                                       global_level_list=global_level_list, local_level_list=local_level_list)
    
    if use_wavelet:
        model_1 = gen_model_2_levels(inputs, output_classes, binary_classification=binary_classification, l1=l1, l2=l2, dropout=dropout, global_view=num_bins_global, local_view=num_bins_local)
        
    else:
        model_1 = gen_astronet(inputs, output_classes)
    
    if use_wavelet:
        if binary_classification:
            model_1.compile(loss = 'binary_crossentropy', optimizer=tf.keras.optimizers.Adam(),
                            metrics=['accuracy', tf.keras.metrics.Recall(), tf.keras.metrics.Precision(),])
        else:
            
            # model_1.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(),
            #                 metrics=[F1_Score(),])
    
            
            count = pd.DataFrame({'col': np.argmax(y_entire, axis=1)}).reset_index(drop=False).groupby('col').index.count()
            print("count:",  count[0]/count[1]*frac)
            model_1.compile(loss=WeightedCategoricalCrossentropy(weights=[1.0, count[0]/count[1]*frac]), optimizer=tf.keras.optimizers.Adam(),
                            metrics=[entrenamiento.F1_Score(),])
    
    else:
        
        count =  pd.DataFrame({'col': y_entire}).reset_index(drop=False).groupby('col').index.count()
        print("count:",  count[0]/count[1]*frac)
        # from FBetaScore import DifferentiableFBetaScore
        k_fold = None
        model_1.compile(loss=WeightedBinaryCrossentropy(weights=[0.01, 0.1]), optimizer=tf.keras.optimizers.Adam(),
                        metrics=['accuracy', tf.keras.metrics.Recall(), tf.keras.metrics.Precision(), F1_Score(beta=count[0]/count[1]*frac)])
    
    tf.keras.utils.plot_model(model_1, f"{model_name}.png")
    tf.keras.utils.model_to_dot(model_1).write(f"{model_name}.dot")
    print("model_1 has", sum(count_params(layer) for layer in model_1.trainable_weights), "parameters")
    return model_1, sum(count_params(layer) for layer in model_1.trainable_weights)

if globals().get("model_1"):
    print("Erasing model_1")
    del model_1
    import gc
    gc.collect()
    tf.keras.backend.clear_session()
# device = cuda.get_current_device()
# device.reset()
# model_1, weights = get_model_wrapper(lightcurves, use_wavelet=use_wavelet, binary_classification=binary_classification, frac=frac)

# %%
# %pdb on
def train_model(lightcurves, use_wavelet=True, binary_classification=False, k_fold=None, global_level_list=None, local_level_list=None, epochs=200, batch_size=128):
    
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    cp_callback = tf.keras.callbacks.ModelCheckpoint(log_dir, monitor='val_loss', save_best_only=True)
    best_callback = GetBest(monitor='val_loss', verbose=0, mode='min')
    
    # cp_callback = tf.keras.callbacks.BackupAndRestore(log_dir)
    history_1 =  pd.DataFrame()
    if k_fold is None:
        inputs, X_train, X_test, y_train, y_test, y, kepid_test, kepid_train, num2class, \
            output_classes = get_data_split(lightcurves, binary_classification=binary_classification, use_wavelet=use_wavelet, global_level_list=global_level_list, local_level_list=local_level_list)
        temp = model_1.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test),
                                callbacks=[cp_callback, best_callback])
        history_1 = history_1.append(pd.DataFrame(temp.history))
    
    else:
        lightcurves_kfold, lightcurves_val = train_test_split(lightcurves, test_size=0.2, shuffle=True)
        _, _, X_test, _, y_test, _, kepid_test, _, num2class, \
            _ = get_data_split(lightcurves_kfold, binary_classification=binary_classification, use_wavelet=use_wavelet, test_size=1.0,
                               global_level_list=global_level_list, local_level_list=local_level_list)
        for ind in tqdm(range(k_fold)):
            inputs, X_train, X_test_kfold, y_train, y_test_kfold, y, kepid_test_kfold, kepid_train, num2class, \
                output_classes = get_data_split(lightcurves_kfold, binary_classification=binary_classification, use_wavelet=use_wavelet,
                                                global_level_list=global_level_list, local_level_list=local_level_list)
            inputs, X_train, X_test_kfold, y_train, y_test_kfold, y, kepid_test_kfold, kepid_train, num2class, \
                output_classes = get_data_split(lightcurves_kfold, binary_classification=binary_classification, use_wavelet=use_wavelet, k_fold=k_fold, ind=ind,
                                                global_level_list=global_level_list, local_level_list=local_level_list)
            temp = model_1.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test),
                                    callbacks=[best_callback])
            history_1 = history_1.append(pd.DataFrame(temp.history))
    
    history_1 = history_1.reset_index().rename(columns={"index": "epoch"})
    
    return history_1, num2class, X_test, y_test


# %%
# history_1, num2class, X_test, y_test = train_model(lightcurves,
#                                                    use_wavelet=use_wavelet, binary_classification=binary_classification,
#                                                    k_fold=k_fold, global_level_list=global_level_list, local_level_list=local_level_list, epochs=epochs, batch_size=batch_size)
    
def get_metrics(num2class, X_test, y_test, model_1, β=1):
    num2class_vec = np.vectorize(num2class.get)
    y_predict = model_1.predict(X_test)
    # Escoger la clase que tiene mayor probabilidad
    if binary_classification:
        y_test_sampled = y_test
        y_predict_sampled = (np.squeeze(y_predict) > 0.5).astype(int)
    
    else:
        y_predict_sampled = y_predict.argmax(axis=1)
        y_test_sampled = y_test.argmax(axis=1)


    cm = confusion_matrix(num2class_vec(y_test_sampled), num2class_vec(y_predict_sampled), labels=[str(v) for v in num2class.values()])
    if plot:
        ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[str(v) for v in num2class.values()]).plot(xticks_rotation='vertical')
    
    precision = cm[0][0]/(cm[0][0] + cm[1][0])
    recall = cm[0][0]/(cm[0][0] + cm[0][1])
    F1 = 2*(precision*recall)/(precision+recall)
    Fβ = (1+β**2)*(precision*recall)/(β**2*precision+recall)
    return precision, recall, F1, Fβ, cm

# precision, recall, F1, Fβ, cm = get_metrics(num2class, X_test, y_test, model_1, β=2.0)
# print("P : %f\nR : %f\nF1: %f\nFβ: %f" % (precision, recall, F1, Fβ))

def load_files(file, path):
    try:
        global_local = LightCurveWaveletGlobalLocalCollection.from_pickle(path+file)
    except Exception as e:
        import traceback
        print(f"Error con archivo {path}/{file}")
        traceback.print_exc()
        return None
        
    return global_local

def main(sigma = 20, sigma_upper = 5,
            num_bins_global = 2001, bin_width_factor_global = 1 / 2001,
            num_bins_local = 201, bin_width_factor_local = 0.16,
            levels_global = 6, levels_local = 3, wavelet_family = "sym5",
            use_wavelet = True, binary_classification = True,
            k_fold = 5,
            global_level_list = (1, 5,), local_level_list = (1, 3,),
            l1 = 0.00, l2 = 0.0, dropout = 0.0,
            epochs = 200, batch_size = 128,
            frac = 0.5, β=1.0,
            download_dir="data3/",
            path = "all_data_2024-07-17/",
            df_path = 'cumulative_2024.06.01_09.08.01.csv',
            use_download_cache = True,
            n_proc = 20,
            parallel = True,
            lightcurve_cache=True,
    ):
    
    if lightcurve_cache:
        if use_wavelet:
            files = [file for file in os.listdir(path) if file.endswith(".pickle") and "wavelet" in file]
        else:
            files = [file for file in os.listdir(path) if file.endswith(".pickle") and "wavelet" not in file]
            
        func = partial(load_files, path=path)
        
        lightcurves = progress_imap(func, files, n_cpu=64, total=len(files), executor='processes', error_behavior='raise', chunk_size=len(files)//64//10)
    else:
        results = descarga_process_light_curve(
            df_path=df_path,
            sigma=sigma, sigma_upper=sigma_upper,
            num_bins_global=num_bins_global, bin_width_factor_global=bin_width_factor_global,
            num_bins_local=num_bins_local, bin_width_factor_local=bin_width_factor_local,
            plot=False, plot_comparative=False, save=False, path=path, download_dir=download_dir, plot_folder=None, use_download_cache=use_download_cache,
            parallel=True
        )
        lightcurves = [x for x in results if type(x) in (LightCurveWaveletGlobalLocalCollection, )]

    
    lightcurves = [lc for lc in lightcurves if lc is not None]
    
    model_1, weights = get_model_wrapper(lightcurves, use_wavelet=use_wavelet, binary_classification=binary_classification, frac=frac)
    
    history_1, num2class, X_test, y_test = train_model(lightcurves,
                                                       use_wavelet=use_wavelet, binary_classification=binary_classification,
                                                       k_fold=k_fold, global_level_list=global_level_list, local_level_list=local_level_list, epochs=epochs, batch_size=batch_size)

    
    precision, recall, F1, Fβ, cm = get_metrics(num2class, X_test, y_test, model_1, β=β)
    return precision, recall, F1, Fβ, cm
    
if __name__ == "__main__":
    sigma = 20
    sigma_upper = 5
    num_bins_global = 2001
    bin_width_factor_global = 1 / 2001
    num_bins_local = 201
    bin_width_factor_local = 0.16
    levels_global = 6
    levels_local = 3
    wavelet_family = "sym5"
    
    
    use_wavelet = True
    binary_classification = True
    k_fold = None
    global_level_list = (1, 5,)
    local_level_list = (1, 3,)
    epochs = 10
    batch_size = 128
    l1 = 0.00
    l2 = 0.0
    dropout = 0.0
    β = 2.0
    frac = 0.5 # fracción del porcentaje relativo de datos de cada clase que multiplica a la categorical_crossentropy, o fracción de beta
    
    
    
    download_dir="data3/"
    path = "all_data_2024-07-17/"
    df_path = 'cumulative_2024.06.01_09.08.01.csv'
    use_download_cache = True
    lightcurve_cache = True
    
    n_proc = 20 
    parallel = True

    
    precision, recall, F1, Fβ, cm = main(sigma=sigma, sigma_upper=sigma_upper,
                num_bins_global=num_bins_global, bin_width_factor_global=bin_width_factor_global,
                num_bins_local=num_bins_local, bin_width_factor_local=bin_width_factor_local,
                levels_global=levels_global, levels_local=levels_local, wavelet_family=wavelet_family,
                use_wavelet=use_wavelet, binary_classification=binary_classification,
                k_fold=k_fold,
                global_level_list=global_level_list, local_level_list=local_level_list,
                l1=l1, l2=l2, dropout=dropout,
                epochs=epochs, batch_size=batch_size,
                frac=frac, β=β,
                download_dir=download_dir,
                path=path,
                df_path=df_path,
                use_download_cache=use_download_cache,
                n_proc=n_proc,
                parallel=parallel,
                lightcurve_cache=lightcurve_cache,
        )

    print("P : %f\nR : %f\nF1: %f\nFβ: %f" % (precision, recall, F1, Fβ))
