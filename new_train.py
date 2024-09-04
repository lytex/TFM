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
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings
warnings.filterwarnings(action="ignore", category=DeprecationWarning)
warnings.filterwarnings(action="ignore", category=FutureWarning)
from LCWavelet import *
from tqdm import tqdm
from collections import defaultdict
from parallelbar import progress_imap
from tqdm.contrib.concurrent import process_map
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
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
import pandas as pd
import traceback
mpl.use("agg")
import os
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"


descarga = importlib.import_module("01_descarga")
entrenamiento = importlib.import_module("02_entrenamiento")

GetBest = entrenamiento.GetBest
FilterModel = entrenamiento.FilterModel
F1_Score = entrenamiento.F1_Score
get_data_split = entrenamiento.get_data_split
gen_model_2_levels = entrenamiento.gen_model_2_levels
gen_astronet = entrenamiento.gen_astronet
from weighted_loss import WeightedBinaryCrossentropy,  WeightedCategoricalCrossentropy
process_light_curve = descarga.process_light_curve
load_files = entrenamiento.load_files


def grouper(iterable, n, *, incomplete='fill', fillvalue=None):
    "Collect data into non-overlapping fixed-length chunks or blocks."
    # grouper('ABCDEFG', 3) → ABC DEF
    iterators = [iter(iterable)] * n
    return zip(*iterators)


def process_func_continue_row(row,
    sigma=None, sigma_upper=None,
    num_bins_global=None, bin_width_factor_global=None,
    num_bins_local=None, bin_width_factor_local=None, num_durations=None,
    path=None, download_dir=None, use_download_cache=None,
    levels_global=None, levels_local=None, wavelet_family=None, use_wavelet=None,):

    descarga = importlib.import_module("01_descarga")
    process_light_curve = descarga.process_light_curve
    process_func =  partial(process_light_curve,
                            sigma=sigma, sigma_upper=sigma_upper,
                            num_bins_global=num_bins_global, bin_width_factor_global=bin_width_factor_global,
                            num_bins_local=num_bins_local, bin_width_factor_local=bin_width_factor_local, num_durations=num_durations,
                            levels_global=levels_global, levels_local=levels_local, wavelet_family=wavelet_family, use_wavelet=use_wavelet,
                            plot=False, plot_comparative=False, save=False, path=path, download_dir=download_dir, plot_folder=None, use_download_cache=use_download_cache, cache_dict=dict())
    print(f"Received {row.kepoi_name}")
    try:
        return process_func(row), row
    except Exception as e:
        print(f"Exception on {row.kepoi_name}")
        import traceback
        traceback.print_exc()
        return e, row

def descarga_process_light_curve(
    df_path=None,
    sigma=None, sigma_upper=None,
    num_bins_global=None, bin_width_factor_global=None,
    num_bins_local=None, bin_width_factor_local=None, num_durations=None,
    plot=False, plot_comparative=False, save=False, path=None, download_dir=None, plot_folder=None, use_download_cache=True,
    levels_global=None, levels_local=None, wavelet_family=None, use_wavelet=None,
    parallel=None
):
                        

    process_func =  partial(process_light_curve,
                            sigma=sigma, sigma_upper=sigma_upper,
                            num_bins_global=num_bins_global, bin_width_factor_global=bin_width_factor_global,
                            num_bins_local=num_bins_local, bin_width_factor_local=bin_width_factor_local, num_durations=num_durations,
                            levels_global=levels_global, levels_local=levels_local, wavelet_family=wavelet_family, use_wavelet=use_wavelet,
                            plot=False, plot_comparative=False, save=False, path=path, download_dir=download_dir, plot_folder=None, use_download_cache=use_download_cache, cache_dict=dict())

    def process_func_continue(row):
        try:
            return process_func(row)
        except Exception as e:
            print(f"Exception on {row.kepid}")
            import traceback
            traceback.print_exc()
            return e



    df = pd.read_csv(df_path ,skiprows=144)

    if not parallel:
        results = []
        for _, row in tqdm(df.iterrows(), total=len(df)):
            results.append(process_func_continue(row))
    else:
        n_cpu = multiprocessing.cpu_count()*4

        executor = ProcessPoolExecutor(max_workers=n_cpu)
        futures = {}
        results = []
        failed = []
        for group in list(grouper(df.iterrows(), 100)):
            print("len(group):", len(group))
            group = pd.DataFrame([x[1] for x in group] + [x[1] for x in failed]).drop_duplicates()
            group = list(group.iterrows())
            for _, row in group:
                failed = pd.DataFrame([x[1] for x in failed] + [x[1] for x in group]).drop_duplicates()
                failed = list(failed.iterrows())
                try:
                    future = executor.submit(process_func_continue_row, row,
                    sigma=sigma, sigma_upper=sigma_upper,
                    num_bins_global=num_bins_global, bin_width_factor_global=bin_width_factor_global,
                    num_bins_local=num_bins_local, bin_width_factor_local=bin_width_factor_local, num_durations=num_durations,
                    path=path, download_dir=download_dir, use_download_cache=use_download_cache,
                    levels_global=levels_global, levels_local=levels_local, wavelet_family=wavelet_family, use_wavelet=use_wavelet,
                                             )
                    futures[future] = row
                except BrokenProcessPool:
                    executor.shutdown(wait=False)
                    print("BrokenProcessPool, creating new ProcessPoolExecutor")
                    executor = ProcessPoolExecutor(max_workers=n_cpu)
            for future in tqdm(as_completed(futures, timeout=30*60)):
                try:
                    exc = future.exception()
                    if exc is not None:
                        print("multiprocessing exc 1:", exc)
                        traceback.print_tb(exc.__traceback__)
                except BrokenProcessPool as exc:
                    print("multiprocessing exc 2:", exc)
                    traceback.print_tb(exc.__traceback__)
                    executor.shutdown(wait=False)
                    print("BrokenProcessPool, creating new ProcessPoolExecutor")
                    executor = ProcessPoolExecutor(max_workers=n_cpu)
                except Exception as exc:
                    print("multiprocessing exc 2:", exc)
                    traceback.print_tb(exc.__traceback__)
                try:
                    result, row = future.result()
                    if type(result) in (LightCurveWaveletGlobalLocalCollection, LightCurveShallueCollection):
                        results.append(result)
                        print("lenfailed", len(failed))
                        failed = [x for x in failed if x[1] != row]
                        print("lenfailed", len(failed))
                except BrokenProcessPool as exc:
                    print("multiprocessing exc 3:", exc)
                    traceback.print_tb(exc.__traceback__)
                    executor.shutdown(wait=False)
                    print("BrokenProcessPool, creating new ProcessPoolExecutor")
                    executor = ProcessPoolExecutor(max_workers=n_cpu)
                except Exception as exc:
                    print("multiprocessing exc 3:", exc)
                    traceback.print_tb(exc.__traceback__)

        executor.shutdown()

    return results

# %% [raw]
# [np.sum(x) for x in hist]

# %%

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
def get_model_wrapper(lightcurves, use_wavelet=True, binary_classification=False, frac=0.5, model_name="model", test_size=0.3, global_level_list=None, local_level_list=None, l1=0.0, l2=0.0, dropout=0.0,
                     num_bins_global=None, num_bins_local=None):

    lightcurves = [lc for lc in lightcurves if lc is not None]
    
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
                            metrics=['accuracy', tf.keras.metrics.Recall(), tf.keras.metrics.Precision(), F1_Score(), tf.keras.metrics.AUC(curve='PR')])
        else:
            
            # model_1.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(),
            #                 metrics=[F1_Score(),])
    
            
            count = pd.DataFrame({'col': np.argmax(y_entire, axis=1)}).reset_index(drop=False).groupby('col').index.count()
            print("count:",  count[0]/count[1]*frac)
            model_1.compile(loss=WeightedCategoricalCrossentropy(weights=[1.0, count[0]/count[1]*frac]), optimizer=tf.keras.optimizers.Adam(),
                            metrics=[F1_Score(),])
    
    else:
        
        count =  pd.DataFrame({'col': y_entire}).reset_index(drop=False).groupby('col').index.count()
        print("count:",  count[0]/count[1]*frac)
        # from FBetaScore import DifferentiableFBetaScore
        model_1.compile(loss=WeightedBinaryCrossentropy(weights=[1.0, count[0]/count[1]*frac]), optimizer=tf.keras.optimizers.Adam(),
                        metrics=['accuracy', tf.keras.metrics.Recall(), tf.keras.metrics.Precision(), F1_Score(), tf.keras.metrics.AUC(curve='PR')])
    
    # tf.keras.utils.plot_model(model_1, f"{model_name}.png")
    # tf.keras.utils.model_to_dot(model_1).write(f"{model_name}.dot")
    # parameters = sum(count_params(layer) for layer in model_1.trainable_weights
    # print("model_1 has", parameters, "parameters")
    return model_1

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
def train_model(model_1_lazy, lightcurves, use_wavelet=True, binary_classification=False, k_fold=None,
                global_level_list=None, local_level_list=None, epochs=200, batch_size=128, test_size=0.3,
               save_callback=False, best_callback=True):

    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    
    # cp_callback = tf.keras.callbacks.BackupAndRestore(log_dir)
    history_1 =  pd.DataFrame()

    model_1 = model_1_lazy()

    callbacks = []
    if save_callback:
        callbacks += [tf.keras.callbacks.ModelCheckpoint(log_dir, monitor='val_loss', save_best_only=True)]
    if best_callback:
        callbacks += [GetBest(monitor='val_f1_score', verbose=0, mode='max')]
    callbacks += [FilterModel(epochs=epochs, batch_size=batch_size)]

    
    if k_fold is None:
        lightcurves_kfold, lightcurves_val = train_test_split(lightcurves, test_size=test_size, shuffle=True)
        inputs, X_train, X_test, y_train, y_test, y, kepid_test, kepid_train, num2class, \
            output_classes = get_data_split(lightcurves, binary_classification=binary_classification, use_wavelet=use_wavelet, global_level_list=global_level_list, local_level_list=local_level_list, test_size=test_size)
        _, X_val, X_test, y_val, y_test, _, kepid_test, kepid_val, num2class, \
            _ = get_data_split(lightcurves_val, binary_classification=binary_classification, use_wavelet=use_wavelet, test_size=0.5,
                               global_level_list=global_level_list, local_level_list=local_level_list)
        
        print("y_train:", y_train.shape, np.where(y_train == 0)[0].shape, np.where(y_train == 1)[0].shape)
        print("y_val:", y_val.shape, np.where(y_val == 0)[0].shape, np.where(y_val == 1)[0].shape)
        print("y_test:", y_test.shape, np.where(y_test == 0)[0].shape, np.where(y_test == 1)[0].shape)
        print("total:", y_train.shape[0]+y_val.shape[0]+y_test.shape[0],
                  np.where(y_train == 0)[0].shape[0] +np.where(y_test == 0)[0].shape[0] + np.where(y_test == 0)[0].shape[0],
                  np.where(y_train == 1)[0].shape[0] +np.where(y_test == 1)[0].shape[0] + np.where(y_test == 1)[0].shape[0],
             )
        print("num2class:", num2class)
        temp = model_1.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val),
                                callbacks=callbacks)
        history_1 = history_1.append(pd.DataFrame(temp.history))
    else:
        lightcurves_kfold, lightcurves_val = train_test_split(lightcurves, test_size=test_size, shuffle=True)
        _, X_val, X_test, y_val, y_test, _, kepid_test, kepid_val, num2class, \
            _ = get_data_split(lightcurves_val, binary_classification=binary_classification, use_wavelet=use_wavelet, test_size=0.5,
                               global_level_list=global_level_list, local_level_list=local_level_list)
        for ind in tqdm(range(k_fold)):
            inputs, X_train, X_test_kfold, y_train, y_test_kfold, y, kepid_test_kfold, kepid_train, num2class, \
                output_classes = get_data_split(lightcurves_kfold, binary_classification=binary_classification, use_wavelet=use_wavelet,
                                                global_level_list=global_level_list, local_level_list=local_level_list, test_size=test_size)
            inputs, X_train, X_test_kfold, y_train, y_test_kfold, y, kepid_test_kfold, kepid_train, num2class, \
                output_classes = get_data_split(lightcurves_kfold, binary_classification=binary_classification, use_wavelet=use_wavelet, k_fold=k_fold, ind=ind,
                                                global_level_list=global_level_list, local_level_list=local_level_list, test_size=test_size)
            temp = model_1.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val),
                                    callbacks=callbacks)
            
            temp.history.update({"k_ind": ind, "k_fold": k_fold})
            history_1 = history_1.append(pd.DataFrame(temp.history))
    
    history_1 = history_1.reset_index().rename(columns={"index": "epoch"})
    
    return model_1, history_1, num2class, X_val, y_val, X_test, y_test, kepid_test


# %%
# history_1, num2class, X_test, y_test = train_model(lightcurves,
#                                                    use_wavelet=use_wavelet, binary_classification=binary_classification,
#                                                    k_fold=k_fold, global_level_list=global_level_list, local_level_list=local_level_list, epochs=epochs, batch_size=batch_size)
    
def get_metrics(num2class, X_test, y_test, model_1, β=1.0, binary_classification=False, plot=False, save_failures=False):
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
        
        from sklearn.metrics import precision_recall_curve, auc
        # Calculate precision and recall for various thresholds
        precision, recall, thresholds = precision_recall_curve(y_test_sampled, np.squeeze(y_predict))
        # Calculate the Area Under the Curve (AUC)
        auc_score = auc(recall, precision)
        print("auc_score", auc_score)
        
        # Plot the Precision-Recall curve
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='blue', label=f'PR curve (AUC = {auc_score:.2f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower left")
        plt.grid(True)
        plt.savefig("plot/results/ROC_"+datetime.datetime.utcnow().strftime("%s%f")+".png")
        plt.show()
    if save_failures:
        wrong = y_predict_sampled != y_test_sampled
        
        
        download_dir="data3/"
        import importlib  
        descarga = importlib.import_module("01_descarga")
        process_light_curve = descarga.process_light_curve

        df_path = 'cumulative_2024.06.01_09.08.01.csv'
        df = pd.read_csv(df_path ,skiprows=144)
        df_kepid = pd.DataFrame({"kepid": kepid_test[wrong], "predicted": y_predict_sampled[wrong], "true": y_test_sampled[wrong]})
        df_wrong = pd.merge(df_kepid, df, how="inner", on="kepid")
    
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
        
        results = []
        for _, row in tqdm(df_wrong.iterrows(), total=len(df_wrong)):
            try:
                results.append(process_light_curve(row, title=f" Predicho: {num2class[row.predicted]} Real: {num2class[row.true]}",
                                                   levels_global=6, levels_local=3, wavelet_family="sym5", sigma=20, sigma_upper=5,
                                                   plot=True, plot_comparative=False, save=False, path=path, download_dir=download_dir, plot_folder=log_dir, use_download_cache=True))
            except Exception as e:
                print(f"Exception on {row.kepid}")
                import traceback
                traceback.print_exc()
                results.append(e)



    precision = cm[0][0]/(cm[0][0] + cm[1][0])
    recall = cm[0][0]/(cm[0][0] + cm[0][1])
    F1 = 2*(precision*recall)/(precision+recall)
    Fβ = (1+β**2)*(precision*recall)/(β**2*precision+recall)

    
    return precision, recall, F1, Fβ, auc_score, cm, num2class

# precision, recall, F1, Fβ, cm = get_metrics(num2class, X_test, y_test, model_1, β=2.0)
# print("P : %f\nR : %f\nF1: %f\nFβ: %f" % (precision, recall, F1, Fβ))

def load_files(file, path, use_wavelet=None):
    try:
        if use_wavelet:
            global_local = LightCurveWaveletGlobalLocalCollection.from_pickle(path+file)
        else:
            global_local = LightCurveShallueCollection.from_pickle(path+file)
    except Exception as e:
        import traceback
        print(f"Error con archivo {path}/{file}")
        traceback.print_exc()
        return None
        
    return global_local

def load_files_wrapper(path, use_wavelet=True):
    from new_train import load_files
    if use_wavelet:
        files = [file for file in os.listdir(path) if file.endswith(".pickle") and "wavelet" in file]
    else:
        files = [file for file in os.listdir(path) if file.endswith(".pickle") and "wavelet" not in file]
        
    func = partial(load_files, path=path, use_wavelet=use_wavelet)
            
    lightcurves = progress_imap(func, files, n_cpu=multiprocessing.cpu_count()*4, total=len(files), executor='processes', error_behavior='raise', chunk_size=len(files)//multiprocessing.cpu_count()//4//10)
    return lightcurves

def main(sigma = 20, sigma_upper = 5,
            num_bins_global = 2001, bin_width_factor_global = 1 / 2001,
            num_bins_local = 201, bin_width_factor_local = 0.16, num_durations=4,
            levels_global = 6, levels_local = 3, wavelet_family = "sym5",
            use_wavelet = None, binary_classification = None,
            k_fold = 5,
            global_level_list = (1, 5,), local_level_list = (1, 3,),
            l1 = 0.00, l2 = 0.0, dropout = 0.0,
            epochs = 200, batch_size = 128, test_size=0.3,
            frac = 0.5, β=1.0,
            download_dir=None,
            path = None,
            df_path = 'cumulative_2024.06.01_09.08.01.csv',
            file_path="",
            use_download_cache = True,
            n_proc = 20,
            parallel = True,
            lightcurve_cache=True,
            return_lightcurves=False,
            lightcurves=None,
            apply_candidates=False,
    ):

    if lightcurves is None:
        if lightcurve_cache:
            lightcurves = load_files_wrapper(path=path, use_wavelet=use_wavelet)
        else:
            results = descarga_process_light_curve(
                df_path=df_path,
                sigma=sigma, sigma_upper=sigma_upper,
                num_bins_global=num_bins_global, bin_width_factor_global=bin_width_factor_global,
                num_bins_local=num_bins_local, bin_width_factor_local=bin_width_factor_local, num_durations=num_durations,
                levels_global=levels_global, levels_local=levels_local, wavelet_family=wavelet_family,
                plot=False, plot_comparative=False, save=False, path=path, download_dir=download_dir, plot_folder=None, use_download_cache=use_download_cache,
                use_wavelet=use_wavelet, parallel=parallel
            )
            lightcurves = [x for x in results if type(x) in (LightCurveWaveletGlobalLocalCollection, LightCurveShallueCollection)]
        
            
        lightcurves = [lc for lc in lightcurves if lc is not None]

    model_1_lazy = lambda : get_model_wrapper(lightcurves, use_wavelet=use_wavelet, binary_classification=binary_classification, frac=frac,  test_size=test_size,
                                         global_level_list=global_level_list, local_level_list=local_level_list,
                                        l1=l1, l2=l2, dropout=dropout,
                                        num_bins_global=num_bins_global,
                                        num_bins_local=num_bins_local)

    if k_fold is None:
        model_1, history_1, num2class, X_val, y_val, X_test, y_test, recall_val = train_model(model_1_lazy, lightcurves,
                                                           use_wavelet=use_wavelet, binary_classification=binary_classification,
                                                           k_fold=k_fold, global_level_list=global_level_list, local_level_list=local_level_list, epochs=epochs, batch_size=batch_size, test_size=test_size)
    
        
        precision_val, recall_val, F1_val, Fβ_val, auc_val, cm_val, num2class = get_metrics(num2class, X_val, y_val, model_1, β=β, binary_classification=binary_classification, plot=True)
    else:
        # TODO añadir en el caso de k-fold
        precision_val, recall_val, F1_val, Fβ_val, auc_val, cm_val, num2class = get_metrics(num2class, X_val, y_val, model_1, β=β, binary_classification=binary_classification, plot=True)

    precision, recall, F1, Fβ, auc, cm, num2class = get_metrics(num2class, X_test, y_test, model_1, β=β, binary_classification=binary_classification, plot=False)



    if apply_candidates:
        if use_wavelet:
            lightcurves_candidate = [lc for lc in lightcurves if lc.headers["class"] == "CANDIDATE"]
            lightcurves_candidate = sorted(lightcurves_candidate, key=lambda lc: lc.headers["Kepler_name"])
            def set_class(lc):
                lc.headers["class"] = ""
                return lc
            lightcurves_candidate = [set_class(lc) for lc in lightcurves_candidate]
        else:
            lightcurves_candidate = [lc for lc in lightcurves if lc.headers["koi_disposition"] == "CANDIDATE"]
            lightcurves_candidate = sorted(lightcurves_candidate, key=lambda lc: lc.headers["kepoi_name"])
            def set_class(lc):
                lc.headers["koi_disposition"] = ""
                return lc
            lightcurves_candidate = [set_class(lc) for lc in lightcurves_candidate]

        _, _, X_candidate, _, y_candidate, _, kepid_candidate, _, _, \
            _ = get_data_split(lightcurves_candidate, binary_classification=binary_classification, use_wavelet=use_wavelet, test_size=1.0,
                              global_level_list=global_level_list, local_level_list=local_level_list)
        
        num2class_vec = np.vectorize(num2class.get)
        y_predict = model_1.predict(X_candidate)
        # Escoger la clase que tiene mayor probabilidad
        if binary_classification:
            y_candidate_sampled = y_candidate
            y_predict_sampled = (np.squeeze(y_predict) > 0.5).astype(int)
        
        else:
            y_predict_sampled = y_predict.argmax(axis=1)
            y_candidate_sampled = y_candidate.argmax(axis=1)

        df_candidate = pd.DataFrame({"id": kepid_candidate, "class": num2class_vec(y_predict_sampled)})
        df_candidate.to_csv(f"{path+file_path}/candidate.csv")

    from shutil import copyfile
    study_name = "example-study"  # Unique identifier of the study.
    copyfile(f"{study_name}.db", f"{path+file_path}/{study_name}.db")
    print("copiando db a ", f"{path+file_path}/{study_name}.db")
    print(os.listdir(path+file_path+"/"))
    if return_lightcurves:
        return precision, recall, F1, Fβ, auc, cm, num2class, precision_val, recall_val, F1_val, Fβ_val, auc_val, cm_val, history_1, lightcurves
    else:
        return precision, recall, F1, Fβ, auc, cm, num2class, precision_val, recall_val, F1_val, Fβ_val, auc_val, cm_val, history_1
    
if __name__ == "__main__":

    # %matplotlib inline

    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    from multiprocessing import set_start_method
    set_start_method("spawn", force=True)

    sigma = 20
    sigma_upper = 5
    num_bins_global = 2001
    bin_width_factor_global = 0.0004997501249375
    num_bins_local = 201
    bin_width_factor_local = 0.16
    num_durations = 4
    levels_global = 6
    levels_local = 3
    wavelet_family = 'sym5'
    use_wavelet = True
    binary_classification = True
    k_fold = None
    global_level_list = (1, 3)
    local_level_list = (1,)
    l1 = 0.0031991408399097
    l2 = 0.0033802040346427
    dropout = 0.0065163715298307
    epochs = 100
    batch_size = 128
    frac = 0.9342089267108736
    β = 2.0

    
    
    download_dir="data3/"
    path = "all_data_2024-07-17/"
    df_path = 'cumulative_2024.06.01_09.08.01.csv'
    use_download_cache = True
    lightcurve_cache = True
    
    n_proc = 20 
    parallel = True


    
    precision, recall, F1, Fβ, auc, cm, num2class, precision_val, recall_val, F1_val, Fβ_val, auc_val, cm_val, history_1 = main(sigma=sigma, sigma_upper=sigma_upper,
                num_bins_global=num_bins_global, bin_width_factor_global=bin_width_factor_global,
                num_bins_local=num_bins_local, bin_width_factor_local=bin_width_factor_local, num_durations=num_durations,
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
                apply_candidates=True,
        )


    import datetime
    # %matplotlib inline
    print("val_auc", history_1.sort_values(by="val_f1_score", ascending=False).iloc[0].val_auc)

    ConfusionMatrixDisplay(confusion_matrix=cm_val, display_labels=[str(v) for v in sorted(num2class.values(), reverse=True)]).plot(xticks_rotation='vertical')
    plt.subplots_adjust(left=0.15, bottom=0.5)
    plt.savefig("plot/results/cm_val.png")
    print("P_val : %f\nR_val : %f\nF1_val: %f\naccuracy: %f\nFβ_val: %f" % (precision_val, recall_val, F1_val, cm_val.trace()/cm_val.sum(), Fβ_val))

    print(cm_val)
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[str(v) for v in sorted(num2class.values(), reverse=True)]).plot(xticks_rotation='vertical')
    plt.subplots_adjust(left=0.15, bottom=0.5)
    plt.savefig("plot/results/cm.png")
    print("P : %f\nR : %f\nF1: %f\naccuracy: %f\nFβ: %f" % (precision, recall, F1, cm.trace()/cm.sum(), Fβ))
    print(cm)
    print(pd.DataFrame({"dataset": ["Validación" , "Test", ] ,"AUC": [auc_val, auc], "Accuracy": [cm_val.trace()/cm_val.sum(), cm_val.trace()/cm_val.sum()],
                  "Precision": [precision_val, precision], "Recall": [recall_val, recall], "F1": [F1_val, F1]}).to_latex(index=False))
    history_1_old = history_1

# %%
if __name__ == "__main__":
    # %matplotlib inline
    import datetime
    if not binary_classification:
        # history_1 = history_1_old[100:]
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
        best_index =  history_1.sort_values(by="val_f1_score", ascending=False).index[0]
        # summarize history_1 for accuracy
        plt.plot(history_1['accuracy'])
        plt.plot(history_1['val_accuracy'])
        plt.plot(history_1['epoch'].iloc[best_index], history_1['val_accuracy'].iloc[best_index], 'k+', markersize=12)
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.savefig("plot/results/accuracy.png")
        plt.show()

        # summarize history_1 for precision
        plt.plot(history_1['precision'])
        plt.plot(history_1['val_precision'])
        plt.plot(history_1['epoch'].iloc[best_index], history_1['val_precision'].iloc[best_index], 'k+', markersize=12)
        plt.title('model precision')
        plt.ylabel('precision')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.savefig("plot/results/precision.png")
        plt.show()

        # summarize history_1 for recall
        plt.plot(history_1['recall'])
        plt.plot(history_1['val_recall'])
        plt.plot(history_1['epoch'].iloc[best_index], history_1['val_recall'].iloc[best_index], 'k+', markersize=12)
        plt.title('model recall')
        plt.ylabel('recall')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.savefig("plot/results/recall.png")
        plt.show()

        plt.plot(history_1['loss'])
        plt.plot(history_1['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.savefig("plot/results/loss.png")
        plt.show()


        plt.plot(history_1['f1_score'])
        plt.plot(history_1['val_f1_score'])
        plt.plot(history_1['epoch'].iloc[best_index], history_1['val_f1_score'].iloc[best_index], 'k+', markersize=12)
        plt.title('model f1_score')
        plt.ylabel('f1_score')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.savefig("plot/results/f1_score.png")
        plt.show()


        plt.plot(history_1['auc'])
        plt.plot(history_1['val_auc'])
        plt.plot(history_1['epoch'].iloc[best_index], history_1['val_auc'].iloc[best_index], 'k+', markersize=12)
        plt.title('model auc')
        plt.ylabel('auc')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.savefig("plot/results/auc.png")
        plt.show()


        plt.plot(history_1['val_precision'])
        plt.plot(history_1['val_recall'])
        plt.title('precision vs recall')
        plt.ylabel('recall')
        plt.xlabel('epoch')
        plt.legend(['val precision', 'val recall'], loc='lower center')
        plt.savefig("plot/results/precision_vs_recall.png")
        plt.show()
