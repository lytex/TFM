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
print("optuna_trial.py, new code")
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
from new_train import main, load_files_wrapper
import pandas as pd
import gc
import os
import tensorflow as tf
import optuna
import multiprocessing
import datetime
import logging
import sys
from shutil import copyfile
from functools import partial, reduce

from optuna.artifacts import FileSystemArtifactStore
from optuna.artifacts import upload_artifact
from taguchi import generate_taguchi
import warnings
warnings.filterwarnings(action="ignore", category=UserWarning)


base_path = "./all_data_2024-07-17/all_data_2024-07-17/"
os.makedirs(base_path, exist_ok=True)
artifact_store = FileSystemArtifactStore(base_path=base_path)

path = "all_data_2024-07-17/all_data_2024-07-17/"
file_path = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
os.makedirs(path+file_path, exist_ok=True)
print("trial folder created:", path+file_path)

lightcurves_wavelet = load_files_wrapper(path=path, use_wavelet=True)
lightcurves_no_wavelet = load_files_wrapper(path=path, use_wavelet=False)

def objective(trial, global_level_list=None, local_level_list=None, use_wavelet=None):
    global lightcurves_wavelet
    global lightcurves_no_wavelet


    # binary_classification = trial.suggest_categorical("binary_classification", [True, False])
    # use_wavelet = use_wavelet or not trial.params["binary_classification"]
    binary_classification = True
    levels_global = 6
    levels_local = 3
    # global_level_list = trial.suggest_categorical("global_level_list", [tuple(reduce(lambda x, y: x+y, [[i+1]*bool(x&(2**i)) for i in range(levels_global)], [])) for x in range(2**(levels_global+1))])
    # local_level_list = trial.suggest_categorical("local_level_list", [tuple(reduce(lambda x, y: x+y, [[i+1]*bool(x&(2**i)) for i in range(levels_local)], [])) for x in range(2**(levels_local+1))])
    local_level_list = tuple()
    global_level_list = tuple()
    use_wavelet = False

    # global_level_list = (1, 5,)
    # local_level_list = (1, 3,)

    # if len(trial.params["global_level_list"]) == 0 and len(trial.params["local_level_list"]) == 0:
    # # if len(trial.params["global_level_list"]) == 0:
    #     use_wavelet = False
    # else:
    #     use_wavelet = True

    if use_wavelet:
        lightcurves = lightcurves_wavelet
    else:
        lightcurves = lightcurves_no_wavelet

    sigma = 20
    sigma_upper = 5
    num_bins_global = 2001
    bin_width_factor_global = 1 / 2001
    num_bins_local = 201
    bin_width_factor_local = 0.16
    num_durations = 4
    wavelet_family = "sym5"
    
    
    k_fold = None
    epochs = 100
    batch_size = 128
    l1 = trial.suggest_float("l1", 0.0, 0.1)
    # l1 = 0.0
    l2 = trial.suggest_float("l2", 0.0, 0.1)
    # dropout = 0.0
    dropout = trial.suggest_float("dropout", 0.0, 0.3)
    
    β = 2.0
    # frac = 1.43
    frac =  trial.suggest_float("frac", 0.1, 1.9)
    
    
    
    download_dir="data3/data3/"
    path = "all_data_2024-07-17/all_data_2024-07-17/"
    df_path = 'cumulative_2024.06.01_09.08.01.csv'
    use_download_cache = True
    lightcurve_cache = True
    
    n_proc = int(multiprocessing.cpu_count()*1.25)
    parallel = True

    
    try:
        precision, recall, F1, Fβ, auc, cm, num2class, precision_val, recall_val, F1_val, Fβ_val, auc_val, cm_val, history_1  = main(sigma=sigma, sigma_upper=sigma_upper,
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
                    lightcurves=lightcurves,
                    file_path=file_path,
            )

    except Exception as exc:

        print("optuna exc:", exc)
        import traceback
        traceback.print_tb(exc.__traceback__)
        precision = 0
        recall = 0
        F1 = 0
        Fβ = 0
        auc = 0
        cm = [[0, 0], [0, 0]]
        num2class = {0: None, 1: None}
        precision_val = 0
        recall_val = 0
        F1_val = 0
        Fβ_val = 0
        auc_val = 0
        cm_val = [[0, 0], [0, 0]]
        history_1 = []

    gc.collect()
    tf.keras.backend.clear_session()

    variables = ["sigma", "sigma_upper",
                "num_bins_global", "bin_width_factor_global",
                "num_bins_local", "bin_width_factor_local", "num_durations",
                "levels_global", "levels_local", "wavelet_family",
                "use_wavelet", "binary_classification",
                "k_fold",
                "global_level_list", "local_level_list",
                "l1", "l2","dropout",
                "epochs", "batch_size",
                "frac", "β",
                "download_dir",
                "path",
                "df_path",
                "use_download_cache",
                "n_proc",
                "parallel",
                "lightcurve_cache",
                ]
    local_dict = locals()
    variables_dict = {variable: local_dict.get(variable, trial.params.get(variable)) for variable in variables}
    variables_dict.update({
        "precision": precision, "recall": recall, "F1": F1, "Fβ": Fβ, "auc": auc,
        "precision_val": precision_val, "recall_val": recall_val, "F1_val": F1_val, "Fβ_val": Fβ_val, "auc_val": auc_val,
                  "cm_val_00": cm_val[0][0], "cm_val_01": cm_val[0][1], "cm_val_10": cm_val[1][0], "cm_val_11": cm_val[1][1],
                  "cm_00": cm[0][0], "cm_01": cm[0][1], "cm_10": cm[1][0], "cm_11": cm[1][1], "0": num2class[0], "1": num2class[1]})
    result_df = pd.DataFrame([variables_dict])

    now = datetime.datetime.now().strftime("%s")
    # upload_artifact(trial, path+file_path+"/"+now, artifact_store)
    result_df.to_csv(path+file_path+"/"+now+".csv", index=False)
    print("guardando csv en", path+file_path+"/"+now+".csv")
    print(os.listdir(path+file_path+"/"))
    print("P : %f\nR : %f\nF1: %f\nFβ: %f" % (precision, recall, F1, Fβ))
    print(cm)

    return F1_val


# Add stream handler of stdout to show the messages
optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
study_name = "example-study"  # Unique identifier of the study.

# storage = optuna.storages.JournalStorage(
#    optuna.storages.JournalFileStorage("./journal.log"),
# )

storage = "sqlite:///{}.db".format(study_name)

study = optuna.create_study(direction="maximize", storage=storage)

study.optimize(objective, n_trials=None, gc_after_trial=True,  show_progress_bar=True)

trial = study.best_trial

print("Accuracy: {}".format(trial.value))
print("Best hyperparameters: {}".format(trial.params))

from optuna.importance import PedAnovaImportanceEvaluator
evaluator = PedAnovaImportanceEvaluator()
evaluator.evaluate(study)
copyfile(f"{study_name}.db", f"{path+file_path}/{study_name}.db")
print("copiando db a ", f"{path+file_path}/{study_name}.db")
print(os.listdir(path+file_path+"/"))
import subprocess
process = subprocess.Popen(f"cd {path+file_path}; head -n +1 $(for file in *.csv; do echo $file; done | head -n 1) > all.csv; tail -q -n 1 *.csv >> all.csv", shell=True,
                           stdout=sys.stdout,
                           stderr=sys.stderr)

