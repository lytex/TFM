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
if __name__ == "__main__":
    from multiprocessing import set_start_method
    set_start_method("spawn", force=True)
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    from new_train import main, load_files_wrapper
    import pandas as pd
    import gc
    import os
    import tensorflow as tf
    import optuna
    import traceback
    import multiprocessing
    import datetime
    # import logging
    import sys
    from shutil import copyfile
    from functools import partial

    from optuna.artifacts import FileSystemArtifactStore
    from optuna.artifacts import upload_artifact
    from tqdm import tqdm
    from taguchi import generate_taguchi


    base_path = "./all_data_2024-07-17/all_data_2024-07-17/"
    os.makedirs(base_path, exist_ok=True)
    artifact_store = FileSystemArtifactStore(base_path=base_path)

    path = "all_data_2024-07-17/all_data_2024-07-17/"
    file_path = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    os.makedirs(path+file_path, exist_ok=True)
    print("trial folder created:", path+file_path)

    # lightcurves_wavelet = load_files_wrapper(path=path, use_wavelet=True)
    # lightcurves_no_wavelet = load_files_wrapper(path=path, use_wavelet=False)

    def objective(trial, global_level_list=None, local_level_list=None, use_wavelet=None):
        # global lightcurves_wavelet
        # global lightcurves_no_wavelet
        #
        # if use_wavelet:
        #     lightcurves = lightcurves_wavelet
        # else:
        #     lightcurves = lightcurves_no_wavelet
        #
        sigma = 20
        sigma_upper = 5
        num_bins_global = 2001
        bin_width_factor_global = 1 / 2001
        num_bins_local = 201
        bin_width_factor_local = 0.16
        num_durations = 4
        levels_global = 6
        levels_local = 3


        num_bins_global = trial.suggest_int("num_bins_global", 201, 20001, step=20)
        bin_width_factor_global = 1 / trial.params.get("num_bins_global")
        num_bins_local = trial.suggest_int("num_bins_local", trial.params.get("num_bins_global")//5, trial.params.get("num_bins_global")*100//5, step=4)
        bin_width_factor_local = trial.suggest_float("bin_width_factor_local", 0.016, 1.6)
        num_durations = trial.suggest_int("num_durations", 1, 6)

        wavelet_family = trial.suggest_categorical("wavelet_family", [f"sym{N}" for N in range(2, 7)] + [f"db{N}" for N in range(1, 7)])
        # wavelet_family = "sym5"


        binary_classification = True
        k_fold = None
        # global_level_list = (1, 5,)
        # local_level_list = (1, 3,)
        epochs = 100
        batch_size = 128
        l1 = trial.suggest_float("l1", 0.0, 0.1)
        l2 = trial.suggest_float("l2", 0.0, 0.1)
        dropout = trial.suggest_float("dropout", 0.0, 0.3)

        β = 2.0
        frac =  trial.suggest_float("frac", 0.1, 1.9)



        download_dir="data3/data3/"
        path = "all_data_2024-07-17/all_data_2024-07-17/"
        df_path = 'cumulative_2024.06.01_09.08.01.csv'
        use_download_cache = True
        lightcurve_cache = False
        lightcurves = None

        n_proc = int(multiprocessing.cpu_count()*1.25)
        parallel = True


        try:
            precision, recall, F1, Fβ, cm, num2class, precision_val, recall_val, F1_val, Fβ_val, cm_val, history_1 = main(sigma=sigma, sigma_upper=sigma_upper,
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
                )
        except Exception as exc:

            print("optuna exc:", exc)
            traceback.print_tb(exc.__traceback__)
            precision = 0
            recall = 0
            F1 = 0
            Fβ = 0
            cm = [[0, 0], [0, 0]]
            num2class = {0: None, 1: None}
            precision_val = 0
            recall_val = 0
            F1_val = 0
            Fβ_val = 0
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
            "precision": precision, "recall": recall, "F1": F1, "Fβ": Fβ,
            "precision_val": precision_val, "recall_val": recall_val, "F1_val": F1_val, "Fβ_val": Fβ_val,
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

    taguchi_list = list(generate_taguchi(levels_global=6, levels_local=3))
    for global_level_list, local_level_list in tqdm(taguchi_list):
        if len(global_level_list) == 0 and len(local_level_list) == 0:
            use_wavelet = False
        else:
            use_wavelet = True

        print(f"Wavelet list: global: {global_level_list}, local: {local_level_list}, use_wavelet: {use_wavelet}")
        objective_func = partial(objective, global_level_list=global_level_list, local_level_list=local_level_list, use_wavelet=use_wavelet)
        study.optimize(objective_func, n_trials=100)

        trial = study.best_trial

        print("Accuracy: {}".format(trial.value))
        print("Best hyperparameters: {}".format(trial.params))

    copyfile(f"{study_name}.db", f"{path+file_path}/{study_name}.db")
    print("copiando db a ", f"{path+file_path}/{study_name}.db")
    print(os.listdir(path+file_path+"/"))
    import subprocess
    process = subprocess.Popen(f"cd {path+file_path}; head -n +1 $(for file in *.csv; do echo $file; done | head -n 1) > all.csv; tail -q -n 1 *.csv >> all.csv", shell=True,
                               stdout=sys.stdout,
                               stderr=sys.stderr)
    import pandas as pd
    df = pd.read_csv(path+file_path+'/all.csv')
    print("\n".join([f"{k} = {v}" for k, v in df.sort_values(by="F1_val", ascending=False).iloc[0, 0:22].items()]))


# %%
