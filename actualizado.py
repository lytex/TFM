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

# %% editable=true slideshow={"slide_type": ""}
import pandas as pd
import lightkurve as lk
import numpy as np
import pywt
import pickle
import os
import matplotlib.pyplot as plt
from LCWavelet import *
import json
# %matplotlib inline

df_path = 'cumulative_2022.09.30_09.06.43.csv'
df = pd.read_csv(df_path ,skiprows=144)

# generate_dataset lee cumulative_2022.09.30_09.06.43.csv y escribe light_curves_<espectro>_stars_filter.csv con sólo confirmados y falsos positivos y light_curves_<espectro>_stars.csv con todo
# En mi caso no interesa filtrar por tipo espectral

wavelet_family='sym5'
level = 9
save_lc = True
save_path = 'all_data_2024-05-06/'
wavelet_windows = 15000

with open(save_path+"params.json", "w") as f:
    json.dump({"df": df_path, "df_hash": str(pd.util.hash_pandas_object(df).sum()), "wavelet_family": wavelet_family, "level": level, "wavelet_windows": wavelet_windows}, f)

completed = os.listdir(save_path)
if 'errors.txt' in completed:
  completed.remove('errors.txt')
completed_id = []
for element in completed:
  completed_id.append(element.replace('.pickle','').replace('kic ',''))

# en Process_light_curve_dataset.ipynb, process_dataset se guarda en save_path las wavelets procesadas finales como pickles

errores = process_dataset(df,repeat_completed=False, completed=completed_id, save_path=save_path, wavelet_family=wavelet_family, level=level, save_lc=save_lc, wavelet_windows=wavelet_windows)

# en Train_LCWavelet_Model.ipynb se cargan esos pickles utilizando dataset_path y se entrenan distintos modelos

dataset_path= save_path
train_split = .80

# %%
