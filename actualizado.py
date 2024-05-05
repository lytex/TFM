# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.0
#   kernelspec:
#     display_name: TFM
#     language: python
#     name: tfm
# ---

# %%
import pandas as pd

df_light_curves = pd.read_csv('cumulative_2022.09.30_09.06.43.csv',skiprows=144)
# generate_dataset lee cumulative_2022.09.30_09.06.43.csv y escribe light_curves_<espectro>_stars_filter.csv con sólo confirmados y falsos positivos  y light_curves_<espectro>_stars.csv con todo
# En mi caso no interesa filtrar por tipo espectral
df = pd.read_csv('light_curves_K_stars_filter.csv')

import pandas as pd
import lightkurve as lk
import numpy as np
import pywt
import pickle
import os
import matplotlib.pyplot as plt
from LCWavelet import *
# %matplotlib inline

wavelet_family='sym5'
level = 9
save_lc = True
save_path = 'waveletsG/'
kep_id = 10583180
kep_id_2 =9021075
wavelet_windows = 15000

# en Process_light_curve_dataset.ipynb, process_dataset se guarda en save_path las wavelets procesadas finales como pickles

# en Train_LCWavelet_Model.ipynb se cargan esos pickles utilizando dataset_path y se entrenan distintos modelos

dataset_path= 'waveletsK/'
df_path = 'light_curves_K_stars_filter.csv'
train_split = .80
