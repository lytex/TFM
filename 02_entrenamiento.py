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

path = "all_data_2024-06-01/"
files = [file for file in os.listdir(path) if file.endswith(".pickle")]
lightcurves = []
for file in tqdm(files):
    lightcurves.append(LightCurveWaveletGlobalLocalCollection.from_pickle(path+file))
