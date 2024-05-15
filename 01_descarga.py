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
import pandas as pd
import lightkurve as lk
import numpy as np
import pywt
import pickle
import os
import matplotlib.pyplot as plt
from LCWavelet import *

df_path = 'cumulative_2022.09.30_09.06.43.csv'
df = pd.read_csv(df_path ,skiprows=144)


columns = ['kepid','kepoi_name','koi_disposition','koi_period','koi_time0bk']

def process_light_curve(row, mission="Kepler", download_dir="data/", sigma=20, sigma_upper=4, wavelet_window=None, wavelet_family=None, levels=None, cut_border_percent=0.1,
                        plot = False, plot_comparative=False,save=False, path=""):

    kic = f'KIC {row.kepid}'
    lc_search = lk.search_lightcurve(kic, mission=mission)
    light_curve_collection = lc_search.download_all(download_dir=download_dir)

    lc_collection = lk.LightCurveCollection([lc.remove_outliers(sigma=sigma, sigma_upper=sigma_upper) for lc in light_curve_collection])

    lc_ro = lc_collection.stitch()

    lc_nonans = lc_ro.remove_nans()
    lc_fold = lc_nonans.fold(period = row.koi_period,epoch_time = row.koi_time0bk)
    lc_odd = lc_fold[lc_fold.odd_mask]
    lc_even = lc_fold[lc_fold.even_mask]

    if wavelet_window is not None:
        print('Aplicando ventana ...')
        lc_impar = cut_wavelet(lc_odd, wavelet_window)
        lc_par = cut_wavelet(lc_even, wavelet_window)
    else:
        lc_impar = lc_odd
        lc_par = lc_even

    lc_w_par = apply_wavelet(lc_par, wavelet_family, levels, cut_border_percent=cut_border_percent)
    lc_w_impar = apply_wavelet(lc_impar, wavelet_family, levels, cut_border_percent=cut_border_percent)

    headers = {
        "period": row.koi_period,
        "epoch": row.koi_time0bk,
        "class": row.koi_disposition,
        "wavelet_family":wavelet_family,
        "levels":levels,
        "window":wavelet_window,
        "border_cut":cut_border_percent,
        "Kepler_name":row.kepoi_name
    }
    lc_wavelet_collection = LightCurveWaveletCollection(row.kepid, headers, lc_w_par, lc_w_impar)

    if(plot):
        print('graficando wavelets obtenidas...')
        lc_w_par.plot()
        lc_w_inpar.plot()
    if(plot_comparative):
        print('graficando wavelets obtenidas...')
        lc_wavelet_collection.plot_comparative()
    if(save):
        print('guardando wavelets obtenidas...')
        lc_wavelet_collection.save(path)
    return lc_wavelet_collection

process_light_curve(df.iloc[1], levels=[1], wavelet_family="sym5", plot=False, plot_comparative=False, save=False)
