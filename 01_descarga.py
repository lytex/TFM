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
import pandas as pd
import lightkurve as lk
import numpy as np
import pywt
import pickle
import os
import matplotlib.pyplot as plt
from LCWavelet import *
from binning import global_view, local_view
from parallelbar import progress_map
from functools import partial
import logging


df_path = 'cumulative_2022.09.30_09.06.43.csv'
df = pd.read_csv(df_path ,skiprows=144)


def process_light_curve(row, mission="Kepler", download_dir="data3/",
                        sigma=20, sigma_upper=4,
                        wavelet_window=None,
                        wavelet_family=None, levels=None, cut_border_percent=0.1,
                        plot = False, plot_comparative=False,save=False, path=""):

    
    FORMAT = '%(asctime)s [%(levelname)s] [%(process)d] :%(name)s:%(message)s'
    logger = logging.getLogger(__name__)
    formatter =  logging.Formatter(FORMAT)
    
    logger.setLevel("INFO")
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 1. Bajarse los datos con lightkurve o cargarlos si ya están bajados
    logger.info(f"Bajando datos para {row.kepid}...");
    kic = f'KIC {row.kepid}'
    lc_search = lk.search_lightcurve(kic, mission=mission)
    light_curve_collection = lc_search.download_all(download_dir=download_dir)

    # 2. Generar la colección, juntarlos todos y quitarles Nan
    logger.info("Juntando colleción de curvas...")
    lc_collection = lk.LightCurveCollection([lc for lc in light_curve_collection])
    lc_ro = lc_collection.stitch()
    lc_nonans = lc_ro.remove_nans()

    # 3. Plegar en fase y dividir en pares e impares
    logger.info("Plegando en fase pares/impares...")
    lc_fold = lc_nonans.fold(period = row.koi_period,epoch_time = row.koi_time0bk)
    lc_odd = lc_fold[lc_fold.odd_mask]
    lc_even = lc_fold[lc_fold.even_mask]

    # 4. Aplicar bineado en local y global y normalizar
    logger.info("Bineando en vista global y vista local...")
    lc_odd_global =  global_view(np.arange(lc_odd), lc_odd, row.koi_period, normalize=True)
    lc_even_global =  global_view(np.arange(lc_even), lc_even, row.koi_period, normalize=True)
    
    lc_odd_local =  local_view(np.arange(lc_odd), lc_odd, row.koi_period)
    lc_even_local =  local_view(np.arange(lc_even), lc_even, row.koi_period)

    if wavelet_window is not None:
        logger.info('Aplicando ventana ...')
        lc_impar_global = cut_wavelet(lc_odd_global, wavelet_window)
        lc_par_global = cut_wavelet(lc_even_global, wavelet_window)
        lc_impar_local = cut_wavelet(lc_odd_local, wavelet_window)
        lc_par_local = cut_wavelet(lc_even_local, wavelet_window)
    else:
        lc_impar_local = lc_odd_local
        lc_par_local = lc_even_local

    
    # para quitar oscilaciones en los bordes (quizás mejor no guardar los datos con esto quitado)
    logger.info("Quitando oscilaciones en los bordes...")
    lc_w_par_global = apply_wavelet(lc_par_global, wavelet_family, levels, cut_border_percent=cut_border_percent)
    lc_w_impar_global = apply_wavelet(lc_impar_global, wavelet_family, levels, cut_border_percent=cut_border_percent)
    lc_w_par_local = apply_wavelet(lc_par_local, wavelet_family, levels, cut_border_percent=cut_border_percent)
    lc_w_impar_local = apply_wavelet(lc_impar_local, wavelet_family, levels, cut_border_percent=cut_border_percent)

    headers = {
        "period": row.koi_period,
        "koi_period_err1": row.koi_period_err1,
        "koi_period_err2": row.koi_period_err2,
        "depth": row.koi_depth,
        "depth_err1": row.koi_depth_err1,
        "depth_err2": row.koi_depth_err2,
        "duration": row.koi_duration,
        "duration_err1": row.koi_duration_err1,
        "duration_err2": row.koi_duration_err2,
        "steff": row.koi_steff,
        "steff_err1": row.koi_steff_err1,
        "steff_err2": row.koi_steff_err2,
        "impact": row.koi_impact,
        "impact_err1": row.koi_impact_err1,
        "impact_err2": row.koi_impact_err2,
        "class": row.koi_disposition,
        "wavelet_family":wavelet_family,
        "levels":levels,
        "window":wavelet_window,
        "border_cut":cut_border_percent,
        "Kepler_name":row.kepoi_name
    }
    lc_wavelet_collection = LightCurveWaveletGlobalLocalCollection(row.kepid, headers, lc_w_par_global, lc_w_impar_global, lc_w_par_gobal)

    if(plot):
        logger.info('graficando wavelets obtenidas...')
        lc_w_par_global.plot()
        lc_w_impar_global.plot()
        lc_w_par_local.plot()
        lc_w_impar_local.plot()
    if(plot_comparative):
        logger.info('graficando wavelets obtenidas...')
        lc_wavelet_collection.plot_comparative()
    if(save):
        logger.info(f'guardando wavelets obtenidas en {path}...')
        lc_wavelet_collection.save(path)
    return lc_wavelet_collection


process_func = partial(process_light_curve, levels=[1, 2, 3, 4], wavelet_family="sym5", plot=False, plot_comparative=False, save=True, path="all_data_2024-06-01/")

progress_map(process_func, [row for _, row in df.iterrows()], n_cpu=16, total=len(df))
# process_light_curve
