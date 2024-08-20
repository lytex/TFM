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
# %matplotlib -l
import pandas as pd
df_path = 'cumulative_2024.06.01_09.08.01.csv'
df = pd.read_csv(df_path ,skiprows=144)
pd.merge(df, df.groupby('kepid')['kepoi_name'].filter(lambda x: len(x)> 1), on='kepoi_name').query("koi_disposition != 'CANDIDATE'").sort_values(by=['kepid', 'kepoi_name'])

# %%
# %matplotlib agg
# %pdb on

from IPython.display import display
import warnings
import traceback
warnings.filterwarnings(action="ignore", category=DeprecationWarning)
warnings.filterwarnings(action="ignore", category=FutureWarning)
import pandas as pd
import lightkurve as lk
import numpy as np
import pywt
import pickle
import time
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
from LCWavelet import *
from binning import global_view, local_view
from parallelbar import progress_map, progress_imap
from tqdm import tqdm
from functools import partial
import logging

# plt.ion()
# mpl.use("module://mplcairo.qt")
mpl.use("agg")

df_path = 'cumulative_2024.06.01_09.08.01.csv'
df = pd.read_csv(df_path ,skiprows=144)
# Nos quedamos sólamente con los kepid que tienen más de un kepoi_name
# df = pd.merge(df, df.groupby('kepid')['kepoi_name'].filter(lambda x: len(x)> 1), on='kepoi_name').sort_values(by=["kepid", "kepoi_name"])
# failure = pd.read_csv('all_data_2024-07-17/failure_1721424039.csv')
# df = pd.merge(failure[['kepoi_name']], df, on="kepoi_name")


def process_light_curve(row, mission="Kepler", download_dir="data3/",
                        sigma=20, sigma_upper=4,
                        wavelet_window=None,
                        num_bins_global=2001, bin_width_factor_global=1 / 2001,
                        num_bins_local=201, bin_width_factor_local=0.16, num_durations=4,
                        wavelet_family=None, levels_global=None, levels_local=None, cut_border_percent=0.1,
                        plot = False, plot_comparative=False,save=False, path="", plot_folder=None, use_download_cache=False, df_path=None, title="",
                        cache_dict=None, use_wavelet=True) -> LightCurveWaveletGlobalLocalCollection|LightCurveShallueCollection:
    """

    Args:
        row Fila del csv a procesar: 
        mission (): misión de la que descargarse los  
        download_dir (): directorio de descarga de lightkurve  (si ya existe un archivo en esta ruta no vuelve a bajarlo )
        sigma (): 
        sigma_upper (): 
        wavelet_window (): 
        wavelet_family (): 
        levels_global, levels_local (): 
        cut_border_percent (): 
        plot (): 
        plot_comparative (): 
        save (): si es True, guarda los datos procesados en el directorio path
        path (): directorio donde guardar los datos
        plot_folder (): Por defecto None. Si no es None, entonces en vez de enseñar el gráfico lo guarda en {plot_folder}/plot/

    Returns: LightCurveWaveletGlobalLocalCollection

    """

    if cache_dict is None:
        cache_dict = dict()


    FORMAT = '%(asctime)s [%(levelname)s] :%(name)s:%(message)s'
    logger = logging.getLogger(f"process_light_curve[{os.getpid()}]")
    if logger.hasHandlers():
        logger.handlers.clear()
    formatter =  logging.Formatter(FORMAT)
    
    logger.setLevel("INFO")
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if plot_folder is not None:
        os.makedirs(os.path.dirname(f"{plot_folder}/plot/"), exist_ok=True)
    
    # 1. Bajarse los datos con lightkurve o cargarlos si ya están bajados
    kic = f'KIC {row.kepid}'
    file_name = download_dir+kic+".pickle"
    if "lc_search" in cache_dict.keys():
        lc_search = cache_dict['lc_search']
    else:
        if use_download_cache:
            logger.info(f"Cargando datos de caché para {mission} {row.kepid}...");
            with open(file_name, "rb") as f:
                lc_search = pickle.load(f)
        else:
            logger.info(f"Bajando datos para {mission} {row.kepid}...");
            lc_search = lk.search_lightcurve(kic, mission=mission)
            with open(file_name, "wb") as f:
                pickle.dump(lc_search, f)

    if "light_curve_collection" in cache_dict.keys():
        light_curve_collection = cache_dict["light_curve_collection"]
    else:
        logger.info(f"Abriendo o descargando curvas de luz en la ruta {download_dir}...");
        light_curve_collection = lc_search.download_all(download_dir=download_dir)

    if "lc_collection" in cache_dict.keys():
        lc_collection = cache_dict["lc_collection"]
    else:
        # 2. Generar la colección, juntarlos todos y quitarles Nan
        logger.info("Juntando colección de curvas...")
        lc_collection = lk.LightCurveCollection([lc for lc in light_curve_collection])
        lc_collection = lc_collection.stitch()

    logger.info(f"Eliminando outliers sigma={sigma}, sigma_upper={sigma_upper}...")
    lc_nonans = lc_collection.remove_nans()
    from astropy.stats.sigma_clipping import sigma_clip
    res, min_, max_ = sigma_clip(data=lc_nonans.flux, sigma=sigma, sigma_upper=sigma_upper, return_bounds=True)
    outlier_mask = res.mask
    lc_ro_1 = lc_nonans.copy()[~outlier_mask]
    lc_ro = lc_ro_1.flatten()
    if plot_folder is not None:
        logger.info("Graficando serie completa")
        mask_lc_ro = lc_ro.create_transit_mask(period=row.koi_period, duration=row.koi_duration/24.0, transit_time=row.koi_time0bk)
        fig, ax = plt.subplots(figsize=(16, 12))
        df_lc_ro = lc_ro.to_pandas().reset_index()[['time', 'flux']]
        ax.scatter(df_lc_ro.time[~mask_lc_ro], df_lc_ro.flux[~mask_lc_ro], c='b', marker='.')
        ax.scatter(df_lc_ro.time[mask_lc_ro], df_lc_ro.flux[mask_lc_ro], c='r', marker='*')
        ax.set_title(f'KIC {row.kepid} {row.kepoi_name}: {row.koi_disposition}'+title)
        plt.close('all')
    else:
        plt.show()
        plt.pause(.001)
    # lc_ro, mask = lc_collection.remove_outliers(sigma=sigma, sigma_upper=sigma_upper, return_mask=True)

    # 3. Plegar en fase
    logger.info("Plegando en fase...")
    lc_fold = lc_ro.fold(period = row.koi_period,epoch_time = row.koi_time0bk)
    if plot:
        logger.info('graficando series plegadas en fase...')
        # Ploteamos con los outliers
        fig, ax = plt.subplots(figsize=(12, 16))
        ax = lc_nonans.fold(period = row.koi_period,epoch_time = row.koi_time0bk).plot(ax=ax)
        ax.set_title(f'KIC {row.kepid} {row.kepoi_name}: {row.koi_disposition}'+title)
        ax.axhline(min_, color='r')
        ax.axhline(max_, color='r')
        if plot_folder is not None:
            plt.savefig(f"{plot_folder}/plot/kic_{row.kepid}_{row.kepoi_name}_00_plegado.png")
            plt.close('all')
        else:
            plt.show()
            plt.pause(.001)

    if save:
        LightCurveShallueCollection(row.kepoi_name, row,
                                    global_view(lc_fold.time.to_value("jd"), lc_fold.flux.to_value(), row.koi_period, normalize=True),
                                    local_view(lc_fold.time.to_value("jd"), lc_fold.flux.to_value(), row.koi_period, row.koi_duration/24.0, normalize=True)
                                   ).save(path)
    if not use_wavelet:
        return LightCurveShallueCollection(row.kepoi_name, row,
                                    global_view(lc_fold.time.to_value("jd"), lc_fold.flux.to_value(), row.koi_period, normalize=True),
                                    local_view(lc_fold.time.to_value("jd"), lc_fold.flux.to_value(), row.koi_period, row.koi_duration/24.0, normalize=True)
                                   )
        

     # 4. Dividir en pares e impares
    logger.info("Diviendo en curvas pares e impares...")
    lc_odd = lc_fold[lc_fold.odd_mask]
    lc_even = lc_fold[lc_fold.even_mask]
    if plot:
        logger.info("Graficando curvas pares e impares...")
        fig, (ax1, ax2) = plt.subplots(figsize=(12, 16), nrows=2, ncols=1)
        ax1 = lc_odd.plot(ax=ax1)
        ax2 = lc_even.plot(ax=ax2)
        fig.suptitle(f'KIC {row.kepid} {row.kepoi_name}: {row.koi_disposition}'+title)
        if plot_folder is not None:
            plt.savefig(f"{plot_folder}/plot/kic_{row.kepid}_{row.kepoi_name}_01_par_impar.png")
            plt.close('all')
        else:
            plt.show()
            plt.pause(.001)

    # 5. Aplicar bineado en local y global y normalizar
    logger.info("Bineando en vista global y vista local...")
    
    lc_odd.sort("time")
    lc_even.sort("time")
    
    lc_odd_global_flux =  global_view(lc_odd.time.to_value("jd"), lc_odd.flux.to_value(), row.koi_period, normalize=False, num_bins=num_bins_global, bin_width_factor=bin_width_factor_global)
    lc_even_global_flux =  global_view(lc_even.time.to_value("jd"), lc_even.flux.to_value(), row.koi_period, normalize=False, num_bins=num_bins_global, bin_width_factor=bin_width_factor_global)
    lc_odd_global = lk.lightcurve.FoldedLightCurve(time=np.arange(len(lc_odd_global_flux)), flux=lc_odd_global_flux)
    lc_even_global = lk.lightcurve.FoldedLightCurve(time=np.arange(len(lc_even_global_flux)), flux=lc_even_global_flux)

    # Revisar los valores de los periodos, printearlos
    # Mirar paso a paso y ver si tiene sentido aplicarlo
    # Mirar pliegue antes de binear (plotear) en par e impar
    # Comrpobar unidades (está en horas la duración)
    # Mirar sistemas con múltiples planetas . Se ven varias curvas? No quitarlo de train porque se puede encontrar con esos casos, pero saber que sucede
    # Normalizar sólamente las wavelets
    lc_odd_local_flux =  local_view(lc_odd.time.to_value("jd"), lc_odd.flux.to_value(), row.koi_period, row.koi_duration/24.0, normalize=False,
                                    num_bins=num_bins_local, bin_width_factor=bin_width_factor_local, num_durations=num_durations)
    lc_even_local_flux =  local_view(lc_even.time.to_value("jd"), lc_even.flux.to_value(), row.koi_period, row.koi_duration/24.0, normalize=False,
                                     num_bins=num_bins_local, bin_width_factor=bin_width_factor_local, num_durations=num_durations)
    lc_odd_local = lk.lightcurve.FoldedLightCurve(time=np.arange(len(lc_odd_local_flux)), flux=lc_odd_local_flux,)
    lc_even_local = lk.lightcurve.FoldedLightCurve(time=np.arange(len(lc_even_local_flux)), flux=lc_even_local_flux)
    lc_gl_collection = LightCurveGlobalLocalCollection(row.kepoi_name, row, lc_odd_global, lc_even_global, lc_even_local, lc_odd_local)

    if plot:
        logger.info('graficando series bineadas...')
        # fig, ((ax1, ax2), (ax3, ax4)) = lc_gl_collection.plot()
        fig, (ax1, ax2) = lc_gl_collection.plot_comparative()
        fig.suptitle(f'KIC {row.kepid} {row.kepoi_name}: {row.koi_disposition}'+title)
        if plot_folder is not None:
            plt.savefig(f"{plot_folder}/plot/kic_{row.kepid}_{row.kepoi_name}_02_bineado.png")
            plt.close('all')
        else:
            plt.show()
            plt.pause(.001)

    # para quitar oscilaciones en los bordes (quizás mejor no guardar los datos con esto quitado)
    if wavelet_window is not None:
        logger.info("Quitando oscilaciones en los bordes en la ventana seleccionada...")
        lc_odd_global = cut_wavelet(lc_odd_global, wavelet_window)
        lc_even_global = cut_wavelet(lc_even_global, wavelet_window)
        lc_odd_local = cut_wavelet(lc_odd_local, wavelet_window)
        lc_even_local = cut_wavelet(lc_even_local, wavelet_window)
    
    logger.info("Calculando wavelets...")
    lc_w_even_global = apply_wavelet(lc_even_global, wavelet_family, levels_global, cut_border_percent=cut_border_percent, normalize=True)
    lc_w_odd_global = apply_wavelet(lc_odd_global, wavelet_family, levels_global, cut_border_percent=cut_border_percent, normalize=True)
    lc_w_even_local = apply_wavelet(lc_even_local, wavelet_family, levels_local, cut_border_percent=cut_border_percent, normalize=True)
    lc_w_odd_local = apply_wavelet(lc_odd_local, wavelet_family, levels_local, cut_border_percent=cut_border_percent, normalize=True)

    headers = {
        "id": row.kepid,
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
        "levels_global": levels_global,
        "levels_local": levels_local,
        "window":wavelet_window,
        "border_cut":cut_border_percent,
        "df_path": df_path,
        "Kepler_name":row.kepoi_name
    }
    lc_wavelet_collection = LightCurveWaveletGlobalLocalCollection(row.kepoi_name, headers,
                                                                   lc_w_even_global,
                                                                   lc_w_odd_global,
                                                                   lc_w_even_local,
                                                                   lc_w_odd_local,
                                                                   levels_global, levels_local)

    if(plot):
        logger.info('graficando wavelets obtenidas...')
        if plot_folder is not None:
            # gi, gp, li, lp: global impar, global par, local impar, local par
            figure_paths = (f"{plot_folder}/plot/kic_{row.kepid}_{row.kepoi_name}_03_wavelet_gi.png",
             f"{plot_folder}/plot/kic_{row.kepid}_{row.kepoi_name}_03_wavelet_gp.png",
             f"{plot_folder}/plot/kic_{row.kepid}_{row.kepoi_name}_03_wavelet_li.png",
             f"{plot_folder}/plot/kic_{row.kepid}_{row.kepoi_name}_03_wavelet_lp.png",
             )
            figure_paths = (f"{plot_folder}/plot/kic_{row.kepid}_{row.kepoi_name}_03_wavelet_global.png",
             f"{plot_folder}/plot/kic_{row.kepid}_{row.kepoi_name}_03_wavelet_local.png",
             )
            lc_wavelet_collection.plot_comparative(figure_paths=figure_paths, title=f'KIC {row.kepid} {row.kepoi_name}: {row.koi_disposition}' + title)
        else:
            lc_wavelet_collection.plot()
            plt.show()
            plt.pause(.001)
    if(plot_comparative):
        logger.info('graficando wavelets obtenidas...')
        lc_wavelet_collection.plot_comparative()
    if(save):
        logger.info(f'guardando wavelets obtenidas en {path}...')
        lc_wavelet_collection.save(path)
    return lc_wavelet_collection

if __name__ == "__main__":
    path = "all_data_2024-08-04/"
    download_dir="data4/"
    process_func =  partial(process_light_curve, levels_global=6, levels_local=3, wavelet_family="sym5", sigma=20, sigma_upper=5,
                            plot=True, plot_comparative=False, save=True, path=path, download_dir=download_dir, df_path=df_path, plot_folder=path, use_download_cache=False)
    
    def process_func_continue(row, retry=True):
        try:
            return process_func(row)
        except Exception as e:
            print(f"Exception on {row.kepid}")
            traceback.print_exc()
            path = str(e).replace('Not recognized as a supported data product:\n', '') \
            .replace('\nThis file may be corrupt due to an interrupted download. Please remove it from your disk and try again.', '')
            if path != str(e) and retry:
                os.remove(path)
                process_func_continue(row, retry=False)
            return e
    
    
    results = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        results.append(process_func_continue(row))

    # n_proc = 20; results = progress_imap(process_func, [row for _, row in df.iterrows()], n_cpu=n_proc, total=len(df), error_behavior='coerce', chunk_size=len(df)//n_proc//10)
    
    failures_idx = [n for n, x in enumerate(results) if type(x) != LightCurveWaveletGlobalLocalCollection]
    failures = [x for x in results if type(x) != LightCurveWaveletGlobalLocalCollection]
    
    now = int(time.time())
    df_fail = df.loc[failures_idx].copy()
    df_fail['exception'] = failures
    df_fail.to_csv(path+f"/failure_{now}.csv", index=False)
    
    # process_light_curve

# %%
df[['rowid', 'koi_disposition']].groupby('koi_disposition')['rowid'].count(), df[['rowid', 'koi_pdisposition']].groupby('koi_pdisposition')['rowid'].count()
# df.keys()
