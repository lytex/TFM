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
# %matplotlib agg
import warnings
import traceback
warnings.filterwarnings(action="ignore", category=DeprecationWarning)
warnings.filterwarnings(action="ignore", category=FutureWarning)
import pandas as pd
import lightkurve as lk
import numpy as np
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


def process_light_curve(row, mission="Kepler", download_dir="data3/",
                        sigma=20, sigma_upper=4,
                        wavelet_window=None,
                        num_bins_global=2001, bin_width_factor_global=1 / 2001,
                        num_bins_local=201, bin_width_factor_local=0.16,
                        wavelet_family=None, levels_global=None, levels_local=None, cut_border_percent=0.1,
                        plot = False, plot_comparative=False,save=False, path="", plot_folder=None, use_download_cache=False, df_path=None, title="",
                        cache_dict=None) -> LightCurveWaveletGlobalLocalCollection:
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
    
    logger.info(f"Cargando datos de caché para {mission} {row.kepid}...");
    with open(file_name, "rb") as f:
        lc_search = pickle.load(f)

    logger.info(f"Abriendo o descargando curvas de luz en la ruta {download_dir}...");
    light_curve_collection = lc_search.download_all(download_dir=download_dir)

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

    # 3. Plegar en fase
    logger.info("Plegando en fase...")
    lc_fold = lc_ro.fold(period = row.koi_period,epoch_time = row.koi_time0bk)
    lc_fold.meta.update(row.to_dict())
    logger.info("Guardando...")
    with open(path + "/" + lc_fold.meta["LABEL"] + lc_fold.meta["kepoi_name"], "wb") as f:
        pickle.dump(lc_fold, f)
    # lc_fold.save(path=path)

path = "raw_freq/"
download_dir="data3/"
process_func =  partial(process_light_curve, levels_global=6, levels_local=3, wavelet_family="sym6", sigma=20, sigma_upper=5,
                        plot=True, plot_comparative=False, save=False, path=path, download_dir=download_dir, df_path=df_path, plot_folder=None, use_download_cache=False)

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


# results = []
# for _, row in tqdm(df.iterrows(), total=len(df)):
#     results.append(process_func_continue(row))

# n_proc = 20; results = progress_imap(process_func, [row for _, row in df.iterrows()], n_cpu=n_proc, total=len(df), error_behavior='coerce', chunk_size=len(df)//n_proc//10)

# failures_idx = [n for n, x in enumerate(results) if type(x) != LightCurveWaveletGlobalLocalCollection]
# failures = [x for x in results if type(x) != LightCurveWaveletGlobalLocalCollection]

# now = int(time.time())
# df_fail = df.loc[failures_idx].copy()
# df_fail['exception'] = failures
# df_fail.to_csv(path+f"/failure_{now}.csv", index=False)

# %%
import os
from new_train import load_files
from parallelbar import progress_map
from functools import partial
from tqdm import tqdm
import traceback

files = [file for file in os.listdir(path)]

def func(file):
    # FORMAT = '%(asctime)s [%(levelname)s] :%(name)s:%(message)s'
    # logger = logging.getLogger(f"process_light_curve[{os.getpid()}]")
    # if logger.hasHandlers():
    #     logger.handlers.clear()
    # formatter =  logging.Formatter(FORMAT)
    
    # logger.setLevel("INFO")
    # console_handler = logging.StreamHandler()
    # console_handler.setFormatter(formatter)
    # logger.addHandler(console_handler)

    # logger.info(f"Cargando archivo {path+file}...")
    with open(path+file, "rb") as f:
        contents = pickle.load(f)
    # logger.info(f"{path+file} cargado!")
    return contents

# lightcurves = progress_map(func, files, n_cpu=2, total=len(files), executor='processes', error_behavior='raise')

lightcurves = []
for file in tqdm(files, total=len(files)):
    try:
        lightcurves.append(func(file))
    except:
        traceback.print_exc()
        pass
# Guardarlo como .fits que parece que va más rápido?

# %%
import matplotlib.pyplot as plt

shape = [lc.time.value.shape[0] for lc in lightcurves]
def aggregate_bin(value, lightcurvres, nbins=10000):
    nbins =  (max(value) - min(value))//nbins
    
    hist = [np.histogram(v, range=(min(value), max(value)), bins=nbins)[0] for v in value]
    _, bins = np.histogram(value[0], range=(min(value), max(value)), bins=nbins)
    bins = bins[:-1]
    hist_1d = np.vstack(hist).sum(axis=0)
    print(np.vstack(hist).shape)
    print(hist_1d.shape)
    print(bins.shape)
    return hist_1d, bins


def aggregate_time(value, lightcurves, nbins=100):
    max_ = max([max(x) for x in value])
    min_ = min([min (x) for x in value])
    nbins =  int((max_ - min_)*nbins)
    hist = [np.histogram(v, range=(min_, max_), bins=nbins)[0] for v in value]
    _, bins = np.histogram(value[0], range=(min_, max_), bins=nbins)
    bins = bins[:-1]
    hist_1d = np.vstack(hist).sum(axis=0)
    print(np.vstack(hist).shape)
    print(hist_1d.shape)
    print(bins.shape)
    return hist_1d, bins


# %%
# %pdb off
# np.array(hist)
# %matplotlib qt
plt.ion()
# time = [np.diff(lc.time.value) for lc in lightcurves]
# hist_1d, bins = aggregate_time(time, lightcurves)
hist_1d, bins = aggregate_bin(shape, lightcurves)
# plt.plot(bins, hist_1d)
# plt.bar(bins, np.log(hist_1d), width=np.min(np.diff(bins)))
plt.bar(bins, hist_1d, width=np.max(np.diff(bins)))
np.median(shape)
# for vv in (30, -30, 60, -60, 90, -90):
#     plt.axvline(vv, 0, 12, c='k')
# hist_1d.shape, bins.shape
# hist[1].shape
# bins, hist_1d

# %%
# np.array(hist)
# %matplotlib qt
plt.ion()
time = [np.diff(lc.time.value)/lc.meta["koi_period"] for lc in lightcurves]
time = [np.diff(lc.time.value)/lc.meta["koi_duration"] for lc in lightcurves]
nbins = 10000
hist_1d, bins = aggregate_time(time, lightcurves, nbins=nbins)
# hist_1d_, bins_ = aggregate_bin(shape, lightcurves, nbins=nbins)
# plt.plot(bins, hist_1d)
plt.bar(bins, np.log(hist_1d+1e-7), width=np.min(np.diff(bins)))
# plt.bar(bins, hist_1d, width=np.min(np.diff(bins)))
# np.median(time, axis=0)
# for vv in (30, -30, 60, -60, 90, -90):
#     plt.axvline(vv, 0, 12, c='k')
# hist_1d.shape, bins.shape
# hist[1].shape
# bins, hist_1d

# %%
ttime = np.hstack(time)
# ttime = ttime[ttime]
plt.hist(ttime, bins=1000)
# plt.bar(bins, np.log(hist_1d+1e-7), width=np.min(np.diff(bins)))

# %%

# %%
bins[-1]

# %%
import lightkurve
# lightkurve.__version__

ls = [lc.to_periodogram(method='lombscargle', frequency=np.linspace(0.0001, 1000000, 40000)) for lc in lightcurves]

# %%

# %%
# %matplotlib qt
ls[3].frequency.value.shape
freq, power = np.vstack([l.frequency.value for l in ls]), np.vstack([l.power.value for l in ls])
freq = freq[0]
power = np.max(power, axis=0)
plt.plot(1/freq*24*3600, power)

# %%
max_T, min_T = max([max(np.diff(x.time.value)) for x in lightcurves]), min([min(np.diff(x.time.value)) for x in lightcurves])
max_TT, min_TT = max([np.ptp(x.time.value) for x in lightcurves]), min([np.ptp(x.time.value) for x in lightcurves])
max_T, min_T, max_TT, min_TT, 6/3600/24, df.koi_period.describe(), df.koi_period.min()*24
# Los fotogramas duran 6s, con lo que 6/3600/24 ~ 7e-5 es un valor más sensato para resolución entre puntos
# El periodo más corto detectado es de 0.24 días, 5.8 horas

# %%
# %matplotlib qt
df.query("koi_period < 120_000").koi_period.hist(bins=2500)
