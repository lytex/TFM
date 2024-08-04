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
# %matplotlib qt

import pandas as pd
import logging
import lightkurve as lk
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

df_path = 'cumulative_2024.06.01_09.08.01.csv'
df = pd.read_csv(df_path ,skiprows=144).query("koi_disposition != 'CANDIDATE'")
df_groupby =  df.groupby('kepid').filter(lambda x: len(x)> 1)['kepoi_name']
df = pd.merge(df.copy(), df_groupby.copy(), on='kepoi_name').sort_values(by=['kepid', 'kepoi_name'])
plt.ion()


def process_light_curve(kepid):
    download_dir="data3/"
    use_download_cache=True
    mission = "Kepler"
    
    FORMAT = '%(asctime)s [%(levelname)s] :%(name)s:%(message)s'
    logger = logging.getLogger(f"process_light_curve[{os.getpid()}]")
    if logger.hasHandlers():
        logger.handlers.clear()
    formatter =  logging.Formatter(FORMAT)

    
    df_path = 'cumulative_2024.06.01_09.08.01.csv'
    df = pd.read_csv(df_path ,skiprows=144).query("koi_disposition != 'CANDIDATE'")
    df_groupby =  df.groupby('kepid').filter(lambda x: len(x)> 1)['kepoi_name']
    df = pd.merge(df.copy(), df_groupby.copy(), on='kepoi_name').sort_values(by=['kepid', 'kepoi_name'])
    plt.ion()
    
    logger.setLevel("INFO")
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    kic = f'KIC {kepid}'
    file_name = download_dir+kic+".pickle"
    if use_download_cache:
        logger.info(f"Cargando datos de caché para {mission} {kepid}...")
        with open(file_name, "rb") as f:
            lc_search = pickle.load(f)
    else:
        logger.info(f"Bajando datos para {mission} {row.kepid}...")
        lc_search = lk.search_lightcurve(kic, mission=mission)
        with open(file_name, "wb") as f:
            pickle.dump(lc_search, f)

    logger.info(f"Abriendo o descargando curvas de luz en la ruta {download_dir}...")
    lc_collection = lc_search.download_all(download_dir=download_dir)
    logger.info("Stitcheando, aplanando, y quitando outliers...")
    lc = lc_collection.stitch().flatten(window_length=901).remove_outliers().remove_nans()

    planets = df.query(f"kepid == {kepid}")
    max_period = planets.koi_period.max()
    period = np.linspace(1, 10*max_period, 10000)
    logger.info("Calculando BLS...")
    bls = lc.to_periodogram(method='bls', period=period, frequency_factor=500)
    logger.info("Terminado BLS")
    ax = bls.plot()
    plt.show()
    plt.pause(.1)
    logger.info("Ploteado BLS")
    ax = lc.scatter()
    masked_lc = lc
    prop_cycle = plt.rcParams['axes.prop_cycle']
    logger.info(str(planets.koi_period))
    for i, planet in tqdm(planets.reset_index().iterrows()):
        planet_bls = bls.get_transit_model(period=planet.koi_period, 
                                           transit_time=planet.koi_time0, 
                                           duration=planet.koi_duration/24.0)
        planet_bls.plot(ax=ax, c=prop_cycle.by_key()['color'][i], label=f'{planet.kepoi_name}', linewidth=2)
        plt.show(block=False)
        plt.pause(.1)
        logger.info(f"Terminado plot de planeta {i}")
    plt.show()


    
# process_light_curve(np.sort(df.kepid.unique())[1])
process_light_curve(2306756)
plt.pause(3600)
# for kepid in tqdm(np.sort(df.kepid.unique())[:]):
#     print(f"process_light_curve({kepid})")
#     process_light_curve(kepid)
