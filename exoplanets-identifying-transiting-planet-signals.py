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

# %% [markdown] id="bYYGJvqP-Csl"
# # Identifying transiting exoplanet signals in a light curve

# %% [markdown] id="gOFEJAKe-vjA"
# ## Learning Goals
#
# By the end of this tutorial, you will:
#
#  - Understand the "Box Least Squares" (BLS) method for identifying transit signals.
#  - Know how to use the Lightkurve [BoxLeastSquaresPeriodogram](https://docs.lightkurve.org/reference/api/lightkurve.periodogram.BoxLeastSquaresPeriodogram.from_lightcurve.html?highlight=boxleastsquaresperiodogram) to identify a transiting planet.
#  - Be able to estimate the period, epoch, and duration of the transit.
#  - Be able to plot the phase-folded transit light curve.
#  - Be familiar with the interactive Box Least Squares periodogram in Lightkurve.

# %% [markdown] id="7NGNkiQB-sKQ"
# ## Introduction
#
# The *Kepler* and *TESS* missions are optimized for finding new transiting exoplanets. [Lightkurve](http://docs.lightkurve.org/) provides a suite of tools that help make the process of identifying and characterizing planets convenient and accessible.
#
# In this tutorial, we will show you how to conduct your own search for transiting exoplanets in *Kepler* and *TESS* light curves. [Lightkurve](http://docs.lightkurve.org/) uses the [Astropy](https://www.astropy.org/) implementation of the Box Least Squares (BLS) method to identify transit signals. This tutorial demonstrates the basics of how to optimally use Lightkurve's BLS tools.

# %% [markdown] id="EAzRgQPpAcBR"
# ## Imports
# This tutorial requires the [**Lightkurve**](http://docs.lightkurve.org/) package, which uses [**Matplotlib**](https://matplotlib.org/) for plotting.

# %% id="azlcSd2lAjhy"
import lightkurve as lk
# %matplotlib inline

# %% [markdown] id="IChTo8Ir3-Ju"
# ---

# %% [markdown] id="r6p8DOLURorT"
# ## 1. Downloading a Light Curve and Removing Long-Term Trends

# %% [markdown] id="QQLSlKKPOxEW"
# As an example, we will download all available [*Kepler*](https://archive.stsci.edu/kepler) observations for a known multi-planet system, [Kepler-69](https://iopscience.iop.org/article/10.1088/0004-637X/768/2/101).

# %% colab={"base_uri": "https://localhost:8080/", "height": 387} executionInfo={"elapsed": 72588, "status": "ok", "timestamp": 1601412555227, "user": {"displayName": "Susan Mullally", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GhoAkTlu5JCXqC32438ISJU86DPSZdvoBOLwtOQMfU=s64", "userId": "01921813910966567332"}, "user_tz": 240} id="6mO0bXd6Rw5X" outputId="0bbf9a53-3897-483a-d2a0-7737981a6494"
# Search for Kepler observations of Kepler-69
import pandas as pd
import matplotlib.pyplot as plt
df_path = 'cumulative_2024.06.01_09.08.01.csv'
df = pd.read_csv(df_path ,skiprows=144)
# df_groupby =  df.groupby('kepid').filter(lambda x: len(x) == 2)['kepoi_name']
# df = pd.merge(df.copy(), df_groupby.copy(), on='kepoi_name').sort_values(by=['kepid', 'kepoi_name'])
# random_state = 7
# random_state = 8
random_state = 9
# random_state = 21
# random_state = 11
row = df.query("koi_disposition == 'CONFIRMED' and koi_score > 0.99").sample(n=1, random_state=random_state).iloc[0]
# row = df.query("kepid == 10227020").sample(n=1, random_state=random_state).iloc[0]
# row = df.query("kepid == 6185476").iloc[0]
row = df.query("kepid == 4852528").iloc[0]
planets = df.query(f"kepid == {row.kepid}").sort_values(by='koi_period', ascending=False)
from IPython.display import display
display(planets[["kepid", "kepoi_name", "kepler_name", "koi_disposition", "koi_period", "koi_time0"]])
row = planets.iloc[planets.koi_period.argmax()]
display(row[["kepoi_name", "kepler_name", "koi_disposition", "koi_period", "koi_time0"]])
KIC = "KIC " + str(row["kepid"])
search_result = lk.search_lightcurve(KIC, author='Kepler', cadence='long')
# search_result = lk.search_lightcurve("Kepler-30", author='Kepler', cadence='long')
# Download all available Kepler light curves
lc_collection = search_result.download_all(download_dir="data3/")

# fig, ax = plt.subplots(figsize=(32,12))
lc_collection.plot()
plt.legend([KIC])
plt.savefig(f"plot/{KIC}_00.png")

# %% [markdown] id="v3CQzkKfPO8e"
# Each observation has a different offset, so in order to successfully search this light curve for transits, we first need to normalize and flatten the full observation. This can be performed on a stitched light curve. For more information about combining multiple observations of the same target, please see the companion tutorial on combining multiple quarters of *Kepler* data with Lightkurve.

# %% colab={"base_uri": "https://localhost:8080/", "height": 442} executionInfo={"elapsed": 72582, "status": "ok", "timestamp": 1601412555229, "user": {"displayName": "Susan Mullally", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GhoAkTlu5JCXqC32438ISJU86DPSZdvoBOLwtOQMfU=s64", "userId": "01921813910966567332"}, "user_tz": 240} id="V0XVebt-AM-4" outputId="29fd2769-0dcc-4700-9314-1fae68ccdd70"
search_result

# %% colab={"base_uri": "https://localhost:8080/", "height": 405} executionInfo={"elapsed": 74448, "status": "ok", "timestamp": 1601412557102, "user": {"displayName": "Susan Mullally", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GhoAkTlu5JCXqC32438ISJU86DPSZdvoBOLwtOQMfU=s64", "userId": "01921813910966567332"}, "user_tz": 240} id="iXDKVb9aBBNx" outputId="d93fb7aa-bbd1-48a3-f3a9-e777770460e0"
# Flatten the light curve 
lc = lc_collection.stitch(corrector_func=lambda x: x)
lc.plot();
plt.savefig(f"plot/{KIC}_10.png")
lc = lc.flatten(window_length=901)
lc.plot();
plt.savefig(f"plot/{KIC}_20.png")
lc = lc.remove_outliers()
lc.plot();
plt.savefig(f"plot/{KIC}_30.png")

# %% [markdown] id="c03Y0DPv-Csl"
# ## 2. The Box Least Squares Method for Finding Transiting Planets

# %% [markdown] id="Bz1vMoZWTQRm"
# The most common method used to identify transiting exoplanets is the Box Least Squares (BLS) periodogram analysis. BLS works by modeling a transit using an upside-down top hat with four parameters: period, duration, depth, and reference time. These can be seen in the figure below, from the [astropy.timeseries](https://docs.astropy.org/en/stable/timeseries/) implementation of BLS.
#
# <img style="float: right;" src="https://docs.astropy.org/en/stable/timeseries/bls-1.png" alt="Box Least Squares" width="600px"/>
#
# These parameters are then optimized by minimizing the square difference between the BLS transit model and the observation. For more information about BLS, please see the [Astropy documentation](https://docs.astropy.org/en/stable/timeseries/bls.html).
#
# Lightkurve has two types of periodogram available to anaylze periodic trends in light curves:
# * `LombScarglePeriodogram`
# * `BoxLeastSquaresPeriodogram`
#
# Please see the companion tutorial on how to create periodograms and identify significant peaks for an example of the `LombScarglePeriodogram`.

# %% [markdown] id="QWoWOLfELCTX"
# ## 3. Searching for Transiting Planets in a *Kepler* Light Curve Using BLS
#
# To create a `BoxLeastSquaresPeriodogram`, use the `LightCurve` method [to_periodogram](https://docs.lightkurve.org/reference/api/lightkurve.LightCurve.to_periodogram.html?highlight=to_periodogram), and pass in the string `'bls'` to specify the type of periodogram object you want to create. This method also optionally takes an array of periods (in days) to search, which we will set from 1–20 days to limit our search to short-period planets. We do so using the [numpy.linspace](https://numpy.org/doc/stable/reference/generated/numpy.linspace.html) function.

# %% colab={"base_uri": "https://localhost:8080/", "height": 390} executionInfo={"elapsed": 81201, "status": "ok", "timestamp": 1601412563862, "user": {"displayName": "Susan Mullally", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GhoAkTlu5JCXqC32438ISJU86DPSZdvoBOLwtOQMfU=s64", "userId": "01921813910966567332"}, "user_tz": 240} id="iLOVr_ZuRP2n" outputId="abef2286-e635-4663-b1f5-a6950d76ceda"
import numpy as np
import astropy
# Create array of periods to search
period = np.linspace(planets.koi_period.min()/10, planets.koi_period.max()*10, 10000)
# Create a BLSPeriodogram
try:
    bls = lc.to_periodogram(method='bls', period=period, frequency_factor=5000);
except ValueError:
    period = np.linspace(planets.koi_period.min()/2, planets.koi_period.max()*2, 10000)
    bls = lc.to_periodogram(method='bls', period=period, frequency_factor=5000);
fig, ax = plt.subplots(figsize=(8.485, 4))
bls.plot(ax=ax);
for i, planet in planets.iterrows():
    ax.axvline(planet.koi_period, 0, float(bls.max_power), color='k', linestyle=":", zorder=-10, alpha=0.5)

plt.legend([KIC]+[((str(planet.kepler_name))*(not pd.isna(planet.kepler_name)) or planet.kepoi_name ) + " (%.2f d)" % planet.koi_period for _, planet in planets.iterrows()])

plt.savefig(f"plot/{KIC}_35.png")
from IPython.display import display
display(planets[["kepoi_name", "kepler_name", "koi_disposition", "koi_period", "koi_time0"]])
# lc_remove = lc
# transit_mask = bls.get_transit_mask(period=bls.period_at_max_power, 
#                          transit_time=bls.transit_time_at_max_power, 
#                          duration=bls.duration_at_max_power)
# lc_remove = lc_remove[~transit_mask]
for i, planet in list(planets.reset_index(drop=True).iterrows())[::1]:
    print(planet.koi_period, planet.koi_time0bk)
    lc.fold(period=planet.koi_period, epoch_time=planet.koi_time0bk,).plot_river()
    plt.savefig(f"plot/{KIC}_4{i}.png")
    # bls2 = lc_remove.to_periodogram(method='bls', period=period, frequency_factor=500);
    # transit_mask = bls2.get_transit_mask(period=bls2.period_at_max_power, 
    #                          transit_time=bls2.transit_time_at_max_power, 
    #                          duration=bls2.duration_at_max_power)
    # lc_remove = lc_remove[~transit_mask]

# %% [markdown] id="a-8gU6D2A7a6"
# The plot above shows the power, or the likelihood of the BLS fit, for each of the periods in the array we passed in. This plot shows a handful of high-power peaks at discrete periods, which is a good sign that a transit has been identified. The highest power spike shows the most likely period, while the lower power spikes are fractional harmonics of the period, for example, 1/2, 1/3, 1/4, etc. 
#
# We can pull out the most likely BLS parameters by taking their values at maximum power — we will refer to this transiting object as "planet b."

# %% colab={"base_uri": "https://localhost:8080/", "height": 37} executionInfo={"elapsed": 81197, "status": "ok", "timestamp": 1601412563864, "user": {"displayName": "Susan Mullally", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GhoAkTlu5JCXqC32438ISJU86DPSZdvoBOLwtOQMfU=s64", "userId": "01921813910966567332"}, "user_tz": 240} id="3c24rWxnR7_M" outputId="fdc82ff2-19ab-4809-cf43-77676daa5a40"
planet_b_period = bls.period_at_max_power
planet_b_t0 = bls.transit_time_at_max_power
planet_b_dur = bls.duration_at_max_power

# Check the value for period
planet_b_period

# %% [markdown] id="4rlYrX_tLjRu"
# To confirm that this period and transit time (epoch) correspond to a transit signal, we can phase-fold the light curve using these values and plot it.

# %% colab={"base_uri": "https://localhost:8080/", "height": 405} executionInfo={"elapsed": 82802, "status": "ok", "timestamp": 1601412565475, "user": {"displayName": "Susan Mullally", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GhoAkTlu5JCXqC32438ISJU86DPSZdvoBOLwtOQMfU=s64", "userId": "01921813910966567332"}, "user_tz": 240} id="KOM3l2NbSDft" outputId="173ae47e-b646-43d3-cab1-fa8f24302249"
ax = lc.fold(period=planet_b_period, epoch_time=planet_b_t0).scatter()
ax.set_xlim(-5, 5);
plt.savefig(f"plot/{KIC}_50.png")

# %% [markdown] id="QpfdGLSoMDUb"
# The phase-folded light curve shows a strong transit signal with the identified period and transit time of maximum BLS power.

# %% [markdown] id="IhdPdcMOSRYy"
# ## 4. Retrieving a Transit Model and Cadence Mask

# %% [markdown] id="Yj6Oym9HMMvw"
# The BLS periodogram has features that make it possible to search for multiple planets in the same system. If we want to identify additional transit signals, it will be much more convenient if we first remove the previously identified signal. This will prevent the high-power periodicity of the first planet, planet b, from dominating the BLS periodogram, and will allow us to find lower signal-to-noise ratio (SNR) transits.
#
# We can create a cadence mask for the light curve using the transit parameters from the `BoxLeastSquaresPeriodogram`.

# %% id="l03MOLu3Nm1J"
# Create a cadence mask using the BLS parameters
planet_b_mask = bls.get_transit_mask(period=planet_b_period, 
                                     transit_time=planet_b_t0, 
                                     duration=planet_b_dur)

# %% [markdown] id="gh0XVIJ4OBSu"
# Now, we can create a masked version of the light curve to search for additional transit signals. The light curve is shown below, with masked cadences marked in red.

# %% colab={"base_uri": "https://localhost:8080/", "height": 405} executionInfo={"elapsed": 88300, "status": "ok", "timestamp": 1601412570982, "user": {"displayName": "Susan Mullally", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GhoAkTlu5JCXqC32438ISJU86DPSZdvoBOLwtOQMfU=s64", "userId": "01921813910966567332"}, "user_tz": 240} id="l_vzNQZwOKI0" outputId="7a50a3b5-a962-45c0-fab3-c59ba3fe5f2f"
masked_lc = lc[~planet_b_mask]
ax = masked_lc.scatter();
lc[planet_b_mask].scatter(ax=ax, c='r', label='Masked');
plt.savefig(f"plot/{KIC}_55.png")

# %% [markdown] id="S7SJfPWnOqE7"
# We can also create a BLS model to visualize the transit fit. This returns a `LightCurve` object with the BLS model in the flux column.

# %% id="QptjZH66OwR1"
# Create a BLS model using the BLS parameters
planet_b_model = bls.get_transit_model(period=planet_b_period, 
                                       transit_time=planet_b_t0, 
                                       duration=planet_b_dur)

# %% [markdown] id="qzfYLeCTPKjD"
# We can plot this over the folded light curve to confirm that it accurately represents the transit.

# %% colab={"base_uri": "https://localhost:8080/", "height": 405} executionInfo={"elapsed": 89329, "status": "ok", "timestamp": 1601412572021, "user": {"displayName": "Susan Mullally", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GhoAkTlu5JCXqC32438ISJU86DPSZdvoBOLwtOQMfU=s64", "userId": "01921813910966567332"}, "user_tz": 240} id="rHqMZmfYSX1E" outputId="00d53e85-ce58-44fc-e2be-7fc629642582"
ax = lc.fold(planet_b_period, planet_b_t0).scatter()
planet_b_model.fold(planet_b_period, planet_b_t0).plot(ax=ax, c='r', lw=2)
ax.set_xlim(-5, 5);
plt.savefig(f"plot/{KIC}_60.png")

# %% [markdown] id="goco294YUFjs"
# ## 5. Identifying Additional Transiting Planet Signals in the Same Light Curve

# %% [markdown] id="OQboGya1PZF-"
# Now that we have created a light curve with the first identified planet masked out, we can search the remaining light curve for additional transit signals. Here, we search for long-period planets by increasing our range of periods to 1–300 days.

# %% colab={"base_uri": "https://localhost:8080/", "height": 431} executionInfo={"elapsed": 96255, "status": "ok", "timestamp": 1601412578953, "user": {"displayName": "Susan Mullally", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GhoAkTlu5JCXqC32438ISJU86DPSZdvoBOLwtOQMfU=s64", "userId": "01921813910966567332"}, "user_tz": 240} id="MSUsDbMJUM-y" outputId="6d2b2bf8-2e57-4232-f3b8-257c94571b26"
# period = np.linspace(1, 300, 10000)
bls = masked_lc.to_periodogram('bls', period=period, frequency_factor=500)
fig, ax = plt.subplots(figsize=(8.485, 4))
bls.plot(ax=ax);
for i, planet in planets.iterrows():
    ax.axvline(planet.koi_period, 0, float(bls.max_power), color='k', linestyle=":", zorder=-10, alpha=0.5)

plt.legend([KIC]+[((str(planet.kepler_name))*(not pd.isna(planet.kepler_name)) or planet.kepoi_name ) + " (%.2f d)" % planet.koi_period for _, planet in planets.iterrows()])
plt.savefig(f"plot/{KIC}_65.png")

# %% [markdown] id="4GJ7WfRR31lT"
# While no peaks in this BLS periodogram display a power as high as the previous transit signal, there is a definite peak near ~240 days. We can pull out the corresponding period and transit time to check the signal.

# %% colab={"base_uri": "https://localhost:8080/", "height": 37} executionInfo={"elapsed": 96253, "status": "ok", "timestamp": 1601412578957, "user": {"displayName": "Susan Mullally", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GhoAkTlu5JCXqC32438ISJU86DPSZdvoBOLwtOQMfU=s64", "userId": "01921813910966567332"}, "user_tz": 240} id="YVhdUd32P0Sx" outputId="6bae3db5-2a2e-48ba-fb02-a490a85727d2"
planet_c_period = bls.period_at_max_power
planet_c_t0 = bls.transit_time_at_max_power
planet_c_dur = bls.duration_at_max_power

# Check the value for period
planet_c_period

# %% [markdown] id="xbk7G1in4NPm"
# We can again plot the phase-folded light curve to examine the transit.

# %% colab={"base_uri": "https://localhost:8080/", "height": 405} executionInfo={"elapsed": 102368, "status": "ok", "timestamp": 1601412585079, "user": {"displayName": "Susan Mullally", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GhoAkTlu5JCXqC32438ISJU86DPSZdvoBOLwtOQMfU=s64", "userId": "01921813910966567332"}, "user_tz": 240} id="6FKmoM70UOl3" outputId="dbba469e-0edb-4f8d-c1b5-be123b256ebe"
ax = masked_lc.fold(planet_c_period, planet_c_t0).scatter()
masked_lc.fold(planet_c_period, planet_c_t0).bin(.1).plot(ax=ax, c='r', lw=2, 
                                                          label='Binned Flux')
ax.set_xlim(-5, 5);

plt.savefig(f"plot/{KIC}_70.png")

# %% [markdown] id="ebk1C7aa4ctu"
# This signal is lower SNR because there are fewer transits due to the longer period, and the shallower depth implies that the planet is smaller. To help see the transit more clearly, we have overplotted the binned flux, combining consecutive points taken over a span of 0.1 days.
#
# We have now successfully identified two planets in the same system! We can use the BLS models to visualize the transit timing in the light curve.

# %% id="SRngiK4vQNWv"
planet_c_model = bls.get_transit_model(period=planet_c_period, 
                                       transit_time=planet_c_t0, 
                                       duration=planet_c_dur)

# %% colab={"base_uri": "https://localhost:8080/", "height": 405} executionInfo={"elapsed": 105743, "status": "ok", "timestamp": 1601412588463, "user": {"displayName": "Susan Mullally", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GhoAkTlu5JCXqC32438ISJU86DPSZdvoBOLwtOQMfU=s64", "userId": "01921813910966567332"}, "user_tz": 240} id="XEXsEtfHQOE8" outputId="5486451f-9f6a-48c4-c7c9-262510efaa00"
ax = lc.scatter();
planet_b_model.plot(ax=ax, c='dodgerblue', label='Planet b Transit Model');
planet_c_model.plot(ax=ax, c='r', label='Planet c Transit Model');
# ax.set_xlim(200, 300);
plt.savefig(f"plot/{KIC}_75.png")

# %%

# %%
lc.plot_river(period=planet_b_period, bin_points=2, method='mean')
plt.savefig(f"plot/{KIC}_80.png")
lc.plot_river(period=planet_c_period, bin_points=2, method='mean')
plt.savefig(f"plot/{KIC}_85.png")

# %% [markdown] id="ow66EyEhRZ98"
# ## 6. Using the Interactive BLS Periodogram in Lightkurve

# %% [markdown] id="0UwgAQO45MIR"
# Lightkurve also has a tool that enables you to interactively perform a BLS search. A quick demo of this feature is shown below. 
#
# To use the [LightCurve.interact_bls()](https://docs.lightkurve.org/reference/api/lightkurve.LightCurve.interact_bls.html?highlight=lightcurve%20interact_bls#lightkurve.LightCurve.interact_bls) method, zoom in on peaks in the BLS periodogram using the interactive plotting tools. To improve the fit, you can change the transit duration. The phase-folded light curve panel in the top right and the full light curve below it will automatically update to plot the highest power BLS model. The BLS parameters with highest power are noted in the bottom right of the figure.

# %% id="q-Xkjz6sKwlx"
lc.interact_bls()

# %% [markdown] id="aG4fgysD_Brp"
# ## About this Notebook
#
# **Authors:** Nicholas Saunders (nksaun@hawaii.edu)
#
# **Updated On:** 2020-09-28

# %% [markdown] id="MptPsdBQ_Qju"
# ## Citing Lightkurve and Astropy
#
# If you use `lightkurve` or its dependencies in your published research, please cite the authors. Click the buttons below to copy BibTeX entries to your clipboard.

# %% colab={"base_uri": "https://localhost:8080/", "height": 151} executionInfo={"elapsed": 105740, "status": "ok", "timestamp": 1601412588467, "user": {"displayName": "Susan Mullally", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GhoAkTlu5JCXqC32438ISJU86DPSZdvoBOLwtOQMfU=s64", "userId": "01921813910966567332"}, "user_tz": 240} id="1khvwNHx_QDz" outputId="1a696184-e00e-4286-8b8b-f8a665305138"
lk.show_citation_instructions()

# %% [markdown] id="ZhdRVU3B_Zn2"
# <img style="float: right;" src="https://raw.githubusercontent.com/spacetelescope/notebooks/master/assets/stsci_pri_combo_mark_horizonal_white_bkgd.png" alt="Space Telescope Logo" width="200px"/>
#
