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
from IPython.display import display
# df_path = 'cumulative_2022.09.30_09.06.43.csv'
df_path = 'cumulative_2024.06.01_09.08.01.csv'
df = pd.read_csv(df_path ,skiprows=144)
len(df.kepid), len(df.kepoi_name.unique())

df_bin = pd.read_csv("kepler_binaries.csv", skiprows=7)
df_bin["kepid"] = df_bin["#KIC"]

df = pd.merge(df, df_bin, on="kepid", how="left")

df["#KIC"].count()

df.groupby('koi_disposition')["#KIC"].count(), df.loc[~(df["#KIC"].isna()) & (df["koi_disposition"] == "CONFIRMED")]["kepoi_name"].count(), df.groupby(["koi_disposition"])["kepoi_name"].count(), \
df.loc[~(df["#KIC"].isna())].groupby('koi_disposition')["#KIC"].count()


# %%
df.loc[~(df["#KIC"].isna()) & (df["koi_disposition"] == "CONFIRMED")].to_csv("CONFIRMED_binaries.csv", index=False)
df.loc[~(df["#KIC"].isna()) & (df["koi_disposition"] == "FALSE POSITIVE")].to_csv("FALSE_POSITIVE_binaries.csv", index=False)
df.loc[~(df["#KIC"].isna()) & (df["koi_disposition"] == "CANDIDATE")].to_csv("CANDIDATE_binaries.csv", index=False)

# %%
df_companions = pd.read_csv("companions.csv")
df_pri = pd.merge(df, df_companions.assign(kepid=df_companions["KIC-pri"]))
df_comp = pd.merge(df, df_companions.assign(kepid=df_companions["KIC-comp"]))
display(df_pri)
display(df_comp)
