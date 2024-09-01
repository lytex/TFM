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

# %% [markdown]
# # 

# %%
import pandas as pd
import optuna
import warnings
import numpy as np
warnings.filterwarnings(action="ignore", category=UserWarning)
study = optuna.study.load_study(study_name=None, storage="sqlite:///example-study.db")
df2 = pd.read_csv('20240825-115903/all.csv')
import seaborn as sns
sns.set_theme(style="darkgrid")
from optuna.importance import PedAnovaImportanceEvaluator
from optuna_fast_fanova import FanovaImportanceEvaluator
quantile = 0.475
evaluator = PedAnovaImportanceEvaluator(baseline_quantile=quantile)
d = evaluator.evaluate(study)
print(d)
print(pd.DataFrame([d]).to_latex(index=False))
print(sum(d.values()))
# d2 = optuna.importance.get_param_importances(study, evaluator=FanovaImportanceEvaluator())
# print({k: v*sum(d.values()) for k, v in d2.items()})
# print({k: v for k, v in d2.items()})
# dir(evaluator)
# evaluator
df = pd.DataFrame([{**{"F1_val": x.values[0] if x.values is not None else 0}, **x.params, } for x in study.get_trials()])

def optuna_to_pandas(path):
    import pandas as pd
    import optuna
    import warnings
    warnings.filterwarnings(action="ignore", category=UserWarning)
    study = optuna.study.load_study(study_name=None, storage=f"sqlite:///{path}")
    df = pd.DataFrame([{**{"F1_val": x.values[0] if x.values is not None else 0}, **x.params, } for x in study.get_trials()])
    return df
    


# %%
import seaborn as sns
import numpy as np
# sns.catplot(data=df.assign(global_level_list=df.global_level_list.apply(str)).query("F1_val > 0.95"), x="global_level_list", y="l2")

kv = {str(k): v for v, k in enumerate(np.unique(df.global_level_list))}
vk = {v: str(k) for v, k in enumerate(np.unique(df.global_level_list))}
enum = df.assign(global_level_list=df.assign(global_level_list=df.global_level_list.apply(str)).global_level_list.replace(to_replace=kv).apply(int)).query("F1_val > 0.95")

fig = sns.kdeplot(
    data=enum,
    x="global_level_list",
    y='l2',
    thresh=0.01,
    cmap='plasma',
    fill=True,
    bw_method=0.05,
).get_figure()
fig.suptitle('global_level_list vs l2', size=32, va='baseline', y=0.90)
fig.axes[0].set_xlabel("")
fig.axes[0].set_ylabel("")
fig.axes[0].tick_params(labelsize=20)
enum["l2_bin"] = pd.cut(enum.l2, [0, 0.03, 0.10])
{k: [vk.get(v) for v in val] for k, val in enum.groupby("l2_bin").groups.items()}
{k: [eval(vk[enum.loc[v].global_level_list]) for v in val] for k, val in enum.groupby("l2_bin").groups.items()}


pd.concat(
[
pd.DataFrame({k: pd.Series([max(eval(vk[enum.loc[v].global_level_list]), default=0) for v in val]).describe() for k, val in enum.groupby("l2_bin").groups.items()}).T \
    .add_prefix("max_"),# .reset_index()
pd.DataFrame({k: pd.Series([min(eval(vk[enum.loc[v].global_level_list]), default=0) for v in val]).describe() for k, val in enum.groupby("l2_bin").groups.items()}).T \
    .add_prefix("min_"),# .reset_index()
pd.DataFrame({k: pd.Series([len(eval(vk[enum.loc[v].global_level_list])) for v in val]).describe() for k, val in enum.groupby("l2_bin").groups.items()}).T \
    .add_prefix("len_"),#.reset_index()
], join='inner', axis=1)
# {k: [v for v in val] for k, val in enum.groupby("l3_bin").groups.items()}
# enum.iloc[5]


# %%
import matplotlib.pyplot as plt
num2 = df.query("F1_val!=0 and F1_val > 0.95")[["l1", "l2", "global_level_list"]]
num2["l2_bin"] = pd.cut(num2.l2, [0, 0.03, 0.10])
num2["global_level_list"] = num2["global_level_list"].apply(lambda row: [int(x in row) for x in range(1, 7)])
# num2["global_level_list_min"] = num2["global_level_list"].apply(lambda row: [int(x in row) for x in range(1, 7)])
# num2["global_level_list_max"] = num2["global_level_list"].apply(lambda row: [int(x in row) for x in range(1, 7)])
# num2["global_level_list_std"] = num2["global_level_list"].apply(lambda row: [int(x in row) for x in range(1, 7)])
num2
def explode_columns(row):
    return pd.DataFrame(pd.DataFrame(row)[row.name].tolist(), index=row.index)

def my_mean(row):
    if row.name == "global_level_list":
        temp = pd.DataFrame(pd.DataFrame(row).global_level_list.tolist(), index= row.index)
        return pd.Series(temp.mean().to_list(), name=row.name)
    else:
        return row
def my_min(row):
    if row.name == "global_level_list":
        temp = pd.DataFrame(pd.DataFrame(row).global_level_list.tolist(), index= row.index)
        return pd.Series(temp.min().to_list(), name=row.name)
    else:
        return row
def my_max(row):
    if row.name == "global_level_list":
        temp = pd.DataFrame(pd.DataFrame(row).global_level_list.tolist(), index= row.index)
        return pd.Series(temp.max().to_list(), name=row.name)
    else:
        return row
def my_std(row):
    if row.name == "global_level_list":
        temp = pd.DataFrame(pd.DataFrame(row).global_level_list.tolist(), index= row.index)
        return pd.Series(temp.std().to_list(), name=row.name)
    else:
        return row

grouped_df = num2.groupby("l2_bin").agg([my_mean, my_std]).rename(columns={"my_std": "std", "my_mean": "mean"}).reset_index()

# melted_df = grouped_df
# melted_df = grouped_df.melt(id_vars='l2_bin', var_name='Global_Level', value_name='Count')
# grouped_df = num2.groupby("l2_bin").agg(["mean", "std"]).reset_index().drop(columns=["l1", "l2",])

# melted_df = grouped_df.melt(id_vars='l2_bin', var_name='Global_Level', value_name='Count')

# Plotting
# import matplotlib.pyplot as plt
# fig = plt.figure(figsize=(12, 12))
# ax = sns.pointplot(data=melted_df, x='l2_bin', y='Count', hue='Global_Level',
#               dodge=0.2,  errwidth=1.5, capsize=0.2, palette='coolwarm',
#               # errorbar=None,
#              )
# plt.setp(ax.get_legend().get_texts(), fontsize='22') # for legend text
# plt.setp(ax.get_legend().get_title(), fontsize='32') # for legend title
# fig.suptitle("Global L2", size=32, va='baseline', y=0.90)
# fig.axes[0].tick_params(labelsize=20)
# fig.axes[0].set_xlabel("")
# fig.axes[0].set_ylabel("")

# grouped_df
# grouped_df.global_level_list_mean.apply()
# pd.DataFrame(grouped_df[["global_level_list_mean"]].tolist(),
def group(grouped_df, key="l2_bin"):
    aa = pd.merge(grouped_df[key].reset_index(),
                  explode_columns(grouped_df.global_level_list["mean"]).add_prefix("global_level_list_").reset_index()
                 ).drop(columns="index").reset_index()
    return aa.groupby(key).agg(["mean"]).drop(columns="index").reset_index()
def group(groped_df):
    aa = pd.merge(pd.merge(
        grouped_df["l2_bin"].reset_index(),
        explode_columns(grouped_df.global_level_list["mean"]).add_prefix("global_level_list_").reset_index()
        ).drop(columns="index").reset_index(),
        explode_columns(grouped_df.global_level_list["std"]).add_prefix("global_level_list_std_").reset_index())
    aa
    bb = aa.groupby("l2_bin").agg(["mean", "std"]).drop(columns="index").reset_index()
    for x in range(6):
        bb[f"global_level_list_{x}"] = bb.loc[:, f"global_level_list_{x}"].assign(std=aa[f"global_level_list_std_{x}"])
        bb = bb.drop(columns=f"global_level_list_std_{x}")
    return bb

grouped_df_old = grouped_df
grouped_df = group(grouped_df)
melted_df = grouped_df.melt(id_vars='l2_bin', var_name='Global_Level', value_name='Count')

# Plotting
fig = plt.figure(figsize=(12, 12))
ax = sns.pointplot(data=melted_df, x='l2_bin', y='Count', hue='Global_Level',
              dodge=0.2,  errwidth=1.0, capsize=0.2, palette='coolwarm',
              # errorbar=None,
             )
plt.setp(ax.get_legend().get_texts(), fontsize='22') # for legend text
plt.setp(ax.get_legend().get_title(), fontsize='32') # for legend title
fig.suptitle("Global L2", size=32, va='baseline', y=0.90)
fig.axes[0].tick_params(labelsize=20)
fig.axes[0].set_xlabel("")
fig.axes[0].set_ylabel("")
melted_df

# %%
# explode_columns(num2.global_level_list), df.query("F1_val!=0 and F1_val > 0.95")["global_level_list"]
# explode_columns(df.global_level_list)

num2 = df.query("F1_val!=0 and F1_val > 0.95")
num2["l2_bin"] = pd.cut(num2.l2, [0, 0.03, 0.10])
num2.keys()
# gdf = num2.groupby(["l2_bin"]).agg([my_mean, my_std]).rename(columns={"my_std": "std", "my_mean": "mean"}).reset_index()
# gdf

# %%
# explode_columns(melted_df.Count)
# melted_df2
# grouped_df.global_level_list.apply(explode_columns, axis=0)
explode_columns(grouped_df_old.global_level_list["mean"]).add_prefix("global_level_list_")
# grouped_df_old
aa = pd.merge(pd.merge(
    grouped_df_old["l2_bin"].reset_index(),
    explode_columns(grouped_df_old.global_level_list["mean"]).add_prefix("global_level_list_").reset_index()
    ).drop(columns="index").reset_index(),
    explode_columns(grouped_df_old.global_level_list["std"]).add_prefix("global_level_list_std_").reset_index())
aa
bb = aa.groupby("l2_bin").agg(["mean", "std"]).drop(columns="index").reset_index()
bb
# aa
# aa[[f"global_level_list_std_{x}" for x in range(6)]]
for x in range(6):
    # print(bb.loc[:, f"global_level_list_{x}"].assign(std=aa[f"global_level_list_std_{x}"]))
    # bb[f"global_level_list_{x}"]["std"] = aa[f"global_level_list_std_{x}"]
    bb[f"global_level_list_{x}"] = bb.loc[:, f"global_level_list_{x}"].assign(std=aa[f"global_level_list_std_{x}"])
    bb = bb.drop(columns=f"global_level_list_std_{x}")


bb
# melted_df
# grouped_df

# %%
import matplotlib.pyplot as plt
df.l2.hist(bins=50)
plt.figure()
df.query('l1 < 0.02').l1.hist(bins=50)
# def mean_row(row):
#     try:
#         return np.mean(row)
#     except ValueError:
#         return pd.NA
# def std_row(row):
#     try:
#         return np.std(row)
#     except ValueError:
#         return pd.NA
# def max_row(row):
#     try:
#         return np.max(row)
#     except ValueError:
#         return 0
# def min_row(row):
#     try:
#         return np.min(row)
#     except ValueError:
#         return 0
        
for x in range(1, 7):
    df[f"global_level_list_{x}"] = df[f"global_level_list"].apply(lambda row: x in row).astype(int)
# df["global_level_list_mean"] = df[f"global_level_list"].apply(mean_row)
# df["global_level_list_std"] = df[f"global_level_list"].apply(std_row)
# df["global_level_list_max"] = df[f"global_level_list"].apply(max_row)
# df["global_level_list_min"] = df[f"global_level_list"].apply(min_row)
for x in range(1, 4):
    df[f"local_level_list_{x}"] = df[f"local_level_list"].apply(lambda row: x in row).astype(int)
# from IPython.display import display
# display(df.query('l2 < 0.03')[[f"global_level_list_{x}" for x in range(1, 7)]].astype(int).describe())
# display(df.query('l2 > 0.03')[[f"global_level_list_{x}" for x in range(1, 7)]].astype(int).describe())
# display(df.query('l2 < 0.002')[[f"global_level_list_{x}" for x in range(1, 7)]].astype(int).describe())
# display(df.query('l2 > 0.002')[[f"global_level_list_{x}" for x in range(1, 7)]].astype(int).describe())


num2 = df.query("F1_val!=0 and F1_val > 0.90")[ ["l1", "l2"] + [f"global_level_list_{x}" for x in range(1, 7)]]
# num2 = df.query("F1_val!=0 and F1_val > 0.90")[ ["l1", "l2"] + [f"global_level_list_{x}" for x in ("mean", "std", "max", "min")]]
num2["l2_bin"] = pd.cut(num2.l2, [0, 0.03, 0.10])
grouped_df = num2.groupby("l2_bin").agg(["sum"]).reset_index().drop(columns=["l1", "l2",
# "global_level_list_1",
# "global_level_list_2",
# "global_level_list_3",
# "global_level_list_4",
# "global_level_list_5",
# "global_level_list_6",
                                                                                    ])

melted_df = grouped_df.melt(id_vars='l2_bin', var_name='Global_Level', value_name='Count')

# Plotting
fig = plt.figure(figsize=(12, 12))
ax = sns.pointplot(data=melted_df, x='l2_bin', y='Count', hue='Global_Level',
              dodge=0.2,  errwidth=1.5, capsize=0.2, palette='coolwarm',
              # errorbar=None,
             )
plt.setp(ax.get_legend().get_texts(), fontsize='22') # for legend text
plt.setp(ax.get_legend().get_title(), fontsize='32') # for legend title
fig.suptitle("Global L2", size=32, va='baseline', y=0.90)
fig.axes[0].tick_params(labelsize=20)
fig.axes[0].set_xlabel("")
fig.axes[0].set_ylabel("")
fig.savefig("plot/optuna/group/global_l2.png")

num2["l1_bin"] = pd.cut(num2.l1, [0, 0.001, 0.10])
grouped_df = num2.groupby("l1_bin").agg(["sum"]).reset_index().drop(columns=["l1", "l2",
# "global_level_list_1",
# "global_level_list_2",
# "global_level_list_3",
# "global_level_list_4",
# "global_level_list_5",
# "global_level_list_6",
                                                                                    ])

melted_df = grouped_df.melt(id_vars='l1_bin', var_name='Global_Level', value_name='Count')

# Plotting
fig = plt.figure(figsize=(12, 12))
ax = sns.pointplot(data=melted_df, x='l1_bin', y='Count', hue='Global_Level',
              dodge=0.2, errwidth=1.5, capsize=0.2, palette='coolwarm',
              # errorbar=None,
             )
plt.setp(ax.get_legend().get_texts(), fontsize='22') # for legend text
plt.setp(ax.get_legend().get_title(), fontsize='32') # for legend title
fig.suptitle("Global L1", size=32, va='baseline', y=0.90)
fig.axes[0].tick_params(labelsize=20)
fig.axes[0].set_xlabel("")
fig.axes[0].set_ylabel("")
fig.savefig("plot/optuna/group/global_l1.png")




num2 = df.query("F1_val!=0 and F1_val > 0.90")[ ["l1", "l2"] + [f"local_level_list_{x}" for x in range(1, 4)]]
num2["l2_bin"] = pd.cut(num2.l2, [0, 0.03, 0.10])
grouped_df = num2.groupby("l2_bin").agg(["sum"]).reset_index().drop(columns=["l1", "l2",
# "local_level_list_1",
# "local_level_list_2",
# "local_level_list_3",
# "local_level_list_4",
# "local_level_list_5",
# "local_level_list_6",
                                                                                    ])

melted_df = grouped_df.melt(id_vars='l2_bin', var_name='local_Level', value_name='Count')

grouped_df2 = num2.groupby("l2_bin").std().reset_index().drop(columns=["l1", "l2", 
                                                                      ])

melted_df2 = grouped_df.melt(id_vars='l2_bin', var_name='local_Level', value_name='Count')

# Plotting
fig = plt.figure(figsize=(12, 12))
ax = sns.pointplot(data=melted_df, x='l2_bin', y='Count', hue='local_Level',
              dodge=0.2,  errwidth=1.5, capsize=0.2, palette='coolwarm',
              # errorbar=None,
             )

plt.setp(ax.get_legend().get_texts(), fontsize='22') # for legend text
plt.setp(ax.get_legend().get_title(), fontsize='32') # for legend title
fig.suptitle("Local L2", size=32, va='baseline', y=0.90)
fig.axes[0].tick_params(labelsize=20)
fig.axes[0].set_xlabel("")
fig.axes[0].set_ylabel("")
fig.savefig("plot/optuna/group/local_l2.png")

num2["l1_bin"] = pd.cut(num2.l1, [0, 0.001, 0.10])
grouped_df = num2.groupby("l1_bin").agg(["sum"]).reset_index().drop(columns=["l1", "l2",
# "local_level_list_1",
# "local_level_list_2",
# "local_level_list_3",
# "local_level_list_4",
# "local_level_list_5",
# "local_level_list_6",
                                                                                    ])

melted_df = grouped_df.melt(id_vars='l1_bin', var_name='local_Level', value_name='Count')

grouped_df2 = num2.groupby("l1_bin").std().reset_index().drop(columns=["l1", "l2", 
                                                                      ])

melted_df2 = grouped_df.melt(id_vars='l1_bin', var_name='local_Level', value_name='Count')

# Plotting
fig = plt.figure(figsize=(12, 12))
ax = sns.pointplot(data=melted_df, x='l1_bin', y='Count', hue='local_Level',
              dodge=0.2, errwidth=1.5, capsize=0.2, palette='coolwarm',
              # errorbar=None,
             )

plt.setp(ax.get_legend().get_texts(), fontsize='22') # for legend text
plt.setp(ax.get_legend().get_title(), fontsize='32') # for legend title
fig.suptitle("Local L1", size=32, va='baseline', y=0.90)
fig.axes[0].tick_params(labelsize=20)
fig.axes[0].set_xlabel("")
fig.axes[0].set_ylabel("")
fig.savefig("plot/optuna/group/local_l1.png")


# %%
# df["global_level_list"] = df["global_level_list"].apply(tuple)
# df["local_level_list"] = df["local_level_list"].apply(tuple)
# df = df.reset_index()
# df2 = df2.reset_index()
df.shape, df2.shape
merge = pd.merge(df, df2, on=["F1_val"], how="inner")
# notm = merge[merge.l1_x.isna() | merge.l1_y.isna()]
# print(notm)
# notm.keys()
# from IPython.display import display
# display(df.iloc[notm.index_x])
# 'F1_val', 'global_level_list', 'local_level_list', 'l1', 'l2', 'dropout', 'frac'

# %%
# ax = df.query('F1_val != 0').query('F1_val > 0.0').reset_index().plot.scatter('index', 'F1_val', figsize=(14, 10))
# df.query('F1_val != 0').query('F1_val > 0.9').reset_index().F1_val.plot(figsize=(14, 10))

# df.query('F1_val != 0').query('F1_val > 0.9').reset_index().F1_val.describe()
colors=['blue' if x else 'orange' for x in merge.where((merge.recall_val < 0.9) | (merge.precision_val < 0.9)).isna().F1_val.to_numpy() ]
ax = merge.query('F1_val != 0').query('F1_val > 0.0').reset_index().plot.scatter('index', 'F1_val', figsize=(14, 10), c=colors)
ax.axhline(y=df.F1_val.quantile(quantile))
fig = ax.get_figure() 

fig.savefig("plot/optuna/split.png")

# %%
import statsmodels.api as sm
from statsmodels.formula.api import ols
import pandas as pd
numeric_level_list = True
df = optuna_to_pandas("example-study.db")

df['global_level_list'] = df['global_level_list'].apply(tuple)
df['local_level_list'] = df['local_level_list'].apply(tuple)

# import graycode

if numeric_level_list:
    # df['global_level_list'] = df['global_level_list'].apply(lambda row: sum([2**i for i in row]))
    # df['local_level_list'] = df['local_level_list'].apply(lambda row: sum([2**i for i in row]))
    df['global_level_list'] = df['global_level_list'].apply(lambda row: len(row))
    df['local_level_list'] = df['local_level_list'].apply(lambda row: len(row))
    # Numero de puntos por cada uno
    df['global_level_list_original'] = df['global_level_list']
    df['local_level_list_original'] = df['local_level_list']
    # df['global_level_list'] = df['global_level_list'].apply(lambda row: sum([2**(6-i) for i in row]))
    # df['local_level_list'] = df['local_level_list'].apply(lambda row: sum([2**(3-i) for i in row]))
    # df['global_level_list'] = df['global_level_list'].apply(lambda row: int(1 in row))
    # df['local_level_list'] = df['local_level_list'].apply(lambda row: int(1 in row))
    # df['global_level_list'] = df['global_level_list'].apply(lambda row: graycode.tc_to_gray_code(sum([2**i for i in row])))
    # df['local_level_list'] = df['local_level_list'].apply(lambda row: graycode.tc_to_gray_code(sum([2**i for i in row])))
    
    # df['global_level_list'] = df['global_level_list'].apply(lambda row: int(str(row).replace(",", "").replace(" ", "").replace("()", "0").replace("(", "").replace(")", "")))
    # df['local_level_list'] = df['local_level_list'].apply(lambda row: int(str(row).replace(",", "").replace(" ", "").replace("()", "0").replace("(", "").replace(")", "")))
df_red = df.loc[:, (df != df.iloc[0]).any()]
# df[[
#     'global_level_list',
#        'local_level_list', 'l1', 'l2', 'dropout', 
#        'frac',  'F1_val',
# ]] .sort_values(by="F1_val", ascending=False).style.background_gradient(cmap='coolwarm_r').to_excel("all.xlsx", engine="openpyxl", index=False)

if numeric_level_list:
    lm = ols('F1_val ~ global_level_list + local_level_list + l1 + l2 + dropout + frac',
             data=df_red).fit()
else:
    lm = ols('F1_val ~ C(global_level_list) + C(local_level_list) + l1 + l2 + dropout + frac',
             data=df_red).fit()
table = sm.stats.anova_lm(lm, typ=1, robust="hc3") # Type 2 ANOVA DataFrame
print(lm.summary())
print(table.to_latex())
table

# %%
import matplotlib.pyplot as plt

if "global_level_list" not in df.keys():
    shallue = True
    df_red = df.loc[:, (df != df.iloc[0]).any()]
else:
    shallue = False

# for x in range(1, 7):
#     df_red[f"global_level_list_{x}"] = df_red[f"global_level_list_original"].apply(lambda row: x in row).astype(int)
# for x in range(1, 4):
#     df_red[f"local_level_list_{x}"] = df_red[f"local_level_list_original"].apply(lambda row: x in row).astype(int)

num = df_red.query("F1_val!=0 and F1_val > 0.95 and l1 < 0.06")
pd.plotting.scatter_matrix(num, alpha=0.2, figsize=(32, 18), hist_kwds={"bins": 50})
plt.savefig('filename.svg', format='svg')
plt.close('all')
None

# from IPython.display import display
num = num[['F1_val', 'l1', 'l2', 'dropout', 'frac',  ] ]
# [f"global_level_list_{x}" for x in range(1, 7)] + [f"local_level_list_{x}" for x in range(1, 4)]]

for key in num.keys():
    if key == 'F1_val':
        continue
    plt.figure()
    pd.plotting.scatter_matrix(num[['F1_val', key]])

    
    # Set up the figure
    f, ax = plt.subplots(figsize=(8, 8))
    # ax.set_aspect("equal")

    # Draw a contour plot to represent each bivariate density
    fig = sns.kdeplot(
        data=num,
        x=key,
        y='F1_val',
        thresh=0.01,
        cmap='plasma',
        fill=True,
        clip=(0, max(num[key].max(), 1)*1.05),
    ).get_figure()
    fig.suptitle(key, size=32, va='baseline', y=0.90)
    fig.axes[0].set_xlabel("")
    fig.axes[0].set_ylabel("")
    fig.axes[0].tick_params(labelsize=20)
    if shallue:
        fig.savefig(f"plot/optuna/shallue/{key}.png")
    else:
        fig.savefig(f"plot/optuna/{key}.png")
    # fig.axes[1].tick_params(labelsize=20)
    
    # ax = num.plot.scatter(x=key, y='F1_val')

num

# %%
# %matplotlib inline
if not shallue:

    from mpl_toolkits.mplot3d import Axes3D
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import numpy as np
    from scipy.interpolate import griddata

    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import gaussian_kde
    
    # Extract x, y, and F1_val data
    x = num['global_level_list'].to_numpy()
    # y = num['l1'].to_numpy()
    f1_val = num['F1_val'].to_numpy()

    for y, label in [(num['l1'].to_numpy(), "L1"), (num['l2'].to_numpy(), "L2")]:
    
        # Stack x and y for KDE
        xy = np.vstack([x, y])
        
        # Compute KDE with F1_val as weights
        # kde = gaussian_kde(xy, weights=0.0+((f1_val.max() - f1_val )< 0.01), bw_method=0.1)
        kde = gaussian_kde(xy, weights=np.exp(f1_val.max()-f1_val), bw_method=0.3)
        # kde = gaussian_kde(xy, weights=np.ones_like(f1_val), bw_method=0.3)
        # kde2 = gaussian_kde(xy, weights=np.median(f1_val)*np.ones_like(f1_val), bw_method=0.2)
        
        # Create a grid for evaluation
        x_min, x_max = x.min(), x.max()
        y_min, y_max = y.min(), y.max()
        X, Y = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
        
        
        # Evaluate the KDE on the grid
        Z = kde(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)
        Z = Z/Z.max()*f1_val.max()
        
        # Create a figure and plot the weighted KDE
        fig, ax = plt.subplots(figsize=(12, 9))
        contour = ax.contourf(X, Y, Z, levels=30, cmap='viridis')
        
        # Set labels for each axis
        ax.set_xlabel('Global Level List')
        ax.set_ylabel(label)
        
        # Add a color bar
        cbar = fig.colorbar(contour, ax=ax)
        cbar.set_label('Weighted Density (by F1_val)')
        
        plt.show()

    


# %%
# num.sort_values(by='F1_val', ascending=False)
# from IPython.display import display
# with pd.option_context('display.max_rows', 1000, 'display.max_columns', 100, 'display.max_colwidth', 1000):
#     display(df[['global_level_list',
#         'global_level_list_original',
#         'local_level_list',
#         'local_level_list_original']])

# %%
if not shallue:

    from mpl_toolkits.mplot3d import Axes3D
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import numpy as np
    from scipy.interpolate import griddata
    
    # Create a figure
    fig, ax = plt.subplots(figsize=(12, 9))
    
    # Create a grid of points
    x_range = np.linspace(num['global_level_list'].min(), num['global_level_list'].max(), 100)
    y_range = np.linspace(num['l2'].min(), num['l2'].max(), 100)
    # x_range = num['global_level_list'].to_numpy()
    # y_range = num['l2'].to_numpy()
    X, Y = np.meshgrid(x_range, y_range)
    
    # Interpolate the data
    Z = griddata((num['global_level_list'], num['l2']), num['F1_val'], (X, Y), method='nearest')
    
    # Create the contour plot
    contour = ax.contourf(X, Y, Z, levels=100, cmap='viridis')
    
    # Plot the original data points
    # scatter = ax.scatter(num['global_level_list'], num['l2'], c=num['F1_val'], 
    #                      s=50, cmap='viridis', edgecolor='black')
    
    # Set labels for each axis
    ax.set_xlabel('Global Level List')
    ax.set_ylabel('l2')
    
    # Add a color bar
    cbar = fig.colorbar(contour, ax=ax)
    cbar.set_label('F1 Value')


# %%
from scipy.stats import binned_statistic_2d
global_level_list = num['global_level_list'].values
l2 = num['l2'].values
F1_val = num['F1_val'].values
# Define the number of bins in each dimension
n_bins_x = 20
n_bins_y = 20

# Bin the data and calculate the mean F1_val in each bin
statistic, x_edge, y_edge, _ = binned_statistic_2d(global_level_list, l2, F1_val, 
                                                   statistic='max', 
                                                   bins=[n_bins_x, n_bins_y])

# Calculate the bin centers
x_center = (x_edge[:-1] + x_edge[1:]) / 2
y_center = (y_edge[:-1] + y_edge[1:]) / 2

# Create a meshgrid for plotting
X, Y = np.meshgrid(x_center, y_center)

plt.figure(figsize=(10, 8))

# Plot the binned data using pcolormesh
plt.pcolormesh(X, Y, statistic.T, cmap='viridis', shading='auto')

# Add a colorbar
plt.colorbar(label='Average F1_val')

# Label the axes
plt.xlabel('global_level_list')
plt.ylabel('l2')
plt.title('Averaged F1_val in Binned Zones')

# Show the plot
plt.show()


# %%
df3 = optuna_to_pandas("example-study.db").sort_values(by='F1_val', ascending=False)
# print("\n".join([f"{k} = {v if k != 'wavelet_family' and not pd.isna(v) else repr(v) if k =='wavelet_family' else 'None'}" for k, v in df3.sort_values(by="F1_val", ascending=False).iloc[0].items()]))
df2 = pd.read_csv('20240825-115903/all.csv')
print("\n".join([f"{k} = {v if k != 'wavelet_family' and not pd.isna(v) else repr(v) if k =='wavelet_family' else 'None'}" for k, v in df2.sort_values(by="F1_val", ascending=False).iloc[0, 0:22].items()]))
# print("\n".join([f"{k} = {v if k != 'wavelet_family' and not pd.isna(v) else repr(v) if k =='wavelet_family' else 'None'}" for k, v in df3.sort_values(by="F1_val", ascending=False).iloc[0].items()]))
df2.sort_values(by="F1_val", ascending=False).F1

# %%
# %pdb
# list(df3.sort_values(by="F1_val", ascending=False).iloc[0].items())
list(df2.sort_values(by="F1_val", ascending=False).iloc[0, 0:22].items())
print("\n".join([f"{k} = {v if k != 'wavelet_family' and not pd.isna(v) else repr(v) if k =='wavelet_family' else 'None'}" for k, v in df3.sort_values(by="F1_val", ascending=False).iloc[0].items()]))

# %%
