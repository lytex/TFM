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
import statsmodels.api as sm
from statsmodels.formula.api import ols
import pandas as pd
numeric_level_list = True

df = pd.read_csv('all_data_2024-07-17/20240823-232405/all.csv').query('recall<0.99')
df['global_level_list'] = df['global_level_list'].apply(eval)
df['local_level_list'] = df['local_level_list'].apply(eval)
if numeric_level_list:
    # df['global_level_list'] = df['global_level_list'].apply(lambda row: sum([2**i for i in row]))
    # df['local_level_list'] = df['local_level_list'].apply(lambda row: sum([2**i for i in row]))
    # df['global_level_list'] = df['global_level_list'].apply(lambda row: len(row))
    # df['local_level_list'] = df['local_level_list'].apply(lambda row: len(row))
    # Numero de puntos por cada uno
    df['global_level_list'] = df['global_level_list'].apply(lambda row: sum([2**(max(row)-i) for i in row]))
    df['local_level_list'] = df['local_level_list'].apply(lambda row: sum([2**(max(row)-i) for i in row]))
    # df['global_level_list'] = df['global_level_list'].apply(lambda row: int(str(row).replace(",", "").replace(" ", "").replace("()", "0").replace("(", "").replace(")", "")))
    # df['local_level_list'] = df['local_level_list'].apply(lambda row: int(str(row).replace(",", "").replace(" ", "").replace("()", "0").replace("(", "").replace(")", "")))
df_red = df.loc[:, (df != df.iloc[0]).any()][['use_wavelet',  'global_level_list', 'local_level_list', 'l1',
       'l2', 'dropout', 'frac',  'F1', 'F1_val', ]]
df[[
    'use_wavelet',
       'binary_classification', 'global_level_list',
       'local_level_list', 'l1', 'l2', 'dropout', 
       'frac',  'precision', 'recall', 'F1',
       'Fβ', 'precision_val', 'recall_val', 'F1_val', 'Fβ_val',
]] .sort_values(by="F1_val", ascending=False).style.background_gradient(cmap='coolwarm_r').to_excel("all.xlsx", engine="openpyxl", index=False)

if numeric_level_list:
    lm = ols('F1_val ~ use_wavelet + global_level_list + local_level_list + l1 + l2 + dropout + frac',
             data=df_red).fit()
else:
    lm = ols('F1_val ~  use_wavelet + C(global_level_list) + C(local_level_list) + l1 + l2 + dropout + frac',
             data=df_red).fit()
table = sm.stats.anova_lm(lm, typ=2, robust="hc3") # Type 2 ANOVA DataFrame
print(lm.summary())
print(table.to_latex())
table

# %%
df.keys()

# %%
import matplotlib.pyplot as plt
num = df_red.assign(use_wavelet=df_red.use_wavelet.astype(int))
pd.plotting.scatter_matrix(num, alpha=0.2, figsize=(32, 18), hist_kwds={"bins": 50})
None
for key in num.keys():
    if key == 'F1_val':
        continue
    plt.figure()
    pd.plotting.scatter_matrix(num[['F1_val', key]])
    # ax = num.plot.scatter(x=key, y='F1_val')
num

# %%
from IPython.display import display
display(df.query("F1_val < 0.8").describe())
display(df.query("F1_val >= 0.8").describe())
display(num.query("F1_val < 0.8").describe())
display(num.query("F1_val >= 0.8").describe())
# df.query("F1_val < 0.8").to_csv("menor.csv", index=False)
# df.query("F1_val >= 0.8").to_csv("mayor.csv", index=False)

# %%
lm.params.sort_values(ascending=False)
# pd.get_dummies(df_red, columns=["global_level_list", "local_level_list"])
# df_red.explode('global_level_list').explode('local_level_list')
df_red
from sklearn.preprocessing import MultiLabelBinarizer
mlb = MultiLabelBinarizer(sparse_output=True)
mlb = MultiLabelBinarizer()
df_red_oh = df_red.copy()
df_red_oh = df_red_oh.join(pd.DataFrame(mlb.fit_transform(df_red_oh.pop('global_level_list')),
                          columns=[f"global_{x}" for x in mlb.classes_],
                          index=df_red_oh.index))

df_red_oh = df_red_oh.join(pd.DataFrame(mlb.fit_transform(df_red_oh.pop('local_level_list')),
                          columns=[f"local_{x}" for x in mlb.classes_],
                          index=df_red_oh.index))
df_red_oh

categorical = False
if categorical:
    lm = ols(f'F1_val ~ {"+".join([f"C({k})" for k in df_red_oh.keys() if k.startswith("global")])} + \
        {"+".join([f"C({k})" for k in df_red_oh.keys() if k.startswith("local")])}',
             data=df_red_oh).fit()

else:
    lm = ols(f'F1_val ~ {"+".join([f"{k}" for k in df_red_oh.keys() if k.startswith("global")])} + \
        {"+".join([f"{k}" for k in df_red_oh.keys() if k.startswith("local")])}',
             data=df_red_oh).fit()
table = sm.stats.anova_lm(lm, typ=2, robust="hc3") # Type 2 ANOVA DataFrame
print(lm.summary())
table



# %%
df = pd.read_csv('all_data_2024-07-17/20240819-190659/all.csv')
print("\n".join([f"{k} = {v if k != 'wavelet_family' and not pd.isna(v) else repr(v) if k =='wavelet_family' else 'None'}" for k, v in df.sort_values(by="F1_val", ascending=False).iloc[0, 0:22].items()]))

# %%
import statsmodels.api as sm
from statsmodels.formula.api import ols
import pandas as pd
numeric_level_list = True
df = pd.read_csv('all_data_2024-07-17/20240819-190659/all.csv')
df['global_level_list'] = df['global_level_list'].apply(eval)
df['local_level_list'] = df['local_level_list'].apply(eval)
if numeric_level_list:
    df['global_level_list'] = df['global_level_list'].apply(lambda row: sum([2**i for i in row]))
    df['local_level_list'] = df['local_level_list'].apply(lambda row: sum([2**i for i in row]))
df_red = df.loc[:, (df != df.iloc[0]).any()][['use_wavelet',  'global_level_list', 'local_level_list', 'l1',
       'l2', 'dropout', 'frac',  'F1', 'F1_val', ]]

if numeric_level_list:
    lm = ols('F1_val ~ global_level_list + local_level_list + l1 + l2 + dropout + frac',
             data=df_red).fit()
else:
    lm = ols('F1_val ~ C(use_wavelet) + C(global_level_list) + C(local_level_list) + l1 + l2 + dropout + frac',
             data=df_red).fit()
table = sm.stats.anova_lm(lm, typ=2, robust="hc3") # Type 2 ANOVA DataFrame
table

# %%
from scipy.stats import f_oneway
df = pd.read_csv('all_data_2024-07-17/20240823-232405/all.csv')
f_oneway(df.query("use_wavelet").F1_val.dropna().to_numpy(), df.query("~use_wavelet").F1_val.dropna().to_numpy())
df.query("use_wavelet").F1_val.dropna().describe(), df.query("~use_wavelet").F1_val.dropna().describe()
