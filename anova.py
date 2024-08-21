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
df = pd.read_csv('all.csv')
df['global_level_list'] = df['global_level_list'].apply(eval)
df['local_level_list'] = df['local_level_list'].apply(eval)
df_red = df.loc[:, (df != df.iloc[0]).any()][['use_wavelet',  'global_level_list', 'local_level_list', 'l1',
       'l2', 'dropout', 'frac',  'F1', 'F1_val', ]]

lm = ols('F1_val ~ C(use_wavelet) + C(global_level_list) + C(local_level_list) + l1 + l2 + dropout + frac',
         data=df_red).fit()
table = sm.stats.anova_lm(lm, typ=3, robust="hc3") # Type 2 ANOVA DataFrame
table

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


lm = ols(f'F1_val ~ C(use_wavelet) + {"+".join([f"C({k})" for k in df_red_oh.keys() if k.startswith("global")])} + \
    {"+".join([f"C({k})" for k in df_red_oh.keys() if k.startswith("local")])} + l1 + l2 + dropout + frac',
         data=df_red_oh).fit()

lm = ols(f'F1_val ~ use_wavelet + {"+".join([f"{k}" for k in df_red_oh.keys() if k.startswith("global")])} + \
    {"+".join([f"{k}" for k in df_red_oh.keys() if k.startswith("local")])} + l1 + l2 + dropout + frac',
         data=df_red_oh).fit()
table = sm.stats.anova_lm(lm, typ=3, robust="hc3") # Type 2 ANOVA DataFrame
table


