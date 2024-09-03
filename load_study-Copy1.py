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
df.global_level_list = df.global_level_list.fillna(lambda: []).apply(lambda x: x() if callable(x) else x)
df.local_level_list = df.local_level_list.fillna(lambda: []).apply(lambda x: x() if callable(x) else x)


df2.global_level_list = df.global_level_list.fillna(lambda: []).apply(lambda x: x() if callable(x) else x)
df2.local_level_list = df.local_level_list.fillna(lambda: []).apply(lambda x: x() if callable(x) else x)

# %%
import matplotlib.pyplot as plt
df.l2.hist(bins=50)
plt.figure()
df.query('l1 < 0.02').l1.hist(bins=40)
for x in range(1, 7):
    df[f"global_level_list_{x}"] = df[f"global_level_list"].apply(lambda row: x in row).astype(int)
df[f"no_global_wavelet"] = df[f"global_level_list"].apply(lambda row: len(row) == 0).astype(int)
df[f"global_wavelet_len"] = df[f"global_level_list"].apply(lambda row: len(row)).astype(int)
for x in range(1, 4):
    df[f"local_level_list_{x}"] = df[f"local_level_list"].apply(lambda row: x in row).astype(int)
df[f"no_local_wavelet"] = df[f"global_level_list"].apply(lambda row: len(row) == 0).astype(int)
df[f"local_wavelet_len"] = df[f"local_level_list"].apply(lambda row: len(row)).astype(int)

num2 = df.query("F1_val!=0")[ ["F1_val", "l1", "l2", "no_global_wavelet", "no_local_wavelet", "local_wavelet_len", "global_wavelet_len"] \
            + [f"global_level_list_{x}" for x in range(1, 7)] + [f"local_level_list_{x}" for x in range(1, 4)]]
num2["l2_bin"] = pd.cut(num2.l2, [0, 0.03, 1.0])
num2["l1_bin"] = pd.cut(num2.l1, [0, 0.001, 0.10])
num2["dropout_bin"] = pd.cut(num2.l1, [0, 0.10, 1.00])
num2["F1_val_bin"] = pd.cut(num2.F1_val, 100)

def plot_group_f1_per_level(num2, key="l2_bin", var_name="global_level_list_1", drop_cols=("l1", "l2", "F1_val"), plot_sum=False):
# def plot_group_f1(num2, key="l2_bin", , drop_cols=("l1", "l2", "F1_val"), plot_sum=False):
    grouped_df = num2.groupby([key, "F1_val_bin"]).agg("sum").reset_index().drop(columns=list(drop_cols))
    grouped_df[[key, "F1_val_bin", var_name]]
    plt.figure(figsize=(12, 8))
    pivot_df = grouped_df.pivot(index='F1_val_bin', columns=key, values=var_name).sort_index()
    pivot_df = pivot_df[(pivot_df[pivot_df.keys()[0]] > 0) | (pivot_df[pivot_df.keys()[1]] > 0)]
    pivot_df2 = pivot_df.apply(lambda row: row/(row[0]+row[1]), axis=1)

def plot_group_f1(num2, key="l2_bin", var_name="global_level", drop_cols=("l1", "l2", "F1_val"), plot_sum=False):
    grouped_df = num2.groupby([key, "F1_val_bin"]).agg(["sum"]).reset_index().drop(columns=list(drop_cols))
    
    melted_df = grouped_df.melt(id_vars=[key, "F1_val_bin"], var_name=var_name, value_name='Count')
    melted_df = melted_df.groupby([key, "F1_val_bin"]).sum().reset_index()
    
    pivot_df = melted_df.pivot(index='F1_val_bin', columns=key, values='Count').sort_index()
    pivot_df = pivot_df[(pivot_df[pivot_df.keys()[0]] > 0) | (pivot_df[pivot_df.keys()[1]] > 0)]
    pivot_df2 = pivot_df.apply(lambda row: row/(row[0]+row[1]), axis=1)

    figs = []
    # Create the heatmap
    if plot_sum:
        figs.append(plt.figure(figsize=(12, 8)))
        sns.heatmap(pivot_df, annot=True, cmap='YlOrRd', fmt='g')
    
    
    figs.append(plt.figure(figsize=(12, 8)))
    sns.heatmap(pivot_df2, annot=True, cmap='YlOrRd', fmt='g')
    
    plt.title('mean')
    plt.xlabel(key)
    plt.ylabel('F1_val_bin')
    
    plt.tight_layout()
    plt.show()
    if len(figs) == 1:
        return figs[0]
    else:
        return figs

fig_sum, fig_mean = plot_group_f1(num2, key="l2_bin", drop_cols=("l1", "l2", "F1_val"), var_name="global_level", plot_sum=True)
fig_sum.savefig("plot/optuna/heatmap/l2_sum.png")
fig_mean.savefig("plot/optuna/heatmap/l2_mean.png")
fig_sum, fig_mean = plot_group_f1(num2, key="l1_bin", drop_cols=("l1", "l2", "F1_val"), var_name="global_level", plot_sum=True)
fig_sum.savefig("plot/optuna/heatmap/l1_sum.png")
fig_mean.savefig("plot/optuna/heatmap/l1_mean.png")
# fig_sum, fig_mean = plot_group_f1(num2, key="dropout_bin", drop_cols=("l1", "l2", "F1_val"), var_name="global_level", plot_sum=True)


# %%
def plot_f1_per_level(num2, key="l2_bin", var_name="global_level_list_1", drop_cols=("l1", "l2", "F1_val"), agg="sum", plot_sum=False, plot_only_sum=False):
# def plot_group_f1(num2, key="l2_bin", , drop_cols=("l1", "l2", "F1_val"), plot_sum=False):
    grouped_df = num2.groupby([key, "F1_val_bin"]).agg(agg).reset_index().drop(columns=list(drop_cols))
    grouped_df[[key, "F1_val_bin", var_name]]
    pivot_df = grouped_df.pivot(index='F1_val_bin', columns=key, values=var_name).sort_index()
    if agg == "mean":
        pivot_df = pivot_df.fillna(0)
    pivot_df = pivot_df[(pivot_df[pivot_df.keys()[0]] > 0) | (pivot_df[pivot_df.keys()[1]] > 0)]
    pivot_df = pivot_df[(~(pivot_df[pivot_df.keys()[0]].isna())) | (~(pivot_df[pivot_df.keys()[1]].isna()))]
    pivot_df2 = pivot_df.apply(lambda row: row/(row[0]+row[1]), axis=1)


    figs = []
    # Create the heatmap
    if plot_sum:
        figs.append(plt.figure(figsize=(12, 8)))
        sns.heatmap(pivot_df, annot=True, cmap='YlOrRd', fmt='g')
        plt.title(f'{agg} {var_name}')
        plt.xlabel(key)
        plt.ylabel('F1_val_bin')

    if not plot_only_sum:
        if agg == "sum":
            figs.append(plt.figure(figsize=(12, 8)))
            sns.heatmap(pivot_df2, annot=True, cmap='YlOrRd', fmt='g')
            
            plt.title(f'mean {var_name}')
            plt.xlabel(key)
            plt.ylabel('F1_val_bin')
        elif agg == "mean":
            grouped_df_mean = num2.groupby([key, "F1_val_bin"]).agg("count").reset_index().drop(columns=list(drop_cols))
            pivot_df_mean = grouped_df_mean.pivot(index='F1_val_bin', columns=key, values=var_name).sort_index()
            pivot_df_mean = pivot_df_mean[(pivot_df_mean[pivot_df_mean.keys()[0]] > 0) | (pivot_df_mean[pivot_df_mean.keys()[1]] > 0)]
            pivot_df_mean = pivot_df_mean[(~(pivot_df_mean[pivot_df_mean.keys()[0]].isna())) | (~(pivot_df_mean[pivot_df_mean.keys()[1]].isna()))]
            figs.append(plt.figure(figsize=(12, 8)))
            sns.heatmap(pivot_df_mean, annot=True, cmap='YlOrRd', fmt='g')
            
            plt.title(f'count {var_name}')
            plt.xlabel(key)
            plt.ylabel('F1_val_bin')
    
    plt.tight_layout()
    plt.show()

    if len(figs) == 1:
        return figs[0]
    else:
        return figs


fig_sum, fig_mean = plot_f1_per_level(num2, key="l2_bin", var_name=f"no_global_wavelet", plot_sum=True) 
fig_sum, fig_mean = plot_f1_per_level(num2, key="l2_bin", var_name=f"no_local_wavelet", plot_sum=True) 


fig_sum, fig_mean = plot_f1_per_level(num2, key="l1_bin", var_name=f"no_global_wavelet", plot_sum=True) 
fig_sum, fig_mean = plot_f1_per_level(num2, key="l1_bin", var_name=f"no_local_wavelet", plot_sum=True) 

fig_sum = plot_f1_per_level(num2, key="l2_bin", var_name=f"global_wavelet_len", plot_sum=True, plot_only_sum=False, agg="mean") 
fig_sum = plot_f1_per_level(num2, key="l2_bin", var_name=f"local_wavelet_len", plot_sum=True, plot_only_sum=False, agg="mean") 

fig_sum = plot_f1_per_level(num2, key="l1_bin", var_name=f"global_wavelet_len", plot_sum=True, plot_only_sum=False, agg="mean") 
fig_sum = plot_f1_per_level(num2, key="l1_bin", var_name=f"local_wavelet_len", plot_sum=True, plot_only_sum=False, agg="mean") 
# num2.keys()

for x in range(1, 7):
    fig_sum, fig_mean = plot_f1_per_level(num2, key="l2_bin", var_name=f"global_level_list_{x}", plot_sum=True) 
    fig_mean.savefig(f"plot/optuna/heatmap/per_level/global_level{x}_l2_mean.png")
    fig_sum.savefig(f"plot/optuna/heatmap/per_level/global_level{x}_l2_sum.png")
    fig_sum, fig_mean = plot_f1_per_level(num2, key="l1_bin", var_name=f"global_level_list_{x}", plot_sum=True) 
    fig_mean.savefig(f"plot/optuna/heatmap/per_level/global_level{x}_l1_mean.png")
    fig_sum.savefig(f"plot/optuna/heatmap/per_level/global_level{x}_l1_sum.png")
for x in range(1, 4):
    fig_sum, fig_mean = plot_f1_per_level(num2, key="l2_bin", var_name=f"local_level_list_{x}", plot_sum=True) 
    fig_mean.savefig(f"plot/optuna/heatmap/per_level/local_level{x}_l2_mean.png")
    fig_sum.savefig(f"plot/optuna/heatmap/per_level/local_level{x}_l2_sum.png")
    fig_sum, fig_mean = plot_f1_per_level(num2, key="l1_bin", var_name=f"local_level_list_{x}", plot_sum=True) 
    fig_mean.savefig(f"plot/optuna/heatmap/per_level/local_level{x}_l1_mean.png")
    fig_sum.savefig(f"plot/optuna/heatmap/per_level/local_level{x}_l1_sum.png")

# %%
grouped_df = num2.groupby(["l2_bin", "F1_val_bin"]).agg("sum").reset_index().drop(columns=["l1", "l2", "F1_val"])
grouped_df[["l2_bin", "F1_val_bin", "global_level_list_1"]]
melted_df = grouped_df.melt(id_vars='l2_bin', var_name='global_Level', value_name='count')
# melted_df
# fig = plt.figure(figsize=(12, 12))
# ax = sns.pointplot(data=melted_df, x='l2_bin', y='Count', hue='global_level',
#               dodge=0.2,  errwidth=1.5, capsize=0.2, palette='coolwarm',
#               # errorbar=None,
#                   )


# %%
pivot_df.sort_index()
