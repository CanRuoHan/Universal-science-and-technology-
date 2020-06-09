# -*- coding: utf-8 -*-


import numpy as np
import pandas as pd


def option(series, start):
    data = np.where(series < start, series, np.nan)
    new_series = pd.Series(np.where(series < start, series, np.nan))

    new_series[len(data) +1] = 1 if np.all(data) else 0

    return new_series


df = pd.DataFrame({'key1': np.array([1, 2, 3, 6, 0, 1]),
                   'key2': np.arange(6, 12)})
print(df)

df_a = df.quantile(q=0.9)
print(type(df_a))
print(df_a)

col_list = ['key1', 'key2']
new_col = ['key1', 'key2', 'label']
df['label'] = np.nan
df.loc[:, new_col] = df.loc[:, col_list].apply(func=option, axis=1, args=(df_a,)).values

print("=========================================")
print(df)
