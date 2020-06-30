# -*- coding: utf-8 -*-


import pandas as pd


def col_join(series):
    if series.isnull().any():
        series.dropna()
    
    ret = ''.join(map(str, series))
    ret = ret.replace('0', '')
    return ret


def main():
    data = pd.read_csv('./concat_data.csv')
    print(data.head(10))
    columns = data.columns

    data['res'] = data.loc[:, columns].apply(func=col_join, axis=1)
    print(data.head(10))


if __name__ == '__main__':
    main()

