# -*- coding: utf-8 -*-


import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('max_colwidth', 100)



def process_data(data):
    data["iforests_abn"] = np.nan
    zero_index = data.loc[:, "experts_99"] == 0
    one_index = data.loc[:, "experts_99"] == 1
    data.loc[zero_index, 'iforests_abn'] = 0
    data.loc[one_index, 'iforests_abn'] = 2

    # 统计experts中异常数据，进行iforests训练
    coulmns_yd = ['total_byte', 'client_tcp_fin_packet', 'server_tcp_fin_packet', 'client_max_ack_delay',
                  'server_max_ack_delay', 'client_avg_ack_delay', 'server_avg_ack_delay', 'slowconnect',
                  'tcp_transaction_worse_count', 'tcp_transaction_max_rtt']

    index = data["iforests_abn"] == 2
    # 生成待入模的特征标签
    X_yd = data.loc[index, coulmns_yd]
    # 生成特定参数的训练器
    iforest = IsolationForest(n_estimators=20, max_samples="auto", max_features=10, contamination=0.10)
    # 用X_yd的数据训练训练器
    iforest.fit(X_yd)
    # 用训练好的训练器预测X_yd中的数据看是否有异常样本，如果结果为1是正常样本，结果为-1是异常样本
    y_pred_yd = iforest.predict(X_yd)

    # data['y_pred_yd'] = np.nan
    data.loc[index, 'iforests_abn'] = y_pred_yd
    ret = data.loc[:, 'iforests_abn']
    print(ret)

def main():
    data = pd.read_csv('./data.csv', encoding='utf-8')

    process_data(data)


if __name__ == '__main__':
    main()
