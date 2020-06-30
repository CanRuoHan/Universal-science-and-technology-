import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
import warnings
warnings.filterwarnings("ignore")  #消除警告

def iforests_model(data,columns):
    columns_yd=columns[:5]
    ifore=IsolationForest(n_estimators=100,max_samples="auto",contamination=0.1)
    X_yd=data[columns_yd]
    ifore.fit(X_yd)
    y_pred_yd=ifore.predict(X_yd)
    data["forest"]=np.array(y_pred_yd)
    print(data.shape)
    # data.to_csv("res_data.csv",index=False,mode="a")

    est_dict, est_values, = {}, []
    sort_dict, sort_values = {}, []
    count = 0
    for col, tree in zip(ifore.estimators_features_, ifore.estimators_):
        count += 1
        for i, j, k in zip(col, tree.feature_importances_, columns_yd):
            est_dict.update({str(i).replace(str(i), k): float('%.8f' % (j))})  # 存放每一个维度对应的feature
        est_values.append(est_dict)
        sort_res = sorted(est_dict.items(), key=lambda d: d[1], reverse=True)
        print("这是第【{}】棵树，按照其特征从大到小排列：\n【{}】".format(count, sort_res))
    est_res = pd.DataFrame(est_values)
    mean = pd.DataFrame(est_res.loc[:, est_res.columns].mean()).rename(columns={0: "mean"}).sort_values(by="mean",
              axis=0,ascending=False)
    print("每个特征名称的均值是：\n【{}】".format(mean))

    """
    需求：将对应的平均数mean,替换结果为-1，所对应的原来的学习数据，五个字段的值。
    """

def load_data():
    data = pd.read_csv("./iforests_data_1.csv", encoding="ansi")
    print("原始数据形状是",data.shape)
    columns=data.columns[8:]
    iforests_model(data, columns)
load_data()
