import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
import warnings
warnings.filterwarnings("ignore")  #消除警告

def coacat_columns_values(series):
    string = '—'.join(map(str, series))
    return string

def iforests_model(data,columns):
    columns_yd=columns[:5]
    ifore=IsolationForest(n_estimators=100,max_samples="auto",contamination=0.1)
    X_yd=data[columns_yd]
    ifore.fit(X_yd)
    y_pred_yd=ifore.predict(X_yd)
    data["forest"]=np.array(y_pred_yd)
    print(data.shape)
    # data.to_csv("res_data.csv",index=False,mode="a")

    print("finish>>>>>>>>>")

def load_data():
    data = pd.read_csv("./iforests_data_1.csv", encoding="ansi")
    print("原始数据形状是",data.shape)
    columns=data.columns[8:]
    iforests_model(data, columns)
load_data()