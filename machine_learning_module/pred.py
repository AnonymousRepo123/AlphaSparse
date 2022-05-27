# 这里采用不同的方法处理
import utils
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
from joblib import load

# 将模型和预处理方法都从文件中读出来
scaler = load("trained_scaler.bin")
reg = load("trained_model.m")

new_x_float = np.array([[1.000000,4.000000,2048.000000,512.000000,0.000000,32.000000,1.000000,32.000000,1.000000,172.000000,128.000000,1.000000]])

for i in range(5):
    new_x_float[0][-1] = pow(2, i)
    new_x_float_std = scaler.transform(new_x_float)    
    y = reg.predict(new_x_float_std)
    print(y)