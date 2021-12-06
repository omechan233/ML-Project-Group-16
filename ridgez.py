# -*- coding: utf-8 -*-
"""RidgeZ.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1t8lq-c-MqTW-7VF-Jty-5IL682YICjlv
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd

from sklearn.linear_model import Ridge
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

dataset = pd.read_csv("cleaned_data_youtube.csv", encoding='GBK')
raw_dataset = dataset.to_numpy()

dur = raw_dataset[:,[0]]
sub = raw_dataset[:,[1]]

views = raw_dataset[:,[2]]
train_Out = views

train_data = np.column_stack((dur,sub))

print(train_data.shape)
print(train_Out.shape)

from sklearn.preprocessing import  PolynomialFeatures

mean_error = []
std_error = []
c_range = [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 0.5, 1.0]
#c_range = [0.1,0.5,1,2,3,4,5]

from sklearn.metrics import mean_squared_error
from  sklearn.model_selection import KFold

train_data = PolynomialFeatures(1,include_bias=False).fit_transform(train_data)
X_train, X_test, y_train, y_test = train_test_split(train_data, views.ravel(), test_size=0.3, random_state=1)

for c in c_range:
        model = Ridge(alpha=c)
        temp = []

        kf = KFold(n_splits = 5)
        for train,test in kf.split(train_data):
                model.fit(train_data[train], train_Out[train])
                score = model.score(X_test, y_test)
                print(score)

                Out_pred = model.predict(train_data[test])
                temp.append(mean_squared_error(train_Out[test], Out_pred))
        mean_error.append(np.array(temp).mean())
        std_error.append(np.array(temp).std())

print(mean_error)
print(std_error)

print(mean_error)

plt.errorbar(c_range, mean_error, yerr=std_error)
plt.title('Ridge Regression: Compare C Values')
plt.xlabel('C Value')
plt.ylabel('Mean square error')
plt.xlim((0, 1))

plt.show()

temp = []
mean_error = []

model = Ridge(alpha=1/(2*0.2))
model.fit(X_train, y_train)
score = model.score(X_test, y_test)
print(score)

Out_pred = model.predict(X_test).reshape(-1,1)
temp.append(mean_squared_error(y_test, Out_pred))
mean_error.append(np.array(temp).mean())

print(mean_error)

x = np.linspace(1, 559, num = 559).reshape(-1,1)

print(Out_pred.shape)
print(x.shape)

plt.title('Ridge Regression Predictions')
plt.xlabel('X')
plt.ylabel('Views')
plt.scatter(x, Out_pred, color='red')
plt.scatter(x, y_test)
plt.legend(['train', 'value'])