# -*- coding: utf-8 -*-
"""
Created on Fri Jun 10 11:17:43 2022

@author: Алиса
"""

#сем 10.06 работаем с данными которые были в деревьях

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

from sklearn.ensemble import BaggingRegressor
from sklearn.decomposition import PCA

data = pd.read_csv('sdss_redshift.csv')
X = data[list('ugriz')]
y = data['redshift']

x_train, x_test, y_train, y_test = train_test_split(X, y)

#делаем случайный лес с помощью склерна
rfr = RandomForestRegressor(n_estimators=100)

#обучаем модель
rfr.fit(x_train, y_train)
#предсказываем резы
y_pred = rfr.predict(x_test)

#plt.plot(y_test, y_pred, 'o')

#напишем функцию которая на вход принимает модель машинного обучения возвращать график в заголовке которого корень из дисперсии? я хз
def fit_plot(regressor, _x_train=x_train, _x_test=x_test, _y_train=y_train, _y_test=y_test, q=0.05):
    regressor.fit(_x_train, _y_train)
    y_pred = regressor.predict(_x_test)
    res = np.square(_y_test - y_pred)
    #выкинем по 5% лучших и худших резов переменная q
    res = np.sort(res)
    idx = slice(int(q*res.size), int((1-q)*res.size))
    res = res[idx]
    std = np.sqrt(np.mean(res))
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(_y_test, y_pred)
    ax.set_title('{:.5f}'.format(std))
    ax.set_aspect('equal') #чтобы длина ширина одинаковая была

#скормим функции наш регрессор
fit_plot(rfr) #nice

#теперь делаем чето другое
br = BaggingRegressor(n_estimators=100) #в BaggingRegressor можно подать не только дерево
fit_plot(br)

pca = PCA(n_components=5)
x_train_new = pca.fit_transform(x_train)
x_test_new = pca.transform(x_test)

rfr = RandomForestRegressor()
fit_plot(rfr, _x_train=x_train_new, _x_test=x_test_new)

#градиентный бустинг: тоже деревья но обучаются по-другому, как-то на ошибках предыдущего алгоритма
#первое дерево на исходном датасете, дальше правят ошибки?? я хз
#еще здесь прек в том что все деревья неглубокие (чтобы не переобучаться)

#мы не будем писать градиентный бустинг просто юзаем библы (в склерне кстати плохо реализован)

#я знаю что не надо импортировать посередине кода но это как бы новый этап
from xgboost import XGBRegressor
xgb = XGBRegressor(n_estimators=20, max_depth=5) #ну тут параметры так себе надо подбирать
fit_plot(xgb)

import catboost
cbr = catboost.CatBoostRegressor(n_estimators=20, max_depth=5)
fit_plot(cbr)






