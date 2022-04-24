# -*- coding: utf-8 -*-
"""
Created on Fri Apr 15 11:16:28 2022

@author: Алиса
"""

#модель арима: сумма авторегрессионной и скользящего среднего хз
import pandas as pd
import numpy as np
import statsmodels
import statsmodels.api as sm
import matplotlib
import matplotlib.pyplot as plt
import json

with open('sun_indices_spot.json', encoding="utf-8-sig") as f:
    x = json.load(f)
# print(x.keys()) - чтобы чекнуть какие названия
data = x['data']
#"щас будем делать пандасовский датафрейм"
time = pd.to_datetime(data['x'])
area = np.asarray(data['area'], dtype=np.float32)
wolf = np.asarray(data['wolf'], dtype=np.float32)

#один из способов сделать датафрейм - словарь
#пишут датафрейм это табличная структура данных
df = pd.DataFrame({"time":time, "area":area, "wolf":wolf})
df = df.set_index('time')
df_year = df.resample('365d').mean()

plt.figure(figsize=(15,6))
plt.plot(df.area, ".")
plt.plot(df_year.area, "-")

#щас будет какой-то новый тип графика но мы разберемся
fig, ax = plt.subplots(figsize=(15,6))
statsmodels.graphics.gofplots.qqplot(df_year.area, fit=True, line='45', ax=ax)
plt.axis('equal')
#мы хотели посмотреть насколько похожи два распределения

from scipy.stats import boxcox
from scipy.special import inv_boxcox

area_, lmbd = boxcox(np.asarray(df_year.area).reshape(-1)) #функция подбирает такой параметр лямбда когда распределние становится максимально близко к нормальному
fig, ax = plt.subplots(figsize=(15,6))
statsmodels.graphics.gofplots.qqplot(area_, fit=True, line='45', ax=ax)
plt.axis('equal') #видно что график лучше ложится

#чето смотрим автокорреляционную функцию для ряда с нулевым средним
fig, ax = plt.subplots(figsize=(15, 6))
area_mean_ = np.mean(area_)
area_ = area_ - area_mean_
sm.graphics.tsa.plot_acf(area_, ax=ax, lags=50, alpha=0.05) #alpha описывает доверительный интервал

#смотрим две модели
#работает для статсмоделс больше 12 версий

N_forecast = 30
model = sm.tsa.arima.ARIMA(area_, order=(3,0,0)).fit() #в ордере порядок авторегресионной, порядок разности, порядок скользящего среднего, типа 3 поэтому ar.L3 я хз
#3 в скобке потому что типа три точки выпадают из нашей области если мы посмотрим на график автокорреляционной функции
print(model.summary())

#построим еще одну автокорреляционную функцию но уже на другую оценку
fig, ax = plt.subplots(figsize=(15,6))
sm.graphics.tsa.plot_acf(model.resid, lags=50, alpha=0.01, ax=ax)

#возьмем еще одну модельку, где мы добавим сезонность
model1 = sm.tsa.arima.ARIMA(area_, order=(1,0,0), seasonal_order=(4, 0, 0, 11)).fit()
#в seasons 4 предыдущих 11летних цикла
print(model1.summary())

fig, ax = plt.subplots(figsize=(15,6))
sm.graphics.tsa.plot_acf(model1.resid, lags=50, alpha=0.01, ax=ax)

#выведем предсказание 
forecast = model.get_forecast(N_forecast) #N_forecast показывает сколько точек надо взять
f1_mean = inv_boxcox(forecast.predicted_mean + area_mean_, lmbd) #чето там обратное преобразование?
f1_conf = inv_boxcox(forecast.conf_int(alpha=0.05) +area_mean_, lmbd)
f_time = pd.date_range(df.index.values[-1], periods=N_forecast, freq='365D') #здесь просто временную шкалу добавляем

plt.figure(figsize=(15,6))
plt.plot(df.area, '.') #изначальные данные
plt.plot(f_time, f1_mean, '.-') #вроде прогнозы - первое предсказание
plt.fill_between(f_time, *f1_conf.T, color='red', alpha=0.1) #доверительный интервал, альфа тут просто прозрачность


forecast1 = model1.get_forecast(N_forecast)
f2_mean = inv_boxcox(forecast1.predicted_mean + area_mean_, lmbd) #чето там обратное преобразование?
f2_conf = inv_boxcox(forecast1.conf_int(alpha=0.05) +area_mean_, lmbd)
plt.plot(f_time, f2_mean, '.-') #вроде прогнозы - второе предсказание
plt.fill_between(f_time, *f2_conf.T, color='green', alpha=0.1)

#вроде все работает ура
