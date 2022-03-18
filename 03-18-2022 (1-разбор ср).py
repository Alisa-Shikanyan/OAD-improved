# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 11:11:07 2022

@author: Алиса
"""

#разбор самостоятельной
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#в блокноте про пандас с сайта буквально первая строчка как вводить данные
names = ("id", "n", "Na", "Mg", "Al", "Si", "K", "Ca", "Ba", "Fe", "Type")
data = pd.read_csv('glass.csv', header=None, names=names, index_col=0)
#print(data.describe())
print(data['Mg'].max(), data['Mg'].mean())
print(np.sort(data['Mg'].unique())[1]) #это типа юник-уникальные знач, отсортировали и выбрали первое то есть не ноль

for i in names[2:-1]:
    print(i)
    print(data[i].max(), data[i].mean())
    min_val = np.sort(data[i].unique())[0] if np.sort(data[i].unique())[0] != 0.0 else np.sort(data[i].unique())[1]
    print(min_val)
    #в словарь запишем сами
print(data['n'].max(), data['n'].min()) #показатель преломления
fig = plt.figure(figsize=(5,20))
for j, i in enumerate(names[2:-1]): #enumerate добавляет порядковый номер который нужен сабплоту
    ax = fig.add_subplot(8, 1, j+1)
    ax.plot(data[i], data['n'], 'x')
    ax.set_title(i)
    #ну ваще нам в ср не нужен был сабплот, нам просто сохранить надо было

#ок, про мнк
y = np.asarray(data['n'])
A = np.ones(len(y))
for j, i in enumerate(names[2:-1]):
    A = np.vstack((A, data[i]))
print(A) #прекрасная матрица А где у нас единички и все другие данные
res = np.linalg.lstsq(A.T, y, rcond=None) #чето про мнк
#print(res)
y_pred = A.T @ res[0]
plt.plot(y, y_pred, "*") #вот это короче то что реальные данные от пофиченных но тут надо нормально построить в новый график


