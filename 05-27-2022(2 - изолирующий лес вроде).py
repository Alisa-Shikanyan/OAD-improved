# -*- coding: utf-8 -*-
"""
Created on Fri May 27 12:01:23 2022

@author: Алиса
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.ensemble import IsolationForest

data = pd.read_csv('lrs.csv', index_col=0)
x = np.asarray(data.iloc[:, 10:53])
wl = np.asarray(list(map(float, data.columns[10:53])))

"""
for i in range(x.shape[0]):
    plt.plot(wl, x[i], alpha=0.1, color='blue') #видно что какие-то прям выбиваются, их и хотим задетектить
"""


#попробуем сделать анализ главных компонент
pca = PCA(n_components=2, random_state=1)
z = pca.fit_transform(x)
# plt.plot(wl, pca.components_[0,:], '-')
# plt.plot(wl, -pca.components_[1,:], '-')
#график норм, это вроде главные компоненты я хз что это
"""
tsne = TSNE(n_components=2) #массив графиков конвертируем так что каждый график превратится в точку с двумя координатами
u = tsne.fit_transform(x)
#plt.scatter(u[:,0], u[:,1])
#найс но это нам не помогло :(
"""


#поищем с помощью изолирующего леса
ifor = IsolationForest(n_estimators=1000, contamination='auto')
pred = ifor.fit_predict(x)
scores = ifor.decision_function(x) #в некотором смысле средняя глубина каждого, вроде чем меньше тем больше вероятность аномалии? 
#plt.hist(scores)

ind = np.argsort(scores)
x_sorted = x[ind]
"""
for i in range(x.shape[0]): #все графики
    plt.plot(wl, x[i], alpha=0.025, color='blue')

for i in range(10): #первые 10 которые дерево считает максимально аномальными
    plt.plot(wl, x_sorted[i], alpha=0.25, color='red')
"""

#посмотрим как это будет в базисе главных компонент
z_sorted = pca.transform(x_sorted)
plt.scatter(z_sorted[10:,0], z_sorted[10:, 1], s=0.5, color='blue')
plt.scatter(z_sorted[:10,0], z_sorted[:10, 1], s=1.5, color='red')






