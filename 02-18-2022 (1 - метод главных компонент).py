# -*- coding: utf-8 -*-
"""
Created on Fri Feb 18 11:13:10 2022

@author: Student
"""
#Метод главных компонент

#var из прошлого сема/домашки это вроде бы матрица ковариации
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

#сгенерируем данные
 #шум с центром в нуле и единичной дисперсией:
data = stats.multivariate_normal(np.zeros(50,)).rvs(10000) #генерит 50 нормальных распределений? чето 50 параметров в 10000 экспериментах
# шум большей амплитуды:
data = data + 100 * np.random.rand(10000, 1) + np.random.rand(10000, 1) * np.arange(25, 75) #каждому значению увеличиваем шум и чето там линейную штуку добавляем тоже зашумленную
# хотим центрировать данные - чтобы среднее было равно нулю
mean = np.mean(data, axis = 0)
data = data - mean

u, s, v = np.linalg.svd(data)
#print(u.shape) #10000 на 10000 как и должно быть, у какой-то другой размер ок 50 на 50

"""
for i in range(4):
    plt.plot(v[i, :], label=str(i))
plt.legend() #чето где-то шум я хз
"""
#я буду закомменчивать все плоты - они нормальные, но мешают

"""
#как понять сколько из свд компонент оставить? построим сингулярные числа
plt.plot(s, '*-') 
plt.yscale('log')
#видно что только первые две компоненты норм, остальные какая-то фигня - шум
"""

x = data[0, :]
decoder = v[0:2, :] #берем первые две компоненты
x_new = np.dot(decoder.T, np.dot(decoder, x))
plt.plot(x, label='origin')
plt.plot(x_new, label='decoded')
plt.legend() #origin то что было,  decoded то что мы вот сделали - отбросили 48 компонент, но чето вернулись в исходный базис и вот как красиво зафичено без шумов я хз

#обещали выложить блокнот про смайлики
"""
чето точки смайлика переводят в массив одномерный, потом соединяют все смайлики в один массив
дальше бахаем свд,чето анализируем компоненты - сингулярные значения - но (кроме последней) нельзя особо сказать что чето можно выкинуть
ну нет и нет, строим главные компоненты (обратно в формате картинок)
ок, берем сколько-то главных компонент (допустим 20), берем какой-то смайлик - прямое и обратное преобразование - и смотрим че как
соответственно если взять больше компонент (50 например) то получим уже точнее, если все 90 - то исходную картинку получим да
ну и как бы выбирай сам - качество или скорость работы алгоритма
PCA это кстати класс - но это тоже самое просто короче пишется
"""
