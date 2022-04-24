# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 12:14:52 2022

@author: Алиса
"""

#04-22-2022 (2 часть)
import numpy as np
import matplotlib.pyplot as plt
#чето про фурье в нампае

#хотим фурье-образ от прямоугольного сигнала
x = np.linspace(-100, 100, 200)
test_arr = np.zeros_like(x)
test_arr[np.abs(x)<4] = 1
#plt.plot(x, test_arr)

fft = np.fft.fft(test_arr)
#plt.plot(np.abs(fft)) #чето там не совсем то что ожидали

# x1 = np.fft.fftfreq(len(test_arr))
# plt.plot(x1, np.abs(fft)) #???

fft1 = np.fft.fftshift(fft) #короче шифт какая-то классная штука
#plt.plot(np.abs(fft1)) #вот это то что надо

#усреднение по времени = усреднение 3д массив по одной из осей
#чето среднее мощности - сначала вычисляем каждое, потом усредняем

arr = np.ones((40,40))
for i in range(arr.shape[0]):
    arr[i] = i
plt.imshow(arr)
plt.colorbar()

arr[30, 35] = 1000 #чето добавляем шум
#plt.imshow(arr) #плохо, видим только шум чето там
plt.imshow(arr, vmin=0, vmax=np.quantile(arr, 0.99)) #функция возвращает значение чето где больше 99% точек, позволяет видеть область где 99%


