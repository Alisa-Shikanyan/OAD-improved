# -*- coding: utf-8 -*-
"""
Created on Fri May 13 11:53:20 2022

@author: Алиса
"""

#13.05
#задачка на развитие чего-то там фурье


import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

data = fits.open('domecam.fits.gz')[0].data
#plt.imshow(data[1], cmap='Greys')

data_demean = data - data.mean()

data_fft = np.fft.fft2(data_demean)

lag = 4 #чето там сдвиг в 4 картинки (в 400 миллисекунд?)
acf = np.fft.fftshift(np.fft.ifft2(np.mean(data[:-lag] * np.conj(data_fft[lag:]), axis=0))) #в одном выкинули последние 4 элемента в другом первые - из=за этого сдвиг
acf /= np.prod(acf.shape) #нормировка
acf_extent = np.array([-data_demean.shape[2]//2, data_demean.shape[2]//2, -data_demean.shape[1]//2, data_demean.shape[1]//2]) #что-то там оси, равномерное распределение по пикселям

plt.imshow(np.abs(acf), extent=acf_extent)
plt.axis('equal')
plt.colorbar()
#чето красиво конечно но не то

