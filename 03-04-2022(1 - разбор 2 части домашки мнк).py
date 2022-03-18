# -*- coding: utf-8 -*-
"""
Created on Fri Mar  4 11:16:59 2022

@author: Student
"""

#разбор 2 части домашки про МНК?
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
with fits.open('ccd.fits.gz') as hdu:
    data = hdu[0].data.astype(np.float) #важно перевести во флоат
    print(data.shape)
bias = np.mean(data[0], axis=0) #смещение?
flux = np.mean(data-bias, axis=(-3,-2,-1)) #вычитаем смещение и усредняем? я хз
flux_var = np.var(np.diff(data, axis=1), axis=(-2,-1)) #выборочная дисперсия? diff это разность элементов массива?
plt.plot(flux, flux_var, '*') #по одной оси сигма х выборочная дисперсия, по другой хз
A = np.vstack([flux, np.ones_like(flux)]).T
b = flux_var.reshape(-1) #на всякий случай решейп в одномерный массив если вдруг че не так было
x, res, *_ = np.linalg.lstsq(A, b, rcond=None)
g = 2 / x[0]
ron = np.sqrt(x[1] / 2.0 * g**2)
print(g, ron)


