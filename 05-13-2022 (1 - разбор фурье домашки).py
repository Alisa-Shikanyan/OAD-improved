# -*- coding: utf-8 -*-
"""
Created on Fri May 13 11:08:23 2022

@author: Алиса
"""

#сем 13.05
#разбор говнодомашки номер 5 

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.ndimage import rotate

data = fits.open('speckledata.fits')[2].data
dmean = data.mean(axis=0)

#plt.imshow(dmean)

spec = np.fft.fftshift((np.abs(np.fft.ifft2(data))**2).mean(axis=0)) #я возьму обратное преобразование ну там числа просто поменьше будут

#plt.imshow(spec, vmax = np.quantile(spec, 0.95))

nx, ny = np.indices(data.shape[1:]) #какие-то квадратные массивы я хз
rr = np.hypot(nx - nx[nx.shape[0]//2, nx.shape[1]//2], ny - ny[ny.shape[0]//2, ny.shape[1]//2])
mask = rr < 50

maspec = np.ma.masked_array(spec, mask=mask)
#plt.imshow(maspec)

spec -= maspec.mean()

rotaver = np.stack([rotate(spec, x, reshape=False) for x in np.linspace(0, 180, 360)]).mean(axis=0)

#plt.imshow(rotaver,vmax=np.quantile(rotaver, 0.95))

spec /= rotaver
maspec = np.ma.masked_array(spec, mask=~mask) #~ это операция not над бинарным массивом
#plt.imshow(maspec)

acf = np.abs(np.fft.fft2(maspec.filled(0)))
plt.imshow(np.fft.fftshift(acf))
plt.xlim(75,125)
plt.ylim(75,125)

#крутяк



