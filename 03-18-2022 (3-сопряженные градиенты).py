# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 12:01:32 2022

@author: Алиса
"""

#про метод сопряженных градиентов
#щас все запрогаем
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt

def conj_grad(c, Hfun, init_x): #алгос просто из пдфки
    init_x = np.atleast_1d(init_x)
    c = np.atleast_1d(c)
    initial_grad = c + Hfun(init_x)
    x = init_x
    g = initial_grad
    g_norm2 = np.dot(g, g)
    p = -g
    for i in range(init_x.shape[0] * 2):
        Hp = Hfun(p)
        alpha = g_norm2 / np.dot(p, Hp)
        x = x + alpha * p
        g = g + alpha * Hp
        next_g_norm2 = np.dot(g, g)
        beta = next_g_norm2 / g_norm2
        g_norm2 = next_g_norm2
        p = -g + beta * p
    return x

with fits.open('ccd.fits') as hdu:
    data = hdu[0].data.astype(np.float32)
    print(data.shape)

bias = np.mean(data[0], axis = 0)
flux = np.mean(data - bias, axis=(-3,-2,-1))
flux_var = np.var(np.diff(data, axis = 1), axis=(-2, -1))
plt.plot(flux, flux_var)
A = np.vstack([flux, np.ones_like(flux)]).T
b = flux_var.reshape(-1)
H = A.T @ A
x = conj_grad(-A.T @ b, lambda vec: H @ vec, np.zeros(A.shape[1]))
print(x) 
gain = 2.0 / x[0]
ron = np.sqrt(x[1] / 2 * gain**2)
print(gain, ron)

    
