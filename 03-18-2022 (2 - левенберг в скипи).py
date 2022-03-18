# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 11:36:45 2022

@author: Алиса
"""

#левенберг и всякие такие методы из домашки 3 уже запроганы в скипи, ща покажем
from scipy.optimize import least_squares
from scipy.integrate import quad
import numpy as np
def d(z, H0, omega): #функция d из второго уравнения
    #quad(function,  нижний предел, верхний предел)
    i = np.empty_like(z)
    for q in range(z.size):
        i[q] = quad(lambda x: 1.0/np.sqrt((1-omega)*(1+x)**3+omega), 0, z[q])[0] #интеграл
    return 3e11 * (1+z) * i * 1.0 / H0

def f(z, H0, omega): #функция мю из первого уравнения
    # print(z)
    # print(H0)
    # print(omega)
    # print("next")
    return 5.0 * np.log10(d(z, H0, omega)) - 5 #тут под логарифмом вылезает отрицательное что-то и все брокается

def j(z, H0, omega): #якобиан
    jac = np.empty((z.size, 2), dtype=np.float)
    jac[:, 0] = - 5.0 / (np.log(10)*H0) #производная по Н0
    for q in range(z.size): #производная по omega
        i = quad(lambda xa: 1/np.sqrt((1-omega)*(1+xa)**3+omega), 0, z[q])[0] #исх инт
        di = quad(lambda s: -(1-(s+1)**3)/(2.0*(omega+(1-omega)*(s+1)**3)**(3/2)), 0, z[q])[0]
        #print('a', i, 'b', di)
        jac[q, 1] = 5.0 * di / i
    return jac
a = np.genfromtxt('jla_mub.txt', delimiter = ' ', names = True) #считали текст

z = a['z']
mu = a['mu']
r = least_squares(fun=lambda args: f(z, *args) - mu, x0=(50, 0.5), jac = lambda args: j(z, *args), xtol=1e-6, method = 'lm')
print(r.x) #то что должно было выйти ура

#другая минимизация? решаем не мнк а минимизацию
from scipy.optimize import minimize
def F(z, H0, Omega):
    res = f(z, H0, Omega) - mu
    val = 0.5 * np.inner(res, res)
    return val

def grad(z, H0, Omega):
    res = f(z, H0, Omega) - mu
    jac = j(z, H0, Omega)
    val = np.dot((jac.T), res)
    return val

r = minimize(fun=lambda args: F(z, *args), x0=(50, 0.5), method='trust-constr',
             jac=lambda args: grad(z, *args), bounds=((0,200),(0,1)), tol=1e-6)
print(r.x)
    
    


