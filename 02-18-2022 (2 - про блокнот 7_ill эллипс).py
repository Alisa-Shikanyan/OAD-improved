# -*- coding: utf-8 -*-
"""
Created on Fri Feb 18 12:02:04 2022

@author: Student
"""

#что-то про блокнот 7_ill_posed

import numpy as np
def lstsq_svd(a, b, rcond=None):
    a = np.atleast_2d(a)
    b = np.atleast_1d(b)
    u, s, vh = np.linalg.svd(a, full_matrices=False)
    if rcond is None:
        where = (s != 0.0)
    else:
        where = s > s[0] * rcond
    x = vh.T @ np.divide(u.T[:s.shape[0],:] @ b, s, out=np.zeros(a.shape[1]), where=where)
    r = a @ x - b
    cost = np.inner(r, r)
    sigma0 = cost / (b.shape[0] - x.shape[0])
    var = vh.T @ np.diag(s**(-2)) @ vh * sigma0
    
    return x, cost, var

def generate_ellipse(phi, width, height, size): #функция которая генерит точки эллипса (с шумами?)
    x = np.vstack([width / 2 * np.cos(phi), height / 2 * np.sin(phi)]).T
    X = x + np.random.normal(0.0, 0.0125, (size,*x.shape))
    return X
def fit_ellipse(x, rcond=None): #мы хз что это эллипс поэтому просто чето для кривой второго рода
    A = np.hstack([x**2,x.prod(axis=1).reshape(-1,1),x]) #A матрица с нашими данными
    cond = np.linalg.cond(A) # чето про число обусловленности
    ret = lstsq_svd(A, np.ones(x.shape[0]), rcond=rcond)
    return ret + (cond,) #ret это массив, cond - число и мы фигачим скобки чтобы как быдобавить число к массиву

width, height = 1.25, 1.25

lim = 5.0 / 180.0 * np.pi
phi = np.linspace(-lim, lim, 4)
x = generate_ellipse(phi, width, height, 4)
params, cost, cov, cond = fit_ellipse(x.reshape(-1,2))

#хз чекни блокнот там все написано
"""
когда фитили просто кривой второго рода получилась херня
In[15]: теперь поставим в фит параметр rcond = 0.01 то есть выкидываются все сингулярные значения которые меньше чем 0.01 от первого
получилась все равно херня потому что наверное мы выкинули слагаемые и это стало прямой
зато если перегенерить данные то прямая остается примерно такой же хз
если сгенерить четыре части данных (то есть на круге в 12-9-6-3 часах, а не только в 3 как было) - то все круто (один из методов)

еще один метод - замена переменных - описан в блокноте
"""

