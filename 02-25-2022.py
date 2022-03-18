# -*- coding: utf-8 -*-
"""
Created on Fri Feb 25 11:14:49 2022

@author: Student

"""
#чето когда нет обратной матрицы хз (J.T * J)^-1 = J.T * J +lambda(k)*I, 
#where lambda(k)=lambda(k-1), lambda(k)=lambda(k-1)/nu, lambda(k)=lambda(k-1)*omega типа смотрим какое уменьшает производную и его юзаем?

import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple

def f(t, a, b, c): #просто придумали функцию
    return b * np.exp(-a*t) * t**2 + c
def j(t, a, b, c):
    jac = np.empty((t.size, 3), dtype=np.float)
    jac[:, 0] = -b * t**3 * np.exp(-a * t) #производная по а
    jac[:, 1] = np.exp(-a*t) * t**2 #производная по b
    jac[:, 2] = 1.0 
    return jac

a = 0.75
b = 2
c = 0.5
x = (a, b, c)
n = 30
t = np.linspace(0, 10, n)
y = f(t, *x) + np.random.normal(0, 0.1, n) #шумы добавили еще
#plt.plot(t, y, '*')

Result = namedtuple('Result',('nfev', 'cost', 'grandnorm', 'x')) #число итераций, ошибкка, значение градиента, х
#t известно, хотим как-то определить a, b, c, alpha размер шага, tol на обработку ошибок типа значения не меняются
def grad_descent(y, f, j, x0, alpha = 0.01, tol=1e-4, max_iter=1000): #maxiter чтобы если вдруг чето ломается чтобы остановилсь
    x = np.asarray(x0, dtype=np.float)
    i = 0 #счетчик
    cost = []
    while True:
        i += 1
        res = f(*x) - y
        cost.append(0.5 * np.dot(res, res))
        jac = j(*x)
        g = np.dot(jac.T, res)
        g_norm = np.linalg.norm(g)
        delta_x = g / g_norm * alpha
        x = x - delta_x
        if i > max_iter: #условия остановки
            break
        if len(cost) > 2 and np.abs(cost[-1] - cost[-2]) < tol * cost[-1]: #хз чето про ошибку
            break
    cost = np.array(cost)
    return Result(nfev=i, cost=cost, grandnorm=np.linalg.norm(g), x=x)

r = grad_descent(y, 
                 lambda *args: f(t, *args),
                 lambda *args: j(t, *args),
                 (1, 1, 1),
                 alpha=0.01,
                 tol=1e-5,
                 max_iter=10000)
print(x)
#print(r)
print(r.nfev, "dd", r.x)
#r.x должны быть близки к х вообще-то...
"""
plt.plot(t, y, '*', label = 'data')
plt.plot(t, f(t, *r.x), label = 'fit')
plt.legend()
"""
plt.plot(r.cost)
#ну этот график в общем фитить должен


#поменяем функцию чтобы она работала Гауссом-Ньютоном
def gauss_newton(y, f, j, x0, k = 1, tol=1e-4, max_iter=1000): #maxiter чтобы если вдруг чето ломается чтобы остановилсь
    x = np.asarray(x0, dtype=np.float)
    i = 0 #счетчик
    cost = []
    while True:
        i += 1
        res = y - f(*x)
        cost.append(0.5 * np.dot(res, res))
        jac = j(*x)
        g = np.dot(jac.T, res)
        #g_norm = np.linalg.norm(g)
        delta_x = np.linalg.solve(np.dot(jac.T, jac), g) #это (J.T*J)^-1*g или J.T*J*x=g как раз
        x = x + k * delta_x
        if i > max_iter: #условия остановки
            break
        if np.linalg.norm(delta_x) <= tol * np.linalg.norm(x):
            break
    cost = np.array(cost)
    return Result(nfev=i, cost=cost, grandnorm=np.linalg.norm(g), x=x)
r = gauss_newton(y, 
                 lambda *args: f(t, *args),
                 lambda *args: j(t, *args),
                 (1, 1, 1),
                 k=0.1,
                 tol=1e-4,
                 max_iter=10000)
print(x)
print(r.nfev, 'aa', r.x)

"""
домаха
z = (lambda-lambda0)/lambda ну типа лямбда длина волны, лямбда0 то какая должна была быть
знаем mu и z, нам надо вытянуть H0 и Omega
для интегралов
from scipy.integrate import quad
quad(function,  нижний предел, верхний предел)
он написал quad(lambda x: x, 0, 1)
"""


