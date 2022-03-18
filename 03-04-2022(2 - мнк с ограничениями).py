# -*- coding: utf-8 -*-
"""
Created on Fri Mar  4 11:36:29 2022

@author: Student
"""

#МНК с ограничениями (типа тоже минимизация ||Ax-f||^2 но теперь х>0 я хз, из распечатки)
# напишем класс......
import numpy as np
from collections import namedtuple
Result = namedtuple('Result', ('x', 'fun', 'grad'))

class NNLS(): #просто написали класс чтобы все функции были в одном месте
    def __init__(self, A, b):
        self.A = np.asarray(A)
        self.b = np.asarray(b)
        self._x = np.zeros(A.shape[1]) #x - массив с результатами, нижнее подчеркивание просто соглашение; х - нач приближение?
        self.P = np.zeros_like(self._x, dtype = bool) #массив индексов вектора х
        self.w = np.empty_like(self._x) #величина градиента
        self.recalc_w = True
        self.t = None #переменная для текущего индекса
        
    def loss(self, x): #функция потерь
        return np.linalg.norm(self.A @ x - self.b)**2
    
    def result(self):
        return Result(self.x, self.loss(self.x), self.w)
    
    def negative_grad(self, x):
        return self.A.T @ (self.b - self.A @ x)
    
    @property #про эти собачки в файле от 04.02 (это - декоратор)
    def x(self):
        return self._x
    
    @x.setter
    def x(self, v):
        self._x[:] = v
        self._x[self._x <= 0] = 0 #если значения отрицательны надо заменить на нули
        self.P[self._x == 0] = False #если какие=то компоненты стали отрицательными то и сами индексы надо убить
    
    @property
    def L(self):
        return ~self.P #во-первых, ~ это логическое не, во-вторых, он втирал что L это массив всех индексов а P только тех что больше нуля (при этом чето куда-то переносим и меняется тогда там сами массивы видимо 0 1 я хз)
    #изначально массив L содержит все индексы
    
    @staticmethod
    def argmax_index(x, idx):
        i = np.argmax(x[idx])
        return np.arange(x.size)[idx][i]
    
    def solve(self):
        if self.recalc_w: #это пункт 2 (распечатки) градиент
            self.w[:] = self.negative_grad(self.x)
        
        if np.all(self.w <= 1e-8) or np.all(self.P): #условие остановки? пункт 3
            return self.result()
        
        self.t = self.argmax_index(self.w, self.L) #ищем компоненту х для которой значение градиента наибольшее, пункт 4
        self.P[self.t] = True
        self.inner_solve()
        return self.solve()
    #мы в решении чето там ищем наибольший градиент хз: считаем градиент, выбираем все компоненты которые лежат в массиве L, компоненту с максимальным значение градиента перенесли в P, на следующем шаге рассматриваем все остальное кроме той перенесенной компоненты (типа итерационный метод)
    
    
    def inner_solve(self): #здесь решаем задачу наименьших квадратов но не с нашей начальной матрицей хз пункт 6? внутри решаем МНК обычную, но матрица А не изначальная, а имеет нули везде кроме столбцов которые есть в массиве Р
        l = np.zeros_like(self.x)
        l[self.P] = np.linalg.lstsq(self.A[:, self.P], self.b)[0]
        if l[self.t] <= 0: #если вдруг какой-то там макс(мин?) градиент оказался неположительным
            self.w[self.t] = 0
            self.recalc_w = False
            return
        else:
            self.recalc_w = True
        if np.all(l[self.P]>0):
            self.x = 1
            return
        i_l_nonpos = (l <= 0) & self.P #если какие-то коэфты l меньше нуля то надо что-то делать
        alpha = np.min(self.x[i_l_nonpos] / (self.x[i_l_nonpos] - l[i_l_nonpos])) #это мы как-то преобразовываем х если у него какие-то компоненты отрицтаельные (отрицательные становятся после преобразования нулевыми?)
        self.x = self.x + alpha * (l - self.x)
        self.inner_solve()

#теперь чето решаем??
A = np.random.rand(5, 2)
b = np.random.rand(5)
print(np.linalg.lstsq(A, b, rcond=None)[0])

n = NNLS(A, b)
print(n.solve())

