# -*- coding: utf-8 -*-
"""
Created on Fri May 20 11:14:18 2022

@author: Алиса
"""

#семинар 20.05
#чето там деревья лес кайфарик

import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from collections import namedtuple

Leaf = namedtuple('Leaf',('value')) #зададим классы для листов и узлов
#if регрессор, в листе среднее значение того что попало в лист я хз
#(то есть нам дома надо написать почти то же самое только среднее)
#если не регрессор а чето другое (классификатор?), то вроде выбирает самое популярное значение
Node = namedtuple('Node', ('feature', 'value', 'impurity', 'left', 'right')) #left right это левые и правые деревья которые образовались

#напишем класс
class BaseDecisionTree:
    def __init__(self, x, y, max_depth=np.inf):
        self.x = np.atleast_2d(x)
        self.y = np.atleast_1d(y)
        self.max_depth = max_depth
        
        self.features = x.shape[1]
        
        self.depth = 0
        
        #начинаем строить дерево с корневой вершины
        self.root = self.build_tree(self.x, self.y)
    
    def build_tree(self, x, y, depth=1): #depth=1 передаем туда текущую глубину
        if depth > self.max_depth or self.criteria(y) < 1e-6: #критерий остановки
            if depth > self.depth: #это мы тут хотели глубину посчитать
                self.depth = depth
            return Leaf(self.leaf_value(y))
        
        feature, value, impurity = self.find_best_split(x, y) #находим параметр разбиения?
        
        left_xy, right_xy = self.partition(x, y, feature, value) #само разбиение
        left = self.build_tree(*left_xy, depth=depth+1)
        right = self.build_tree(*right_xy, depth=depth+1)
        
        return Node(feature, value, impurity, left, right)
    
    def leaf_value(self, y):
        raise NotImplementedError #чето если забудем что-то объявить то получим такой error
        
    def partition(self, x, y, feature, value): #разбиение
        i = x[:, feature] >= value #i - массив индексов
        j = np.logical_not(i) #массив чето в другую сторону
        return (x[j], y[j]), (x[i], y[i])
    
    def find_best_split(self, x, y): #номер столбца по которому осуществялем разбиение, пороговое значение и текущее значение
        
        best_feature, best_value, best_impurity = 0, x[0,0], np.inf #начальные значения
        for feature in range(self.features): #features это кстати параметры эксперимента (?)
            if x.shape[0] > 2:
                x_interval = np.sort(x[:, feature])
                res = optimize.minimize_scalar(self.impurity_partition, args=(feature, x, y),
                                               bounds=(x_interval[1], x_interval[-1]),
                                               method='Bounded')
                assert res.success
                value = res.x
                impurity = res.fun
            else:
                value = np.max(x[:, feature])
                impurity = self.impurity_partition(value, feature, x, y)
            if impurity < best_impurity:
                best_feature, best_value, best_impurity = feature, value, impurity
            
            return best_feature, best_value, best_impurity
    
    def impurity_partition(self, value, feature, x, y):
        (_, left), (_, right) = self.partition(x, y, feature, value)
        return self.impurity(left, right)
    
    def impurity(self, left, right):
        raise NotImplementedError #чето потому что они разные для классификатора и регрессора поэтому напишем потом
    
    def criteria(self, y):
        raise NotImplementedError
   
    def predict(self, x):
        x = np.atleast_2d(x)
        y = np.empty(x.shape[0], dtype=self.y.dtype)
        for i, row in enumerate(x):
            node = self.root
            while not isinstance(node, Leaf):
                if row[node.feature] >= node.value:
                    node = node.right
                else:
                    node = node.left
            y[i] = node.value
        return y


class DecisionTreeClassifier(BaseDecisionTree): #чето наследуем, то есть доступны все функции из предыдущего класса
    def __init__(self, x, y, *args, random_state=None, **kwargs):
        y = np.asarray(y, dtype=int)
        self.random_state = np.random.RandomState(random_state)
        self.classes = np.unique(y)
        super().__init__(x, y, *args, **kwargs) #super значит здесь инициализируем родительский класс
    
    def impurity(self, left, right):
        h_l = self.criteria(left)
        h_r = self.criteria(right)
        return (left.size * h_l + right.size * h_r) / (left.size + right.size)
    
    def criteria(self, y):
        p = np.sum(y == self.classes.reshape(-1, 1), axis=1) / y.size
        return np.sum(p * (1 - p))
    
    def leaf_value(self, y):
        class_counts = np.sum(y == self.classes.reshape(-1, 1), axis=1) #конструкция покажет сколько значений каждого класса тут присутствует
        m = np.max(class_counts) #номер макс значения
        most_common = self.classes[class_counts==m]
        if most_common.size == 1:
            return most_common[0]
        #необязательно else так как выход из функции все равно по return
        return self.random_state.choice(most_common)
        
#btw в папке seminars написано decision_tree там чето похожее
#np.std надо возвести в квадрат чтобы получить дисперсию

#надо еще добавить функцию predict !!! из файла где seminars (UPD: я добавила)
        
#обучающий где-то 80% данных



#сем 27.05
COLORS = np.array([[1., 0., 0.], [0., 0., 1.]])

n = 100
rs = np.random.RandomState(0)
x = rs.normal(0, 1, size=(n,2))
y = x[:, 0] > 0 
#plt.scatter(*x.T, color=COLORS[np.array(y, dtype=int)], s=10) #как должно быть

'''
#обучим наше дерево
dtc = DecisionTreeClassifier(x, y)
plt.scatter(*x.T, color=COLORS[dtc.predict(x)], s=10) #то что надо, ровно как в строчке выше
plt.vlines(0, x[:,1].min(), x[:,1].max(), 'k')
'''

'''
dtc = DecisionTreeClassifier(x, y)
n = 100
rs = np.random.RandomState(0)
x = rs.normal(0, 1, size=(n,2))
y = x[:, 1] > 2 * x[:, 0]
#plt.scatter(*x.T, color=COLORS[np.array(y, dtype=int)], s=10) #как должно быть
plt.scatter(*x.T, color=COLORS[dtc.predict(x)], s=10) #класс, то же что и выше
x0 = np.linspace(x[:,0].min(), x[:,0].max(), 100)
plt.plot(x0, 2*x0, 'k')
'''

'''
#вроде бы добавим шум
x1 = rs.normal(0, 1, size=(n, 2))
plt.scatter(*x.T, color=COLORS[dtc.predict(x1)], s=30, marker='v') #вроде какой-то кал, должны были добавиться точки (апд: поправила, может будет не кал хз)
'''

'''
n = 100
r = 1
rs = np.random.RandomState(0)
x = rs.normal(0, 1, size=(n,2))
y = (x[:,0]**2 + x[:,1]**2) < r**2

dtc = DecisionTreeClassifier(x, y)
plt.scatter(*x.T, color=COLORS[dtc.predict(x)], s=10)
circ = plt.Circle(xy=(0,0), radius=r, facecolor=(0,0,0,0), edgecolor=(0.,0.,0.,1.))
plt.axes().add_artist(circ)
#вроде рисует что надо

x_grid = np.linspace(-3, 3, 100)
yy, xx = np.meshgrid(x_grid, x_grid) #хотим сетку 100 на 100? я хз
xx_test = np.stack((xx, yy), axis=-1).reshape(-1,2)
pred = dtc.predict(xx_test).reshape(xx.shape) #предсказываем на этом массиве
plt.contourf(xx, yy, pred, cmap='pink_r', alpha=0.5) #чето нам нужен только контур где меняются метки?
#вроде у меня опять кал построился, должно быть почти закрашено внутри круга
'''

n = 1000
rs = np.random.RandomState(0)
x = rs.normal(0, 1, size=(n,2))
y = x[:, 0] > 0
rand_ind = np.random.choice(y.shape[0], int(0.1*y.shape[0]))
y[rand_ind] = np.logical_not(y[rand_ind])
y = np.array(y, dtype=int)

dtc = DecisionTreeClassifier(x, y) #он сюда иногда добавляет ,max_depth=число, чтобы чето дерево не переобучалось
plt.scatter(*x.T, color=COLORS[dtc.predict(x)], s=10) #вроде норм картинка
plt.vlines(0, x[:,1].min(), x[:,1].max(), 'k')




























