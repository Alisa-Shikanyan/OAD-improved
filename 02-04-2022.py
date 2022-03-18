# -*- coding: utf-8 -*-
"""
Created on Fri Feb  4 11:08:33 2022

@author: Student
"""

#январь 1979 - декабрь 2020 это отсчет времени в домашке
#тема: прикладной аспект объекто-ориентированного проганья в питоне.....

from collections import namedtuple #namedtuple какая-то надстройка на кортежи, где каждому элементу массива присвоеноимя
TuplePoint = namedtuple('Point', ['latitude', 'longitude']) #создаем новый класс - тип данных? массив который состоит из широты и долготы
x = TuplePoint(latitude=10, longitude=20)
print(x)
print(x.longitude)
#кстати Tuple неизменяемый тип данных, то есть x.longitude = 30 не сработает

#ок, создаем новый класс (Point это имя)
class Point:
    def __init__(self, latitude, longitude): #напишем что должно происходить при инициализации класса; self - ключевое слово
        self.longitude = longitude
        self.latitude = latitude #means к этим переменным можно будет напрямую обращаться
    @property #декоратор
    def latitude(self): #что должно происходить при чтении
        print('get latitude')
        return self._latitude #то есть обращаясь к latitude нам вернется вот то что тут
    
    @latitude.setter #что должно происходить когда записываем, например тут можно обработку ошибок бахнуть
    def latitude(self, new_latitude): #кстати если перед new ... бахнуть * то это массив
        self._latitude = new_latitude
        print('set latitude')
    
    @property
    def longitude(self):
        return self._longitude
    
    @longitude.setter
    def longitude(self, new_longitude):
        self._longitude = new_longitude
        print('set longitude')
x = Point(20, 30)
print(x.latitude) #вот обращаемся к этой штуке (объяснение к селф было)
print(x._latitude) #это короче просто переменная без всяких действий? где подчеркивание - лучше не обращаться

#что-то про декораторы
""" 
def fun1(fun):
    pass
def fun2(a, b, c):
    pass
#хотим передать функцию 1 в функцию 2 - с помощью декоратора
@fun 
"""

#будем писать класс который посчитает кратчайшее расстояние между точками на земном шаре (дугу)
import math

class EarthPoint(Point): #в скобках значит что мы наследуем класс, то есть тут мы наследум от поинт, то есть внутри нового класса доступны все функции/формулы из родительского класса
    RADIUS = 6371.0
    #у нас из поинта уже есть широта/долгота, осталось задать функцию расстояния
    
    def __init__(self, latitude, longitude):
        super().__init__(latitude, longitude) #то есть инициализируем класс поинт с двумя параметрами, это инициализация родительского класса
    
    @staticmethod
    def deg_to_rad(deg): #в math все в радианах, а у нас градусы, поэтому нужна такая функция
        return deg / 180.0 * math.pi
    
    def distance_to(self, other): #задаем функцию расстояния
        delta_long = math.cos(self.deg_to_rad(self._longitude - other._longitude)) #формула просто из вики, по слагаемым делаем
        sin_lat1 = math.sin(self.deg_to_rad(self._latitude))
        sin_lat2 = math.sin(self.deg_to_rad(other._latitude))
        cos_lat1 = math.cos(self.deg_to_rad(self._latitude))
        cos_lat2 = math.cos(self.deg_to_rad(other._latitude))
        return self.RADIUS * math.acos(sin_lat1*sin_lat2 + cos_lat1*cos_lat2*delta_long)
    
    @staticmethod
    def distance(first, second): #в отлчие от дистанс_ту тут можно не инициализировать переменные + это штука от двух перменных, а дистанс-ту это метогд от одной
        return first.distance_to(second)
x = EarthPoint(1.0, 0.0)
z = EarthPoint(0.0, 0.0)
print(x.distance_to(z)) #вроде работает)))))))
print(EarthPoint.distance(x, z))


#напишем еще раз но чето про радианы, комменты скопированы
class RadPoint(Point): 

    
    def __init__(self, latitude, longitude):
        self._longitude = self.deg_to_rad(longitude)
        self._latitude = self.deg_to_rad(latitude)
    
    @staticmethod
    def deg_to_rad(deg): #в math все в радианах, а у нас градусы, поэтому нужна такая функция
        return deg / 180.0 * math.pi
    def rad_to_deg(rad):
        return rad / math.pi * 180.0

    @property #декоратор
    def latitude(self): #что должно происходить при чтении
        print('get latitude')
        return self.rad_to_deg(self._latitude)
    
    @latitude.setter #что должно происходить когда записываем, например тут можно обработку ошибок бахнуть
    def latitude(self, new_latitude): #кстати если перед new ... бахнуть * то это массив
        self._latitude = self.deg_to_rad(new_latitude)
        print('set latitude')
    
    @property
    def longitude(self):
        return self.rad_to_deg(self._longitude)
    
    @longitude.setter
    def longitude(self, new_longitude):
        self._longitude = self.deg_to_rad(new_longitude)
        print('set longitude')
    
    def __del__(self):
        print('deleted') #это встроенная штучка, когда мы удаляем значение (напр было х=1, потом задали х=4, то есть предыдущее стерлось) выполняется эта штука и сюда можно запихнуть что-то
    def __str__(self):
        return 'to_string' #уже не помню
    def __repr__(self):
        return 'repr' #короче это общие методы, их можно загуглить
x = RadPoint(10, 20)
print(x.latitude, x._latitude)
