# -*- coding: utf-8 -*-
"""
Created on Fri Feb 11 11:16:34 2022

@author: Student
"""

#разбираемся про **kwargs, нужно для домашки
def fun(x, y, z):
    print(x, y, z)
#как передать туда?
fun (1, 2, 3) #1 вариант
a = (1, 2, 3)
#fun(a) - будет не работать
fun(*a) #теперь со звездочкой работает - как бы распаковка
print(a)
print(*a) #see the difference

#именнованые аргументы
fun(x=1, y=2, z=3)
fun(y=2, x=1, z=3) #видно что порядок не важен т.к. имена есть у аргументов

#создадим словарь...
d = {'z':3,
     'x':1,
     'y':2}
#fun(d) не работает
fun(*d) #подали массив как бы z x y, но это не то
fun(**d) #теперь все воркс - читаются имена, значения, порядок не важен, все круто

#как читать файлы fits
from astropy.io import fits
with fits.open('sombrero.fits') as f:
    f = fits.open('sombrero.fits')
print(len(f))
hdu0 = f[0]
print(hdu0) #выводитфигню
print(hdu0.data) #а теперь норм вот так вот данные и надо читать
data = hdu0.data
print(type(data)) #видно что это массив нампи

import matplotlib.pyplot as plt

plt.imshow(data) #должен строить картинку
#тут еще было f.close() типа раз открыли надо закрыть но я хз прога должна быть внтури with open или нет
