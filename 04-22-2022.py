# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 11:11:25 2022

@author: Алиса
"""


#семинар 22.04.2022
#что-то про фильтрацию и предстоящую (4) домашку

import numpy as np
import matplotlib.pyplot as plt
import komm
from komm import BarkerSequence

code = BarkerSequence(11) #короче это хотели задать так массив
#code = np.array([1, 1, 1, -1, -1, -1, 1, -1, -1, 1, -1]) #это наш массив
# plt.plot(code.autocorrelation()) #autocorrelation это из библы автокорреляционная функция

#мы щас реализуем класс но в домашке можно делать как хочеца
class WiFi:
    seq = np.asarray(code.polar_sequence, dtype=np.int8) #используем 8битный потому что кто-то там восьмибитный тоже
    length = 5
    encoding = 'ascii'
    
    def __init__(self):
        pass
    
    @property
    def code_signal(self):
        return np.repeat(self.seq, self.length) #возвращаем нашу последовательность 5 раз
    #есть битовое представление, кодируем каждый символ последовательностью кого-то, чтобы получить битовый массив
    def bits2encoder(self, bits): 
        data = 2 * bits.astype(np.int8) - 1
        return np.outer(data, self.seq).flatten() #чето внешнее умножение и разворачиваем в 1д массив
    
    
    def str2bits(self, s):
        byte_str = s.encode(encoding = self.encoding) #подается строка, метод encode возвращает байтовый объект
        byte_array = np.frombuffer(byte_str, dtype=np.uint8)
        bits = np.unpackbits(byte_array).flatten()
        return bits
    
    def encode(self, s):
        bits = self.str2bits(s)
        return self.bits2encoder(bits)
    
    def signal(self, s):
        return np.repeat(self.encode(s), self.length)
    
w = WiFi()
info = 'test'
#print(len(w.str2bits(info)))  #все работает ура

signal = w.signal(info)
#print(len(signal))

y = signal + np.random.normal(0, 1, size=signal.size) #добавляем шум с амплитудой 1
figure = plt.figure(figsize=(40, 2))
# plt.plot(y)
# plt.plot(signal) #графики шума и сигнала

#вроде декодируем хз
corr = np.correlate(y, np.repeat(w.seq, 5), mode='full')
corr -= corr.mean() #вычли средее чтобы было легче декодировать
std = np.std(corr)

fig = plt.figure(figsize=(10,5))
plt.plot(corr[:550]) #построили первые 10 бит?? видно что на фоне шума есть какие-то выбросы вверх/вниз
#нам и надо задетектить пики - это и есть пики (авто?)корелляционной функции

#как можно выделить пики? построим горизонтальные линии
level = 2*std
plt.hlines([level, -level], *plt.xlim()) #круто, осталось детектить все что за пределами двух этих линий (двух стандартных отклонений)

#все с этим методом

