# -*- coding: utf-8 -*-
"""
Created on Fri Jan 28 11:19:00 2022

@author: Student
"""

from time import monotonic #make sense только разность
with open('D:/sh/pushkin.txt', encoding='utf-8') as f:
    text = f.read()
words = text.split() #разбили текст на слова
print (len(words)) #чекнули количествослов это 21574
t = monotonic()

most_common = None
max_count = 0

for word in words:
    count = words.count(word) #ищем слово самое популярное (ворд - переменная, вордс - массив)
    if count > max_count:
        most_common = word
        max_count = count

print(monotonic()-t, 'seconds') #вот за сколько выполняется операция
print(most_common, max_count) #просто чекаем что у нас за популярное слово и сколько раз

#теперь метод со словарем
t = monotonic()

d = {} #словарь
for word in words:
    if word in d: #если новое слово - создаем (ключ вроде слово), если есть то как бы значение на единицу повышаем
        d[word] += 1
    else:
        d[word] = 1
most_common = max(d, key=d.get) #с помощью d.get получаем количество вхождений (типа гет обращается к значению штуки с определенным ключом)
print(monotonic()-t, 'seconds') #мач быстрее

#чето еще короче
from collections import Counter
t = monotonic()
c = Counter(words) #заменяет наш цикл
most_common = c.most_common(1)[0][0]

print(monotonic()-t) #короче 0 значит функция монотоник не успевает сработать
print(c.most_common(5)) #типа топ5 слов

#откроем файл другим способом
t = monotonic()
c = Counter()
with open('D:/sh/pushkin.txt', encoding='utf-8') as f: #f будет итератором который хранит строки (итератор чето типа список но пройти один раз и хранит 1 значение)
    for x in map(lambda x: x.split(), f): #первое в мэпе функция, второе - массив к которому применяется функция; лямбда функция - способ записать функцию в одну строчку (подается х возвр сплит х)
        c.update(x) #делает как микроцикл: если слова нет - заводит, если слово есть - увеличивает счетчик

most_common = c.most_common(1)[0][0]

print(monotonic()-t)

