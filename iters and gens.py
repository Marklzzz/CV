class Counter:
    def __init__(self, max_value):
        self.max_value = max_value
        self.current = 0

    def __iter__(self):
        return self  # Возвращает сам объект как итератор

    def __next__(self):
        if self.current < self.max_value:
            self.current += 1
            return self.current
        else:
            raise StopIteration  # Элементы закончились

    def reset(self):  # ← Метод для сброса
        self.current = 0


# Использование
print("---------------  Итерация по объекту")
counter = Counter(3)
for num in counter:
    print(num)  # Вывод: 1, 2, 3

print("---------------  Сброс и новая итерация по объекту")
counter.reset()
for _ in range(3):
    print(next(counter))


def counter(max_value):
    current = 0
    while current < max_value:
        current += 1
        yield current  # Возвращает значение и "замораживает" состояние


print("---------------  Генератор")
gen = counter(3)
print(next(gen))  # 1
print(next(gen))  # 2
print(next(gen))  # 3
