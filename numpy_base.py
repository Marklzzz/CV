import numpy as np

# 1. Создание массивов
print("1. Создание массивов:")
array_a = np.array([1, 2, 3, 4, 5])
array_b = np.arange(0, 10, 2)  # Массив от 0 до 10 с шагом 2
matrix = np.array([[1, 2], [3, 4]])

print(f"array_a: {array_a}")
print(f"array_b: {array_b}")
print(f"Матрица 2x2:\n{matrix}\n")

# 2. Арифметические операции
print("2. Арифметические операции:")
print(f"Сложение: {array_a + array_b}")
print(f"Умножение: {array_a * 2}")
print(f"Возведение в квадрат: {array_b ** 2}\n")

# 3. Матричные операции
print("3. Матричные операции:")
vector = np.array([5, 6])
print(f"Умножение матрицы на вектор: {matrix.dot(vector)}")
print(f"Транспонирование матрицы:\n{matrix.T}\n")

# 4. Математические функции
print("4. Математические функции:")
print(f"Синус: {np.sin(array_a)}")
print(f"Экспонента: {np.exp(matrix)}\n")

# 5. Статистика
print("5. Статистические вычисления:")
random_data = np.random.normal(0, 1, 100)  # 100 случайных чисел из нормального распределения
print(f"Среднее: {np.mean(random_data):.2f}")
print(f"Стандартное отклонение: {np.std(random_data):.2f}")
print(f"Медиана: {np.median(random_data):.2f}")
print(f"Сумма элементов array_a: {np.sum(array_a)}")
print(f"Максимум в array_b: {np.max(array_b)}")