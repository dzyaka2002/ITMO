import numpy as np
import psutil
import tracemalloc
import os
import time

def load_data(directory):
    X = []
    y = []
    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            with open(os.path.join(directory, filename), 'r') as f:
                matrix = [list(map(int, line.strip().split())) for line in f]
                X.append(np.array(matrix).flatten())
                if 'circle' in filename:
                    y.append([1, 0, 0])
                elif 'square' in filename:
                    y.append([0, 1, 0])
                elif 'triangle' in filename:
                    y.append([0, 0, 1])
    return np.array(X), np.array(y)
X_train, y_train = load_data('c:/python/1lab/train')
X_test, y_test = load_data('c:/python/1lab/validate')
print(y_test)
print(len([10])+1)
# # Сигмоидальная функция активации
# def sigmoid(x):
#     return 1 / (1 + np.exp(-x))

# # Производная сигмоидальной функции
# def sigmoid_derivative(x):
#     return x * (1 - x)

# # Функция для загрузки данных из файлов
# def load_data(directory):
#     X = []
#     y = []
#     for filename in os.listdir(directory):
#         if filename.endswith('.txt'):
#             # Загрузка матрицы из файла
#             with open(os.path.join(directory, filename), 'r') as f:
#                 matrix = [list(map(int, line.strip().split())) for line in f]
#                 X.append(np.array(matrix).flatten())  # Преобразуем 7x7 в 1D массив
#                 # Определяем метку на основе имени файла
#                 if 'circle' in filename:
#                     y.append([1, 0, 0])  # Класс 0
#                 elif 'square' in filename:
#                     y.append([0, 1, 0])  # Класс 1
#                 elif 'triangle' in filename:
#                     y.append([0, 0, 1])  # Класс 2
#     return np.array(X), np.array(y)

# # Функция для обучения нейронной сети методом обратного распространения ошибки
# # X_train: входные данные для обучения (матрица признаков). 
# # y_train: целевые значения (матрица меток), векторы, которые кодируют класс каждой фигуры.
# # hidden_neurons: количество нейронов в скрытом слое.
# # learning_rate: скорость обучения (параметр, определяющий, насколько сильно обновляются веса на каждой итерации).
# # epochs: количество эпох (число итераций обучения).
# def train_neural_network(X_train, y_train, hidden_neurons, learning_rate, epochs):
# # Инициализация весов сети
#     input_neurons = X_train.shape[1]
#     output_neurons = y_train.shape[1]
# #Веса между входным и скрытым слоями (weights_input_hidden) и между скрытым и выходным слоями (weights_hidden_output) инициализируются случайными значениями.
#     weights_input_hidden = np.random.uniform(size=(input_neurons, hidden_neurons))
#     weights_hidden_output = np.random.uniform(size=(hidden_neurons, output_neurons))

#     for epoch in range(epochs):
#     # Прямой ход
#     # Вычисляется вход для скрытого слоя и затем его выход с помощью функции активации sigmoid. np.dot - умножение массивов
#         hidden_input = np.dot(X_train, weights_input_hidden)
#         hidden_output = sigmoid(hidden_input)
#     # Вычисляется аналогично вход для выходного слоя и затем его выход с помощью функции активации sigmoid.
#         output_input = np.dot(hidden_output, weights_hidden_output)
#         output = sigmoid(output_input)
#     # Ошибка определяется как разница между реальными значениями и предсказанными.
#         error = y_train - output
    
#     # Каждые 100 эпох выводится средняя абсолютная ошибка.
#         if epoch % 100 == 0:
#             print('Error:', np.mean(np.abs(error)))
#     # Обратный ход
#     # Вычисляется градиент ошибки для выходного слоя и скрытого слоя, используя производную сигмоиды.
#         d_output = error * sigmoid_derivative(output)
#         error_hidden = d_output.dot(weights_hidden_output.T)
#         d_hidden = error_hidden * sigmoid_derivative(hidden_output)

#     # Обновление весов. Веса обновляются на основе рассчитанных градиентов и скорости обучения.
#         weights_hidden_output += hidden_output.T.dot(d_output) * learning_rate
#         weights_input_hidden += X_train.T.dot(d_hidden) * learning_rate
    
#     # Функция возвращает обновлённые веса для скрытого и выходного слоёв.
#     return weights_input_hidden, weights_hidden_output

# # Функция предсказания класса и вероятности
# # X_test: тестовые данные, на которых будет производиться предсказание.
# # weights_input_hidden: веса между входным и скрытым слоями.
# # weights_hidden_output: веса между скрытым и выходным слоями.
# def predict(X_test, weights_input_hidden, weights_hidden_output):
# # hidden_input: вычисляется вход для скрытого слоя, используя матричное умножение входных данных X_test на веса weights_input_hidden.
#     hidden_input = np.dot(X_test, weights_input_hidden)
# # hidden_output: выход скрытого слоя, вычисленный с помощью функции активации sigmoid.
#     hidden_output = sigmoid(hidden_input)
# # output_input: вычисляется вход для выходного слоя, используя выход скрытого слоя и веса weights_hidden_output.
#     output_input = np.dot(hidden_output, weights_hidden_output)
# # output: выход нейронной сети, также с использованием функции активации sigmoid.
#     output = sigmoid(output_input)
# # predicted_class: класс, который имеет наибольшее значение в выходных данных (предсказанный класс для каждого примера).
#     predicted_class = np.argmax(output, axis=1)
# # probability: вероятность предсказанного класса, равная максимальному значению в выходных данных. В результате для каждой строки возвращается максимальное значение.
#     probability = np.max(output, axis=1)
#     return predicted_class, probability, output

# # Загрузка обучающих и тестовых данных
# X_train, y_train = load_data('c:/python/1lab/train')
# X_test, y_test = load_data('c:/python/1lab/validate')

# # Обучение нейронной сети
# start_time = time.time()
# weights_input_hidden, weights_hidden_output = train_neural_network(X_train, y_train, 10, 0.2, 600)
# end_time = time.time()

# # Тестирование
# start_time2 = time.time()
# predicted_class, probability, output = predict(X_test, weights_input_hidden, weights_hidden_output)
# end_time2 = time.time()
# classes = ['Circle', 'Square', 'Triangle']

# # for i in range(len(X_test)):
# #     original_figure = classes[np.argmax(y_test[i])]
# #     predicted_figure = classes[predicted_class[i]]
# #     print("Original Figure:", original_figure, "- Predicted Figure:", predicted_figure, "with probability:", probability[i])
# for i in range(len(X_test)):
#     original_figure = classes[np.argmax(y_test[i])]
#     predicted_probabilities = output[i]
#     predicted_figure = classes[np.argmax(predicted_probabilities)]
#     print("Original Figure:", original_figure)
#     print("Predicted Figure Probabilities:")
#     for j in range(len(classes)):
#         print("Probability of being a", classes[j], ":", predicted_probabilities[j])
#     print("\n")
# elapsed_time = end_time - start_time
# elapsed_time2 = end_time2 - start_time2
# print('Time for training: ', elapsed_time)
# print('Time for testing: ', elapsed_time2)