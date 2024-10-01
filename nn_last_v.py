import numpy as np
import tracemalloc
import os
import time


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# Производная сигмоидальной функции
def sigmoid_derivative(x):
    return x * (1 - x)

# Функция загрузки данных
# В x хранится вложенный массив с 0 и 1 из файла, в y хранится массив из 100, 010, 001
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

# Обучение нейронной сети
# X_train: входные данные для обучения (матрица признаков). 
# y_train: целевые значения (матрица меток), векторы, которые кодируют класс каждой фигуры.
# hidden_neurons: количество слоев и количество нейронов в них, вложенный список.
# learning_rate: скорость обучения (параметр, определяющий, насколько сильно обновляются веса на каждой итерации).
# epochs: количество эпох (число итераций обучения).
def train_neural_network(X_train, y_train, hidden_neurons, learning_rate, epochs):
    #Подсчет количество колонок - это кол-во входных нейронов
    input_neurons = X_train.shape[1]
    #Подсчет количество колонок - это кол-во выходных слоев
    output_neurons = y_train.shape[1]

    # Инициализация весов сети
    weights = []
    # Цикл работает по количеству скрытых слоев с нейронами, весы инициализируются рандомно, в зависимости от того.ю какой слой - входной, скрытый последний или иной скрытый
    for i in range(len(hidden_neurons) + 1):
        if i == 0:
            weights.append(np.random.uniform(size=(input_neurons, hidden_neurons[i])))
        elif i == len(hidden_neurons):
            weights.append(np.random.uniform(size=(hidden_neurons[i-1], output_neurons)))
        else:
            weights.append(np.random.uniform(size=(hidden_neurons[i-1], hidden_neurons[i])))
    
    for epoch in range(epochs):
        # Прямой ход
        hidden_outputs = [] # пустой список, который будет хранить выходы каждого скрытого слоя
        output = X_train # входные данные
        
        for i in range(len(hidden_neurons) + 1): 
            if i == len(hidden_neurons): # если мы нахожимся на выходном слое
                output_input = np.dot(output, weights[i]) # Выход вычисляется как скалярное произведение текущего выхода и весов для этого слоя, а затем применяется функция активации sigmoid.
                output = sigmoid(output_input)
            else: # иначе это скрытый слой
                hidden_input = np.dot(output, weights[i])
                hidden_output = sigmoid(hidden_input)
                hidden_outputs.append(hidden_output) # lj
                output = hidden_output #  Выход этого слоя хранится в hidden_outputs и становится входом для следующего слоя.
        
        # Ошибка определяется как разница между реальными значениями и предсказанными.
        error = y_train - output
        
        # Каждые 100 эпох выводится средняя абсолютная ошибка.
        if epoch % 100 == 0:
            print('Error:', np.mean(np.abs(error)))
        
        # Обратный ход
        d_outputs = [] # пустой список, который будет хранить градиенты ошибки для каждого слоя
        d_hidden = error * sigmoid_derivative(output) # инициализируется как ошибка, умноженная на производную функции sigmoid, вычисленную для выхода.
        d_outputs.append(d_hidden) 
        for i in range(len(hidden_neurons)): # цикл повторяется по количеству скрытых слоев
            # Градиент ошибки для этого слоя вычисляется как скалярное произведение предыдущего градиента ошибки
            # и транспонированных весов для этого слоя, а затем умножается на производную функции sigmoid, вычисленную для выхода этого слоя.
            d_hidden = d_outputs[-1].dot(weights[len(hidden_neurons) - i].T) * sigmoid_derivative(hidden_outputs[len(hidden_neurons) - i - 1])
            d_outputs.append(d_hidden)

        # Обновление весов. Веса обновляются на основе рассчитанных градиентов и скорости обучения.
        for i in range(len(hidden_neurons) + 1):
            if i == 0: # если входной слой
                # Веса обновляются как скалярное произведение транспонированных входных данных и градиента ошибки, а затем умножается на скорость обучения.
                weights[i] += X_train.T.dot(d_outputs[-1]) * learning_rate 
            elif i == len(hidden_neurons): # если слой скрытый последний
                # cкалярное произведение транспонированных выходов предыдущего слоя (hidden_outputs[-1].T) и первого градиента ошибки (d_outputs[0]), а затем умножаем на скорость обучения
                weights[i] += hidden_outputs[-1].T.dot(d_outputs[0]) * learning_rate
            else:
                weights[i] += hidden_outputs[-i-1].T.dot(d_outputs[-i]) * learning_rate
    return weights

#Функция для предсказания вероятности
# X_test: тестовые данные, на которых будет производиться предсказание.
# weights: вычисленные на обучении веса
def predict(X_test, weights):
    output = X_test
    for i in range(len(weights)):
        hidden_input = np.dot(output, weights[i])
        output = sigmoid(hidden_input)
        output = np.round(output, 6)
    return output



# Загрузка данных
X_train, y_train = load_data('c:/python/1lab/train')
X_test, y_test = load_data('c:/python/1lab/test')

hidden_neurons = [10]  # задаем количество скрытых слоев и количество нейронов в них
learning_rate = 0.1 # скорость обучения
epochs = 2000

# Обучение сети
start_time = time.time()
tracemalloc.start()
weights = train_neural_network(X_train, y_train, hidden_neurons, learning_rate, epochs)
current, peak = tracemalloc.get_traced_memory()
#Память для обучения сети
print(f"Текущий размер памяти: {current / 10**6} МБ")
print(f"Пиковый размер памяти: {peak / 10**6} МБ")
tracemalloc.stop()
end_time = time.time()

#Вычисление выходов
start_time2 = time.time()
tracemalloc.start()
output = predict(X_test, weights)
current2, peak2 = tracemalloc.get_traced_memory()
#Память для валидации
print(f"Текущий размер памяти2: {current2 / 10**6} МБ")
print(f"Пиковый размер памяти2: {peak2 / 10**6} МБ")

# Вывод результатов
classes = ['Circle', 'Square', 'Triangle']

for i in range(len(X_test)):
    original_figure = classes[np.argmax(y_test[i])]
    predicted_probabilities = output[i]
    predicted_figure = classes[np.argmax(predicted_probabilities)]
    print("Original Figure:", original_figure)
    print("Predicted figure:", predicted_figure)
    print("Predicted Figure Probabilities:")
    for j in range(len(classes)):
        print("Probability of being a", classes[j], ":", predicted_probabilities[j])
    print("\n")
end_time2 = time.time()

# Расчет времени
elapsed_time = end_time - start_time
elapsed_time2 = end_time2 - start_time2
print('Time for training: ', elapsed_time)
print('Time for testing: ', elapsed_time2)
