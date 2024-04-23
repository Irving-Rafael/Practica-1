import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def initialize_weights(input_size, hidden_size, output_size):
    weights_input_hidden = np.random.uniform(-0.5, 0.5, size=(input_size, hidden_size))
    weights_hidden_output = np.random.uniform(-0.5, 0.5, size=(hidden_size, output_size))
    return weights_input_hidden, weights_hidden_output

def forward_propagation(X, weights_input_hidden, weights_hidden_output):
    hidden_input = np.dot(X, weights_input_hidden)
    hidden_output = sigmoid(hidden_input)
    output_input = np.dot(hidden_output, weights_hidden_output)
    output = sigmoid(output_input)
    return hidden_output, output

def backpropagation(X, y, hidden_output, output, weights_hidden_output):
    output_error = y - output
    output_delta = output_error * sigmoid_derivative(output)
    hidden_error = output_delta.dot(weights_hidden_output.T)
    hidden_delta = hidden_error * sigmoid_derivative(hidden_output)
    return output_delta, hidden_delta

def update_weights(X, hidden_output, output_delta, hidden_delta, weights_input_hidden, weights_hidden_output, learning_rate):
    weights_hidden_output += hidden_output.T.dot(output_delta) * learning_rate
    weights_input_hidden += X.T.dot(hidden_delta) * learning_rate
    return weights_input_hidden, weights_hidden_output

def train(X_train, y_train, input_size, hidden_size, output_size, epochs, learning_rate):
    weights_input_hidden, weights_hidden_output = initialize_weights(input_size, hidden_size, output_size)
    for _ in range(epochs):
        hidden_output, output = forward_propagation(X_train, weights_input_hidden, weights_hidden_output)
        output_delta, hidden_delta = backpropagation(X_train, y_train, hidden_output, output, weights_hidden_output)
        weights_input_hidden, weights_hidden_output = update_weights(X_train, hidden_output, output_delta, hidden_delta, weights_input_hidden, weights_hidden_output, learning_rate)
    return weights_input_hidden, weights_hidden_output

def predict(X, weights_input_hidden, weights_hidden_output):
    _, output = forward_propagation(X, weights_input_hidden, weights_hidden_output)
    return output.argmax(axis=1)

def accuracy_score(y_true, y_pred):
    return np.mean(y_true == y_pred)

def leave_one_out_cross_validation(X, y, input_size, hidden_size, output_size, epochs, learning_rate):
    n = X.shape[0]
    accuracies = []
    for i in range(n):
        X_train = np.delete(X, i, axis=0)
        y_train = np.delete(y, i, axis=0)
        X_test = X[i].reshape(1, -1)
        y_test = y[i].reshape(1, -1)
        weights_input_hidden, weights_hidden_output = train(X_train, y_train, input_size, hidden_size, output_size, epochs, learning_rate)
        y_pred = predict(X_test, weights_input_hidden, weights_hidden_output)
        accuracies.append(accuracy_score(y_test.argmax(axis=1), y_pred))
    return np.mean(accuracies), np.std(accuracies)

def leave_k_out_cross_validation(X, y, input_size, hidden_size, output_size, epochs, learning_rate, k=5):
    n = X.shape[0]
    indices = np.arange(n)
    np.random.shuffle(indices)
    fold_size = n // k
    accuracies = []
    for i in range(k):
        start = i * fold_size
        end = min((i + 1) * fold_size, n)
        test_indices = indices[start:end]
        train_indices = np.concatenate([indices[:start], indices[end:]])
        X_train, X_test = X[train_indices], X[test_indices]
        y_train, y_test = y[train_indices], y[test_indices]
        weights_input_hidden, weights_hidden_output = train(X_train, y_train, input_size, hidden_size, output_size, epochs, learning_rate)
        y_pred = predict(X_test, weights_input_hidden, weights_hidden_output)
        accuracies.append(accuracy_score(y_test.argmax(axis=1), y_pred))
    return np.mean(accuracies), np.std(accuracies)

def plot_2d_projection(X, y, title):
    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Set1, edgecolor='k')
    plt.xlabel('Longitud del sépalo')
    plt.ylabel('Anchura del sépalo')
    plt.title(title)
    plt.show()

# Cargar los datos
data = pd.read_csv('irisbin.csv', header=None)
X = data.iloc[:, :-3].values  # características (dimensiones de los pétalos y sépalos)
y = data.iloc[:, -3:].values  # etiquetas (códigos binarios de las especies)

# Normalizamos los datos
X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

# Dividir en conjunto de entrenamiento y prueba
n_samples = X.shape[0]
n_train = int(n_samples * 0.8)
X_train, X_test = X[:n_train], X[n_train:]
y_train, y_test = y[:n_train], y[n_train:]

input_size = X_train.shape[1]
hidden_size = 10
output_size = 3
epochs = 1000
learning_rate = 0.1

# Entrenamiento
weights_input_hidden, weights_hidden_output = train(X_train, y_train, input_size, hidden_size, output_size, epochs, learning_rate)

# Clasificación
y_pred = predict(X_test, weights_input_hidden, weights_hidden_output)
accuracy = accuracy_score(y_test.argmax(axis=1), y_pred)
print("Precisión en el conjunto de prueba:", accuracy)

# Proyección en 2D
plot_2d_projection(X_test[:, :2], y_test.argmax(axis=1), "Proyección de dos dimensiones de las especies de Iris")

# Validación cruzada leave-one-out
leave_one_out_accuracy, leave_one_out_std = leave_one_out_cross_validation(X, y, input_size, hidden_size, output_size, epochs, learning_rate)
print("Precisión media con leave-one-out:", leave_one_out_accuracy)
print("Desviación estándar con leave-one-out:", leave_one_out_std)

# Validación cruzada leave-k-out
k = 5
leave_k_out_accuracy, leave_k_out_std = leave_k_out_cross_validation(X, y, input_size, hidden_size, output_size, epochs, learning_rate, k)
print(f"Precisión media con leave-{k}-out:", leave_k_out_accuracy)
print(f"Desviación estándar con leave-{k}-out:", leave_k_out_std)
