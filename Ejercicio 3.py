import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Paso 1: Preparación de datos
def generate_circle_data(n_samples, radius_inner, radius_outer):
    # Generar puntos aleatorios dentro y fuera del círculo
    theta = np.random.rand(n_samples) * 2 * np.pi
    r_inner = np.random.rand(n_samples) * radius_inner
    r_outer = np.random.rand(n_samples) * (radius_outer - radius_inner) + radius_inner

    x_inner = r_inner * np.cos(theta)
    y_inner = r_inner * np.sin(theta)

    x_outer = r_outer * np.cos(theta)
    y_outer = r_outer * np.sin(theta)

    # Combinar puntos dentro y fuera del círculo
    X = np.vstack((np.hstack((x_inner, x_outer)), np.hstack((y_inner, y_outer)))).T
    y = np.hstack((np.zeros(n_samples), np.ones(n_samples))).reshape(-1, 1)

    return X, y

def plot_data(X, y):
    plt.scatter(X[:, 0], X[:, 1], c=y.flatten(), cmap=plt.cm.coolwarm)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Dataset')
    plt.show()

# Paso 2: Definición de la red neuronal
class MLP:
    def __init__(self, input_size, hidden_layers, output_size):
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.output_size = output_size

        # Inicializar pesos y sesgos
        self.weights = [np.random.randn(input_size, hidden_layers[0])]
        self.biases = [np.random.randn(hidden_layers[0])]
        for i in range(1, len(hidden_layers)):
            self.weights.append(np.random.randn(hidden_layers[i-1], hidden_layers[i]))
            self.biases.append(np.random.randn(hidden_layers[i]))

        self.weights.append(np.random.randn(hidden_layers[-1], output_size))
        self.biases.append(np.random.randn(output_size))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def forward(self, x):
        self.activations = [x]
        self.z_values = []

        for i in range(len(self.weights)):
            z = np.dot(self.activations[-1], self.weights[i]) + self.biases[i]
            self.z_values.append(z)
            activation = self.sigmoid(z)
            self.activations.append(activation)

        return self.activations[-1]

    def backward(self, x, y, learning_rate):
        error = y - self.activations[-1]
        delta = error * self.sigmoid_derivative(self.activations[-1])

        for i in range(len(self.weights)-1, -1, -1):
            self.weights[i] += learning_rate * np.dot(self.activations[i].T, delta)
            self.biases[i] += learning_rate * np.sum(delta, axis=0)
            delta = np.dot(delta, self.weights[i].T) * self.sigmoid_derivative(self.activations[i])

    def train(self, x_train, y_train, epochs, learning_rate):
        for epoch in range(epochs):
            for x, y in zip(x_train, y_train):
                x = x.reshape(1, -1)
                self.forward(x)
                self.backward(x, y, learning_rate)

# Generar 1000 puntos aleatorios en dos círculos
X_random, y_random = generate_circle_data(500, 5, 10)

# Visualizar datos de prueba aleatorios
plot_data(X_random, y_random)

# Prueba de la red neuronal con los datos aleatorios
input_size = X_random.shape[1]
hidden_layers = [4, 3]  # Cantidad de neuronas en cada capa oculta
output_size = 1
mlp = MLP(input_size, hidden_layers, output_size)
mlp.train(X_random, y_random, epochs=1000, learning_rate=0.1)

# Prueba de la red neuronal con los datos aleatorios
predictions_random = []
for x in X_random:
    x = x.reshape(1, -1)
    prediction = mlp.forward(x)
    predictions_random.append(prediction)

predictions_random = np.array(predictions_random)
predictions_random[predictions_random >= 0.5] = 1
predictions_random[predictions_random < 0.5] = 0

# Calcular precisión
accuracy_random = np.mean(predictions_random == y_random)
print("Precisión de la red neuronal en el conjunto de prueba aleatorio:", accuracy_random)
