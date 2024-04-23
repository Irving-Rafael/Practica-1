import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Perceptron:
    def __init__(self, num_inputs, learning_rate=0.01, max_epochs=1000):
        self.weights = np.random.rand(num_inputs + 1) * 2 - 1  
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs

    def activation(self, x):
        return 1 if x >= 0 else -1

    def predict(self, inputs):
        summation = np.dot(inputs, self.weights[1:]) + self.weights[0]  
        return self.activation(summation)

    def train(self, training_inputs, labels):
        for epoch in range(self.max_epochs):
            errors = 0
            for inputs, label in zip(training_inputs, labels):
                prediction = self.predict(inputs)
                error = label - prediction
                self.weights[1:] += self.learning_rate * error * inputs
                self.weights[0] += self.learning_rate * error  
                errors += int(error != 0)
            if errors == 0:
                print(f"Converged in epoch {epoch + 1}")
                break
        else:
            print("Convergencia terminada.")

# Función para visualizar los datos y la línea de separación
def plot_data_and_line(X, y, perceptron):
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired, marker='o', edgecolors='k', label='Data Points')
    plt.xlabel('X1')
    plt.ylabel('X2')

    # Plotting the decision boundary
    slope = -perceptron.weights[1] / perceptron.weights[2]
    intercept = -perceptron.weights[0] / perceptron.weights[2]
    x_vals = np.linspace(-2, 2, 100)
    y_vals = slope * x_vals + intercept
    plt.plot(x_vals, y_vals, '-r', label='Decision Boundary')
    plt.legend()
    plt.title('Perceptron Decision Boundary')
    plt.show()

# Leer datos de entrenamiento y prueba desde archivos CSV para XOR
train_data_xor = pd.read_csv("XOR_trn.csv", header=None)
test_data_xor = pd.read_csv("XOR_tst.csv", header=None)

# Dividir datos de XOR en entradas (características) y salidas (etiquetas)
X_train_xor = train_data_xor.iloc[:, :-1].values
y_train_xor = train_data_xor.iloc[:, -1].values
X_test_xor = test_data_xor.iloc[:, :-1].values
y_test_xor = test_data_xor.iloc[:, -1].values

# Datos de entrenamiento y prueba de la compuerta OR
X_train_or = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_train_or = np.array([0, 1, 1, 1])
X_test_or = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_test_or = np.array([0, 1, 1, 1])

# Definir parámetros de entrenamiento
learning_rate = 0.1
max_epochs = 1000

# Entrenar y probar el perceptrón para XOR
print("Entrenamiento y prueba de XOR:")
perceptron_xor = Perceptron(num_inputs=2, learning_rate=learning_rate, max_epochs=max_epochs)
perceptron_xor.train(X_train_xor, y_train_xor)
predictions_test_xor = [perceptron_xor.predict(inputs) for inputs in X_test_xor]
accuracy_xor = np.mean(predictions_test_xor == y_test_xor) * 100
print(f"Precisión en datos de prueba para XOR: {accuracy_xor:.2f}%")
plot_data_and_line(X_test_xor, y_test_xor, perceptron_xor)

# Entrenar y probar el perceptrón para OR
print("\nEntrenamiento y prueba de OR:")
perceptron_or = Perceptron(num_inputs=2, learning_rate=learning_rate, max_epochs=max_epochs)
perceptron_or.train(X_train_or, y_train_or)
predictions_test_or = [perceptron_or.predict(inputs) for inputs in X_test_or]
accuracy_or = np.mean(predictions_test_or == y_test_or) * 100
print(f"Precisión en datos de prueba para OR: {accuracy_or:.2f}%")
plot_data_and_line(X_test_or, y_test_or, perceptron_or)
