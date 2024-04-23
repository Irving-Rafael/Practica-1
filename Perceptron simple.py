import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Perceptron:
    def __init__(self, num_inputs, learning_rate=0.01):
        self.weights = np.random.rand(num_inputs + 1)
        self.learning_rate = learning_rate

    def predict(self, inputs):
        summation = np.dot(inputs, self.weights[1:]) + self.weights[0]
        return 1 if summation > 0 else 0

    def train(self, training_inputs, labels, max_epochs):
        for epoch in range(max_epochs):
            for inputs, label in zip(training_inputs, labels):
                prediction = self.predict(inputs)
                self.weights[1:] += self.learning_rate * (label - prediction) * inputs
                self.weights[0] += self.learning_rate * (label - prediction)

def read_data(file_path):
    data = pd.read_csv(file_path, header=None)
    inputs = data.iloc[:, :-1].values
    labels = data.iloc[:, -1].values
    return inputs, labels

def plot_data(inputs, labels, perceptron):
    plt.scatter(inputs[:,0], inputs[:,1], c=labels, cmap='bwr')
    plt.xlabel('')
    plt.ylabel('')

    if len(perceptron.weights) == 3: # If two inputs, plot decision boundary
        x_values = np.array([np.min(inputs[:, 0]), np.max(inputs[:, 0])])
        y_values = (-1/perceptron.weights[2]) * (perceptron.weights[1] * x_values + perceptron.weights[0])
        plt.plot(x_values, y_values, label='Decision Boundary')
    
    plt.legend()
    plt.show()

def main():
    # Step 1: Read training data
    training_inputs, training_labels = read_data("XOR_trn.csv")
    test_inputs, test_labels = read_data("XOR_tst.csv")

    # Step 2: Create perceptron
    num_inputs = training_inputs.shape[1]
    perceptron = Perceptron(num_inputs)

    # Step 3: Train perceptron
    max_epochs = 2000
    perceptron.train(training_inputs, training_labels, max_epochs)

    # Step 4: Test perceptron
    accuracy = sum(1 for inputs, label in zip(test_inputs, test_labels) if perceptron.predict(inputs) == label) / len(test_labels)
    print("Accuracy:", accuracy)

    # Step 5: Plot data and decision boundary
    plot_data(training_inputs, training_labels, perceptron)

if __name__ == "__main__":
    main()
