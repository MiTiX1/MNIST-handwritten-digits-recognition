import numpy as np
from losses import mse, mse_prime
import matplotlib.pyplot as plt

class Network:
    def __init__(self, epochs, learning_rate):
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.network = []

    def add(self, layer):
        self.network.append(layer)

    def fit(self, x, y):
        self.x = x
        self.y = y
        self.cost_history = np.zeros(len(self.y))

        for i in range(self.epochs):
            error = 0
            for j, k in zip(self.x, self.y):
                output = j
                for layer in self.network:
                    output = layer.forward(output)

                error += mse(k, output)

                grad = mse_prime(k, output)
                for layer in reversed(self.network):
                    grad = layer.backward(grad, self.learning_rate)

            error /= len(self.y)
            self.cost_history[i] = error

    def predict(self, x):
        pred = np.zeros(len(x), dtype='int64')
        i = 0
        for j in x:
            output = j
            for layer in self.network:
                output = layer.forward(output)
            pred[i] = np.argmax(output)
            i += 1
        return pred

    def get_accuracy(self, y, y_pred):
        y_cat = np.zeros(len(y), dtype='int64')
        for i in range(len(y)):
	        y_cat[i] = np.argmax(y[i])
        return np.sum(y_cat == y_pred) / len(y_cat)

    def display_cost_history(self):
        plt.plot(self.cost_history)
        plt.show()