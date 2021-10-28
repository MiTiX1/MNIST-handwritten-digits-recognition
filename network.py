import numpy as np
from losses import mse, mse_prime
import matplotlib.pyplot as plt

class Network:
    def __init__(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train
        self.network = []
        self.cost_history = []
        self.accuracy = 0

    def add(self, layer):
        self.network.append(layer)

    def train(self, epochs, learning_rate):
        dataset_size = len(self.x_train)
        for i in range(epochs):
            error = 0
            for x, y in zip(self.x_train, self.y_train):
                output = x
                for layer in self.network:
                    output = layer.forward(output)

                error += mse(y, output)

                grad = mse_prime(y, output)
                for layer in reversed(self.network):
                    grad = layer.backward(grad, learning_rate)

            error /= dataset_size
            self.cost_history.append(error)

    def predict(self, x_test, y_test):
        correct_pred = 0
        pred = []
        for x, y in zip(x_test, y_test):
            output = x
            for layer in self.network:
                output = layer.forward(output)
            if np.argmax(output) == np.argmax(y):
                correct_pred += 1
            pred.append(np.argmax(output))
        self.accuracy = correct_pred / len(y_test)
        return pred

    def display_cost_history(self):
        plt.plot(self.cost_history)
        plt.show()

    def get_accuracy(self):
        return self.accuracy