from activation import Activation
import numpy as np

class Sigmoid(Activation):
	def __init__(self):
		sigmoid = lambda x: 1 / (1 + np.exp(-x))
		sigmoid_prime = lambda x: sigmoid(x) * (1 - sigmoid(x))
		super().__init__(sigmoid, sigmoid_prime)