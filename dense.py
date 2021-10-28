import numpy as np

class Dense:
	def __init__(self, input_size, output_size):
		self.W = np.random.randn(output_size, input_size)
		self.b = np.random.randn(output_size, 1)

	def forward(self, input):
		self.input = input
		return np.dot(self.W, self.input) + self.b

	def backward(self, output_gradient, learning_rate):
		W_gradient = np.dot(output_gradient, self.input.T)
		self.W -= learning_rate * W_gradient
		self.b -= learning_rate * output_gradient
		return np.dot(self.W.T, output_gradient)