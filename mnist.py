from dense import Dense
from activations import Sigmoid
from network import Network
from keras.datasets import mnist
from keras.utils.np_utils import to_categorical

def prep_data(x, y):
	x = x.reshape(x.shape[0], 784, 1)
	x = x.astype('float32')
	x /= 255
	y = to_categorical(y)
	y = y.reshape(y.shape[0], 10, 1)
	return x, y

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, y_train = prep_data(x_train, y_train)
x_test, y_test = prep_data(x_test, y_test)

layers = [
    Dense(784, 16),
    Sigmoid() , 
    Dense(16, 16),
    Sigmoid() ,
	Dense(16, 16),
	Sigmoid() ,
	Dense(16, 10),
	Sigmoid() 
]

network = Network(x_train, y_train)

for i in layers:
	network.add(i)

epochs = 100
learning_rate = 0.1

network.train(epochs, learning_rate)
network.predict(x_test, y_test)
network.display_cost_history()
print(network.get_accuracy())