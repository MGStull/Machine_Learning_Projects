import numpy as np
from keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
import matplotlib.pyplot as plt
import numpy as np



class DenseLayer:
    def __init__(self, input_size, output_size):
        self.W = np.random.randn(input_size, output_size) * 0.01
        self.b = np.zeros((1, output_size))

    def forward(self, x):
        self.z = x@self.W + self.b
        self.a = self._apply_activation(self.z)
        return self.a

    def _apply_activation(self, x):
        return np.exp(-x)
    def activationDerivative(self,x):
        return -np.exp(-x)

        
class Neural_Network:
    def __init__(self, layers_config):
        self.layers = []
        for config in layers_config:
            self.layers.append(DenseLayer(*config))
    
    def guess(self, x):
        a = self.forward(x)
        return np.argmax(a)

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x
    
    def backprop(self,batch):
        predicted = self.forward(batch.input)
        expected = batch.expected
        self.weightGradient(expected,predicted)
        return self.error(batch)

    def error(self,batch):
        return np.linalg.norm(batch.predicted-batch.expeceted,ord=2)/batch.size
    def errorGradient(self,batch):
        return 2*(batch.predicted-batch.expected)/batch.size

    #This is done by the matrix linear algebra definiont of backword pass formula for Neural Networks./ A little bit overkill but we will see how my computer does
    def weightGradient(self,expected,predicted,weights):
        pass

    def errorGradient(expected,predicted,weights):
        pass
    
    def biasGradient(self,expected,predicted,weights):
        pass


(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

class batch:
    def __init__(self,size):
        self.size=size
        self.input = self.Grab(size)[0]
        self.expected = self.Grab(size)[1]
    #Pull Data in a K cross format to prevent overfitting
    def Grab(self,size):
        # Grab the first `size` samples from test data
        X_batch = x_test[:size]  
        Y_batch = y_test[:size]  
        X_batch = X_batch.reshape(size, -1)/255  # Normalize pixel values to [0, 1]
        Y_batch_one_hot = np.zeros((size, 10))
        Y_batch_one_hot[np.arange(size), Y_batch] = 1
        return [X_batch, Y_batch_one_hot]


NN = Neural_Network([
    [784,784],
    [784,1]
    ])
print(NN.backprop(batch(3)))


