import numpy as np
from keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential


import matplotlib.pyplot as plt
import numpy as np
class DenseLayer:
    def __init__(self, input_size, output_size, activation):
        self.W = np.random.randn(input_size, output_size) * 0.01
        self.b = np.zeros((1, output_size))
        self.activation = activation

    def forward(self, x):
        self.z = np.dot(x, self.W) + self.b
        self.a = self._apply_activation(self.z)
        return self.a

    def _apply_activation(self, x):
        return np.maximum(0, x)

        
class Neural_Network:
    def __init__(self, layers_config):
        self.layers = []
        for config in layers_config:
            self.layers.append(DenseLayer(*config))

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x
    
    def backprop(self,batch):
        predicted = self.forward(batch.)
        expected = self.forward(batch.inMtrix[2])
        

    def error(self,x,batch):
        pass
        
class batch:
    def __init__(self,size):
        self.input = self.Grab(size)[1]
        self.expected = self.Grab(size)[2]
    def Grab(size):
        return None
    def format(image):
        return None
