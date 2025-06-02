import numpy as np
from keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
import matplotlib.pyplot as plt
import numpy as np
import random as rand


class DenseLayer:
    def __init__(self, input_size, output_size):
        self.W = np.random.randn(input_size, output_size) * 0.01
        self.b = np.zeros((1, output_size))
        self.dW = None
        self.dB = None

    def forward(self, x):
        self.x = np.copy(x)
        self.z = x@self.W + self.b
        self.a = self._apply_activation(self.z)
        return self.a

    def _apply_activation(self, x):
        return np.exp(-x)
    def activationDerivative(self,x):
        return -np.exp(-x)
    def update(self,stepSize=.01):
        self.W -= self.dW*stepSize
        self.b -= self.db*stepSize

        

        
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
        batch.predicted = self.forward(batch.input)
        expected = batch.expected
        self.weightGradient(batch,self.layers)
        for layer in self.layers:
            layer.update()
        return self.error(batch)

    def error(self,batch):
        return np.linalg.norm(batch.predicted-batch.expected,ord=2)/batch.size
    def errorGradient(self,batch):
        return 2*(batch.predicted-batch.expected)/batch.size

    #This is done by the matrix linear algebra definiont of backword pass formula for Neural Networks./ A little bit overkill but we will see how my computer does
    def weightGradient(self,batch,weights):
        delta = self.errorGradient(batch)
        for layer in reversed(self.layers):
            dz=delta*layer.activationDerivative(layer.z)
            dW = layer.x.T @ dz
            db = np.sum(dz,axis=0,keepdims=True)
            delta = dz@layer.W.T
            layer.dW =dW
            layer.db =db

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
class batch:
        def __init__(self,indices):
            self.size=len(indices)
            self.input,self.expected = self.Grab(indices)
            self.indices = indices
        #Pull Data in a K cross format to prevent overfitting
        def Grab(self,indices):
            # Grab the first `self.size` samples from test data
            X_batch = np.zeros((self.size,28,28))
            Y_batch = np.zeros(self.size,dtype=int)

            for i in range(0,len(indices)):    
                X_batch[i] = x_train[indices[i]]  
                Y_batch[i] = y_train[indices[i]]  
            X_batch = X_batch.reshape(self.size, -1)/255  # Normalize pixel values to [0, 1]
            Y_batch_one_hot = np.zeros((self.size, 10))
            Y_batch_one_hot[np.arange(self.size), Y_batch] = 1
            return [X_batch, Y_batch_one_hot]
class Epoch:
    def __init__(self,k):
        self.indicesList = [[]]*k
        self.batches = []
        randHold = list(range(0,60000))
        print(self.indicesList)
        print(int(len(x_train)/k))
        for i in range(0,k):
            for j in range(0,int(len(x_train)/k)):
                #print("randHold Length: ",len(randHold))
                x =rand.randint(0,len(randHold)-1)
                #print("Random Index: ",x)
                #print("self.indicesList[i]",print(len(self.indicesList),"Does not match: ",i))
                self.indicesList[i].append(randHold.pop(x))
        print("Complete",len(randHold),len(self.indicesList),self.indicesList[1])
        for i in self.indicesList:
            self.batches.append(batch(i))
    def randTrain(self,NN):
        for i in self.batches:
            NN.backprop(i) 
        print("Complete")  


NN = Neural_Network([
    [784,128],
    [128,128],
    [128,64],
    [64,32],
    [32,10]
    ])
Tbatch = batch([1,10,100,99,10000,1000])

def TEST(testSize,NN):
    # Grab the first `size` samples from test data
    X_batch = x_test[:testSize]
    X_batch = X_batch.reshape(testSize, -1)/255 
    Y_batch =y_test[:testSize]
    sum=0
    for i in range(testSize):
        if NN.guess(X_batch[i]) == Y_batch[i]:
            sum +=1
    return (sum/testSize)

NN.backprop(Tbatch)
print(TEST(100,NN))
