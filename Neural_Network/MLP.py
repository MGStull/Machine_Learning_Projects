import numpy as np
from keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
import matplotlib.pyplot as plt
import numpy as np
import random as rand


class DenseLayer:
    def __init__(self, input_size, output_size):
        self.W = np.random.randn(input_size, output_size) *np.sqrt(2/input_size)
        self.B = np.zeros((1, output_size))
        self.dW = None
        self.dB = None

    def forward(self, x):
        self.x = np.copy(x)
        self.z = x@self.W + self.B
        self.a = self._apply_activation(self.z)
        return self.a

    #From Geeks for Geeks
    def _apply_activation(self,x,alpha=.01):
        return np.maximum(alpha * x, x)

    #From Geeks for Geeks
    def activationDerivative(self,x,alpha=.01):
        dx = np.ones_like(x)
        dx[x < 0] =alpha
        return dx

    def update(self,stepSize):
        self.W -= self.dW*stepSize
        self.B -= self.dB*stepSize

        

        
class Neural_Network:
    def __init__(self, layers_config,stepSize=.01):
        self.layers = []
        self.stepSize = stepSize
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
        self.weightGradient(batch,self.layers)
        for layer in self.layers:
            layer.update(self.stepSize)
        return self.error(batch)

    def error(self,batch):
        return np.linalg.norm(batch.predicted-batch.expected,ord=2)/batch.size
    def errorGradient(self,batch):
        return 2*(batch.predicted-batch.expected)/batch.size

    def weightGradient(self,batch,weights):
        delta = self.errorGradient(batch)
        for layer in reversed(self.layers):
            dz=delta*layer.activationDerivative(layer.z)
            dW = layer.x.T @ dz
            dB = np.sum(dz,axis=0,keepdims=True)
            delta = dz@layer.W.T
            layer.dW =dW
            layer.dB =dB

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
class batch:
        def __init__(self,indices):
            self.size=len(indices)
            self.input,self.expected = self.Grab(indices)
            self.indices = indices
        def Grab(self,indices):
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
        self.indicesList = [[] for _ in range(k)]
        self.batches = []
        randHold = list(range(len(x_train)))
        rand.shuffle(randHold)
        chunk_size = len(x_train) // k
        for i in range(k - 1):
            self.indicesList[i] = randHold[i*chunk_size:(i+1)*chunk_size]
        self.indicesList[k - 1] = randHold[(k - 1)*chunk_size:]

        for i in self.indicesList:
            self.batches.append(batch(i))
    def randTrain(self,NN):
        for i in self.batches:
            NN.backprop(i) 
        #print("Complete")  


NN = Neural_Network([
    [784,128],
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


TrainEpoch = Epoch(600)
acc =[]
acc.append(TEST(10000, NN))
for i in range(10):
    TrainEpoch.randTrain(NN)
    acc.append(TEST(10000, NN))
    TrainEpoch = Epoch(600)

plt.title("Test Accuracy Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.plot(acc)
plt.show()
print(acc)


