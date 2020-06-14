import numpy as np
import pandas as pd
import DynamicNeuralNetwork as NN

# training_data = pd.read_csv("./MNIST_dataset/mnist_train.csv")
# test_data = pd.read_csv("./MNIST_dataset/mnist_test.csv")

X = np.array((
    [0,0],
    [0,1],
    [1,0],
    [1,1]),dtype = float)

Y = np.array((
    [0],
    [1],
    [1],
    [0]),dtype = float)

NN = NN.NeuralNetwork([2,3,5,3,1],0.1)
for i in range(10000):
    NN.train(X,Y)
    if(i%1000 == 0):
        print("Finished epoch %d" %i)

# NN.train(X,Y)
print(NN.predict(X))