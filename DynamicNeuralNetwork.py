import numpy as np

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

class NeuralNetwork(object):
    def __init__(self, layers,lr):
        self.layers = layers
        self.learningRate = lr

        self.weights = []
        self.activations = []

        for i in range(len(self.layers)):
            self.activations.append([0])

        for i in range(len(layers)-1):
            self.weights.append(np.random.randn(self.layers[i], self.layers[i+1]))
    
    def sigmoid(self, x, deriv = False):
        if deriv == True:
            return x * (1-x)
        return 1/(1+np.exp(-x))
    
    def tanh(self, x, deriv = False):
        if deriv == True:
            return 1 - np.tanh(x)**2
        return np.tanh(x)
    
    def feedForward(self,inputs):
        self.inputs = inputs
        activatedInputs = inputs
        self.activations[0] = activatedInputs
        
        for i in range(len(self.weights)):
            dotProduct = activatedInputs.dot(self.weights[i]) 
            activatedInputs = self.tanh(dotProduct)
            self.activations[i+1] = activatedInputs
        
        outputs = self.activations[len(self.activations)-1]
        
        return outputs

    def backPropagate(self, outputs, expectedOutputs):
        error = expectedOutputs - outputs
        deltas = []
        for i in range(len(self.weights)):
            deltas.append([0])

        for i in range(len(self.weights)-1, -1, -1):
            delta = error * self.tanh(self.activations[i+1], deriv=True)
            deltas[len(self.weights)-1 -i] = delta
            if i>0:
                error = delta.dot(self.weights[i].T)

        deltas.reverse()

        for i in range(len(self.weights)):
            self.weights[i] += self.activations[i].T.dot(deltas[i]) * self.learningRate



    def train(self, inputs, expectedOutputs):
        outputs = self.feedForward(inputs)
        self.backPropagate(outputs, expectedOutputs)

    def predict(self, inputs):
        return self.feedForward(inputs)




# NN = NeuralNetwork([2,3,5,3,1],0.1)

# for i in range(10000):
#     NN.train(X,Y)
# print(NN.predict(X))


# NN.train(X,Y)
