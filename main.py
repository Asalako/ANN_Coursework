
import pandas as pd
import numpy as np


class NeuralNetwork(object):
    def __init__(self):
        # parameters
        self.inputSize = 5
        self.outputSize = 1
        self.hiddenSize = 5
        self.learningRate = 0.01

        # weights
        self.W1 = np.random.randn(self.inputSize, self.hiddenSize)  # (3x2) weight matrix from input to hidden layer
        self.W2 = np.random.randn(self.hiddenSize, self.outputSize)  # (3x1) weight matrix from hidden to output layer

    def feedForward(self, X):
        # forward propogation through the network
        self.z = np.dot(X, self.W1)  # dot product of X (input) and first set of weights (3x2)
        self.z2 = self.sigmoid(self.z)  # activation function
        self.z3 = np.dot(self.z2, self.W2)  # dot product of hidden layer (z2) and second set of weights (3x1)
        output = self.sigmoid(self.z3)
        return output

    def sigmoid(self, s, deriv=False):
        if (deriv == True):
            return s * (1 - s)
        return 1 / (1 + np.exp(-s))

    def backward(self, X, y, output):
        # backward propogate through the network
        self.output_error = y.T - output  # error in output
        self.output_delta = self.output_error * self.sigmoid(output, deriv=True)

        self.z2_error = self.output_delta.dot(
            self.W2.T)  # z2 error: how much our hidden layer weights contribute to output error
        self.z2_delta = self.z2_error * self.sigmoid(self.z2, deriv=True)  # applying derivative of sigmoid to z2 error

        self.W1 += X.T.dot(self.z2_delta) * self.learningRate  # adjusting first set (input -> hidden) weights
        self.W2 += self.z2.T.dot(self.output_delta) * self.learningRate # adjusting second set (hidden -> output) weights
        #print(self.mse(self.output_error))

    def train(self, X, y):
        output = self.feedForward(X)
        self.backward(X, y, output)

    def testModel(self, inputSet, desiredOutput):
        desiredOutput = np.array([desiredOutput])
        actualOutput = self.feedForward(inputSet)
        outputError = desiredOutput.T - actualOutput
        return self.mse(outputError)

    def mse(self, error):
        # sum = 0
        # for e in error:
        #     sum += e[0]
        print(len(error))
        mse = (np.sum(error)**2)/len(error)

        return mse

def arrayCon(arr):
    array = [[elem] for elem in arr]
    return array


def dictToList(data, columns):
    li = []
    if len(columns) == 1:
        for row in range(len(data)):
            value = data.iloc[row][columns[0]]
            li.append(value)
        return li

    for row in range(len(data)):
        record = []
        for column in columns:
            record.append(data.iloc[row][column])
        li.append(record)
    return li


def standardisation(inputData, minimum, maximum):
    s = 0.8 * ((inputData - minimum) / (maximum - minimum)) + 0.1
    return round(s, 3)


def standardiseDataset(data, columnNames):
    dataset = pd.DataFrame(data, columns=columnNames)
    dataDict = dataset.to_dict()
    for key in dataset:
        predictor = dataDict[key]
        minimum = dataset[key].min()
        maximum = dataset[key].max()

        for key in predictor:
            s = standardisation(predictor[key], minimum, maximum)
            predictor[key] = s

    return pd.DataFrame(dataDict)



data = pd.read_excel(
    r'C:\Users\ayo-n\Documents\University\Lecture_Files\Year 2\Semester 2\AI\CW\ANNCW\DataWithoutErrors.xlsx')

columns = ["T", "W", "SR", "DSP", "DRH", "PanE"]
dictToList(data, columns)

dataset = standardiseDataset(data, columns)

trainingSet = dataset.sample(frac=0.6, replace=False)
validationSet = dataset.sample(frac=0.2, replace=False)
testingSet =  dataset.sample(frac=0.2, replace=False)
testInput = dictToList(testingSet, ["T", "W", "SR", "DSP", "DRH"])
testOutput = dictToList(testingSet, ["PanE"])

inputSet = dictToList(trainingSet, ["T", "W", "SR", "DSP", "DRH"])
outputSet = dictToList(trainingSet, ["PanE"])

# inputSet = [[1, 0], [2,1]]
# outputSet = [1,1]
inputSet = np.array(inputSet)
outputSet = np.array([outputSet])

NN = NeuralNetwork()

for i in range(10000): #trains the NN 1000 times
    # if (i % 100 == 0):
    #     print("Loss: " + str(np.mean(np.square(y - NN.feedForward(X)))))
    NN.train(inputSet, outputSet)
print("test", NN.testModel(testInput, testOutput))

# epochs = []
# results = []
# p = Mlp(inputSet, outputSet)
# for j in range(1):
#     p.trainNetwork()
    # p.output()
    #if j % 100 == 0:
