
import pandas as pd
import numpy as np

class Mlp:

    def __init__(self, inputData, output, nodes=2, lp=0.1):
        self.size = 2.5
        self.inputs = inputData
        self.desiredOutput = np.array(arrayCon(output), dtype=float)
        self.learningRate = lp

        self.epoch = 0
        network = {
            "inputNodes": 5,
            "outputNodes": 1,
            "hiddenNodes": 2
        }
        network["hiddenNodes"] = nodes
        w1 = np.random.uniform(-self.size, self.size, ( network["inputNodes"], network["hiddenNodes"]) ) #input to hidden layer between given size
        w2 = np.random.uniform(-self.size, self.size, ( network["hiddenNodes"], network["outputNodes"]) )#hidden to output layer between given size
        # w1 = np.array([[3, 6], [4, 5]], dtype=float)  # weights for a node. inputs --> hidden node
        # w2 = np.array([2, 4], dtype=float)
        self.weights = [w1, w2]
        # self.hiddenNodes = np.array([1, -6], dtype=float)  # set to random after
        self.hiddenNodes = np.random.uniform( -self.size, self.size, network["hiddenNodes"] )
        # self.outputNode = [-3.92]
        self.outputNode = np.random.uniform(-self.size, self.size, size=1)

    def output(self):
        print(self.weights[0])
        print(self.weights[1])
        print(self.hiddenNodes)
        print(self.outputNode)


    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoidDerivative(self, x):
        sigD = x * (1 - x)
        return sigD

    def f(self):
        Sj = np.dot(self.inputs, self.weights[0]) + self.hiddenNodes
        uj = self.sigmoid(Sj)
        return uj

    # def feedForward(self, node):
    #     Sj = 0
    #     for i in range( len(self.inputs) ):
    #         Sj += self.inputs[i] * self.weights[0][node][i]
    #     Sj += self.hiddenNodes[node] * self.bias
    #     uj = self.sigmoid(Sj)
    #     return uj

    def backwardPass(self, activations):
        roundNumber = 4
        deltaOutputs = []
        #Finding delta for the output node
        sigODervivative = self.sigmoidDerivative(self.sigmoidOutput)
        deltaO = (self.desiredOutput - self.sigmoidOutput) * sigODervivative

        hiddenError = deltaO * self.weights[1].T
        hiddenDelta =  hiddenError * self.sigmoidDerivative(activations)

        # for i in range(len(activations)):
        #     sigDervivative = self.sigmoidDerivative(activations[i])
        #     delta = self.weights[1][i] * deltaO * sigDervivative
        #     deltaOutputs.append(delta)

        #Updating all weights
        #updating output node
        #print(self.outputNode, deltaO)
        self.outputNode = np.add(self.outputNode, self.learningRate * deltaO)
        #print(self.outputNode)
        self.hiddenNodes = np.add(self.hiddenNodes, self.learningRate * hiddenDelta)
        #print("next",self.hiddenNodes)
        #print(self.weights[1], hiddenDelta, activations )
        self.weights[1] = np.add(self.weights[1].T, self.learningRate * deltaO * activations)
        print(self.weights[1].shape)
        #updating hidden nodes and weights from hidden layer --> output node
        self.weights[0] = np.add(self.weights[0], self.learningRate * hiddenDelta * self.inputs.T)
        #updating weights from inputs --> hidden layer
        #print(hiddenDelta, deltaO)
        #print(self.desiredOutput - self.sigO)
        #print(self.sigO)
        self.mse()
        return

    def trainNetwork(self):
        activations = self.f()
        sumSo = np.dot(activations, self.weights[1]) + self.outputNode
        self.sigmoidOutput = self.sigmoid(sumSo)
        self.backwardPass(activations)
        return

    def mse(self):
        sumDesiredOutput = np.sum(self.desiredOutput, axis=0, dtype=float)
        sumError = np.sum(np.subtract(self.desiredOutput, self.sigmoidOutput))
        mse = sumError**2 / len(self.desiredOutput)
        print(mse)


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

# print(w1, "\n", w2)
# print(weights)
# bias = [ 1 * 1 for i in range(len(dataset["T"]))]
# dataset["Bias"] = bias
# =============================================================================
# trainingSet = dataset.sample(frac=0.6, replace=False)
# validationSet = dataset.sample(frac=0.2, replace=False)
# testingSet =  dataset.sample(frac=0.2, replace=False)
# =============================================================================
trainingSet = dataset.sample(frac=0.6, replace=False)

inputSet = dictToList(trainingSet, ["T", "W", "SR", "DSP", "DRH"])
outputSet = dictToList(trainingSet, ["PanE"])

# inputSet = [[1, 0], [2,1]]
# outputSet = [1,1]
inputSet = np.array(inputSet)
outputSet = np.array(outputSet)

epochs = []
results = []
p = Mlp(inputSet, outputSet)
for j in range(1):
    p.trainNetwork()
    # p.output()
    #if j % 100 == 0:
