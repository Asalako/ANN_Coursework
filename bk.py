# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 00:23:27 2020

@author: ayo-n
"""

import pandas as pd
import numpy as np

class Mlp:

    def __init__(self, inputData, output, nodes=2, lp=0.1):
        self.size = 2.5
        self.inputs = inputData
        self.desiredOutput = output
        self.learningRate = lp

        self.epoch = 0
        self.bias = 1
        network = {
            "inputNodes": 5,
            "outputNodes": 1,
            "hiddenNodes": 2
        }
        network["hiddenNodes"] = nodes
        w1 = np.random.uniform(-self.size, self.size, ( network["hiddenNodes"], network["inputNodes"]) ) #input to hidden layer
        w2 = np.random.uniform(-self.size, self.size,  network["hiddenNodes"] ) #hidden to output layer
        self.weights = [w1, w2]
        self.hiddenNodes = np.random.uniform( -self.size, self.size, network["hiddenNodes"] )
        self.outputNode = np.random.uniform(-self.size, self.size, size=1)[0]

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoidDerivative(self, x):
        sigD = x * (1 - x)
        return sigD

    def feedForward(self, node):
        Sj = 0
        for i in range( len(self.inputs) ):
            Sj += self.inputs[i] * self.weights[0][node][i]
        Sj += self.hiddenNodes[node] * self.bias
        uj = self.sigmoid(Sj)
        return uj

    def backwardPass(self, activations):
        roundNumber = 4
        deltaOutputs = []
        sigODervivative = self.sigmoidDerivative(self.sigO)
        deltaO = (self.desiredOutput - self.sigO) * sigODervivative

        for i in range(len(activations)):
            sigDervivative = self.sigmoidDerivative(activations[i])
            delta = self.weights[1][i] * deltaO * sigDervivative
            deltaOutputs.append(delta)

        #Updating all weights
        #updating output node
        self.outputNode += self.learningRate * deltaO * self.bias
        self.outputNode = round(self.outputNode, roundNumber)

        #updating hidden nodes and weights from hidden layer --> output node
        for i in range( len(deltaOutputs) ):
            delta = deltaOutputs[i]
            self.hiddenNodes[i] += self.learningRate * delta * self.bias
            self.hiddenNodes[i] = round(self.hiddenNodes[i], roundNumber)
            self.weights[1][i] += self.learningRate * delta * activations[i]
            self.weights[1][i] = round(self.weights[1][i], roundNumber)

        #updating weights from inputs --> hidden layer
        for i in range(len(self.weights[0])):
            delta = deltaOutputs[i]
            for j in range(len(self.weights[0][0])):
                weight = self.weights[0][i]
                weight[j] += self.learningRate * delta * self.inputs[j]
                weight[j] = round(weight[j], roundNumber)

        #print(self.desiredOutput - self.sigO)
        #print(self.sigO)
        self.epoch += 1
        return

    def trainNetwork(self):
        changed = False
        activations = []
        for i in range(len(self.hiddenNodes)):
            activations.append(self.feedForward(i))

        sumSo = np.dot(activations, self.weights[1]) + self.outputNode * self.bias
        self.sigO = self.sigmoid(sumSo)
        self.backwardPass(activations)
        return

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

#def Mse(data):

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

# inputSet = [inputSet[0]]
# outputSet = [outputSet[0]]

epochs = []
results = []
nodes = [2, 4, 6, 8, 10]
lps = [0.1, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01]
for node in nodes:
    for lp in lps:
        for i, o in zip(inputSet, outputSet):
            prevSig = -999
            p = Mlp(i, o, node, lp)
            for j in range(10000):
                p.trainNetwork()
                if j % 100 == 0:
                    print(p.desiredOutput - p.sigO)
                if prevSig != p.sigO:
                    prevSig = p.sigO
                else:
                    epochs.append(p.epoch)
                    print(p.epoch)
                    break
        results.append([node, lp, sum(epochs)/len(epochs)])

best = 100000
best_result = []
for result in results:
    print("Number of nodes:", result[0])
    print("Learning Rate:", result[1])
    print("Avg Epoch:", result[2])
    if result[2] < best:
        best_result = result