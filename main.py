# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 00:23:27 2020

@author: ayo-n
"""

import pandas as pd
import numpy as np


# class Percepton:

# =============================================================================
# class Node:
#     def __init__(self, inputData, output, weight):
#         self.weightsToNode = 
# =============================================================================

class Mlp:

    def __init__(self, inputData, output):
        self.inputs = inputData
        self.desiredOutput = output
        network = {
            "inputNodes": 2,
            "outputNodes": 1,
            "hiddenNodes": 2
        }

        self.hiddenNodes = [1, -6]  # set to random after
        w1 = [[3, 4], [6, 5]]  # weights for a node. inputs --> hidden node
        w2 = [2, 4]
        self.weights = [w1, w2]
        self.outputNode = -3.92
        self.learningRate = 0.1
        self.bias = 1
        self.epoch = 0

    # =============================================================================
    #         w1 = np.random.randn(network["inputNodes"], network["hiddenNodes"]) #input to hidden layer
    #         w2 = np.random.randn(network["hiddenNodes"], network["outputNodes"]) #hidden to output layer
    #         self.weights = [w1, w2]
    #         self.learningRate = 0.01
    #         self.target = output
    #         self.hiddenBias = np.random.randn(network["hiddenNodes"], network["outputNodes"])
    #         self.outputBias = 1;
    # =============================================================================

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

    # =============================================================================
    #     def forwardPropagation(self):
    #         sumOf = np.dot(self.inputs, self.weights[0]) #dot product of input and set of weights
    #         #test = np.dot(self.inputs, self.hiddenBias)
    #         self.sigS = self.sigmoid(sumOf) #activation function
    #         s2 = np.dot(self.sigS, self.weights[1]) #dot product of hidden layer and set of weights
    #
    #         return self.sigmoid(s2)
    # =============================================================================

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

        print(self.weights, self.hiddenNodes, self.outputNode)
        print(self.sigO)
        self.epoch += 1
        return

    # backward propagate through the network
    def backwardPropagation(self, actualOutput):
        error = self.target - actualOutput  # error in output
        deltaOutput = error * self.sigmoidDerivative(actualOutput)

        hiddenError = deltaOutput.dot(self.weights[1].T)  # hidden layer weights output error
        deltaHidden = hiddenError * self.sigmoidDerivative(self.sigS)  # apply derivative of sigmoid to hidden error

        # print(inputMatrix, delta[0])
        for delta in deltaHidden:
            self.weights[0] += self.Inputs * delta  # adjusting first set of weights
        self.weights[1] += self.sigS.T.dot(deltaOutput)  # adjusting second set of weights

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
trainingSet = dataset.sample(n=6, replace=False)

inputSet = dictToList(trainingSet, ["T", "W", "SR", "DSP", "DRH"])
outputSet = dictToList(trainingSet, ["PanE"])

inputSet = [[1, 0]]
outputSet = [1]

prevSig = -999
for epoch in range(1):
    for i, o, n in zip(inputSet, outputSet, range(len(inputSet))):
        p = Mlp(i, o)
        for i in range(20000):
            p.trainNetwork()
            if prevSig != p.sigO:
                prevSig = p.sigO
            else:
                print(p.epoch)
                break
        print()
