# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 00:23:27 2020

@author: ayo-n
"""


import pandas as pd
import numpy as np

#class Percepton:
    

class Mlp:
    
    def __init__(self, inputData, output):
        self.inputs = inputData
        self.desiredOutput = output
        network = {
            "inputNodes": 5,
            "outputNodes": 1,
            "hiddenNodes": 2
            }
        w1 = np.random.randn(network["inputNodes"], network["hiddenNodes"]) #input to hidden layer
        w2 = np.random.randn(network["hiddenNodes"], network["outputNodes"]) #hidden to output layer
        self.weights = [w1, w2]
        #self.weights = np.random.uniform(low=-1, high=1, size=len(self.inputs))
        self.learningRate = 0.01
        self.target = output
        self.hiddenBias = np.random.randn(network["hiddenNodes"], network["outputNodes"])
        self.outputBias = 1;
    
    def sigmoid(self,x):
        return 1/(1+np.exp(-x))
    
    def sigmoidDerivative(self,x):
        sig = self.sigmoid(x)
        sigD = sig * (1 - sig)
        return sigD
    
    def forwardPropagation(self):
        sumOf = np.dot(self.inputs, self.weights[0]) #dot product of input and set of weights
        #test = np.dot(self.inputs, self.hiddenBias)
        self.sigS = self.sigmoid(sumOf) #activation function
        s2 = np.dot(self.sigS, self.weights[1]) #dot product of hidden layer and set of weights
        
        return self.sigmoid(s2)
        #return test
    
# =============================================================================
#     def calSumBias(self):
#         
#         return
# =============================================================================
    
    #beckward propagate through the network
    def backwardPropagation(self, actualOutput):
        error = self.target - actualOutput #error in output
        deltaOutput = error * self.sigmoidDerivative(actualOutput)
        
        hiddenError = deltaOutput.dot(self.weights[1].T) #hidden layer weights output error
        deltaHidden = hiddenError * self.sigmoidDerivative(self.sigS) #apply derivative of sigmoid to hidden error
        
        print(inputMatrix, delta[0])
        for delta in deltaHidden:
            print(input)
            self.weights[0] += self.Inputs*delta #adjusting first set of weights
        self.weights[1] += self.sigS.T.dot(deltaOutput) #adjusting second set of weights
            
    
    def trainNetwork(self):
        output = self.forwardPropagation()
        self.backwardPropagation(output)
        return
 

def arrayCon(arr):
    array = [[elem] for elem in arr]
    return array
def dictToList(data, columns):
    li = []
    if len(columns) == 1:
        for row in range ( len(data) ):
            value = data.iloc[row][columns[0]]
            li.append(value)
        return li
            
    for row in range( len(data) ):
        record = []
        for column in columns:
            record.append(data.iloc[row][column])
        li.append(record)
    return li

def standardisation(inputData, minimum, maximum):
    s = 0.8 * ( ( inputData - minimum) / (maximum - minimum) ) + 0.1
    return round(s,3)

def standardiseDataset( data, columnNames):
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

columns = ["T","W", "SR", "DSP", "DRH", "PanE"]
dictToList(data, columns)

dataset = standardiseDataset(data, columns)

 

#print(w1, "\n", w2)
#print(weights)
#bias = [ 1 * 1 for i in range(len(dataset["T"]))]
#dataset["Bias"] = bias
# =============================================================================
# trainingSet = dataset.sample(frac=0.6, replace=False)
# validationSet = dataset.sample(frac=0.2, replace=False)
# testingSet =  dataset.sample(frac=0.2, replace=False)
# =============================================================================
trainingSet = dataset.sample(n=6, replace=False)

inputSet = dictToList(trainingSet, ["T", "W", "SR", "DSP", "DRH"])
outputSet = dictToList(trainingSet, ["PanE"])

for epoch in range(3):
    for i, o, n in zip(inputSet, outputSet, range( len(inputSet) )):
        p = Mlp(i, o)
        p.trainNetwork()
        print()
        print(p.weights[0])
