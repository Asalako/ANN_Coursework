# -*- coding: utf-8 -*-
"""
Created on Sun Mar 15 16:53:38 2020

@author: ayo-n
"""

import pandas as pd
import numpy as np

    
def standardisation(inputData, minimum, maximum):
    s = 0.8 * ( ( inputData - minimum) / (maximum - minimum) ) + 0.1
    return round(s,3)

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


class Perceptron:

    def __init__(self, data, output):
        self.inputs = data
        self.weights = np.random.uniform(low=-1, high=1, size=len(self.inputs))
        self.learningRate = 0.01
        self.target = output

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
    
    def feedForward(self):
        sumOf = 0
        for i in range( len(self.inputs) ):
            sumOf += self.inputs[i]*self.weights[i]
        return self.activationFunction(sumOf)

    def train(self):
        result = self.feedForward()
        error = self.target - result
        print(error)
        #tuning weights
        for i in range( len(self.weights)):
            self.weights[i] += error * self.inputs[i] * self.learningRate
      
    def activationFunction(self,n):
        if n >= 0:
            return 1
        else:
            return -1          
        
data = pd.read_excel(
    r'C:\Users\ayo-n\Documents\University\Lecture_Files\Year 2\Semester 2\AI\CW\ANNCW\DataWithoutErrors.xlsx')

columnNames = ["T", "W", "PanE"]
dataset = pd.DataFrame(data, columns=columnNames)
trainingSet = dataset.sample(n=3, replace=False)
trainingSet = standardiseDataset(trainingSet, columnNames)

inputSet = Perceptron.dictToList(trainingSet, ["T","W"])
outputSet = Perceptron.dictToList(trainingSet, ["PanE"])

print(trainingSet)
#print(inputSet)
#print(outputSet)

for epoch in range(100):
    for i, o, n in zip(inputSet, outputSet, range( len(inputSet) )):
        p = Perceptron(i, o)
        print(n,p.weights)
        p.train()
        print(p.weights)
print("\n", p.feedForward())



