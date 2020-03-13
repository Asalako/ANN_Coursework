# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 00:23:27 2020

@author: ayo-n
"""


import pandas as pd
import numpy as np

def standardisation(inputData, minimum, maximum):
    s = 0.8 * ( ( inputData - minimum) / (maximum - minimum) ) + 0.1
    return round(s,3)

def standardiseDataset(data, columnNames):
    dataset = pd.DataFrame(data, columns=columnNames)
    for key in dataset:
        if key == "PanE":
            continue
        predictor = dataset[key]
        minimum = predictor.min()
        maximum = predictor.max()
        
        for i in range( len(predictor)):
            s = standardisation(predictor[i], minimum, maximum)
            predictor[i] = s
    
    print(dataset)
data = pd.read_excel(
    r'C:\Users\ayo-n\Documents\University\Lecture_Files\Year 2\Semester 2\AI\CW\ANNCW\DataWithoutErrors.xlsx')

standardiseDataset(data, ["T","W", "SR", "DSP", "DRH"])

