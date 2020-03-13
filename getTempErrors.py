# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 12:24:10 2020

@author: ayo-n
"""

import pandas as pd
import numpy as np

def findInterQuartileRange(dataList):
    dataset = pd.DataFrame(dataList, columns=["value"])
    q1 = dataset['value'].quantile(0.25)
    q3 = dataset['value'].quantile(0.75)
    IQR = q3 - q1
    return [ round(q1 - 1.5*IQR, 3), round(q3 + 1.5*IQR,3) ]

def findExtremeTemperatures(data):
    temp = pd.DataFrame(data, columns= ['T']).to_dict()
    temp = temp['T']
    print (len(temp), "This is the length of temp \n" )
    listOfCells = []
    
    columnLetter = "B"
    highestValue = -999
    
    for i in range( len(temp)):
            value = temp[i]
            if value >= 56.7:
                listOfCells.append([columnLetter + str(i + 2),value, i+2])
            if value > highestValue and value <= 56.7:
                highestValue = value
                
    for error in listOfCells:
        print (error[0], error[1])
    print("highest value is ", highestValue)

data = pd.read_excel(
    r'C:\Users\ayo-n\Documents\University\Lecture_Files\Year 2\Semester 2\AI\CW\ANNCW\DataWithoutErrors.xlsx')

findExtremeTemperatures(data)
