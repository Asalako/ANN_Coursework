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
    
    for i in range( len(temp) - 1):
            value = temp[i]
            if value >= 56.7:
                listOfCells.append([columnLetter + str(i + 2),value, i+2])
            if value > highestValue and value <= 56.7:
                highestValue = value
                
    for error in listOfCells:
        print (error[0], error[1])
    print("highest value is ", highestValue)

def findOutlier(data, cn):
    column = pd.DataFrame(data, columns= ['Date',cn]).to_dict()
    listOfCells = []
    columnLetter = "B"
    
    for key in column:
        if key == "W":
            columnLetter = "C"
        elif key == "SR":
            columnLetter = "D"
        elif key == "DSP":
            columnLetter = "E"
        elif key == "DRH":
            columnLetter = "F"        
        elif key == "PanE":
            columnLetter = "G"
        else:
            continue
    
    month = data['Date'][0].month
    i = 0
    listOfDataInMonth = []
    while i <= (len(data['Date']) - 1) :
        date = data['Date'][i]
        if month != date.month:
            qRange = findInterQuartileRange(listOfDataInMonth)
            listOfCells.append([qRange, columnLetter + str(i+2), month])
            listOfDataInMonth = []
            month = date.month
        
        listOfDataInMonth.append(data[cn][i])
        i+= 1
    
    qRange = findInterQuartileRange(listOfDataInMonth)
    listOfCells.append([qRange, columnLetter + str(i+2), month])

    outliers = []
    month = data['Date'][0].month
    i = 0
    lqrt = listOfCells[0][0][0]
    uqrt = listOfCells[0][0][1]
    
    for x in range(len(data['Date'])):
        date = data['Date'][i]
        value = data['W'][i]
        if date.month == listOfCells[0][2]:
            if (value < lqrt ) and (value > uqrt):
                outliers.append([value, listOfCells[0][1]])
        else:
            listOfCells.pop(0)
            lqrt = listOfCells[0][0][0]
            uqrt = listOfCells[0][0][1]
            if (value < lqrt ) and (value > uqrt):
                outliers.append([value, listOfCells[0][1]])
        
        #lqrt = listOfCells[0][0][0]
        #uqrt = listOfCells[0][0][1]
        #print(lqrt, uqrt, i)
        i+= 1
        
    for outlier in outliers:
        print(outlier)
    
        
data = pd.read_excel(r'C:\Users\ayo-n\Documents\University\Lecture_Files\Year 2\Semester 2\AI\CW\ANNCW\Data.xlsx')

findExtremeTemperatures(data)
#findOutlier(data, 'W')
