# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 22:29:23 2020

@author: ayo-n
"""

import pandas as pd
import numpy as np

num = 0

def findInterQuartileRange(dataList, cn):
    q1 = dataList[cn].quantile(0.25)
    q3 = dataList[cn].quantile(0.75)
    IQR = q3 - q1
    return [ round(q1 - 1.5*IQR, 3), round(q3 + 1.5*IQR,3) ]

def findOutliersBySeason(season, cn):
    global num
    data = pd.read_excel(
        r'C:\Users\ayo-n\Documents\University\Lecture_Files\Year 2\Semester 2\AI\CW\ANNCW\DataWithoutErrors.xlsx'
        , season)

    column = pd.DataFrame(data, columns=[cn])
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
    
    qRange = findInterQuartileRange(column, cn)
    
    lqrt = qRange[0]
    uqrt = qRange[1]
    outliers = []
    
    for i in range(len(data[cn])):
        value = data[cn][i]
        if (value < lqrt ) or (value > uqrt):
            outliers.append( [value, columnLetter + str(i+2)] )
    
    print(lqrt, uqrt)
    for outlier in outliers:
        print(outlier)
    num += len(outliers)
        
findOutliersBySeason("Winter", "DRH")

seasons = ["Sheet1"]
#seasons = ["Winter", "Spring", "Summer", "Autumn"]
predictors = ["T", "W", "SR", "DSP", "DRH"]
for season in seasons:
    for predictor in predictors:
        print(season, predictor)
        findOutliersBySeason(season, predictor)
        print()
        
print(num)