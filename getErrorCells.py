# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
import numpy as np

#df.drop([...]) outliers

def sortBy(elem):
    return elem[2]

data = pd.read_excel(r'C:\Users\ayo-n\Documents\University\Lecture_Files\Year 2\Semester 2\AI\CW\ANNCW\Data.xlsx')
temp = pd.DataFrame(data, columns= ['T', 'W', 'SR', 'DSP', 'DRH', 'PanE']).to_dict()
#temp = temp['T']
print (len(temp), "This is the length of temp \n" )
listOfCells = []

columnLetter = "B"

for key in temp:
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
        
    columnData = temp[key]
    
    for i in range( len(columnData) - 1):
        value = columnData[i]
        if pd.isna(value) == True:
            listOfCells.append([columnLetter + str(i + 2), "blank", i+2])

        elif value == "a":
            listOfCells.append([columnLetter + str(i + 2), "letter a", i+2])

        elif abs(value) == 999:
            listOfCells.append([columnLetter + str(i + 2), "999", i+2])

#listOfCells.sort(key=sortBy, reverse=True)
        
for error in listOfCells:
    print (error[0], error[1])