import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics 
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

# 
# This initial part is loading up the data set and then mergeing them into one dataset. 
# we have to do this because we are using multiple years and each dataset is going to have to 
data1 = pd.read_csv('data/2000-01_Regular_box_scores.csv', sep=",")
data2 = pd.read_csv('data/2001-02_Regular_box_scores.csv', sep=",")
data3 = pd.read_csv('data/2002-03_Regular_box_scores.csv', sep=",")
data4 = pd.read_csv('data/2003-04_Regular_box_scores.csv', sep=",")
data5 = pd.read_csv('data/2004-05_Regular_box_scores.csv', sep=",")
data6 = pd.read_csv('data/2005-06_Regular_box_scores.csv', sep=",")
data7 = pd.read_csv('data/2006-07_Regular_box_scores.csv', sep=",")
data8 = pd.read_csv('data/2007-08_Regular_box_scores.csv', sep=",")
data9 = pd.read_csv('data/2008-09_Regular_box_scores.csv', sep=",")
data10 = pd.read_csv('data/2009-10_Regular_box_scores.csv', sep=",")
final_dataset = pd.concat([data1, data2, data3, data4, data5, data6, data7, data8, data9, data10], ignore_index=True)

# print(final_dataset.tail(10))
#
# what below is doing it dropping the columns that I do not think that we are going to need 
final_dataset = final_dataset.drop(columns=['TEAM', 'MATCH UP', 'GAME DATE', 'MIN'])

y = final_dataset[['W/L']]

# PTS = POINTS, FGM = FIELD GOALS MADE, FG% = FIELD GOAL PERCENTAGE, 3PM = THREE POINTS MADE, 3PA = THREE POINTS ATTEMPTED
# FTM = FREE THROWS MADE, FT% = FREE THROW PRECENTATGE, OREB = OFENSIVE REBOUNDS, DREB = DEFENSIVE REBOUNDS, AST = ASSISTS, 
# STL = STEALS, BLK = BLOCKS, TOV = TURNOVERS, PF = PERSONAL FOULS, =/- CALCULATED TEAM RATING (BASED OFF OF THAT GAME)
x = final_dataset[['PTS', 'FGM', 'FGA', 'FG%', '3PM', '3PA', '3P%', 'FTM', 'FT%', 'OREB', 'DREB', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'PF', '+/-']]

#Here we are finding the means or all the columns
MEAN = final_dataset.mean()
PTS_MEAN = MEAN['PTS']
FGM_MEAN = final_dataset['FGM'].mean()
print(PTS_MEAN)

ar = x 
#create a dataset with 1 for above mean and 0 for below mean. 
for column in ar[['PTS']]:
    columnSeriesObj = ar['PTS']
    print(columnSeriesObj.values)
    col_size = len(columnSeriesObj)
    print(col_size) 
    for iteration in range(col_size):
        if columnSeriesObj[iteration] > MEAN['PTS']:
            columnSeriesObj[iteration] = 1
        else:
            columnSeriesObj[iteration] = 0
    ar['PTS'] = columnSeriesObj
print(ar)


# for i in ar[['PTS', 'FGM', 'FGA', 'FG%', '3PM', '3PA', '3P%', 'FTM', 'FT%', 'OREB', 'DREB', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'PF', '+/-']]:
#     if ar[i].values > MEAN[i].values:
#         ar[i].values == 1
#     else:
#         ar[i].values == 0

# print(ar)