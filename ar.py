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
FGM_MEAN = MEAN['FGM']
FGM_MEAN = final_dataset['FGM'].mean()
print(PTS_MEAN)

ar = x 
greater = 0 
less = 0
for i in ar.index:
    if ar.at[i, 'PTS'] > MEAN['PTS']:
        greater += 1
        ar.at[i, 'PTS'] = 1
    else:
        less += 1
        ar.at[i, 'PTS'] = 0
        
    if ar.at[i,'FGM'] > MEAN['FGM']:
        ar.at[i, 'FGM'] = 1
    else:
        ar.at[i, 'FGM'] = 0
        
    if ar.at[i,'FGA'] > MEAN['FGA']:
        ar.at[i, 'FGA'] = 1
    else:
        ar.at[i, 'FGA'] = 0
            
    if ar.at[i,'FG%'] > MEAN['FG%']:
        ar.at[i, 'FG%'] = 1
    else:
        ar.at[i, 'FG%'] = 0
                
    if ar.at[i,'3PM'] > MEAN['3PM']:
        ar.at[i, '3PM'] = 1
    else:
        ar.at[i, '3PM'] = 0
                    
    if ar.at[i,'3PA'] > MEAN['3PA']:
        ar.at[i, '3PA'] = 1
    else:
        ar.at[i, '3PA'] = 0
                
    if ar.at[i,'3P%'] > MEAN['3P%']:
        ar.at[i, '3P%'] = 1
    else:
        ar.at[i, '3P%'] = 0
            
    if ar.at[i,'FTM'] > MEAN['FTM']:
        ar.at[i, 'FTM'] = 1
    else:
        ar.at[i, 'FTM'] = 0
                    
    if ar.at[i,'FT%'] > MEAN['FT%']:
        ar.at[i, 'FT%'] = 1
    else:
        ar.at[i, 'FT%'] = 0
                            
    if ar.at[i,'OREB'] > MEAN['OREB']:
        ar.at[i, 'OREB'] = 1
    else:
        ar.at[i, 'OREB'] = 0
                                    
    if ar.at[i,'DREB'] > MEAN['DREB']:
        ar.at[i, 'DREB'] = 1
    else:
        ar.at[i, 'DREB'] = 0
                                            
    if ar.at[i,'REB'] > MEAN['REB']:
        ar.at[i, 'REB'] = 1
    else:
        ar.at[i, 'REB'] = 0
                                                    
    if ar.at[i,'AST'] > MEAN['AST']:
        ar.at[i, 'AST'] = 1
    else:
        ar.at[i, 'AST'] = 0
        
    if ar.at[i,'STL'] > MEAN['STL']:
        ar.at[i, 'STL'] = 1
    else:
        ar.at[i, 'STL'] = 0
               
    if ar.at[i,'BLK'] > MEAN['BLK']:
        ar.at[i, 'BLK'] = 1
    else:
        ar.at[i, 'BLK'] = 0
                       
    if ar.at[i,'TOV'] > MEAN['TOV']:
        ar.at[i, 'TOV'] = 1
    else:
        ar.at[i, 'TOV'] = 0
                           
    if ar.at[i,'PF'] > MEAN['PF']:
        ar.at[i, 'PF'] = 1
    else:
        ar.at[i, 'PF'] = 0
                           
    if ar.at[i,'+/-'] > MEAN['+/-']:
        ar.at[i, '+/-'] = 1
    else:
        ar.at[i, '+/-'] = 0

print(greater)
print(less)
print(ar)
#create a dataset with 1 for above mean and 0 for below mean. 
# for column in ar[['PTS']]:
#     columnSeriesObj = ar['PTS']
#     print(columnSeriesObj.values)
#     col_size = len(columnSeriesObj)
#     print(col_size) 
#     for iteration in range(col_size):
#         if columnSeriesObj[iteration] > MEAN['PTS']:
#             columnSeriesObj[iteration] = 1
#         else:
#             columnSeriesObj[iteration] = 0
#     ar['PTS'] = columnSeriesObj
# print(ar)


# for i in ar[['PTS', 'FGM', 'FGA', 'FG%', '3PM', '3PA', '3P%', 'FTM', 'FT%', 'OREB', 'DREB', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'PF', '+/-']]:
#     if ar[i].values > MEAN[i].values:
#         ar[i].values == 1
#     else:
#         ar[i].values == 0

# print(ar)