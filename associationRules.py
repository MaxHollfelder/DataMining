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
from pyECLAT import ECLAT

data1 = pd.read_csv('data/2000-01_Regular_box_scores.csv', sep=",", header=None)
data2 = pd.read_csv('data/2001-02_Regular_box_scores.csv', sep=",", header=None)
data3 = pd.read_csv('data/2002-03_Regular_box_scores.csv', sep=",", header=None)
data4 = pd.read_csv('data/2003-04_Regular_box_scores.csv', sep=",", header=None)
data5 = pd.read_csv('data/2004-05_Regular_box_scores.csv', sep=",", header=None)
data6 = pd.read_csv('data/2005-06_Regular_box_scores.csv', sep=",", header=None)
data7 = pd.read_csv('data/2006-07_Regular_box_scores.csv', sep=",", header=None)
data8 = pd.read_csv('data/2007-08_Regular_box_scores.csv', sep=",", header=None)
data9 = pd.read_csv('data/2008-09_Regular_box_scores.csv', sep=",", header=None)
data10 = pd.read_csv('data/2009-10_Regular_box_scores.csv', sep=",", header=None)
final_dataset = pd.concat([data1, data2, data3, data4, data5, data6, data7, data8, data9, data10], ignore_index=True)

final_dataset = final_dataset.drop(final_dataset.columns[[0, 1, 2, 4]], axis=1)
print(final_dataset)

wins = final_dataset[3] == 'W'
print(wins)
only_wins = final_dataset[wins]
print(only_wins)
ar = only_wins.drop(only_wins.columns[[0]], axis=1)
pts_mean = ar.columns[5]
print(pts_mean)
print(ar)
MEAN = ar[[5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]].mean()
print(MEAN)

# for i in ar.index:
#     if ar.at[i, 'PTS'] > MEAN['PTS']:
#         ar.at[i, 'PTS'] = 'PTS'
#     else:
#         ar.at[i, 'PTS'] = 'NaN'
        
#     if ar.at[i,'FGM'] > MEAN['FGM']:
#         ar.at[i, 'FGM'] = 1
#     else:
#         ar.at[i, 'FGM'] = 'NaN'
        
#     if ar.at[i,'FGA'] > MEAN['FGA']:
#         ar.at[i, 'FGA'] = 1
#     else:
#         ar.at[i, 'FGA'] = 'NaN'
            
#     if ar.at[i,'FG%'] > MEAN['FG%']:
#         ar.at[i, 'FG%'] = 1
#     else:
#         ar.at[i, 'FG%'] = 'NaN'
                
#     if ar.at[i,'3PM'] > MEAN['3PM']:
#         ar.at[i, '3PM'] = 1
#     else:
#         ar.at[i, '3PM'] = 'NaN'
                    
#     if ar.at[i,'3PA'] > MEAN['3PA']:
#         ar.at[i, '3PA'] = 1
#     else:
#         ar.at[i, '3PA'] = 'NaN'
                
#     if ar.at[i,'3P%'] > MEAN['3P%']:
#         ar.at[i, '3P%'] = 1
#     else:
#         ar.at[i, '3P%'] = 'NaN'
            
#     if ar.at[i,'FTM'] > MEAN['FTM']:
#         ar.at[i, 'FTM'] = 1
#     else:
#         ar.at[i, 'FTM'] = 'NaN'

#     if ar.at[i,'FTA'] > MEAN['FTA']:
#         ar.at[i, 'FTA'] = 1
#     else:
#         ar.at[i, 'FTA'] = 'NaN'
                    
#     if ar.at[i,'FT%'] > MEAN['FT%']:
#         ar.at[i, 'FT%'] = 1
#     else:
#         ar.at[i, 'FT%'] = 'NaN'
                            
#     if ar.at[i,'OREB'] > MEAN['OREB']:
#         ar.at[i, 'OREB'] = 1
#     else:
#         ar.at[i, 'OREB'] = 'NaN'
                                    
#     if ar.at[i,'DREB'] > MEAN['DREB']:
#         ar.at[i, 'DREB'] = 1
#     else:
#         ar.at[i, 'DREB'] = 'NaN'
                                            
#     if ar.at[i,'REB'] > MEAN['REB']:
#         ar.at[i, 'REB'] = 1
#     else:
#         ar.at[i, 'REB'] = 'NaN'
                                                    
#     if ar.at[i,'AST'] > MEAN['AST']:
#         ar.at[i, 'AST'] = 1
#     else:
#         ar.at[i, 'AST'] = 'NaN'
        
#     if ar.at[i,'STL'] > MEAN['STL']:
#         ar.at[i, 'STL'] = 1
#     else:
#         ar.at[i, 'STL'] = 'NaN'
               
#     if ar.at[i,'BLK'] > MEAN['BLK']:
#         ar.at[i, 'BLK'] = 1
#     else:
#         ar.at[i, 'BLK'] = 'NaN'
                       
#     if ar.at[i,'TOV'] > MEAN['TOV']:
#         ar.at[i, 'TOV'] = 1
#     else:
#         ar.at[i, 'TOV'] = 'NaN'
                           
#     if ar.at[i,'PF'] > MEAN['PF']:
#         ar.at[i, 'PF'] = 1 
#     else:
#         ar.at[i, 'PF'] = 'NaN'
                           
#     if ar.at[i,'+/-'] > MEAN['+/-']:
#         ar.at[i, '+/-'] = 1
#     else:
#         ar.at[i, '+/-'] = 'NaN'


transactions = [
    ['Milk', 'Bread', 'Saffron'],
    ['Milk', 'Saffron'],
    ['Bread', 'Saffron','Wafer'],
    ['Bread','Wafer'],
 ]

# print(transactions)
# import associationRules
# from associationRules import eclat

# data1ECLAT = pd.read_csv('data/2000-01_Regular_box_scores.csv', sep=",")
# eclat_dataset = data1ECLAT
# eclat_instance = ECLAT(data=transactions, verbose=True)
# eclat_instance.df_bin   #generate a binary dataframe, that can be used for other analyzes.
# eclat_instance.uniq_    #a list with all the names of the different items
# get_ECLAT_indexes, get_ECLAT_supports = eclat_instance.fit(min_support=0.08,
#                                                            min_combination=1,
#                                                            max_combination=3,
#                                                            separator=' & ',
#                                                            verbose=True)
# print(get_ECLAT_indexes)
# print(get_ECLAT_supports)

# get_ECLAT_indexes, get_ECLAT_supports = ar.fit(min_support=0.10,min_combination=1, max_combination=3,separator=' & ',verbose=True)

# Now from here we are actually going to have to do one of the association minging rule algorithms on 
# this dataset. 
# run with no headers, drop the first row, drip the columns that are not needed, run through the ifs and elses. 

