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
from pprint import pprint

## just to get the means for each columns (of which is a winning column)
mean1 = pd.read_csv('data/2000-01_Regular_box_scores.csv', sep=",")
mean2 = pd.read_csv('data/2001-02_Regular_box_scores.csv', sep=",")
mean3 = pd.read_csv('data/2002-03_Regular_box_scores.csv', sep=",")
mean4 = pd.read_csv('data/2003-04_Regular_box_scores.csv', sep=",")
mean5 = pd.read_csv('data/2004-05_Regular_box_scores.csv', sep=",")
mean6 = pd.read_csv('data/2005-06_Regular_box_scores.csv', sep=",")
mean7 = pd.read_csv('data/2006-07_Regular_box_scores.csv', sep=",")
mean8 = pd.read_csv('data/2007-08_Regular_box_scores.csv', sep=",")
mean9 = pd.read_csv('data/2008-09_Regular_box_scores.csv', sep=",")
mean10 = pd.read_csv('data/2009-10_Regular_box_scores.csv', sep=",")
mean_dataset = pd.concat([mean1, mean2, mean3, mean4, mean5, mean6, mean7, mean8, mean9, mean10], ignore_index=True)

mean_dataset = mean_dataset.drop(columns=['TEAM', 'MATCH UP', 'GAME DATE', 'MIN'])
wins = mean_dataset['W/L'] == 'W'
only_wins = mean_dataset[wins]
MEAN = mean_dataset.mean()
ar_means = only_wins.drop(columns= ['W/L'])
MEAN = ar_means.mean()
print(ar_means)
## end of just finding the means 

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
wins = final_dataset[3] == 'W'
only_wins = final_dataset[wins]
ar = only_wins.drop(only_wins.columns[[0]], axis=1)
ar = ar.reset_index()
del ar['index']
ar = ar.T.reset_index(drop=True).T
# ar.columns=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]
print(ar)
mean_index = MEAN.reset_index()
del mean_index['index']

for i in range(19):
    for j in ar.index:
        if float(ar[i][j]) > float(mean_index[0][i]):
            if i == 0:
                ar[i][j] = 'PTS'
            elif i == 1:
                ar[i][j] = 'FGM'
            elif i == 2:
                ar[i][j] = 'FGA'
            elif i == 3:
                ar[i][j] = 'FG%'
            elif i == 4:
                ar[i][j] = '3PM'
            elif i == 5:
                ar[i][j] = '3PA'
            elif i == 6:
                ar[i][j] = '3P%'
            elif i == 7:
                ar[i][j] = 'FTM'
            elif i == 8:
                ar[i][j] = 'FTA'
            elif i == 9:
                ar[i][j] = 'FT%'
            elif i == 10:
                ar[i][j] = 'OREB'
            elif i == 11:
                ar[i][j] = 'DREB'
            elif i == 12:
                ar[i][j] = 'REB'
            elif i == 13:
                ar[i][j] = 'AST'
            elif i == 14:
                ar[i][j] = 'STL'
            elif i == 15:
                ar[i][j] = 'BLK'
            elif i == 16:
                ar[i][j] = 'TOV'
            elif i == 17:
                ar[i][j] = 'PF'
            elif i == 18:
                ar[i][j] = '+/-'
        else:
            ar[i][j] = 'NaN'

print(ar)


eclat_instance = ECLAT(data=ar, verbose=True)

eclat_instance.df_bin
eclat_instance.uniq_

get_ECLAT_indexes, get_ECLAT_supports = eclat_instance.fit(min_support=0.37,
                                                           min_combination=1,
                                                           max_combination=19,
                                                           separator=' & ',
                                                           verbose=True)
print("Here are the combinations and the supports that they had.")
pprint(get_ECLAT_supports)




