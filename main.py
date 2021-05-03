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
data11 = pd.read_csv('data/2010-11_Regular_box_scores.csv', sep=",")
data12 = pd.read_csv('data/2011-12_Regular_box_scores.csv', sep=",")
data13 = pd.read_csv('data/2013-14_Regular_box_scores.csv', sep=",")
data14 = pd.read_csv('data/2014-15_Regular_box_scores.csv', sep=",")
data15 = pd.read_csv('data/2015-16_Regular_box_scores.csv', sep=",")
data16 = pd.read_csv('data/2016-17_Regular_box_scores.csv', sep=",")
data17 = pd.read_csv('data/2017-18_Regular_box_scores.csv', sep=",")
data18 = pd.read_csv('data/2018-19_Regular_box_scores.csv', sep=",")
data19 = pd.read_csv('data/2019-20_Regular_box_scores.csv', sep=",")
final_dataset = pd.concat([data1, data2, data3, data4, data5, data6, data7, data8, data9, data10, data11, data12, data13, data14, data15, data16, data17, data18, data19], ignore_index=True)
print(final_dataset.tail(10))
#
# what below is doing it dropping the columns that I do not think that we are going to need 
final_dataset = final_dataset.drop(columns=['TEAM', 'MATCH UP', 'GAME DATE', 'MIN'])

y = final_dataset[['W/L']]

# PTS = POINTS, FGM = FIELD GOALS MADE, FG% = FIELD GOAL PERCENTAGE, 3PM = THREE POINTS MADE, 3PA = THREE POINTS ATTEMPTED
# FTM = FREE THROWS MADE, FT% = FREE THROW PRECENTATGE, OREB = OFENSIVE REBOUNDS, DREB = DEFENSIVE REBOUNDS, AST = ASSISTS, 
# STL = STEALS, BLK = BLOCKS, TOV = TURNOVERS, PF = PERSONAL FOULS, =/- CALCULATED TEAM RATING (BASED OFF OF THAT GAME)
x = final_dataset[['PTS', 'FGM', 'FGA', 'FG%', '3PM', '3PA', '3P%', 'FTM', 'FT%', 'OREB', 'DREB', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'PF', '+/-']]
# 
# This is splitting the merged datasets into tesing and training data sets. for this we are doing. 
# and 80 - 20 split. 80% of that data in training and 20% in testing. 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.20)
# print(x_train.head(10))
# print(y_train.head(10))
# 
# Naive bayes on the data 
gnb = GaussianNB()
gnb.fit(x_train, y_train)
y_prediction = gnb.predict(x_test)
y_prediction = np.ravel(y_prediction)
y_test = np.ravel(y_test)
print(y_prediction)
print(y_test)
# print(len(y_prediction))
# print(len(y_test))

#
#This is printing the accuracy of the naive bayes predictions. 
print("Accuracy of regular season predictions using naive bayes: ", metrics.accuracy_score(y_test, y_prediction))

#
# Here we are developing a confusion matrix so that we can know the amount of true positives etc. 
CMMultilabel = multilabel_confusion_matrix(y_test, y_prediction)
print(CMMultilabel)
CM = confusion_matrix(y_test, y_prediction)
print("Confusion Matrix Below")
print(CM)
TP = CM[0,0]
FN = CM[0,1]
FP = CM[1,0]
TN = CM[1,1]
print("True Positives: ", TP)
print("False Negatives: ", FP)
print("False Positives: ", FP)
print("True Negatives: ", TN)

#
# Here percision score of the test and actually predicted. 
precisionScore = precision_score(y_test, y_prediction, average='micro')
print("Precision Score: ", precisionScore)

#
# Here we are finding the recall 
recallScore = recall_score(y_test, y_prediction, average='micro') 
print("Recall Score: ", recallScore)

##
#user input section.

def naive_bayes_UI():
    num_Input = int(input("How many categories : "))
    cat = []
    for i in range(num_Input):
        userInput = input("Category: ")
        cat.append(userInput)

    input_dataset = final_dataset[cat]
    print(input_dataset)

    x_input_train, x_input_test, y_input_train, y_input_test = train_test_split(input_dataset, y, test_size=.20)
    # Naive bayes on the data 
    gnb = GaussianNB()
    gnb.fit(x_input_train, y_input_train)
    y_input_prediction = gnb.predict(x_input_test)
    y_input_prediction = np.ravel(y_input_prediction)
    y_input_test = np.ravel(y_input_test)
    print(y_input_prediction)
    print(y_input_test)

    #
    #This is printing the accuracy of the naive bayes predictions. 
    print("Accuracy of regular season predictions using naive bayes: ", metrics.accuracy_score(y_input_test, y_input_prediction))

    #
    # Here we are developing a confusion matrix so that we can know the amount of true positives etc. 
    CMMultilabel = multilabel_confusion_matrix(y_input_test, y_input_prediction)
    print(CMMultilabel)
    CM = confusion_matrix(y_input_test, y_input_prediction)
    print("Confusion Matrix Below")
    print(CM)
    TP = CM[0,0]
    FN = CM[0,1]
    FP = CM[1,0]
    TN = CM[1,1]
    print("True Positives: ", TP)
    print("False Negatives: ", FP)
    print("False Positives: ", FP)
    print("True Negatives: ", TN)

    #
    # Here percision score of the test and actually predicted. 
    precisionScore = precision_score(y_input_test, y_input_prediction, average='micro')
    print("Precision Score: ", precisionScore)

    #
    # Here we are finding the recall 
    recallScore = recall_score(y_input_test, y_input_prediction, average='micro') 
    print("Recall Score: ", recallScore)


# naive_bayes_UI()

repeat = "Y"

while repeat[0] not in ("n", "N"): 
    naive_bayes_UI()
    repeat = input("Would you like run this again? (y/n): ")

# Thinking we may want to do something like maybe using a season as a training data and playoffs as a test and see how it works 
# Best teams head to head not sure really how to do this. 
# thinking for this we train the data on what we have. so all of the seasons, when we make the test 



## if shoot above a certain percentage are you more likely to win. 