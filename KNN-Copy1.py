#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV


# In[2]:


ONLYPHONE = pandas.read_csv('ONLYPHONE.csv')
ONLYMOOD  = pandas.read_csv('ONLYMOOD.csv')
MOODPHONE = pandas.read_csv('MOODPHONE.csv')


# In[3]:


col = ['totaltime', 'Phone_Tools', 'WhatsApp Messenger', 'Instagram', 'Social_Networking', 'Facebook', 'Google Chrome', 'Snapchat', 'Entertainment', 'YouTube', 'Internet_Browser', 'Camera', 'Spotify', 'Email', 'News', 'Facebook Messenger', 'Instant_Messaging', 'Google Search', 'Phone', 'Office', 'Other',"Outcome_variable"]
ONLYPHONE = ONLYPHONE[col]

col = ['Cheerful', 'Energy_level', 'Tired', 'Content', 'totaltime', 'Gloomy', 'WhatsApp Messenger', 'Energetic', 'Phone_Tools', 'enjoy', 'Bored', 'Anxious', 'Facebook', 'Stressed', 'Upset', 'Instagram', 'Calm', 'Snapchat', 'Google Chrome', 'inferior', 'Spotify', 'Social_Networking', 'Email', 'Camera', 'Entertainment',"Outcome_variable"]
MOODPHONE = MOODPHONE[col]


# In[4]:


XP = ONLYPHONE.iloc[:, 1:21].values
yP = ONLYPHONE.iloc[:,21].values
yP = yP.astype('int')

XM = ONLYMOOD.iloc[:, 1:15].values
yM = ONLYMOOD.iloc[:,15].values
yM = yM.astype('int')


XMP = MOODPHONE.iloc[:, 1:25].values
yMP = MOODPHONE.iloc[:,25].values
yMP = yMP.astype('int')


# In[5]:


from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=10)
skf.get_n_splits(XMP, yMP)
StratifiedKFold(n_splits=10, random_state=None, shuffle=False)
for train_index, test_index in skf.split(XMP, yMP):
    X_trainMP, X_testMP = XMP[train_index], XMP[test_index]
    y_trainMP, y_testMP = yMP[train_index], yMP[test_index]

skf.get_n_splits(XM, yM)
StratifiedKFold(n_splits=10, random_state=None, shuffle=False)
for train_index, test_index in skf.split(XM, yM):
    X_trainM, X_testM = XM[train_index], XM[test_index]
    y_trainM, y_testM = yM[train_index], yM[test_index]

skf.get_n_splits(XP, yP)
StratifiedKFold(n_splits=10, random_state=None, shuffle=False)
for train_index, test_index in skf.split(XP, yP):
    X_trainP, X_testP = XP[train_index], XP[test_index]
    y_trainP, y_testP = yP[train_index], yP[test_index]


# In[6]:


scalingP = MinMaxScaler(feature_range=(-1,1)).fit(X_trainP)
X_trainP = scalingP.transform(X_trainP)
X_testP = scalingP.transform(X_testP)

scalingM = MinMaxScaler(feature_range=(-1,1)).fit(X_trainM)
X_trainM = scalingM.transform(X_trainM)
X_testM = scalingM.transform(X_testM)

scalingMP = MinMaxScaler(feature_range=(-1,1)).fit(X_trainMP)
X_trainMP = scalingMP.transform(X_trainMP)
X_testMP = scalingMP.transform(X_testMP)


# In[7]:


knn = KNeighborsClassifier()
knn.fit(X_trainP, y_trainP)
print('test score', knn.score(X_testP, y_testP))


# In[8]:


knn = KNeighborsClassifier()
knn.fit(X_trainM, y_trainM)
print('test score', knn.score(X_testM, y_testM))


# In[9]:


knn = KNeighborsClassifier()
knn.fit(X_trainMP, y_trainMP)
print('test score', knn.score(X_testMP, y_testMP))


# In[10]:



k_range = list(range(1, 39))
param_grid = dict(n_neighbors=k_range, weights= ['uniform',"distance"], metric= ['euclidean', 'manhattan' ])


# In[11]:


grid = GridSearchCV(knn, param_grid, cv=10, scoring='accuracy')


# In[12]:


gs_resultsP = grid.fit(X_trainP, y_trainP)
gs_resultsP.best_params_


# In[13]:


gs_resultsM = grid.fit(X_trainM, y_trainM)
gs_resultsM.best_params_


# In[14]:


gs_resultsMP = grid.fit(X_trainMP, y_trainMP)
gs_resultsMP.best_params_


# In[15]:


## Train ONLYPHONE
classifierP = KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='manhattan',
                     metric_params=None, n_jobs=None, n_neighbors=30,
                     weights='distance')
classifierP.fit(X_trainP, y_trainP)

## Train ONLYMOOD
classifierM = KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='manhattan',
                     metric_params=None, n_jobs=None, n_neighbors=27,
                     weights='uniform')
classifierM.fit(X_trainM, y_trainM)

## Train MOODDPHONE
classifierMP = KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='manhattan',
                     metric_params=None, n_jobs=None, n_neighbors=28,
                     weights='uniform')
classifierMP.fit(X_trainMP, y_trainMP)


# In[16]:


y_predP = classifierP.predict(X_testP)
y_predM = classifierM.predict(X_testM)
y_predMP = classifierMP.predict(X_testMP)


# In[17]:


#print(confusion_matrix(y_testP,y_predP))
#print(classification_report(y_testP,y_predP))
print("Only phone Accuracy for KNN on CV data: ",accuracy_score(y_testP, y_predP))
print("only mood Accuracy for KNN on CV data: ",accuracy_score(y_testM, y_predM))
print("MOODPHONE Accuracy for KNN on CV data: ",accuracy_score(y_testMP, y_predMP))


# In[ ]:





# In[ ]:




