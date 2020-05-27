#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas 
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier
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


rfc=RandomForestClassifier(random_state=42)


# In[8]:


param_grid = { 
    'n_estimators': [64, 70,76,82,88,94,100,106,112,118,124,128],
    'max_features': ['auto', 'sqrt', 'log2']
}


# In[9]:


CV_rfcP = GridSearchCV(estimator=rfc, param_grid=param_grid, cv= 5)
CV_rfcP.fit(X_trainP, y_trainP)

CV_rfcM = GridSearchCV(estimator=rfc, param_grid=param_grid, cv= 5)
CV_rfcM.fit(X_trainM, y_trainM)

CV_rfcMP = GridSearchCV(estimator=rfc, param_grid=param_grid, cv= 5)
CV_rfcMP.fit(X_trainMP, y_trainMP)


# In[10]:


CV_rfcP.best_params_


# In[11]:


CV_rfcM.best_params_


# In[12]:


CV_rfcMP.best_params_


# In[13]:


regressorP = RandomForestClassifier(random_state=87, criterion= 'gini', max_features= 'auto', n_estimators= 128)
regressorP.fit(X_trainP, y_trainP)

regressorM = RandomForestClassifier(random_state=87, criterion= 'gini', max_features= 'auto', n_estimators= 106)
regressorM.fit(X_trainM, y_trainM)

regressorMP = RandomForestClassifier(random_state=87, criterion= 'gini', max_features= 'auto', n_estimators= 124)
regressorMP.fit(X_trainMP, y_trainMP)


# In[14]:


y_predP = regressorP.predict(X_testP)
y_predM = regressorM.predict(X_testM)
y_predMP = regressorMP.predict(X_testMP)
print("Only phone Accuracy for Random Forest on CV data: ",accuracy_score(y_testP, y_predP))
print("only mood Accuracy for Random Forest on CV data: ",accuracy_score(y_testM, y_predM))
print("MOODPHONE Accuracy for Random Forest on CV data: ",accuracy_score(y_testMP, y_predMP))


# In[ ]:




