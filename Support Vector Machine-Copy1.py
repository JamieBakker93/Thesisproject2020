#!/usr/bin/env python
# coding: utf-8

# In[17]:


import pandas 
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.svm import SVC
from sklearn import svm
from sklearn.model_selection import GridSearchCV


# In[18]:


ONLYPHONE = pandas.read_csv('ONLYPHONE.csv')
ONLYMOOD  = pandas.read_csv('ONLYMOOD.csv')
MOODPHONE = pandas.read_csv('MOODPHONE.csv')


# In[19]:


col = ['totaltime', 'Phone_Tools', 'WhatsApp Messenger', 'Instagram', 'Social_Networking', 'Facebook', 'Google Chrome', 'Snapchat', 'Entertainment', 'YouTube', 'Internet_Browser', 'Camera', 'Spotify', 'Email', 'News', 'Facebook Messenger', 'Instant_Messaging', 'Google Search', 'Phone', 'Office', 'Other',"Outcome_variable"]
ONLYPHONE = ONLYPHONE[col]

col = ['Cheerful', 'Energy_level', 'Tired', 'Content', 'totaltime', 'Gloomy', 'WhatsApp Messenger', 'Energetic', 'Phone_Tools', 'enjoy', 'Bored', 'Anxious', 'Facebook', 'Stressed', 'Upset', 'Instagram', 'Calm', 'Snapchat', 'Google Chrome', 'inferior', 'Spotify', 'Social_Networking', 'Email', 'Camera', 'Entertainment',"Outcome_variable"]
MOODPHONE = MOODPHONE[col]


# In[20]:


XP = ONLYPHONE.iloc[:, 1:21].values
yP = ONLYPHONE.iloc[:,21].values
yP = yP.astype('int')
print(yP)

XM = ONLYMOOD.iloc[:, 1:15].values
yM = ONLYMOOD.iloc[:,15].values
yM = yM.astype('int')


XMP = MOODPHONE.iloc[:, 1:25].values
yMP = MOODPHONE.iloc[:,25].values
yMP = yMP.astype('int')


# In[27]:


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


# In[30]:


scalingP = MinMaxScaler(feature_range=(0,1)).fit(X_trainP)
X_trainP = scalingP.transform(X_trainP)
X_testP = scalingP.transform(X_testP)

scalingM = MinMaxScaler(feature_range=(0,1)).fit(X_trainM)
X_trainM = scalingM.transform(X_trainM)
X_testM = scalingM.transform(X_testM)

scalingMP = MinMaxScaler(feature_range=(0,1)).fit(X_trainMP)
X_trainMP = scalingMP.transform(X_trainMP)
X_testMP = scalingMP.transform(X_testMP)


# In[31]:


svclassifierP = SVC(kernel = 'rbf', gamma = 'scale', decision_function_shape = 'ovo')
svclassifierP.fit(X_trainP, y_trainP)

svclassifierM = SVC(kernel = 'rbf', gamma = 'scale', decision_function_shape = 'ovo')
svclassifierM.fit(X_trainM, y_trainM)

svclassifierMP = SVC(kernel = 'rbf', gamma = 'scale', decision_function_shape = 'ovo')
svclassifierMP.fit(X_trainMP, y_trainMP)

y_predP = svclassifierP.predict(X_testP)
y_predM = svclassifierM.predict(X_testM)
y_predMP = svclassifierMP.predict(X_testMP)
#print(confusion_matrix(y_testP,y_predP))
#print(classification_report(y_testP,y_predP))
print("Only phone Accuracy for Support vector Machine on CV data: ",accuracy_score(y_testP, y_predP))
print("only mood Accuracy for Support vector Machine on CV data: ",accuracy_score(y_testM, y_predM))
print("MOODPHONE Accuracy for Support vector Machine on CV data: ",accuracy_score(y_testMP, y_predMP))


# In[32]:


svclassifierP =  SVC(kernel='rbf') 
svclassifierP.fit(X_trainP, y_trainP)

svclassifierM = SVC(kernel='rbf') 
svclassifierM.fit(X_trainM, y_trainM)

svclassifierMP = SVC(kernel='rbf') 
svclassifierMP.fit(X_trainMP, y_trainMP)

y_predP = svclassifierP.predict(X_testP)
y_predM = svclassifierM.predict(X_testM)
y_predMP = svclassifierMP.predict(X_testMP)
#print(confusion_matrix(y_testP,y_predP))
#print(classification_report(y_testP,y_predP))
print("Only phone Accuracy for Support vector Machine on CV data: ",accuracy_score(y_testP, y_predP))
print("only mood Accuracy for Support vector Machine on CV data: ",accuracy_score(y_testM, y_predM))
print("MOODPHONE Accuracy for Support vector Machine on CV data: ",accuracy_score(y_testMP, y_predMP))


# In[33]:


svclassifierP =  SVC(gamma = 'auto', decision_function_shape = 'ovr')
svclassifierP.fit(X_trainP, y_trainP)

svclassifierM = SVC(gamma = 'auto', decision_function_shape = 'ovr')
svclassifierM.fit(X_trainM, y_trainM)

svclassifierMP = SVC(gamma = 'auto', decision_function_shape = 'ovr')
svclassifierMP.fit(X_trainMP, y_trainMP)

y_predP = svclassifierP.predict(X_testP)
y_predM = svclassifierM.predict(X_testM)
y_predMP = svclassifierMP.predict(X_testMP)
#print(confusion_matrix(y_testP,y_predP))
#print(classification_report(y_testP,y_predP))
print("Only phone Accuracy for Support vector Machine on CV data: ",accuracy_score(y_testP, y_predP))
print("only mood Accuracy for Support vector Machine on CV data: ",accuracy_score(y_testM, y_predM))
print("MOODPHONE Accuracy for Support vector Machine on CV data: ",accuracy_score(y_testMP, y_predMP))


# In[34]:


svclassifierP =  SVC(kernel='poly', degree=8)
svclassifierP.fit(X_trainP, y_trainP)

svclassifierM = SVC(kernel='poly', degree=8)
svclassifierM.fit(X_trainM, y_trainM)

svclassifierMP = SVC(kernel='poly', degree=8)
svclassifierMP.fit(X_trainMP, y_trainMP)

y_predP = svclassifierP.predict(X_testP)
y_predM = svclassifierM.predict(X_testM)
y_predMP = svclassifierMP.predict(X_testMP)
#print(confusion_matrix(y_testP,y_predP))
#print(classification_report(y_testP,y_predP))
print("Only phone Accuracy for Support vector Machine on CV data: ",accuracy_score(y_testP, y_predP))
print("only mood Accuracy for Support vector Machine on CV data: ",accuracy_score(y_testM, y_predM))
print("MOODPHONE Accuracy for Support vector Machine on CV data: ",accuracy_score(y_testMP, y_predMP))


# In[35]:


svclassifierP =  SVC(kernel='linear')
svclassifierP.fit(X_trainP, y_trainP)

svclassifierM = SVC(kernel='linear')
svclassifierM.fit(X_trainM, y_trainM)

svclassifierMP = SVC(kernel='linear')
svclassifierMP.fit(X_trainMP, y_trainMP)

y_predP = svclassifierP.predict(X_testP)
y_predM = svclassifierM.predict(X_testM)
y_predMP = svclassifierMP.predict(X_testMP)
#print(confusion_matrix(y_testP,y_predP))
#print(classification_report(y_testP,y_predP))
print("Only phone Accuracy for Support vector Machine on CV data: ",accuracy_score(y_testP, y_predP))
print("only mood Accuracy for Support vector Machine on CV data: ",accuracy_score(y_testM, y_predM))
print("MOODPHONE Accuracy for Support vector Machine on CV data: ",accuracy_score(y_testMP, y_predMP))


# In[36]:


svclassifierP =  svm.LinearSVC(multi_class = 'crammer_singer')
svclassifierP.fit(X_trainP, y_trainP)

svclassifierM = svm.LinearSVC(multi_class = 'crammer_singer')
svclassifierM.fit(X_trainM, y_trainM)

svclassifierMP = svm.LinearSVC(multi_class = 'crammer_singer')
svclassifierMP.fit(X_trainMP, y_trainMP)

y_predP = svclassifierP.predict(X_testP)
y_predM = svclassifierM.predict(X_testM)
y_predMP = svclassifierMP.predict(X_testMP)
#print(confusion_matrix(y_testP,y_predP))
#print(classification_report(y_testP,y_predP))
print("Only phone Accuracy for Support vector Machine on CV data: ",accuracy_score(y_testP, y_predP))
print("only mood Accuracy for Support vector Machine on CV data: ",accuracy_score(y_testM, y_predM))
print("MOODPHONE Accuracy for Support vector Machine on CV data: ",accuracy_score(y_testMP, y_predMP))


# In[37]:


svclassifierP =  SVC(gamma = 'auto', decision_function_shape = 'ovo')
svclassifierP.fit(X_trainP, y_trainP)

svclassifierM = SVC(gamma = 'auto', decision_function_shape = 'ovo')
svclassifierM.fit(X_trainM, y_trainM)

svclassifierMP = SVC(gamma = 'auto', decision_function_shape = 'ovo')
svclassifierMP.fit(X_trainMP, y_trainMP)

y_predP = svclassifierP.predict(X_testP)
y_predM = svclassifierM.predict(X_testM)
y_predMP = svclassifierMP.predict(X_testMP)
#print(confusion_matrix(y_testP,y_predP))
#print(classification_report(y_testP,y_predP))
print("Only phone Accuracy for Support vector Machine on CV data: ",accuracy_score(y_testP, y_predP))
print("only mood Accuracy for Support vector Machine on CV data: ",accuracy_score(y_testM, y_predM))
print("MOODPHONE Accuracy for Support vector Machine on CV data: ",accuracy_score(y_testMP, y_predMP))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




