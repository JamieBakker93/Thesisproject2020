#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas 
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import pandas as pd
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler


# In[ ]:


ONLYPHONE = pandas.read_csv('ONLYPHONE.csv')


# In[ ]:


train = ONLYPHONE[0:3223]
test = ONLYPHONE[3224:4029]
print(train)


# In[ ]:


train_y = train['Outcome_variable']
train_x = train
train_x.drop(['user_date_time', 'Outcome_variable'], axis=1, inplace=True)
test_id = test['user_date_time']
del test['user_date_time']


# In[ ]:


rf_clf = RandomForestClassifier(n_estimators=25, random_state=10)

# Train the model
rf_clf.fit(train_x,train_y)


# In[ ]:


imp_feat_rf = pd.Series(rf_clf.feature_importances_, index=train_x.columns).sort_values(ascending=False)
imp_feat_rf[:40].plot(kind='bar', title='Feature Importance Only Phone', figsize=(12,8))
plt.ylabel('Feature Importance values')
plt.subplots_adjust(bottom=0.25)
plt.savefig('FeatImportance.png')
plt.show()


# In[ ]:


# Save indexes of the important features in descending order of their importance
indices = np.argsort(rf_clf.feature_importances_)[::-1]

# list the names of the names of top 25 selected features adn remove the unicode
select_feat =[str(s) for s in train_x.columns[indices][:40]]




# In[ ]:


print(select_feat)


# # # METHOD 2

# In[ ]:


# Feature Extraction with RFE
from pandas import read_csv
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
# load data
dataframe = ONLYPHONE
array = dataframe.values
X = array[:,1:25]
Y = array[:,25]
Y = Y.astype('int')
# feature extraction
model = LogisticRegression(solver='lbfgs')
rfe = RFE(model, 15)
fit = rfe.fit(X, Y)

import warnings
from sklearn.exceptions import ConvergenceWarning
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=ConvergenceWarning)

print("Num Features: %d" % fit.n_features_)
print("Selected Features: %s" % fit.support_)
print("Feature Ranking: %s" % fit.ranking_)



# In[ ]:


# Feature Importance with Extra Trees Classifier
from pandas import read_csv
from sklearn.ensemble import ExtraTreesClassifier
model = ExtraTreesClassifier(n_estimators=500)
model.fit(X, Y)
print(model.feature_importances_)


# # METHOD 3
# 

# In[ ]:



from sklearn.datasets import make_classification
from sklearn.ensemble import ExtraTreesClassifier



# Build a forest and compute the feature importances
forest = ExtraTreesClassifier(n_estimators=250,
                              random_state=0)

forest.fit(X, Y)
importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

for f in range(X.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

# Plot the feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.bar(range(X.shape[1]), importances[indices],
       color="r", yerr=std[indices], align="center")
plt.xticks(range(X.shape[1]), indices)
plt.xlim([-1, X.shape[1]])
plt.show()

