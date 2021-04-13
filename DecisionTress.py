#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Importing Python modules

from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.tree import export_graphviz
import pandas as pd
import numpy as np
import graphviz
import matplotlib.pyplot as plt
from sklearn import tree


# In[2]:


#Use pandas read_csv function to upload Bikebuyer data from the computer folder (Mac) to IPython. For other OS, please refer to Chapter 6 of Wes McKinney’s book on Python for Data Analysis

data=pd.read_csv("C:/Users/user/Downloads/Bikebuyer.csv")
#Extract relevant features from data
X=data[['MaritalStatus','YearlyIncome','TotalChildren','ChildrenAtHome','HouseOwnerFlag','NumberCarsOwned','Age']]
#Convert ‘MartitalStatus’ to nominal data (M, F to 0, 1)
X['MaritalStatus']=pd.get_dummies(X['MaritalStatus'])
#Assign Bikebuyer as target variable y
y=data['BikeBuyer']
#Split X into train and test data: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
#Specify decision tree model as dt using scikit learn “DecisionTreeClassifier” module: https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html#sklearn.tree.DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion='gini',
    splitter='best',
    max_depth=None,
    min_samples_split=800,
    min_samples_leaf=500,
    min_weight_fraction_leaf=0.0,
    max_features=None,
    random_state=None,
    max_leaf_nodes=None,
    min_impurity_decrease=0.0,
    min_impurity_split=None,
    class_weight=None,
    ccp_alpha=0.0)
#Fit the model using fit() method
dt.fit(X_train, y_train)


# In[3]:


#Determining Cost Complexity Parameter (ccp_alpha) for post pruning the tree: https://scikit-learn.org/stable/auto_examples/tree/plot_cost_complexity_pruning.html#sphx-glr-auto-examples-tree-plot-cost-complexity-pruning-py

path = dt.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas, impurities = path.ccp_alphas, path.impurities
#Using matplotlib.pyplot to plot the effect of varying ccp_alpha on error
fig, ax = plt.subplots()
ax.plot(ccp_alphas[:-1],impurities[:-1],marker='o', drawstyle="steps-post")
ax.set_xlabel("effective alpha")
ax.set_ylabel("total impurity of leaves")
ax.set_title("Total Impurity vs effective alpha for training set")
plt.plot(ccp_alphas, impurities)

dts = []
for ccp_alpha in ccp_alphas:
    dt = DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alpha)
    dt.fit(X_train, y_train)
    dts.append(dt)


# In[4]:


#Evaluates prediction accuracy and plots it against ccp_alphas

train_scores = [dt.score(X_train, y_train) for dt in dts]
test_scores = [dt.score(X_test, y_test) for dt in dts]
fig, ax = plt.subplots()
ax.set_xlabel("alpha")
ax.set_ylabel("accuracy")
ax.set_title("Accuracy vs alpha for training and testing sets")
ax.plot(ccp_alphas, train_scores, marker='o', label="train",
        drawstyle="steps-post")
ax.plot(ccp_alphas, test_scores, marker='o', label="test",
        drawstyle="steps-post")
ax.legend()
plt.show()


# In[5]:


#Building final tree

dt=DecisionTreeClassifier(criterion='gini',
    splitter='best',
    max_depth=None,
    min_samples_split=800,
    min_samples_leaf=500,
    min_weight_fraction_leaf=0.0,
    max_features=None,
    random_state=None,
    max_leaf_nodes=None,
    min_impurity_decrease=0.0,
    min_impurity_split=None,
    class_weight=None,
    ccp_alpha=0.0025)
dt.fit(X_train, y_train)


# In[6]:


#Visualizing the tree using graphviz

dot_data = tree.export_graphviz(dt, out_file=None,feature_names=X_train.columns,class_names=['NonBuyer','Buyer'],filled=True, rounded=True,special_characters=True) 
graph = graphviz.Source(dot_data)
graph


# In[7]:



from sklearn.metrics import confusion_matrix
y_pred=dt.predict(X_test)
from sklearn.metrics import classification_report
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
tn, fp, fn, tp
#print(classification_report(y_test, y_pred, target_names=['Buyer','Nonbuyer']))


# In[8]:


#Random Forest

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
X, y = make_classification(n_samples=1000, n_features=4,
                           n_informative=2, n_redundant=0,
                           random_state=0, shuffle=False)
#Generates 1000 subsamples for tree
dt=RandomForestClassifier(n_estimators=5, criterion='gini', 
                          max_depth=None, min_samples_split=800, 
                          min_samples_leaf=500, 	
                          min_weight_fraction_leaf=0.0, 
                          max_features='auto', 
                          max_leaf_nodes=None, 
                          min_impurity_decrease=0.0, 
                          min_impurity_split=None, 
                          bootstrap=True, oob_score=False, 
                          n_jobs=None, 
                          random_state=None, verbose=0, 
                          warm_start=False, 
                          class_weight=None, ccp_alpha=0.0,
                          max_samples=None)
dt.fit(X_train, y_train)
y_pred=dt.predict(X_test)
from sklearn.metrics import classification_report
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
tn, fp, fn, tp
print(classification_report(y_test, y_pred, target_names=['Buyer','Nonbuyer']))   


# In[9]:


#Cross Validation

#simple validation set approach: https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter  
#by default splits into five folds
from sklearn import model_selection
from sklearn.model_selection import cross_validate
from sklearn.metrics import recall_score
scores = cross_validate(dt, X_test, y_test)
k=scores['test_score']
k


# In[10]:


#K-fold

from sklearn.model_selection import KFold
kf = KFold(n_splits=500)
scores=[]
for train_index, test_index in kf.split(X,y):
    X_train, X_test, y_train, y_test = X[train_index], X[test_index],y[train_index], y[test_index]
    dt.fit(X_train, y_train)
    scores.append(dt.score(X_test, y_test)) 
scores


# In[11]:


data.head()


# In[ ]:




