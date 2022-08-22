# -*- coding: utf-8 -*-
"""
Created on Thu Aug 18 11:06:03 2022

@author: esmnralican
"""


from sklearn.datasets import load_iris
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
#%%
iris = load_iris()

x = iris.data
y = iris.target


#%%
# normalization
x = (x - np.min(x)) / (np.max(x) - np.min(x))
#%%
# train test split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3,random_state=42)
#%%
#knn mmodel


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 3) # k : n_neigbours

#%%
# k fold cross validation  , k=10

from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = knn , X = x_train , y = y_train , cv=10)
print("average accuracy : ",np.mean(accuracies))
print("average std : ",np.std(accuracies))

#%%
knn.fit(x_train, y_train)
print("test accuracy : ", knn.score(x_test , y_test))

#%%
 # grid search cross validation
 
 from sklearn.model_selection import GridSearchCV
 
 grid = {"n_neighbors" : np.arange(1,50)}
 knn = KNeighborsClassifier()

knn_cv = GridSearchCV(knn , grid , cv=10 ) # grid serach cv
knn_cv.fit(x,y) 

#%%
#print  hyperparameter KNN algoritmasındaki k değeri
 
 print("tuned hyperparameter K :", knn_cv.best_params_)
 print("tuned paretreye göre en iyi accuracy (best score) : ", knn_cv.best_score_)
 
#%%
# grid search cv with logistic regression (binary classifier)

x = x[:100 , :]
y = y[:100]

from sklearn.linear_model import LogisticRegression
 
grid = {"C":np.logspace(-3 , 3 , 7) , "penalty":["l1", "l2"]} #  regülerazisyon -> L1 : lasso , L2 : ridge
# C değeri büyük ise overfit , küçük ise underfit gerçekleşir

logreg = LogisticRegression()   
logreg_cv = GridSearchCV(logreg , grid , cv = 10)
logreg_cv.fit(x,y)

print("tuned hyperparameters (best parameters): " , logreg_cv.best_params_)
print("accuracy : " , logreg_cv.best_score_  )
 
 
 #%%
logreg2 = LogisticRegression(C=1 , penalty="l1")
logreg2.fit(x_train , y_train)
print("score" , logreg2.score(x_test, y_test))
 
 
 
 
 
 
 
 
 
 
 