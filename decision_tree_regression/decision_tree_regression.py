# -*- coding: utf-8 -*-
"""
Created on Fri Aug  5 14:10:03 2022

@author: esmnralican
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("decision_tree_regression_dataset.csv" ,sep = ";" , header = None)

x = df.iloc[:,0].values.reshape(-1,1)
y = df.iloc[:,1].values.reshape(-1,1)

#%%
#Decision Tree Regression
from sklearn.tree import DecisionTreeRegressor
tree_reg = DecisionTreeRegressor()  
tree_reg.fit(x,y)

tree_reg.predict([[5.5]])
x_ = np.arange(min(x), max(x), 0.01).reshape(-1,1)
y_head = tree_reg.predict(x_)


#%%
#visualization
plt.scatter(x,y, color="red")
plt.plot(x_, y_head , color ="green" )
plt.xlabel("tribun level")
plt.ylabel("price")
plt.show()

 
