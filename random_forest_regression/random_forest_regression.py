# -*- coding: utf-8 -*-
"""
Created on Mon Aug  8 10:08:56 2022

@author: esmnralican
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("random_forest_regression_dataset.csv", sep =";" , header = None)

x = df.iloc[:,0].values.reshape(-1,1)
y = df.iloc[:,1].values.reshape(-1,1)

#%%
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators = 100 , random_state = 42) # n_estimators = 100 : number of tree |  random_forest : aynı random değerlerin tekrar seçilmesini sağlıyor. 
rf.fit(x,y)

print("7.5 seviyesinde fiyat değeri :" , rf.predict([[7.5]]))

x_ = np.arange(min(x), max(x), 0.01).reshape(-1,1)
y_head= rf.predict(x_)

#visualization
plt.scatter(x,y,color="red")
plt.plot(x_ , y_head , color="green")
plt.xlabel("tribun level")
plt.ylabel("ucret")
plt.show()