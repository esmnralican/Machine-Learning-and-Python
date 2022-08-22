# -*- coding: utf-8 -*-
"""
Created on Mon Aug 15 16:22:42 2022

@author: esmnralican
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Aug 15 12:48:36 2022

@author: esmnralican
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#%%
data = pd.read_csv("data.csv")
#%%
data.drop(["id" , "Unnamed: 32"] , axis = 1 , inplace= True )

M = data[data.diagnosis == "M"]
B = data[data.diagnosis == "B"]

#%%
# scatter plot
plt.scatter(M.radius_mean , M.texture_mean , color = "red", label="kotu")
plt.scatter(B.radius_mean , B.texture_mean , color = "green", label="iyi")
plt.xlabel("radius_mean")
plt.ylabel("texture_mean")
plt.show().__bool__()

#%%data.diagnosis = [1 if each == "M" else 0 for each in data.diagnosis]
x_data = data.drop(["diagnosis"], axis=1)   # features
y = data.diagnosis.values    # labels

#%%
# Normalization
x = (x_data - np.min(x_data)) / (np.max(x_data) - np.min(x_data)) 

#%%
# train - test split
from sklearn.model_selection import train_test_split
x_train , x_test , y_train , y_test = train_test_split(x , y , test_size = 0.3 , random_state=42)

#%%
# Naive Bayes
from sklearn.naive_bayes import GaussianNB
nb = GaussianNB() 
nb.fit(x_train , y_train)
#%%
print(" accuracy of naive bayes algorithm ", nb.score(x_test , y_test))


