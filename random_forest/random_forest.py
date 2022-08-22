# -*- coding: utf-8 -*-
"""
Created on Mon Aug 15 17:13:10 2022

@author: esmnralican
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Aug 15 16:43:13 2022

@author: esmnralican
"""

import pandas as pd
import numpy as np

#%%
data = pd.read_csv("data.csv")
data.drop(["id" , "Unnamed: 32"], axis=1 , inplace = True )

#%%
data.diagnosis = [ 1 if each == "M" else 0  for each in data.diagnosis]
y = data.diagnosis.values # class labelları y değerine atandı.
x_data = data.drop(["diagnosis"], axis=1) # diagnosis sütunu kaldırıldı ve içinde bulunan feature^lar x_data ifadesine aktarıldı. 

#%%
# normalization
x = (x_data - np.min(x_data)) / (np.max(x_data) - np.min(x_data)) 

#%%
# train test split
from sklearn.model_selection import train_test_split
x_train , x_test , y_train , y_test = train_test_split(x , y , test_size = 0.15 , random_state=42)

#%%
#decision tree
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()   
dt.fit(x_train , y_train)

print("decision tree score" , dt.score(x_test , y_test))

#%% 
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier( n_estimators= 100 , random_state =42 )
rf.fit(x_train , y_train)
print("random forest algorithm score" , rf.score(x_test , y_test))



