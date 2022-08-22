# -*- coding: utf-8 -*-
"""
Created on Wed Aug  3 11:46:11 2022

@author: esmnralican
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Aug  3 11:46:11 2022

@author: esmnralican
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

df = pd.read_csv("multiple_linear_regression_dataset.csv" , sep =";")


x = df.iloc[:,[0,2]].values
y = df.Salary.values.reshape(-1,1)
#%%
multiple_linear_regression = LinearRegression()
multiple_linear_regression.fit(x,y)

print("b0:" , multiple_linear_regression.intercept_ )
print("b1,b2", multiple_linear_regression.coef_ )

multiple_linear_regression.predict(np.array([[10,35] , [5,35]])) # 10 yıl deneyime sahip birey 11 bin 5 yıllık deneyime sahip birey 4 bin aldı.
