# -*- coding: utf-8 -*-
"""
Created on Wed Aug  3 12:43:25 2022

@author: esmnralican
"""

import pandas as pd
import matplotlib.pyplot as plt

df= pd.read_csv("polynomial_linear_regression_dataset.csv", sep = ";")

y = df.max_car_speed.values.reshape(-1,1)
x = df.car_price.values.reshape(-1,1)

plt.scatter(x, y)
plt.ylabel("max_car_speed")
plt.xlabel("car_price")
plt.show()

# linear regression = y = b0 + b1*x
# multiple linear regression = y = b0 + b1*x1 + b2*x2 + ... bn*xn

#%%
#linear regression
from sklearn.linear_model import LinearRegression

lr = LinearRegression()

lr.fit(x,y)

#%%
y_head = lr.predict(x)
plt.plot(x, y_head , color="red", label="linear") # y_head'i plot ettirdik
plt.show()

lr.predict([[1000]]) # veriler lineer değil bu nedenle bu polynomial yöntem kullanılmalı.


#%% polynomial linear regression = y = b0 + b1*x + b2*x^2 + ... + bn * x^n

from sklearn.preprocessing import PolynomialFeatures
polynomial_regression = PolynomialFeatures(degree = 2)

x_polynomial = polynomial_regression.fit_transform(x) # model oluşturuluyor

#%% fit
linear_regression2 = LinearRegression()
linear_regression2.fit(x_polynomial , y)

#%%
#visualization
y_head2 = linear_regression2.predict(x_polynomial)
plt.plot(x , y_head , color = "green" , label = "poly")
plt.legend()
plt.show()
S