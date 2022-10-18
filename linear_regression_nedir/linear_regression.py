# -*- coding: utf-8 -*-
"""
Created on Tue Aug  2 14:06:15 2022

@author: esmnralican
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Aug  2 14:06:15 2022

@author: esmnralican
"""

import pandas as pd
import matplotlib.pyplot as plt

#import library
df = pd.read_csv("linear_regression_dataset.csv",sep = ";")

#plot data
plt.scatter(df.Experience , df.Salary)
plt.xlabel("Experience")
plt.ylabel("Salary")
plt.show()

#%%
## Linear Regression

#sklearn library
from sklearn.linear_model import LinearRegression

#linear regression modeli linear_reg adlı değişkene eşitlendni.
linear_reg = LinearRegression()

#x = df.Experience # type : pandas
#y = df.Salary.values # type : numpy | x.shape = (14,) bu şekil kullanım sklearn için uygun değildir.
x = df.Experience.values.reshape(-1,1) # x.shape = (14,1)
y = df.Salary.values.reshape(-1,) # x.shape = (14,1)

linear_reg.fit(x,y)

#%% prediction
import numpy as np

b0 = linear_reg.predict([[0]])
print("b0: " , b0)

b0_ = linear_reg.intercept_ # y eksenini kestiği nokta (intercept)
print("b0_: " , b0_)

b1 = linear_reg.coef_ # eğim (slope)
print("b1: " , b1)

# salary = 16.63 + 1138 * experience
salary_new = 1663 + 1138*11
print(salary_new)

print(linear_reg.predict([[11]]))

#experience yani x ekseni
#array = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]) # array.shape çıktısı (16,) ve bu tanım sklearn için uygun değil 
array = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]).reshape(-1, 1) #  array.shape çıktısı (16,1) oldu

plt.scatter(x, y) # blue dots
plt.show()

y_head = linear_reg.predict(array) # approximately salary
plt.plot(array, y_head , color = "red")
linear_reg.predict([[11]]) # deneyim süresü 11 yıl olan çalışanın alacağı maaş yaklaşık olarak 14233.1468454 .


