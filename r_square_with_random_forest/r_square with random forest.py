# -*- coding: utf-8 -*-
"""
Created on Mon Aug  8 11:14:23 2022

@author: esmnralican
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Aug  8 10:08:56 2022

@author: esmnralican
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("r_square with random forest.csv", sep =";" , header = None)

x = df.iloc[:,0].values.reshape(-1,1)
y = df.iloc[:,1].values.reshape(-1,1)

#%%
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators = 100 , random_state = 42) # n_estimators = 100 : number of tree |  random_forest : aynı random değerlerin tekrar seçilmesini sağlıyor. 
rf.fit(x,y)

y_head = rf.predict=(x)

#%%

from sklearn.metrics import r2_score # modelleri değerlendiren şeyler metriklerdir

print("r_score: " , r2_score(y,y_head))


#%%
#r_score with linear regression
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
y = df.Salary.values.reshape(-1,1) # x.shape = (14,1)

linear_reg.fit(x,y)

y_head = linear_reg.predict(x) # maas

plt.plot(x, y_head , color="red") 
plt.show()

#%%
from sklearn.metrics import r2_score 

print("r_square score: " , r2_score(y,y_head))