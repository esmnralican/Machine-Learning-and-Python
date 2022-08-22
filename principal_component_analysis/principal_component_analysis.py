# -*- coding: utf-8 -*-
"""
Created on Wed Aug 17 15:25:09 2022

@author: esmnralican
"""

from sklearn .datasets import load_iris
import pandas as pd
#%%

iris = load_iris()
data = iris.data
feature_names = iris.feature_names
y = iris.target

df = pd.DataFrame(data, columns = feature_names) 
df["sinif"] = y

x = data 

#%%
# PCA
from sklearn.decomposition import PCA
pca = PCA(n_components = 2 , whiten=True) # n_components = 2 :orjinal data kaç boyuta düşürmek istenirse  | whiten=True : normalize eder.
pca.fit(x)

x_pca = pca.transform(x)

print("variance ratio : " , pca.explained_variance_ratio_)

# variance ratio :  [0.92461872 : p1   0.05306648 : p2]
print("sum : " , sum(pca.explained_variance_ratio_)) # 2 boyuttan tek boyuta indirildiğinde yalnızca %3 oranında veri kaybı yaşandı. 

#%%
# 2D
df["p1"] = x_pca[:,0]
df["p2"] = x_pca[:,1]

color = ["red" , "blue" , "pink"]

import matplotlib.pyplot as plt

for each in range(3):    # 3 farklı tür bulunmakta
    plt.scatter(df.p1[df.sinif == each],df.p2[df.sinif == each],color = color[each],label = iris.target_names[each])
    
plt.legend()
plt.xlabel("p1")
plt.ylabel("p2")
plt.show()    
    
























