# -*- coding: utf-8 -*-
"""
Created on Tue Aug 16 17:35:53 2022

@author: esmnralican
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Aug 16 15:48:53 2022

@author: esmnralican
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#%%
# create dataset

#class1
x1 = np.random.normal(25 , 5, 100) # 25 ortalama , 5 sigma değerine sahip 100 adet değer üretir
# oluşan datanın %66'sı 20 ve 30 arasında olucak. bu dağılıma Gaussian Dağılımı denir.
y1 = np.random.normal (25 , 5, 100)

#class2
x2 = np.random.normal (55 , 5, 100)
y2 = np.random.normal (60 , 5, 100)

#class3
x3 = np.random.normal (55 , 5, 100)
y3 = np.random.normal (15 , 5, 100)

x = np.concatenate((x1 , x2 , x3) , axis = 0)
y = np.concatenate((y1 , y2 , y3) , axis = 0)

dictionary = {"x":x , "y":y}

data = pd.DataFrame(dictionary) # data oluşturuldu

plt.scatter(x1 , y1)
plt.scatter(x2 , y2)
plt.scatter(x3 , y3)
plt.show()

#%%
# Algoritma verileri bu şekilde görüyor. Yani verilerin hangi sınıflara ait olduğu bilinmiyor. 
plt.scatter(x1 , y1 , color="black") 
plt.scatter(x2 , y2 , color="black")
plt.scatter(x3 , y3 , color="black")
plt.show()

#%%
#K-Means

from sklearn.cluster import KMeans
wcss = []

for k in range(1,15):
    kmeans = KMeans(n_clusters = k)
    kmeans.fit(data)
    wcss.append(kmeans.inertia_) # inertia_ : her bir k değeri için  wcss değerrini bulma.

plt.plot(range(1,15),wcss)
plt.xlabel("number of k(cluster) value ")
plt.ylabel("wcss")
plt.show() # k = 3 optimum değer

#%%
# k =3 için  
kmeans2 = KMeans(n_clusters=3)
clusters = kmeans2.fit_predict(data) # fit_predict : datayı fit et ,  fit edilen datayı al (kmeans2 modelini) dataya uygula.

data["label"] = clusters # data'ya label'ları ekledik

plt.scatter(data.x[data.label == 0], data.y[data.label == 0], color = "blue" )
plt.scatter(data.x[data.label == 1], data.y[data.label == 1], color = "green" )
plt.scatter(data.x[data.label == 2], data.y[data.label == 2], color = "pink" )
plt.scatter(kmeans2.cluster_centers_[:,0], kmeans2.cluster_centers_[:,1]  ,color = "yellow" )

plt.show()

#%%
# dendogram
from scipy.cluster.hierarchy import linkage , dendrogram

merg = linkage(data, method = "ward") # ward : cluster içindeki varyansları(yayılımları) küçültür
dendrogram(merg , leaf_rotation = 90)
plt.xlabel("data points")
plt.ylabel("euclidean distance")
plt.show()

#%%
#HC
from sklearn.cluster import AgglomerativeClustering

hierartical_cluster = AgglomerativeClustering(n_clusters = 3, affinity= "euclidean" , linkage="ward" )
cluster = hierartical_cluster.fit_predict(data)
 
data["label"] = cluster
plt.scatter(data.x[data.label == 0], data.y[data.label == 0], color = "blue" )
plt.scatter(data.x[data.label == 1], data.y[data.label == 1], color = "green" )
plt.scatter(data.x[data.label == 2], data.y[data.label == 2], color = "pink" )

plt.show()



